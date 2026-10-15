"""Price data layer: yfinance download, on-disk cache and offline demo data.

The app must never hard-fail because of a network problem, so the loading
strategy is layered:

1. read per-ticker CSV cache from ``cache_dir`` (fast, works offline);
2. refresh stale/missing tickers from Yahoo Finance via ``yfinance``;
3. if the network is unavailable, fall back to the stale cache;
4. if nothing real is available at all, synthesise a clearly-labelled *demo*
   book (deterministic, seed-based, with an embedded seasonal component).
   A missing ticker is dropped — never filled with a fake series next to
   real prices, which would invent a rotation.

The returned :class:`MarketData` records which path was taken so charts and
reports can label the data source honestly.
"""

from __future__ import annotations

import hashlib
import logging
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

LOG = logging.getLogger(__name__)

__all__ = ["MarketData", "fetch_prices", "synthetic_prices"]

DEFAULT_CACHE_DIR = Path(".cache/prices")


@dataclass
class MarketData:
    """Adjusted-close price matrix plus provenance metadata."""

    prices: pd.DataFrame
    source: str
    is_demo: bool = False
    missing: tuple[str, ...] = field(default_factory=tuple)

    @property
    def as_of(self) -> pd.Timestamp:
        return self.prices.index[-1]

    def label(self) -> str:
        """Human-readable data-source string for titles and reports."""
        return self.source


# --------------------------------------------------------------------------- #
# public API
# --------------------------------------------------------------------------- #
def fetch_prices(
    tickers: list[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
    *,
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    refresh_days: int = 1,
    offline: bool = False,
    seed: int = 7,
) -> MarketData:
    """Load adjusted-close prices for ``tickers``.

    Parameters
    ----------
    tickers:
        Yahoo Finance symbols (``"UUP"``, ``"BTC-USD"`` ...).
    start / end:
        History window.  ``end`` defaults to today (inclusive).
    cache_dir:
        Where per-ticker CSVs are cached.  ``None`` disables caching.
    refresh_days:
        Consider a cache file fresh if modified within this many days.
    offline:
        Never touch the network; use cache or synthetic demo data.
    seed:
        Seed for the synthetic generator (deterministic demo output).
    """
    cache_dir = Path(cache_dir) if cache_dir else None
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize() if end is not None else pd.Timestamp.today().normalize()

    cached: dict[str, pd.Series] = {}
    stale: list[str] = []
    for ticker in tickers:
        series = _read_cache(cache_dir, ticker)
        if series is None or not _covers(series, start) or _is_stale(cache_dir, ticker, refresh_days):
            stale.append(ticker)
        if series is not None:
            cached[ticker] = series

    fetched: dict[str, pd.Series] = {}
    if stale and not offline:
        fetched = _download(stale, start, end, cache_dir)

    prices: dict[str, pd.Series] = {}
    sources: set[str] = set()
    missing: list[str] = []

    for ticker in tickers:
        if ticker in fetched:
            prices[ticker] = fetched[ticker]
            sources.add("yfinance")
        elif ticker in cached:
            prices[ticker] = cached[ticker]
            sources.add("cache")
        else:
            missing.append(ticker)

    synthetic_used = False
    if not prices:
        LOG.warning("No market data for %s; using the synthetic demo book", ", ".join(tickers))
        demo = synthetic_prices(tickers, start, end, seed=seed)
        prices = {ticker: demo[ticker] for ticker in tickers}
        sources.add("synthetic demo")
        synthetic_used = True
        missing = []
    elif missing:
        LOG.warning(
            "Dropping tickers with no data (not replaced with synthetic): %s",
            ", ".join(missing),
        )

    frame = pd.DataFrame(prices).sort_index()
    frame = frame[~frame.index.duplicated(keep="last")]
    frame = frame.loc[frame.index >= start]
    # Resample every series onto a common business-day grid, carrying the last
    # observation (crypto trades 7 days/week, ETFs 5).
    if not frame.empty:
        frame = frame.reindex(frame.index.union(pd.bdate_range(frame.index[0], frame.index[-1])))
        frame = frame.ffill()

    return MarketData(
        prices=frame,
        source=" + ".join(sorted(sources)),
        is_demo=synthetic_used,
        missing=tuple(missing),
    )


def synthetic_prices(
    tickers: list[str],
    start: str | pd.Timestamp,
    end: str | pd.Timestamp | None = None,
    *,
    seed: int = 7,
) -> pd.DataFrame:
    """Deterministic *demo* price matrix (correlated GBM + seasonal tilt).

    Used offline so every chart and test can run without a data vendor.
    Never confuse this with real history.
    """
    start = pd.Timestamp(start).normalize()
    end = pd.Timestamp(end).normalize() if end is not None else pd.Timestamp.today().normalize()
    dates = pd.bdate_range(start, end)
    n = len(dates)
    if n == 0:
        raise ValueError("Empty date range for synthetic prices")

    rng = np.random.default_rng(seed)
    t = np.arange(n) / 252.0
    market = rng.normal(0.05 / 252.0, 1.0 / np.sqrt(252.0), n)

    out: dict[str, np.ndarray] = {}
    for ticker in tickers:
        # hashlib, not hash(): PYTHONHASHSEED randomises hash() per process.
        digest = hashlib.sha256(f"{seed}:{ticker}".encode()).digest()
        h = int.from_bytes(digest[:8], "little") % (2**31 - 1)
        r = np.random.default_rng(seed + h)
        beta = 0.3 + 0.9 * r.random()
        idio = r.normal(0.0, (0.8 + 0.8 * r.random()) / np.sqrt(252.0), n)
        drift = (0.01 + 0.07 * r.random()) / 252.0
        phase = r.uniform(0, 2 * np.pi)
        amp = 0.03 + 0.06 * r.random()
        seasonal = amp * np.sin(2 * np.pi * t + phase) / 252.0
        rets = drift + beta * market + idio + seasonal
        out[ticker] = 50.0 * np.exp(np.cumsum(rets))

    return pd.DataFrame(out, index=dates)


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #
def _cache_path(cache_dir: Path | None, ticker: str) -> Path | None:
    if cache_dir is None:
        return None
    safe = ticker.replace("/", "-").replace("\\", "-")
    return cache_dir / f"{safe}.csv"


def _read_cache(cache_dir: Path | None, ticker: str) -> pd.Series | None:
    path = _cache_path(cache_dir, ticker)
    if path is None or not path.exists():
        return None
    try:
        series = pd.read_csv(path, index_col=0, parse_dates=True).iloc[:, 0]
        series.index = _naive_index(series.index)
        return series.dropna().sort_index()
    except Exception as exc:  # pragma: no cover - corrupt cache file
        LOG.warning("Could not read cache for %s (%s); ignoring", ticker, exc)
        return None


def _write_cache(cache_dir: Path | None, ticker: str, series: pd.Series) -> None:
    path = _cache_path(cache_dir, ticker)
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    series.to_csv(path, header=["close"])


def _is_stale(cache_dir: Path | None, ticker: str, refresh_days: int) -> bool:
    path = _cache_path(cache_dir, ticker)
    if path is None or not path.exists():
        return True
    age_days = (time.time() - path.stat().st_mtime) / 86400.0
    return age_days > refresh_days


def _covers(series: pd.Series, start: pd.Timestamp) -> bool:
    return series.index[0] <= start + pd.Timedelta(days=7)


def _naive_index(index: pd.Index) -> pd.DatetimeIndex:
    idx = pd.DatetimeIndex(index)
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    return idx


def _download(
    tickers: list[str],
    start: pd.Timestamp,
    end: pd.Timestamp,
    cache_dir: Path | None,
) -> dict[str, pd.Series]:
    """Fetch adjusted closes from Yahoo Finance.  Returns ``{}`` on failure."""
    try:
        import yfinance as yf  # imported lazily so offline use never needs it
    except ImportError:
        LOG.warning("yfinance is not installed; skipping download")
        return {}

    LOG.info("Downloading %d ticker(s) from Yahoo Finance: %s", len(tickers), ", ".join(tickers))
    out: dict[str, pd.Series] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for ticker in tickers:
            try:
                raw = yf.download(
                    ticker,
                    start=start.strftime("%Y-%m-%d"),
                    end=(end + pd.Timedelta(days=1)).strftime("%Y-%m-%d"),
                    auto_adjust=True,
                    progress=False,
                    threads=False,
                )
            except Exception as exc:
                LOG.warning("Download failed for %s: %s", ticker, exc)
                continue
            series = _extract_close(raw, ticker)
            if series is None or series.empty:
                LOG.warning("Empty download for %s", ticker)
                continue
            _write_cache(cache_dir, ticker, series)
            out[ticker] = series
    return out


def _extract_close(raw: pd.DataFrame, ticker: str) -> pd.Series | None:
    if raw is None or raw.empty:
        return None
    frame = raw
    if isinstance(frame.columns, pd.MultiIndex):
        close = frame.xs("Close", axis=1, level=0)
        if close.shape[1] != 1:
            close = close.iloc[:, [0]]
        series = close.iloc[:, 0]
    elif "Close" in frame.columns:
        series = frame["Close"]
    else:
        series = frame.iloc[:, 0]
    series = pd.Series(series.to_numpy(dtype=float), index=_naive_index(series.index), name=ticker)
    return series.dropna().sort_index()
