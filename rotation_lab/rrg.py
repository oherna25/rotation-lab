"""Relative Rotation Graph (RRG) engine.

Implements the Julius de Kempenaer / StockCharts style RRG methodology:

1. **RS** line      = asset price / benchmark price.
2. **RS-Ratio**     = ``100 + z_scale * zscore(log(RS), window)``.
   Values above 100 mean the asset is *stronger than its own recent average*,
   i.e. the rotation is pushing it to the right of the chart.
3. **RS-Momentum**  = ``100 + z_scale * zscore(diff(RS-Ratio), window)``.
   Values above 100 mean relative strength is *accelerating* (moving up).

Quadrants (x = RS-Ratio, y = RS-Momentum):

    Improving (top-left)   |   Leading (top-right)
    -----------------------+-----------------------
    Lagging   (bottom-left)|   Weakening (bottom-right)

The canonical JdK cycle is: Improving -> Leading -> Weakening -> Lagging -> ...
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import pandas as pd

__all__ = ["RRGParams", "RRGResult", "compute_rrg", "quadrant_name", "QUADRANT_ORDER"]

QUADRANT_ORDER: tuple[str, ...] = ("Leading", "Weakening", "Lagging", "Improving")


@dataclass(frozen=True)
class RRGParams:
    """Tunable RRG parameters.

    window:      lookback (trading days) for both normalisations; 63 ~= 3 months.
    z_scale:     chart points per standard deviation; 5 keeps typical readings
                 inside the familiar 95..105 band.
    momentum_lag: periods used to measure the change in RS-Ratio.
    """

    window: int = 63
    z_scale: float = 5.0
    momentum_lag: int = 1


@dataclass
class RRGResult:
    """Wide-format RRG output, one column per asset."""

    rs: pd.DataFrame
    rs_ratio: pd.DataFrame
    rs_momentum: pd.DataFrame
    quadrants: pd.DataFrame
    benchmark: str
    params: RRGParams

    # ----------------------------------------------------------------- API
    def snapshot(self) -> pd.DataFrame:
        """Latest valid RRG reading for every asset.

        Sorted by quadrant (Leading, Weakening, Lagging, Improving) then
        RS-Ratio. ``rs`` is the relative-performance index (100 on the first
        common date). ``heading_deg`` is the direction of the last five
        sessions on the chart: 0 is due east (RS rising), 90 is due north
        (momentum rising).
        """
        rows = []
        for ticker in self.rs_ratio.columns:
            x = self.rs_ratio[ticker].dropna()
            y = self.rs_momentum[ticker].dropna()
            if x.empty or y.empty:
                continue
            xv, yv = float(x.iloc[-1]), float(y.iloc[-1])
            rs_series = self.rs[ticker].dropna()
            rows.append(
                {
                    "ticker": ticker,
                    "rs_ratio": xv,
                    "rs_momentum": yv,
                    "quadrant": quadrant_name(xv, yv),
                    "heading_deg": _heading(x, y),
                    "rs": float(rs_series.iloc[-1]) if not rs_series.empty else np.nan,
                    "as_of": x.index[-1],
                }
            )
        snap = pd.DataFrame(rows)
        if snap.empty:
            return snap
        snap["_q"] = snap["quadrant"].map({q: i for i, q in enumerate(QUADRANT_ORDER)})
        snap = snap.sort_values(["_q", "rs_ratio"], ascending=[True, False]).drop(columns="_q")
        return snap.reset_index(drop=True)

    def trails(self, periods: int = 26, resample: str | None = "W-FRI") -> dict[str, pd.DataFrame]:
        """Return the last ``periods`` points of every asset's RRG path.

        ``resample="W-FRI"`` thins daily data to weekly trails (the classic
        look); pass ``None`` for raw daily trails.
        """
        x = self.rs_ratio
        y = self.rs_momentum
        if resample:
            x = x.resample(resample).last()
            y = y.resample(resample).last()
        out: dict[str, pd.DataFrame] = {}
        for ticker in x.columns:
            frame = pd.DataFrame({"rs_ratio": x[ticker], "rs_momentum": y[ticker]}).dropna()
            if frame.empty:
                continue
            out[ticker] = frame.tail(periods)
        return out

    def long_frame(self) -> pd.DataFrame:
        """Tidy long-form frame of the whole history (for CSV export)."""
        frames = []
        for ticker in self.rs_ratio.columns:
            frames.append(
                pd.DataFrame(
                    {
                        "date": self.rs_ratio.index,
                        "ticker": ticker,
                        "rs": self.rs[ticker].to_numpy(),
                        "rs_ratio": self.rs_ratio[ticker].to_numpy(),
                        "rs_momentum": self.rs_momentum[ticker].to_numpy(),
                        "quadrant": self.quadrants[ticker].to_numpy(),
                    }
                )
            )
        long = pd.concat(frames, ignore_index=True)
        return long.dropna(subset=["rs_ratio", "rs_momentum"])


def compute_rrg(
    prices: pd.DataFrame,
    benchmark: str,
    params: RRGParams | None = None,
) -> RRGResult:
    """Compute the full RRG history for every column of ``prices``.

    ``prices`` must contain the benchmark column; it is excluded from the
    output (an asset can never rotate against itself).
    """
    params = params or RRGParams()
    if benchmark not in prices.columns:
        raise KeyError(f"Benchmark {benchmark!r} not found in price matrix")
    if params.window < 5:
        raise ValueError("RRG window must be >= 5 observations")

    bench = prices[benchmark].astype(float)
    assets = prices.drop(columns=[benchmark]).astype(float)

    rs = assets.div(bench, axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        log_rs = np.log(rs.replace(0.0, np.nan))

    z_ratio = _zscore(log_rs, params.window)
    rs_ratio = (100.0 + params.z_scale * z_ratio).where(z_ratio.notna())

    delta = rs_ratio.diff(params.momentum_lag)
    z_mom = _zscore(delta, params.window)
    rs_momentum = (100.0 + params.z_scale * z_mom).where(z_mom.notna())

    quadrants = pd.DataFrame(
        {col: _quadrant_series(rs_ratio[col], rs_momentum[col]) for col in rs_ratio.columns},
        index=rs_ratio.index,
    )

    # Normalise the RS line to a performance index (100 at the first common date).
    first = rs.dropna(how="all").index
    if len(first):
        base = rs.loc[first[0]]
        rs_indexed = rs.div(base, axis=1) * 100.0
    else:  # pragma: no cover - degenerate input
        rs_indexed = rs * 100.0

    return RRGResult(
        rs=rs_indexed,
        rs_ratio=rs_ratio,
        rs_momentum=rs_momentum,
        quadrants=quadrants,
        benchmark=benchmark,
        params=params,
    )


def quadrant_name(x: float, y: float) -> str:
    """Classify a single (RS-Ratio, RS-Momentum) point."""
    if np.isnan(x) or np.isnan(y):
        return ""
    if x >= 100.0:
        return "Leading" if y >= 100.0 else "Weakening"
    return "Improving" if y >= 100.0 else "Lagging"


def _heading(x: pd.Series, y: pd.Series, lookback: int = 5) -> float:
    """Angle, in degrees, of the last ``lookback`` RRG steps."""
    if len(x) <= lookback or len(y) <= lookback:
        return np.nan
    dx = float(x.iloc[-1] - x.iloc[-1 - lookback])
    dy = float(y.iloc[-1] - y.iloc[-1 - lookback])
    if dx == 0.0 and dy == 0.0:
        return np.nan
    return float(math.degrees(math.atan2(dy, dx)))


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #
def _zscore(frame: pd.DataFrame, window: int) -> pd.DataFrame:
    """Rolling z-score that stays NaN during warm-up and handles zero variance."""
    mean = frame.rolling(window, min_periods=window).mean()
    std = frame.rolling(window, min_periods=window).std(ddof=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        z = (frame - mean).div(std)
    z = z.mask(std == 0.0, 0.0)  # constant over the window -> neutral, not NaN
    z = z.replace([np.inf, -np.inf], np.nan)
    z = z.mask(mean.isna())  # keep warm-up rows empty
    return z


def _quadrant_series(x: pd.Series, y: pd.Series) -> pd.Series:
    values = np.select(
        [(x >= 100) & (y >= 100), (x >= 100) & (y < 100), (x < 100) & (y >= 100), (x < 100) & (y < 100)],
        ["Leading", "Weakening", "Improving", "Lagging"],
        default="",
    )
    values = pd.Series(values, index=x.index)
    values[x.isna() | y.isna()] = ""
    return values

