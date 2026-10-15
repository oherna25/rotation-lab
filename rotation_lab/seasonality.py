"""Larry Williams-style seasonality engine.

Larry Williams popularised *seasonality as a tradable tendency*: instead of
eyeballing "gold usually rallies into year-end", you average many years of
history into a composite seasonal path and trade with (or against) it.

This module builds, for every asset:

* a **seasonal composite** - the average calendar path across all years,
  expressed as an index that starts at 100 in week 1 (equivalent to the
  "average year" overlay in Williams' seasonal studies);
* **percentile bands** (10th/90th) across years, so you can see how uniform
  the pattern is - a tendency that happened once in twelve years is noise;
* **weekly win rates** - what fraction of years were positive in that week;
* a **monthly seasonality table** - mean and median month-by-month returns
  plus win rates (the "right stock at the right time" monthly bias);
* a **forward outlook** - the equal-weighted historical drift over the next N
  ISO weeks (wrapping week 52 into week 1 of the following year), its win
  rate, a cross-sectional z-score and where the current year's
  performance-to-date ranks against prior years.

Everything is computed per-asset from log returns, so results are unit-free
and can be compared across an ETF, a crypto pair and an index.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

__all__ = [
    "MonthlySeasonality",
    "SeasonalCycle",
    "SeasonalOutlook",
    "SeasonalProfile",
    "build_monthly_seasonality",
    "build_seasonal_profile",
    "seasonal_cycle",
    "seasonal_outlook",
]


# --------------------------------------------------------------------------- #
# containers
# --------------------------------------------------------------------------- #
@dataclass
class SeasonalProfile:
    """Weekly seasonal statistics for a whole universe.

    ``composite``/``p10``/``p90``/``win_rate`` are DataFrames indexed by ISO
    week, one column per ticker.  ``per_year`` maps ticker -> DataFrame
    (rows = ISO week, columns = year) of each year's seasonal index path.
    """

    composite: pd.DataFrame
    p10: pd.DataFrame
    p90: pd.DataFrame
    win_rate: pd.DataFrame
    per_year: dict[str, pd.DataFrame] = field(default_factory=dict)
    weekly_stats: pd.DataFrame = field(default_factory=pd.DataFrame)
    current_year: int = 0
    n_years: dict[str, int] = field(default_factory=dict)
    weekly_returns: dict[str, pd.DataFrame] = field(default_factory=dict)

    @property
    def weeks(self) -> pd.Index:
        return self.composite.index


@dataclass
class MonthlySeasonality:
    """Month-by-month statistics, one table per metric (assets x months)."""

    mean_pct: pd.DataFrame
    median_pct: pd.DataFrame
    win_rate_pct: pd.DataFrame
    counts: pd.DataFrame
    years: pd.DataFrame

    def long(self) -> pd.DataFrame:
        """Tidy long-form table for CSV export."""
        frames = []
        for ticker in self.mean_pct.index:
            for month in self.mean_pct.columns:
                frames.append(
                    {
                        "ticker": ticker,
                        "month": int(month),
                        "month_name": _MONTH_NAMES[int(month) - 1],
                        "mean_pct": self.mean_pct.loc[ticker, month],
                        "median_pct": self.median_pct.loc[ticker, month],
                        "win_rate_pct": self.win_rate_pct.loc[ticker, month],
                        "years": self.years.loc[ticker, month],
                    }
                )
        return pd.DataFrame(frames)


@dataclass
class SeasonalOutlook:
    """Forward seasonal expectancy for the next ``weeks_ahead`` ISO weeks."""

    table: pd.DataFrame  # ticker, label, current_week, expected_return_pct, ...

    def __len__(self) -> int:  # pragma: no cover - convenience
        return len(self.table)


_MONTH_NAMES = ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
_WEEK_THRESHOLD = 4  # a year must start by ISO week 4 to anchor the composite


# --------------------------------------------------------------------------- #
# weekly seasonal composite
# --------------------------------------------------------------------------- #
def build_seasonal_profile(prices: pd.DataFrame, min_years: int = 4) -> SeasonalProfile:
    """Average-year seasonal composite for every column of ``prices``."""
    composites: dict[str, pd.Series] = {}
    p10s: dict[str, pd.Series] = {}
    p90s: dict[str, pd.Series] = {}
    win_rates: dict[str, pd.Series] = {}
    per_year: dict[str, pd.DataFrame] = {}
    weekly_returns: dict[str, pd.DataFrame] = {}
    n_years: dict[str, int] = {}
    stats_rows: list[pd.DataFrame] = []
    current_year = int(prices.index[-1].isocalendar().year)

    weeks = pd.Index(range(1, 53), name="week")  # week 53 is folded into 52 upstream

    for ticker in prices.columns:
        series = prices[ticker].dropna()
        if series.empty:
            continue
        weekly_ret = _weekly_log_returns(series)  # DataFrame: year, week, ret
        pivot = weekly_ret.pivot(index="year", columns="week", values="ret")
        pivot.index = pivot.index.astype(int)
        pivot.columns = pivot.columns.astype(int)
        pivot = pivot.reindex(columns=range(1, 53))
        weekly_returns[ticker] = pivot

        # Per-year seasonal index paths: cumulate weekly returns from each
        # year's first week and rebase to 100 there.
        paths: dict[int, pd.Series] = {}
        for year, row in pivot.iterrows():
            ordered = row.dropna().sort_index()
            if ordered.empty or int(ordered.index[0]) > _WEEK_THRESHOLD:
                continue  # partial year without a proper anchor
            cum = 100.0 * np.exp(ordered.cumsum() - ordered.iloc[0])
            paths[int(year)] = cum

        if not paths:
            continue

        frame = pd.DataFrame(paths).reindex(weeks)
        history = frame.drop(columns=[current_year], errors="ignore")
        history = history if len(history.columns) >= min_years else frame

        # Completed years only, so the in-progress week does not bias win rates.
        stats_pivot = pivot.drop(index=[current_year], errors="ignore")
        if len(stats_pivot) < min_years:
            stats_pivot = pivot

        per_year[ticker] = frame
        n_years[ticker] = int(len(history.columns))
        composites[ticker] = history.mean(axis=1)
        p10s[ticker] = history.quantile(0.10, axis=1)
        p90s[ticker] = history.quantile(0.90, axis=1)
        counted = stats_pivot.notna().sum(axis=0).replace(0, np.nan)
        win_rates[ticker] = (stats_pivot > 0).sum(axis=0) / counted

        stats_rows.append(
            pd.DataFrame(
                {
                    "ticker": ticker,
                    "week": stats_pivot.columns,
                    "mean_ret_pct": np.expm1(stats_pivot.mean(axis=0)) * 100.0,
                    "median_ret_pct": np.expm1(stats_pivot.median(axis=0)) * 100.0,
                    "win_rate_pct": (stats_pivot > 0).sum(axis=0) / counted * 100.0,
                    "years": stats_pivot.notna().sum(axis=0),
                }
            )
        )

    weekly_stats = (
        pd.concat(stats_rows, ignore_index=True) if stats_rows else pd.DataFrame()
    )

    return SeasonalProfile(
        composite=pd.DataFrame(composites).sort_index(),
        p10=pd.DataFrame(p10s).sort_index(),
        p90=pd.DataFrame(p90s).sort_index(),
        win_rate=pd.DataFrame(win_rates).sort_index(),
        per_year=per_year,
        weekly_stats=weekly_stats,
        current_year=current_year,
        n_years=n_years,
        weekly_returns=weekly_returns,
    )


# --------------------------------------------------------------------------- #
# monthly seasonality
# --------------------------------------------------------------------------- #
def build_monthly_seasonality(prices: pd.DataFrame, min_years: int = 4, drop_current_month: bool = True) -> MonthlySeasonality:
    """Month-of-year return statistics for every column of ``prices``."""
    mean_pct: dict[str, pd.Series] = {}
    median_pct: dict[str, pd.Series] = {}
    win_rate: dict[str, pd.Series] = {}
    counts: dict[str, pd.Series] = {}
    years_used: dict[str, pd.Series] = {}

    for ticker in prices.columns:
        series = prices[ticker].dropna()
        if series.empty:
            continue
        monthly = _monthly_log_returns(series, drop_current_month=drop_current_month)
        if monthly.empty:
            continue
        pivot = monthly.pivot(index="year", columns="month", values="ret")
        if pivot.shape[0] < min_years:
            pass  # keep the data anyway; the report flags short histories
        mean_pct[ticker] = np.expm1(pivot.mean(axis=0)) * 100.0
        median_pct[ticker] = np.expm1(pivot.median(axis=0)) * 100.0
        win_rate[ticker] = (pivot > 0).sum(axis=0) / pivot.notna().sum(axis=0).replace(0, np.nan) * 100.0
        counts[ticker] = pivot.count(axis=0)
        years_used[ticker] = pivot.notna().sum(axis=0)

    months = list(range(1, 13))
    return MonthlySeasonality(
        mean_pct=pd.DataFrame(mean_pct).T.reindex(columns=months),
        median_pct=pd.DataFrame(median_pct).T.reindex(columns=months),
        win_rate_pct=pd.DataFrame(win_rate).T.reindex(columns=months),
        counts=pd.DataFrame(counts).T.reindex(columns=months),
        years=pd.DataFrame(years_used).T.reindex(columns=months),
    )


# --------------------------------------------------------------------------- #
# forward outlook
# --------------------------------------------------------------------------- #
def seasonal_outlook(
    profile: SeasonalProfile,
    prices: pd.DataFrame,
    weeks_ahead: int = 4,
    labels: dict[str, str] | None = None,
) -> SeasonalOutlook:
    """Expected seasonal drift over the next ``weeks_ahead`` ISO weeks.

    Each historical year contributes the compounded log return of the next
    ``weeks_ahead`` calendar weeks, wrapping week 52 into week 1 of the
    *following* year (a late-December window includes January). The
    in-progress year is not a sample. ``expected_return_pct`` is the
    equal-weighted mean of those windows; ``win_rate_pct`` is the share that
    were positive. Current-year YTD performance is ranked against prior
    years at the same ISO week.
    """
    if weeks_ahead < 1:
        raise ValueError("weeks_ahead must be >= 1")
    labels = labels or {}
    rows = []

    for ticker in profile.composite.columns:
        paths = profile.per_year.get(ticker)
        pivot = profile.weekly_returns.get(ticker)
        series = prices[ticker].dropna()
        if paths is None or pivot is None or pivot.empty or series.empty:
            continue

        current_week = min(max(int(series.index[-1].isocalendar().week), 1), 52)
        target_week = ((current_week - 1 + weeks_ahead) % 52) + 1

        hist_years = [int(year) for year in pivot.index if int(year) != profile.current_year]
        window = np.array(
            [
                value
                for year in hist_years
                if pd.notna(value := _window_log_return(pivot, year, current_week, weeks_ahead))
            ],
            dtype=float,
        )
        if len(window):
            expected = float(np.expm1(window.mean()) * 100.0)
            win = float((window > 0).mean() * 100.0)
            years = int(len(window))
        else:
            expected, win, years = np.nan, np.nan, 0

        rows.append(
            {
                "ticker": ticker,
                "label": labels.get(ticker, ticker),
                "current_week": current_week,
                "target_week": target_week,
                "expected_return_pct": expected,
                "win_rate_pct": win,
                "ytd_vs_history_pct": _ytd_percentile(paths, profile.current_year, current_week),
                "years": years,
            }
        )

    table = pd.DataFrame(rows)
    if table.empty:
        table = pd.DataFrame(
            columns=[
                "rank",
                "ticker",
                "label",
                "current_week",
                "target_week",
                "expected_return_pct",
                "win_rate_pct",
                "ytd_vs_history_pct",
                "years",
                "seasonal_z",
            ]
        )
        return SeasonalOutlook(table=table)

    valid = table["expected_return_pct"]
    mean, std = valid.mean(), valid.std(ddof=0)
    table["seasonal_z"] = 0.0 if not std or np.isnan(std) else (valid - mean) / std
    table = table.sort_values("expected_return_pct", ascending=False, na_position="last").reset_index(drop=True)
    table.insert(0, "rank", np.arange(1, len(table) + 1))
    return SeasonalOutlook(table=table)


# --------------------------------------------------------------------------- #
# annual cycle (dashboard)
# --------------------------------------------------------------------------- #
@dataclass
class SeasonalCycle:
    """Williams-style annual path for one series, scaled to 100.

    ``composite`` is the average of completed years, each rebased to 100 on
    that year's first print, drawn on the calendar of ``year``. ``current``
    is this year's path on the same scale. ``drift`` starts at the last
    print and walks forward ``horizon_days`` calendar days by the average
    historical log return for each month/day. ``seas_idx`` is where today's
    composite sits between its own low and high, from 0 to 100. ``bias_pts``
    is the average historical simple return over the next ``horizon_days``,
    in points of a 100-based index.
    """

    composite: pd.Series
    current: pd.Series
    drift: pd.Series
    as_of: pd.Timestamp
    year: int
    n_years: int
    seas_idx: float
    bias_pts: float
    win_rate_pct: float
    horizon_days: int = 30


def seasonal_cycle(prices: pd.Series, horizon_days: int = 30) -> SeasonalCycle:
    """Build the annual seasonal cycle shown next to the rotation chart.

    Years that do not start in January are left out of the average so a late
    listing does not anchor the composite in June. ``n_years`` is the count
    actually used.
    """
    if horizon_days < 1:
        raise ValueError("horizon_days must be >= 1")

    series = prices.dropna().astype(float).sort_index()
    if isinstance(series.index, pd.DatetimeIndex) and series.index.tz is not None:
        series.index = series.index.tz_localize(None)
    series = series[~series.index.duplicated(keep="last")]
    if series.empty:
        raise ValueError("seasonal_cycle needs at least one price")

    as_of = pd.Timestamp(series.index[-1]).normalize()
    year = int(as_of.year)
    paths = _anchored_year_paths(series, year)
    if paths:
        stacked = pd.concat(paths, axis=1)
        mean = stacked.mean(axis=1).dropna().sort_index()
        n_years = int(stacked.shape[1])
    else:
        mean = pd.Series(dtype=float)
        n_years = 0

    composite = _stamp_month_day(mean, year)
    current = _rebased_year(series, year)
    bias_pts, win_rate = _forward_bias(series, as_of, horizon_days, year)
    return SeasonalCycle(
        composite=composite,
        current=current,
        drift=_defend_drift(series, current, year, horizon_days),
        as_of=as_of,
        year=year,
        n_years=n_years,
        seas_idx=_seasonal_index(composite, as_of),
        bias_pts=bias_pts,
        win_rate_pct=win_rate,
        horizon_days=int(horizon_days),
    )


def _anchored_year_paths(series: pd.Series, current_year: int) -> list[pd.Series]:
    """Completed years rebased to 100, indexed by (month, day)."""
    paths: list[pd.Series] = []
    for yr, chunk in series.groupby(series.index.year):
        if int(yr) >= current_year:
            continue
        chunk = chunk.dropna()
        if chunk.empty or int(chunk.index[0].month) != 1 or len(chunk) < 100:
            continue
        base = float(chunk.iloc[0])
        if base == 0.0 or not np.isfinite(base):
            continue
        rebased = chunk / base * 100.0
        keyed = pd.Series(
            rebased.to_numpy(dtype=float),
            index=pd.MultiIndex.from_arrays(
                [rebased.index.month.astype(int), rebased.index.day.astype(int)],
                names=["month", "day"],
            ),
        )
        paths.append(keyed[~keyed.index.duplicated(keep="last")])
    return paths


def _stamp_month_day(values: pd.Series, year: int) -> pd.Series:
    """Map a (month, day) series onto real dates in ``year``."""
    if values.empty:
        return pd.Series(dtype=float)
    rows: dict[pd.Timestamp, float] = {}
    for key, value in values.items():
        month, day = int(key[0]), int(key[1])
        stamp = _safe_stamp(year, month, day)
        if stamp is None or not np.isfinite(value):
            continue
        rows[stamp] = float(value)
    if not rows:
        return pd.Series(dtype=float)
    out = pd.Series(rows).sort_index()
    return out[~out.index.duplicated(keep="last")]


def _rebased_year(series: pd.Series, year: int) -> pd.Series:
    chunk = series[series.index.year == year].dropna()
    if chunk.empty:
        return pd.Series(dtype=float)
    base = float(chunk.iloc[0])
    if base == 0.0 or not np.isfinite(base):
        return pd.Series(dtype=float)
    out = chunk / base * 100.0
    out.index = pd.DatetimeIndex(out.index).normalize()
    return out


def _seasonal_index(composite: pd.Series, as_of: pd.Timestamp) -> float:
    """0 at the composite's low, 100 at its high, measured at ``as_of``."""
    values = composite.dropna()
    if values.empty:
        return float("nan")
    span = float(values.max() - values.min())
    prior = values.loc[:as_of]
    val = float(prior.iloc[-1] if len(prior) else values.iloc[-1])
    if not np.isfinite(span) or span == 0.0:
        return 50.0
    return float(np.clip((val - float(values.min())) / span * 100.0, 0.0, 100.0))


def _forward_bias(
    series: pd.Series,
    as_of: pd.Timestamp,
    horizon_days: int,
    current_year: int,
) -> tuple[float, float]:
    """Mean point change and win rate of the next ``horizon_days`` across prior years."""
    returns: list[float] = []
    for yr in sorted({int(y) for y in series.index.year}):
        if yr >= current_year:
            continue
        anchor = _safe_stamp(yr, int(as_of.month), int(as_of.day))
        if anchor is None:
            continue
        start_px = _price_on_or_before(series, anchor, tolerance_days=4)
        end_px = _price_on_or_before(series, anchor + pd.Timedelta(days=horizon_days), tolerance_days=4)
        if start_px is None or end_px is None or start_px == 0.0:
            continue
        returns.append(end_px / start_px - 1.0)
    if not returns:
        return float("nan"), float("nan")
    arr = np.asarray(returns, dtype=float)
    return float(arr.mean() * 100.0), float((arr > 0).mean() * 100.0)


def _defend_drift(series: pd.Series, current: pd.Series, current_year: int, horizon_days: int) -> pd.Series:
    """Project the average month/day log return forward from the last print."""
    if current.empty:
        return pd.Series(dtype=float)
    log_ret = np.log(series.where(series > 0.0)).diff().dropna()
    hist = log_ret[log_ret.index.year < current_year]
    if hist.empty:
        return pd.Series(dtype=float)
    avg = pd.Series(
        hist.to_numpy(dtype=float),
        index=pd.MultiIndex.from_arrays([hist.index.month.astype(int), hist.index.day.astype(int)]),
    )
    avg = avg.groupby(level=[0, 1]).mean()
    level = float(current.iloc[-1])
    anchor = pd.Timestamp(current.index[-1]).normalize()
    points = {anchor: level}
    for step in range(1, horizon_days + 1):
        day = anchor + pd.Timedelta(days=step)
        key = (int(day.month), int(day.day))
        change = float(avg[key]) if key in avg.index and np.isfinite(avg[key]) else 0.0
        level *= float(np.exp(change))
        points[day] = level
    return pd.Series(points).sort_index()


def _safe_stamp(year: int, month: int, day: int) -> pd.Timestamp | None:
    """Calendar date, or None when that day does not exist (31 Feb)."""
    try:
        return pd.Timestamp(year=year, month=month, day=day)
    except ValueError:
        if month == 2 and day == 29:
            return pd.Timestamp(year=year, month=2, day=28)
        return None


def _price_on_or_before(series: pd.Series, when: pd.Timestamp, tolerance_days: int) -> float | None:
    window = series.loc[:when]
    if window.empty:
        return None
    age = when - pd.Timestamp(window.index[-1])
    if age > pd.Timedelta(days=tolerance_days):
        return None
    value = float(window.iloc[-1])
    return value if np.isfinite(value) else None


# --------------------------------------------------------------------------- #
# internals
# --------------------------------------------------------------------------- #
def _weekly_log_returns(series: pd.Series) -> pd.DataFrame:
    """Sum of daily log returns per (ISO year, ISO week). Week 53 folds into 52.

    The year is the ISO year, not the calendar year: the last days of December
    often belong to ISO week 1 of the next year, and bucketing them with the
    calendar year parks a January week inside December.
    """
    log_ret = np.log(series).diff()
    iso = series.index.isocalendar()
    frame = pd.DataFrame(
        {
            "year": iso.year.to_numpy(dtype=int),
            "week": np.minimum(iso.week.to_numpy(dtype=int), 52),
            "ret": log_ret.to_numpy(),
        },
        index=series.index,
    )
    frame = frame.dropna(subset=["ret"])
    return frame.groupby(["year", "week"], as_index=False)["ret"].sum()


def _window_log_return(pivot: pd.DataFrame, year: int, start_week: int, n_weeks: int) -> float:
    """Compound the next ``n_weeks`` after ``start_week``, wrapping into ``year + 1``."""
    total = 0.0
    y = year
    w = start_week
    for _ in range(n_weeks):
        w += 1
        if w > 52:
            w = 1
            y += 1
        if y not in pivot.index or w not in pivot.columns:
            return np.nan
        val = pivot.at[y, w]
        if pd.isna(val):
            return np.nan
        total += float(val)
    return total


def _ytd_percentile(paths: pd.DataFrame, current_year: int, week: int) -> float:
    """Percent of prior years whose seasonal index at ``week`` is below this year."""
    if week not in paths.index or current_year not in paths.columns:
        return np.nan
    row = paths.loc[week]
    if isinstance(row, pd.DataFrame):  # duplicate week labels
        row = row.iloc[-1]
    cur = row.get(current_year, np.nan)
    hist = row.drop(labels=[current_year], errors="ignore").dropna()
    if pd.isna(cur) or len(hist) < 3:
        return np.nan
    return float((hist < cur).mean() * 100.0)


def _monthly_log_returns(series: pd.Series, drop_current_month: bool = True) -> pd.DataFrame:
    """Log return per (year, month); the in-progress month is dropped."""
    grouped = series.groupby([series.index.year, series.index.month]).last()
    log_ret = np.log(grouped).diff()
    frame = pd.DataFrame(
        {
            "year": [idx[0] for idx in grouped.index],
            "month": [idx[1] for idx in grouped.index],
            "ret": log_ret.to_numpy(),
        }
    ).dropna(subset=["ret"])
    if drop_current_month and not frame.empty:
        cur = (series.index[-1].year, series.index[-1].month)
        frame = frame[~((frame["year"] == cur[0]) & (frame["month"] == cur[1]))]
    return frame
