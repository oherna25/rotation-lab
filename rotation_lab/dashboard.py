"""Dark single-screen figures for the rotation dashboard.

The layout follows a sector-rotation terminal: one RRG with trails coloured
by the current quadrant, a matrix of those readings plus the seasonal bias,
and the selected name's Williams annual cycle. Plotly is optional and is
imported here so the CLI (matplotlib only) does not depend on it.
"""

from __future__ import annotations

import pandas as pd
import plotly.graph_objects as go

from rotation_lab.config import Asset
from rotation_lab.rrg import RRGResult, quadrant_name
from rotation_lab.seasonality import SeasonalCycle, seasonal_cycle

__all__ = [
    "QUADRANT_COLOR",
    "PLOT_BG",
    "format_bias",
    "rrg_figure",
    "seasonal_board",
    "seasonal_figure",
]

PLOT_BG = "#0c1424"
_PAPER = "#0c1424"
_GRID = "rgba(255,255,255,0.05)"
_AXIS = "#8b98ab"
_CROSS = "rgba(186,198,214,0.55)"
_FONT = "Arial, Helvetica, sans-serif"

QUADRANT_COLOR = {
    "Leading": "#3DDC97",
    "Improving": "#4DA3FF",
    "Weakening": "#E6A817",
    "Lagging": "#E15B64",
}

_QUADRANT_FILL = {
    "Leading": "rgba(61,220,151,0.055)",
    "Improving": "rgba(77,163,255,0.055)",
    "Weakening": "rgba(230,168,23,0.05)",
    "Lagging": "rgba(225,91,100,0.05)",
}


def format_bias(points: float, win_rate: float) -> str:
    """``▲ +4.9 (57%)`` — the matrix cell for the next 30 days."""
    if pd.isna(points) or pd.isna(win_rate):
        return "—"
    arrow = "▲" if points >= 0 else "▼"
    return f"{arrow} {points:+.1f} ({win_rate:.0f}%)"


def seasonal_board(
    snapshot: pd.DataFrame,
    assets: list[Asset],
    prices: pd.DataFrame,
    *,
    benchmark: str,
    relative: bool,
    horizon_days: int = 30,
) -> tuple[pd.DataFrame, dict[str, SeasonalCycle]]:
    """Matrix sorted by RS-Ratio, plus the cycle used for each row.

    ``relative=True`` studies the asset/benchmark ratio instead of the
    outright price. Rows with too little history get a blank seasonal cell
    rather than a fabricated index.
    """
    meta = {asset.ticker: asset for asset in assets}
    columns = ["Ticker", "Class", "RS-Ratio", "RS-Mom", "Quadrant", "Seas Idx", "30d Bias (Win%)"]
    if snapshot.empty:
        return pd.DataFrame(columns=columns), {}

    bench = prices[benchmark] if benchmark in prices.columns else None
    rows: list[dict] = []
    cycles: dict[str, SeasonalCycle] = {}
    ordered = snapshot.sort_values("rs_ratio", ascending=False)
    for rec in ordered.itertuples(index=False):
        ticker = str(rec.ticker)
        asset = meta.get(ticker)
        series = _study_series(prices, ticker, bench, relative)
        cycle = _cycle_or_none(series, horizon_days)
        if cycle is not None:
            cycles[ticker] = cycle
        rows.append(
            {
                "Ticker": ticker,
                "Class": asset.group if asset else "",
                "RS-Ratio": round(float(rec.rs_ratio), 2),
                "RS-Mom": round(float(rec.rs_momentum), 2),
                "Quadrant": rec.quadrant,
                "Seas Idx": None if cycle is None or pd.isna(cycle.seas_idx) else int(round(cycle.seas_idx)),
                "30d Bias (Win%)": "—" if cycle is None else format_bias(cycle.bias_pts, cycle.win_rate_pct),
            }
        )
    return pd.DataFrame(rows), cycles


def rrg_figure(
    result: RRGResult,
    *,
    trail: int,
    weekly: bool,
    title: str,
    height: int = 740,
) -> go.Figure:
    """Quadrant scatter with one coloured trail per asset.

    The whole trail takes the colour of the latest quadrant, which is how a
    rotation terminal reads at a glance: green is leading, blue improving,
    gold weakening, red lagging.
    """
    trails = result.trails(periods=max(int(trail), 2), resample="W-FRI" if weekly else None)
    quadrants = {row.ticker: row.quadrant for row in result.snapshot().itertuples(index=False)}

    xs: list[float] = []
    ys: list[float] = []
    for frame in trails.values():
        xs.extend(float(v) for v in frame["rs_ratio"])
        ys.extend(float(v) for v in frame["rs_momentum"])

    fig = go.Figure()
    if not xs:
        fig.update_layout(title=title, **_base_layout(height))
        fig.add_annotation(text="No RRG history yet", showarrow=False, font=dict(color=_AXIS, size=14))
        return fig

    xmin, xmax = _limits(xs)
    ymin, ymax = _limits(ys)
    _quadrant_rects(fig, xmin, xmax, ymin, ymax)

    for ticker, frame in trails.items():
        last_x = float(frame["rs_ratio"].iloc[-1])
        last_y = float(frame["rs_momentum"].iloc[-1])
        quadrant = quadrants.get(ticker) or quadrant_name(last_x, last_y)
        color = QUADRANT_COLOR.get(quadrant, "#d5dde8")
        n = len(frame)
        fig.add_trace(
            go.Scatter(
                x=frame["rs_ratio"],
                y=frame["rs_momentum"],
                mode="lines+markers+text",
                name=ticker,
                text=[""] * (n - 1) + [ticker],
                textposition="top right",
                textfont=dict(color=color, size=13, family=_FONT),
                cliponaxis=False,
                line=dict(color=color, width=2.4),
                marker=dict(size=[0] * (n - 1) + [8], color=color, line=dict(width=0)),
                customdata=frame.index.strftime("%Y-%m-%d"),
                hovertemplate=(
                    f"<b>{ticker}</b> · {quadrant}<br>"
                    "%{customdata}<br>RS-Ratio %{x:.2f}<br>RS-Momentum %{y:.2f}<extra></extra>"
                ),
                showlegend=False,
            )
        )

    fig.add_hline(y=100, line_dash="dash", line_color=_CROSS, line_width=1)
    fig.add_vline(x=100, line_dash="dash", line_color=_CROSS, line_width=1)
    _corner_labels(fig)
    fig.update_layout(
        title=dict(text=title, x=0.0, xanchor="left", font=dict(size=16, color="#e6edf5", family=_FONT)),
        xaxis=dict(title="JdK RS-Ratio (Relative Strength vs Benchmark)", range=[xmin, xmax], **_axis_style()),
        yaxis=dict(title="JdK RS-Momentum (Rate of Change of Trend)", range=[ymin, ymax], **_axis_style()),
        **_base_layout(height),
    )
    return fig


def seasonal_figure(
    cycle: SeasonalCycle,
    *,
    ticker: str,
    asset_label: str,
    mode_label: str,
    show_drift: bool,
    overlay: SeasonalCycle | None = None,
    overlay_name: str | None = None,
    height: int = 360,
) -> go.Figure:
    """Annual cycle: prior-year average, this year's path, optional forward drift."""
    fig = go.Figure()
    fig.update_layout(
        title=dict(
            text=_cycle_title(cycle, ticker, asset_label, mode_label),
            x=0.0,
            xanchor="left",
            font=dict(size=14, color="#e6edf5", family=_FONT),
        ),
        **_base_layout(height, top=78),
    )
    if cycle.composite.empty and cycle.current.empty:
        fig.add_annotation(text="Not enough history for a seasonal cycle", showarrow=False, font=dict(color=_AXIS))
        return fig

    positive = not pd.isna(cycle.bias_pts) and cycle.bias_pts >= 0
    shade = "rgba(61,220,151,0.45)" if positive else "rgba(225,91,100,0.42)"
    bounds = _series_bounds(
        cycle.composite,
        cycle.current,
        cycle.drift if show_drift else None,
        None if overlay is None else overlay.current,
    )
    # A filled trace, not a layout shape. Plotly drops timestamp shape
    # coordinates, so a vrect never showed the next-30-day window.
    if bounds is not None and not cycle.current.empty:
        y0, y1 = bounds
        start = pd.Timestamp(cycle.current.index[-1])
        end = start + pd.Timedelta(days=int(cycle.horizon_days))
        fig.add_trace(
            go.Scatter(
                x=[start, end, end, start],
                y=[y0, y0, y1, y1],
                fill="toself",
                fillcolor=shade,
                mode="lines",
                line=dict(width=0),
                name="Next 30d Window",
                hoverinfo="skip",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[start, start],
                y=[y0, y1],
                mode="lines",
                line=dict(color="#e85d8c", width=2),
                showlegend=False,
                hoverinfo="skip",
            )
        )

    if not cycle.composite.empty:
        fig.add_trace(
            go.Scatter(
                x=cycle.composite.index,
                y=cycle.composite.to_numpy(),
                name=f"Williams True Seasonal ({cycle.n_years}Y Prior Avg)",
                mode="lines",
                line=dict(color="#E6C229", width=2.2, dash="dash"),
                hovertemplate="Seasonal avg<br>%{x|%b %d}<br>%{y:.1f}<extra></extra>",
            )
        )
    if not cycle.current.empty:
        fig.add_trace(
            go.Scatter(
                x=cycle.current.index,
                y=cycle.current.to_numpy(),
                name=f"{cycle.year} YTD Actual (Scaled)",
                mode="lines",
                line=dict(color="#3D8BFF", width=2.4),
                hovertemplate=f"{cycle.year} actual<br>%{{x|%b %d}}<br>%{{y:.1f}}<extra></extra>",
            )
        )
        fig.add_trace(
            go.Scatter(
                x=[cycle.current.index[-1]],
                y=[float(cycle.current.iloc[-1])],
                mode="markers",
                marker=dict(color="#ff3355", size=9, line=dict(color="#1a0a10", width=0.5)),
                showlegend=False,
                hovertemplate="Now<br>%{x|%b %d}<br>%{y:.1f}<extra></extra>",
            )
        )
    if show_drift and len(cycle.drift) > 1:
        fig.add_trace(
            go.Scatter(
                x=cycle.drift.index,
                y=cycle.drift.to_numpy(),
                name="Defended drift",
                mode="lines",
                line=dict(color="#E6C229", width=1.6, dash="dot"),
                hovertemplate="Defended drift<br>%{x|%b %d}<br>%{y:.1f}<extra></extra>",
            )
        )
    if overlay is not None and not overlay.current.empty:
        fig.add_trace(
            go.Scatter(
                x=overlay.current.index,
                y=overlay.current.to_numpy(),
                name=overlay_name or "Relative (scaled)",
                mode="lines",
                line=dict(color="#9ad7ff", width=1.4),
                hovertemplate="Relative<br>%{x|%b %d}<br>%{y:.1f}<extra></extra>",
            )
        )

    fig.update_layout(
        legend=dict(
            orientation="v",
            x=0.01,
            y=0.98,
            xanchor="left",
            yanchor="top",
            bgcolor="rgba(12,20,36,0.72)",
            borderwidth=0,
            font=dict(size=10, color="#d5dde8", family=_FONT),
        ),
        xaxis=dict(title="", tickformat="%b", dtick="M1", **_axis_style()),
        yaxis=dict(title="Seasonal Index", range=list(bounds) if bounds else None, **_axis_style()),
        hovermode="x unified",
    )
    return fig


def _series_bounds(*series: pd.Series | None) -> tuple[float, float] | None:
    values: list[float] = []
    for series_ in series:
        if series_ is None or len(series_) == 0:
            continue
        values.append(float(series_.min()))
        values.append(float(series_.max()))
    if not values:
        return None
    low, high = min(values), max(values)
    pad = max((high - low) * 0.08, 0.4)
    return low - pad, high + pad


def _study_series(
    prices: pd.DataFrame,
    ticker: str,
    benchmark: pd.Series | None,
    relative: bool,
) -> pd.Series:
    if ticker not in prices.columns:
        return pd.Series(dtype=float)
    series = prices[ticker].dropna()
    if not relative or benchmark is None:
        return series
    aligned = pd.concat([series.rename("asset"), benchmark.rename("bench")], axis=1).dropna()
    if aligned.empty:
        return pd.Series(dtype=float)
    ratio = aligned["asset"] / aligned["bench"].where(aligned["bench"] != 0.0)
    return ratio.replace([float("inf"), float("-inf")], pd.NA).dropna()


def _cycle_or_none(series: pd.Series, horizon_days: int) -> SeasonalCycle | None:
    if len(series.dropna()) < 40:
        return None
    try:
        return seasonal_cycle(series, horizon_days=horizon_days)
    except ValueError:
        return None


def _limits(values: list[float], pad: float = 0.22) -> tuple[float, float]:
    lo = min(min(values), 100.0)
    hi = max(max(values), 100.0)
    span = hi - lo
    if span == 0:
        span = 2.0
    return lo - span * pad, hi + span * pad


def _quadrant_rects(fig: go.Figure, xmin: float, xmax: float, ymin: float, ymax: float) -> None:
    boxes = (
        ("Leading", 100, xmax, 100, ymax),
        ("Improving", xmin, 100, 100, ymax),
        ("Lagging", xmin, 100, ymin, 100),
        ("Weakening", 100, xmax, ymin, 100),
    )
    for name, x0, x1, y0, y1 in boxes:
        fig.add_shape(
            type="rect",
            x0=x0,
            x1=x1,
            y0=y0,
            y1=y1,
            fillcolor=_QUADRANT_FILL[name],
            line=dict(width=0),
            layer="below",
        )


def _corner_labels(fig: go.Figure) -> None:
    spots = (
        ("IMPROVING", 0.012, 0.985, "left", "top", QUADRANT_COLOR["Improving"]),
        ("LEADING", 0.988, 0.985, "right", "top", QUADRANT_COLOR["Leading"]),
        ("LAGGING", 0.012, 0.02, "left", "bottom", QUADRANT_COLOR["Lagging"]),
        ("WEAKENING", 0.988, 0.02, "right", "bottom", QUADRANT_COLOR["Weakening"]),
    )
    for text, x, y, xanchor, yanchor, color in spots:
        fig.add_annotation(
            text=text,
            xref="paper",
            yref="paper",
            x=x,
            y=y,
            xanchor=xanchor,
            yanchor=yanchor,
            showarrow=False,
            font=dict(color=color, size=12, family=_FONT),
        )


def _cycle_title(cycle: SeasonalCycle, ticker: str, asset_label: str, mode_label: str) -> str:
    idx = "—" if pd.isna(cycle.seas_idx) else str(int(round(cycle.seas_idx)))
    if pd.isna(cycle.bias_pts):
        bias, color = "—", "#d5dde8"
    else:
        bias = f"{cycle.bias_pts:+.1f} pts"
        color = "#3DDC97" if cycle.bias_pts >= 0 else "#E15B64"
    win = "—" if pd.isna(cycle.win_rate_pct) else f"{cycle.win_rate_pct:.0f}%"
    return (
        f"Larry Williams Annual Seasonal Cycle: {ticker} {mode_label} ({asset_label})"
        f"<br><span style='font-size:12px;color:#b7c3d4'>Current Idx: {idx}/100"
        f" | 30d Bias: <span style='color:{color}'>{bias}</span>"
        f" | 30d Win Rate: {win}</span>"
    )


def _axis_style() -> dict:
    return dict(
        gridcolor=_GRID,
        zeroline=False,
        showline=False,
        color=_AXIS,
        tickfont=dict(size=11, color=_AXIS, family=_FONT),
        title_font=dict(size=12, color="#9aa8ba", family=_FONT),
        automargin=True,
    )


def _base_layout(height: int, top: int = 58) -> dict:
    return dict(
        height=height,
        paper_bgcolor=_PAPER,
        plot_bgcolor=PLOT_BG,
        font=dict(family=_FONT, color="#d5dde8", size=12),
        margin=dict(l=72, r=28, t=top, b=48),
        hoverlabel=dict(bgcolor="#111a2e", bordercolor="#243044", font=dict(color="#e6edf5", family=_FONT)),
    )
