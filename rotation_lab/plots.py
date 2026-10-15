"""Matplotlib visualisations: RRG with trails, seasonal grid, heatmap, outlook.

All plotting functions return a :class:`matplotlib.figure.Figure` and take a
``theme`` name (``"light"`` or ``"dark"``) so the same code renders equally
well to PNG files or inside the Streamlit app.
"""

from __future__ import annotations

from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg", force=False)  # headless-safe; Streamlit overrides fine

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import to_rgba

from .config import Asset, asset_index
from .rrg import RRGResult
from .seasonality import MonthlySeasonality, SeasonalOutlook, SeasonalProfile

__all__ = [
    "THEMES",
    "Theme",
    "get_theme",
    "plot_monthly_heatmap",
    "plot_rrg",
    "plot_seasonal_grid",
    "plot_seasonal_outlook",
    "save_figure",
]


# --------------------------------------------------------------------------- #
# themes
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Theme:
    name: str
    bg: str
    panel: str
    text: str
    muted: str
    grid: str
    centerline: str
    leading: str
    improving: str
    weakening: str
    lagging: str
    quadrant_alpha: float


THEMES: dict[str, Theme] = {
    "light": Theme(
        name="light",
        bg="#ffffff",
        panel="#f7f9fb",
        text="#1d2733",
        muted="#6b7684",
        grid="#e4e9ef",
        centerline="#aab4c2",
        leading="#2e7d5b",
        improving="#2f6db3",
        weakening="#b98a1d",
        lagging="#bb4d4d",
        quadrant_alpha=0.11,
    ),
    "dark": Theme(
        name="dark",
        bg="#10151d",
        panel="#151b25",
        text="#e6ebf2",
        muted="#8b96a6",
        grid="#242d3a",
        centerline="#3a4556",
        leading="#4fd18b",
        improving="#5aa7f0",
        weakening="#e8c15a",
        lagging="#ef6b6b",
        quadrant_alpha=0.13,
    ),
}


def get_theme(name: str = "light") -> Theme:
    try:
        return THEMES[name]
    except KeyError as exc:  # pragma: no cover - CLI validates choices
        raise ValueError(f"Unknown theme {name!r}; choose from {sorted(THEMES)}") from exc


def _style_axes(ax, theme: Theme, *, grid: bool = True) -> None:
    ax.set_facecolor(theme.panel)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(theme.grid)
    ax.tick_params(colors=theme.muted, labelsize=9)
    if grid:
        ax.grid(True, color=theme.grid, linewidth=0.7, alpha=0.9)
        ax.set_axisbelow(True)


def _empty_figure(theme: str, title: str, message: str):
    """A real figure, used when the sample is too short to draw the chart."""
    th = get_theme(theme)
    fig, ax = plt.subplots(figsize=(8, 3.2))
    fig.patch.set_facecolor(th.bg)
    ax.set_facecolor(th.bg)
    ax.set_axis_off()
    ax.text(0.5, 0.55, message, ha="center", va="center", color=th.text, fontsize=12)
    fig.suptitle(title, color=th.text, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return fig


def _meta_line(source: str, as_of: pd.Timestamp | None = None) -> str:
    stamp = f"as of {as_of:%Y-%m-%d}" if as_of is not None else "as of today"
    return f"{stamp}  ·  data: {source}"


def save_figure(fig, path) -> str:
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return str(path)


# --------------------------------------------------------------------------- #
# RRG
# --------------------------------------------------------------------------- #
def plot_rrg(
    result: RRGResult,
    assets: list[Asset],
    *,
    trail: int = 26,
    resample: str | None = "W-FRI",
    focus: str | None = None,
    theme: str = "light",
    source: str = "",
    title: str | None = None,
) -> "plt.Figure":
    """Relative Rotation Graph with gradient trails and directional arrowheads."""
    th = get_theme(theme)
    meta = asset_index(assets)
    trails = result.trails(periods=trail, resample=resample)
    if not trails:
        return _empty_figure(
            theme,
            title or "Relative Rotation Graph",
            "Not enough history for this RRG window (need about two windows).",
        )

    fig, ax = plt.subplots(figsize=(11.5, 9.5))
    fig.patch.set_facecolor(th.bg)
    _style_axes(ax, th)

    # ---- axis limits (data driven, always containing the 100/100 cross) ----
    xs = np.concatenate([t["rs_ratio"].to_numpy() for t in trails.values()])
    ys = np.concatenate([t["rs_momentum"].to_numpy() for t in trails.values()])
    lo_x, hi_x = min(100.0, float(np.nanmin(xs))), max(100.0, float(np.nanmax(xs)))
    lo_y, hi_y = min(100.0, float(np.nanmin(ys))), max(100.0, float(np.nanmax(ys)))
    span_x = max(hi_x - lo_x, 2.0)
    span_y = max(hi_y - lo_y, 2.0)
    ax.set_xlim(lo_x - 0.20 * span_x, hi_x + 0.20 * span_x)
    ax.set_ylim(lo_y - 0.20 * span_y, hi_y + 0.20 * span_y)

    # ---- quadrant shading ----
    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    quadrants = (
        (100, x1, 100, y1, th.leading, "LEADING", (0.985, 0.985)),
        (x0, 100, 100, y1, th.improving, "IMPROVING", (0.015, 0.985)),
        (100, x1, y0, 100, th.weakening, "WEAKENING", (0.985, 0.015)),
        (x0, 100, y0, 100, th.lagging, "LAGGING", (0.015, 0.015)),
    )
    for qx0, qx1, qy0, qy1, color, label, (tx, ty) in quadrants:
        ax.fill_between([qx0, qx1], qy0, qy1, color=color, alpha=th.quadrant_alpha, lw=0, zorder=0)
        ax.text(
            qx0 + tx * (qx1 - qx0),
            qy0 + ty * (qy1 - qy0),
            label,
            transform=ax.transData,
            ha="right" if tx < 0.5 else "left",
            va="top" if ty > 0.5 else "bottom",
            fontsize=11,
            fontweight="bold",
            color=color,
            alpha=0.85,
            zorder=1,
        )

    ax.axvline(100, color=th.centerline, lw=1.1, ls="--", zorder=2)
    ax.axhline(100, color=th.centerline, lw=1.1, ls="--", zorder=2)

    # ---- trails ----
    for ticker, frame in trails.items():
        asset = meta.get(ticker, Asset(ticker, ticker))
        color = asset.color or "#4C72B0"
        dim = focus is not None and ticker != focus
        x = frame["rs_ratio"].to_numpy()
        y = frame["rs_momentum"].to_numpy()
        n = len(frame)

        if n >= 2:
            points = np.column_stack([x, y]).reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            alphas = np.linspace(0.12, 0.95, len(segments)) * (0.35 if dim else 1.0)
            lc = LineCollection(
                segments,
                colors=[to_rgba(color, float(a)) for a in alphas],
                linewidths=2.0 if not dim else 1.2,
                zorder=4,
            )
            ax.add_collection(lc)
            ax.annotate(
                "",
                xy=(x[-1], y[-1]),
                xytext=(x[-2], y[-2]),
                arrowprops=dict(
                    arrowstyle="-|>",
                    color=to_rgba(color, 0.45 if dim else 0.95),
                    lw=1.8,
                    mutation_scale=16,
                ),
                zorder=5,
            )

        ax.scatter(x[-1], y[-1], s=34, color=color, edgecolor=th.bg, linewidth=0.8, zorder=6, alpha=0.45 if dim else 1.0)
        ax.annotate(
            asset.label,
            xy=(x[-1], y[-1]),
            xytext=(9, 9),
            textcoords="offset points",
            fontsize=9.5,
            fontweight="bold",
            color=color if not dim else to_rgba(color, 0.5),
            bbox=dict(boxstyle="round,pad=0.25", fc=th.bg, ec=color if not dim else th.grid, lw=0.8, alpha=0.9),
            zorder=7,
        )
        if n:
            ax.annotate(
                f"{frame.index[0]:%b %d}",
                xy=(x[0], y[0]),
                xytext=(-4, -12),
                textcoords="offset points",
                fontsize=7.5,
                color=th.muted if not dim else to_rgba(th.muted, 0.4),
                zorder=6,
            )

    ax.set_xlabel(f"RS-Ratio  ·  relative strength vs {result.benchmark} (100 = neutral, right = stronger)", color=th.text, fontsize=10.5)
    ax.set_ylabel(f"RS-Momentum  ·  {result.params.window}d z-score of RS change (100 = neutral, up = accelerating)", color=th.text, fontsize=10.5)

    trail_desc = f"{trail} {'weekly' if resample else 'daily'} points"
    if title:
        fig.suptitle(title, color=th.text, fontsize=15, fontweight="bold", x=0.125, ha="left")
        fig.text(
            0.125,
            0.925,
            f"{trail_desc} per trail  ·  z-scale {result.params.z_scale:g}  ·  {_meta_line(source)}",
            color=th.muted,
            fontsize=9.5,
            ha="left",
        )
    fig.tight_layout(rect=(0, 0, 1, 0.91))
    return fig


# --------------------------------------------------------------------------- #
# seasonal grid
# --------------------------------------------------------------------------- #
def plot_seasonal_grid(
    profile: SeasonalProfile,
    prices: pd.DataFrame,
    assets: list[Asset],
    outlook: SeasonalOutlook | None = None,
    *,
    theme: str = "light",
    source: str = "",
    weeks_ahead: int = 4,
) -> "plt.Figure":
    """Small-multiple average-year chart per asset: history band vs this year."""
    th = get_theme(theme)
    meta = asset_index(assets)
    tickers = [t for t in profile.composite.columns if t in prices.columns]
    if not tickers:
        return _empty_figure(
            theme,
            "Larry Williams-style seasonality",
            "Not enough history for a seasonal composite (need a year that starts by week 4).",
        )

    ncols = min(4, len(tickers))
    nrows = int(np.ceil(len(tickers) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(4.2 * ncols, 3.4 * nrows), squeeze=False, sharex=True)
    fig.patch.set_facecolor(th.bg)

    expected_map = {}
    if outlook is not None and not outlook.table.empty:
        expected_map = dict(zip(outlook.table["ticker"], outlook.table["expected_return_pct"]))

    for ax, ticker in zip(axes.ravel(), tickers):
        _style_axes(ax, th)
        asset = meta.get(ticker, Asset(ticker, ticker))
        color = asset.color or "#4C72B0"
        comp = profile.composite[ticker].dropna()
        p10 = profile.p10[ticker].reindex(comp.index)
        p90 = profile.p90[ticker].reindex(comp.index)

        series = prices[ticker].dropna()
        current_week = min(int(series.index[-1].isocalendar().week), 52)
        paths = profile.per_year.get(ticker)

        ax.fill_between(comp.index, p10, p90, color=color, alpha=0.16, lw=0, label="10-90th pct of years")
        ax.plot(comp.index, comp, color=color, lw=2.0, label="Seasonal composite")

        if paths is not None and profile.current_year in paths.columns:
            cur = paths[profile.current_year].dropna()
            cur = cur[cur.index <= current_week]
            ax.plot(cur.index, cur, color=th.text, lw=1.4, ls="--", label=f"{profile.current_year} actual")

        ax.axhline(100, color=th.centerline, lw=0.9, ls=":", alpha=0.8)
        ax.axvline(current_week, color=th.muted, lw=1.0, ls=":", alpha=0.9)
        _shade_forward(ax, current_week, weeks_ahead, color)
        hit = comp.reindex([current_week]).dropna()
        if not hit.empty:
            ax.scatter([current_week], [hit.iloc[0]], s=30, color=color, edgecolor=th.bg, zorder=5)

        exp = expected_map.get(ticker, np.nan)
        exp_txt = f"  ·  next {weeks_ahead}w {exp:+.1f}%" if pd.notna(exp) else ""
        ax.set_title(f"{asset.label}{exp_txt}", color=th.text, fontsize=11, fontweight="bold", loc="left")
        ax.text(
            0.985,
            0.03,
            f"wk {current_week}",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=8,
            color=th.muted,
        )

    for ax in axes.ravel()[len(tickers):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        if ax.get_visible():
            ax.set_xlabel("ISO week of year", color=th.muted, fontsize=9)
    for row in axes:
        if row[0].get_visible():
            row[0].set_ylabel("Index (100 = year anchor)", color=th.muted, fontsize=9)

    handles, labels = axes.ravel()[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper right", ncol=3, frameon=False, fontsize=9, labelcolor=th.text)

    fig.suptitle(
        "Larry Williams-style seasonality  ·  average-year composite vs history and this year",
        color=th.text,
        fontsize=14,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    fig.text(0.02, 0.955, _meta_line(source), color=th.muted, fontsize=9.5, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    return fig


# --------------------------------------------------------------------------- #
# monthly heatmap
# --------------------------------------------------------------------------- #
def plot_monthly_heatmap(
    monthly: MonthlySeasonality,
    assets: list[Asset],
    *,
    theme: str = "light",
    source: str = "",
    metric: str = "mean",
) -> "plt.Figure":
    """Assets x months heatmap of a monthly seasonality metric."""
    th = get_theme(theme)
    meta = asset_index(assets)
    table = {"mean": monthly.mean_pct, "win_rate": monthly.win_rate_pct, "median": monthly.median_pct}[metric]
    if table.empty:
        return _empty_figure(theme, "Monthly seasonality", "Not enough monthly history to shade a heatmap.")

    labels = [meta.get(t, Asset(t, t)).label for t in table.index]
    data = table.to_numpy(dtype=float)
    vmax = float(np.nanmax(np.abs(data)))
    vmax = vmax if vmax > 0 else 1.0

    fig, ax = plt.subplots(figsize=(1.05 * len(table.columns) + 3.2, 0.52 * len(table.index) + 2.4))
    fig.patch.set_facecolor(th.bg)
    ax.set_facecolor(th.panel)

    im = ax.imshow(data, cmap="RdYlGn", vmin=-vmax, vmax=vmax, aspect="auto")
    ax.set_xticks(range(len(table.columns)))
    ax.set_xticklabels([_month_name(m) for m in table.columns], color=th.text, fontsize=9.5)
    ax.set_yticks(range(len(table.index)))
    ax.set_yticklabels(labels, color=th.text, fontsize=9.5)
    ax.tick_params(colors=th.muted)
    for side in ax.spines.values():
        side.set_visible(False)

    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            val = data[i, j]
            if np.isnan(val):
                continue
            ink = "#1d2733" if abs(val) < 0.75 * vmax else "#ffffff"
            text = f"{val:.1f}" if metric != "win_rate" else f"{val:.0f}%"
            ax.text(j, i, text, ha="center", va="center", fontsize=8.5, color=ink)

    unit = "%" if metric != "win_rate" else "% win rate"
    title = "Monthly seasonality" + (" (mean return, %)" if metric == "mean" else f" ({unit})")
    fig.suptitle(title, color=th.text, fontsize=13, fontweight="bold", x=0.03, ha="left")
    fig.text(0.03, 0.925, _meta_line(source), color=th.muted, fontsize=9, ha="left")
    cbar = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.02)
    cbar.ax.tick_params(colors=th.muted, labelsize=8)
    cbar.outline.set_visible(False)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    return fig


# --------------------------------------------------------------------------- #
# forward outlook bars
# --------------------------------------------------------------------------- #
def plot_seasonal_outlook(
    outlook: SeasonalOutlook,
    assets: list[Asset],
    *,
    theme: str = "light",
    source: str = "",
    weeks_ahead: int = 4,
) -> "plt.Figure":
    """Ranked bars of each asset's expected seasonal drift over the next N weeks."""
    th = get_theme(theme)
    meta = asset_index(assets)
    df = outlook.table.dropna(subset=["expected_return_pct"]) if "expected_return_pct" in outlook.table.columns else outlook.table.iloc[0:0]
    if df.empty:
        return _empty_figure(
            theme,
            "Forward seasonal outlook",
            "No completed historical window for this horizon.",
        )

    df = df.sort_values("expected_return_pct")
    labels = [meta.get(t, Asset(t, t)).label for t in df["ticker"]]
    colors = ["#2e7d5b" if v >= 0 else "#bb4d4d" for v in df["expected_return_pct"]]

    fig, ax = plt.subplots(figsize=(9.5, max(3.2, 0.62 * len(df) + 2.2)))
    fig.patch.set_facecolor(th.bg)
    _style_axes(ax, th, grid=True)

    y = np.arange(len(df))
    ax.barh(y, df["expected_return_pct"], color=colors, alpha=0.85, height=0.62, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, color=th.text, fontsize=10)
    ax.axvline(0, color=th.centerline, lw=1.0)
    ax.set_xlabel(f"Expected seasonal drift, next {weeks_ahead} ISO weeks (%)", color=th.text, fontsize=10)

    for yi, (value, win, years) in enumerate(zip(df["expected_return_pct"], df["win_rate_pct"], df["years"])):
        offset = 0.06 * (df["expected_return_pct"].max() - df["expected_return_pct"].min() + 1e-9)
        x = value + (offset if value >= 0 else -offset)
        ax.text(
            x,
            yi,
            f"{value:+.1f}%  ·  {win:.0f}% win  ·  n={int(years)}y",
            va="center",
            ha="left" if value >= 0 else "right",
            fontsize=8.5,
            color=th.text,
        )

    pad = 0.25 * (df["expected_return_pct"].max() - df["expected_return_pct"].min()) + 1.0
    ax.set_xlim(min(df["expected_return_pct"].min() - pad, 0), max(df["expected_return_pct"].max() + pad, 0))

    fig.suptitle(
        f"Forward seasonal outlook  ·  historical drift over the next {weeks_ahead} weeks",
        color=th.text,
        fontsize=13,
        fontweight="bold",
        x=0.02,
        ha="left",
    )
    fig.text(0.02, 0.93, _meta_line(source), color=th.muted, fontsize=9, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    return fig


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _shade_forward(ax, current_week: int, weeks_ahead: int, color: str) -> None:
    """Mark the outlook window, including the wrap from week 52 back to week 1."""
    if weeks_ahead < 1:
        return
    end = current_week + weeks_ahead
    if end <= 52:
        ax.axvspan(current_week, end, color=color, alpha=0.08, lw=0, zorder=0)
        return
    ax.axvspan(current_week, 52.4, color=color, alpha=0.08, lw=0, zorder=0)
    ax.axvspan(0.6, end - 52, color=color, alpha=0.08, lw=0, zorder=0)


def _month_name(month: int) -> str:
    return (
        "Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec",
    )[int(month) - 1]
