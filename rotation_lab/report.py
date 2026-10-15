"""Markdown report generation: tables + embedded chart images + methodology."""

from __future__ import annotations

import html
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import Asset, asset_index
from .rrg import RRGResult
from .seasonality import MonthlySeasonality, SeasonalOutlook, SeasonalProfile
from .signals import alignment_board

__all__ = ["write_report"]


def write_report(
    path: Path | str,
    *,
    result: RRGResult,
    snapshot: pd.DataFrame,
    profile: SeasonalProfile,
    monthly: MonthlySeasonality,
    outlook: SeasonalOutlook,
    assets: list[Asset],
    benchmark: Asset,
    source: str = "",
    is_demo: bool = False,
    trail: int = 26,
    weeks_ahead: int = 4,
    images: dict[str, str] | None = None,
) -> str:
    """Write a self-contained markdown report and return its path."""
    path = Path(path)
    images = images or {}
    meta = asset_index([*assets, benchmark])

    labels = {ticker: asset.label for ticker, asset in meta.items()}
    board = alignment_board(snapshot, outlook, labels=labels)
    board_out = _format_board(board)

    snap = snapshot.copy()
    if not snap.empty:
        snap["label"] = snap["ticker"].map(lambda t: meta.get(t, Asset(t, t)).label)
        snap["group"] = snap["ticker"].map(lambda t: meta.get(t, Asset(t, t)).group)
        snap["rs_ratio"] = snap["rs_ratio"].map(lambda v: f"{v:.2f}")
        snap["rs_momentum"] = snap["rs_momentum"].map(lambda v: f"{v:.2f}")
        if "heading_deg" in snap.columns:
            snap["heading_deg"] = snap["heading_deg"].map(lambda v: "-" if pd.isna(v) else f"{v:.0f}")
        snap["as_of"] = snap["as_of"].map(lambda d: f"{d:%Y-%m-%d}")
        columns = ["quadrant", "label", "ticker", "rs_ratio", "rs_momentum", "heading_deg", "as_of"]
        snap = snap[[c for c in columns if c in snap.columns]]

    out = outlook.table.copy()
    if not out.empty:
        out["expected_return_pct"] = out["expected_return_pct"].map(lambda v: _signed(v, 2))
        out["win_rate_pct"] = out["win_rate_pct"].map(lambda v: "-" if pd.isna(v) else f"{v:.0f}")
        out["seasonal_z"] = out["seasonal_z"].map(lambda v: _signed(v, 2))
        out["ytd_vs_history_pct"] = out["ytd_vs_history_pct"].map(lambda v: "-" if pd.isna(v) else f"{v:.0f}")
        out = out[["rank", "label", "ticker", "current_week", "target_week", "expected_return_pct", "win_rate_pct", "seasonal_z", "ytd_vs_history_pct", "years"]]

    monthly_out = monthly.mean_pct.round(2)
    monthly_out.columns = [_month_name(int(m)) for m in monthly_out.columns]
    monthly_out.index = [meta.get(t, Asset(t, t)).label for t in monthly_out.index]
    monthly_out.index.name = "asset"

    win_out = monthly.win_rate_pct.round(0)
    win_out.columns = [_month_name(int(m)) for m in win_out.columns]
    win_out.index = [meta.get(t, Asset(t, t)).label for t in win_out.index]
    win_out.index.name = "asset"

    lines: list[str] = []
    add = lines.append

    add("# Intermarket Rotation Report")
    add("")
    add(f"*Generated {datetime.now():%Y-%m-%d %H:%M} · data source: {source or 'unknown'}*")
    add("")

    if is_demo:
        add("> **Warning: demo/synthetic data.** Yahoo Finance was unreachable and no cache was")
        add("> available at run time, so this report was rendered from the built-in synthetic")
        add("> generator. Numbers below illustrate the method, not the market.")
        add("")

    add("## 1. Relative Rotation Graph")
    add("")
    add(
        f"The RRG plots each asset's **RS-Ratio** (relative strength vs {benchmark.label}, "
        f"{benchmark.ticker}) against its **RS-Momentum** (the rate of change of RS-Ratio). "
        f"Values are z-scored over {result.params.window} trading days and scaled by "
        f"{result.params.z_scale:g}; 100 is neutral on both axes."
    )
    add("")
    add(f"Trails show the last {trail} weekly observations per asset; arrowheads mark the latest position.")
    add("`heading_deg` is the direction of the last five sessions: 0 is due east (RS rising), 90 is due north.")
    add("")
    if "rrg" in images:
        add(f"![Relative Rotation Graph]({images['rrg']})")
        add("")
    add("### Latest readings")
    add("")
    add(_md_table(snap))
    add("")
    add(
        "**How to read it:** *Leading* = strong and still accelerating; *Weakening* = still strong "
        "but momentum fading; *Lagging* = weak and still falling; *Improving* = weak but turning up "
        "(the classic early-rotation quadrant). Watch **Improving -> Leading** transitions for "
        "long entries and **Leading -> Weakening** transitions for exits."
    )
    add("")

    add("## 2. Seasonality — Larry Williams style")
    add("")
    add(
        "The seasonal composite averages the price path of every complete year (log returns, "
        "rebased to 100 at each year's first week), so long-term drift is removed and only the "
        "*calendar tendency* remains. The shaded band spans the 10th–90th percentile of "
        "historical years: a narrow band means the pattern is consistent; a wide band means it "
        "is a coin flip dressed up as history."
    )
    add("")
    if "seasonality" in images:
        add(f"![Seasonality grid]({images['seasonality']})")
        add("")

    add("### Forward seasonal outlook (next {} ISO weeks)".format(weeks_ahead))
    add("")
    add(
        "`expected_return_pct` is the equal-weighted drift of the next window across completed "
        "years (week 52 wraps into week 1 of the following year); `win_rate_pct` is the share of "
        "those years that were positive; `seasonal_z` ranks the drift across assets in this "
        "universe; `ytd_vs_history_pct` shows where this year's performance-to-date ranks among "
        "prior years at the same week."
    )
    add("")
    add(_md_table(out))
    add("")
    if "outlook" in images:
        add(f"![Seasonal outlook]({images['outlook']})")
        add("")

    add("### Monthly seasonality — mean return (%)")
    add("")
    add("The in-progress month is excluded so numbers are not polluted by a partial period.")
    add("")
    add(_md_table(monthly_out))
    add("")
    if "heatmap" in images:
        add(f"![Monthly heatmap]({images['heatmap']})")
        add("")
    add("### Monthly seasonality — win rate (%)")
    add("")
    add(_md_table(win_out))
    add("")

    add("## 3. Combining the two lenses")
    add("")
    add(
        "RRG is **relative** (versus the benchmark). Seasonality is **absolute** "
        "(the asset's own calendar). Score = quadrant score "
        "(Leading +2, Improving +1, Weakening -1, Lagging -2) "
        "+ sign of the seasonal drift. Sorted with the most aligned names on top."
    )
    add("")
    add(_md_table(board_out))
    add("")
    add("| RRG quadrant | Seasonal tailwind next {}w | Interpretation |".format(weeks_ahead))
    add("|---|---|---|")
    add("| Leading | positive | Aligned: relative strength and calendar agree - trend continuation candidate |")
    add("| Leading | negative | Late: strength may be late-cycle; seasonal headwind ahead |")
    add("| Improving | positive | Early rotation with seasonal support - classic entry setup |")
    add("| Improving | negative | Turn may stall; wait for confirmation |")
    add("| Weakening | positive | Fading RS but a seasonal bid - trim, don't kill |")
    add("| Weakening | negative | Two negatives stack - reduce exposure |")
    add("| Lagging | positive | Contrarian seasonal play only; no relative strength yet |")
    add("| Lagging | negative | Avoid; nothing supports it |")
    add("")

    add("## 4. Methodology & caveats")
    add("")
    add("- **RRG**: `RS = P_asset / P_benchmark`; `RS-Ratio = 100 + {z:g}·z(log RS, {w}d)`; "
        "`RS-Momentum = 100 + {z:g}·z(ΔRS-Ratio, {w}d)`. Rolling z-scores keep the axes "
        "comparable across regimes. Warm-up is ~2×the window, so early history is excluded.".format(
            z=result.params.z_scale, w=result.params.window
        ))
    add("- **Seasons**: weekly log returns are bucketed by ISO year and ISO week (week 53 "
        "folds into week 52) and cumulated per year; the composite is the cross-year mean, "
        "with 10/90 percentile bands and win rates. Years that do not begin by ISO week 4 "
        "are dropped from the composite as partial. The forward window compounds the next "
        "N ISO weeks and wraps week 52 into week 1 of the following year. The in-progress "
        "year is excluded from the composite, the win rate and the outlook sample.")
    add("- **Seasonality is a tendency, not a guarantee.** Sample sizes are small ({}–{} years "
        "per asset here); treat seasonal edges as a tilt, and always size positions by where "
        "the stop — not the story — sits.".format(
            min(profile.n_years.values(), default=0), max(profile.n_years.values(), default=0)
        ))
    add("- Prices are auto-adjusted closes (dividends/splits handled). Crypto trades 7 days; "
        "ETF holidays are forward-filled.")
    add("")

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")
    index = path.parent / "index.html"
    index.write_text(
        _index_html(
            generated=datetime.now(),
            source=source,
            is_demo=is_demo,
            benchmark=benchmark,
            window=result.params.window,
            z_scale=result.params.z_scale,
            trail=trail,
            weeks_ahead=weeks_ahead,
            n_years=profile.n_years,
            images=images,
            board=board_out,
            snapshot=snap,
            outlook=out,
            monthly=monthly_out,
            win_rate=win_out,
        ),
        encoding="utf-8",
    )
    # GitHub Pages should serve the folder as files, not through Jekyll.
    (path.parent / ".nojekyll").write_text("", encoding="utf-8")
    return str(path)


# --------------------------------------------------------------------------- #
def _index_html(
    *,
    generated: datetime,
    source: str,
    is_demo: bool,
    benchmark: Asset,
    window: int,
    z_scale: float,
    trail: int,
    weeks_ahead: int,
    n_years: dict[str, int],
    images: dict[str, str],
    board: pd.DataFrame,
    snapshot: pd.DataFrame,
    outlook: pd.DataFrame,
    monthly: pd.DataFrame,
    win_rate: pd.DataFrame,
) -> str:
    """Static page for GitHub Pages and for opening the report in a browser."""
    lo = min(n_years.values(), default=0)
    hi = max(n_years.values(), default=0)
    warning = ""
    if is_demo:
        warning = (
            "<p class='banner'>Synthetic demo data. Yahoo Finance was unreachable and no cache "
            "was available, so these figures illustrate the method, not the market.</p>"
        )

    def figure(key: str, caption: str) -> str:
        src = images.get(key)
        if not src:
            return ""
        return (
            f"<figure><img src='{html.escape(src, quote=True)}' alt='{html.escape(caption)}'>"
            f"<figcaption>{html.escape(caption)}</figcaption></figure>"
        )

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Intermarket rotation · {html.escape(benchmark.ticker)}</title>
<style>
  :root {{ color-scheme: light; --ink: #1c1915; --muted: #5c564c; --line: #ddd6c8; --paper: #f6f3ec; --card: #fffdf8; --up: #1f6b45; --down: #9c2f2f; }}
  * {{ box-sizing: border-box; }}
  body {{ margin: 0; background: var(--paper); color: var(--ink); font: 16px/1.5 "Iowan Old Style", Palatino, Georgia, serif; }}
  main {{ max-width: 1080px; margin: 0 auto; padding: 2.5rem 1.25rem 4rem; }}
  h1 {{ font-size: 2.1rem; line-height: 1.15; margin: 0 0 0.4rem; letter-spacing: -0.02em; }}
  h2 {{ font-size: 1.35rem; margin: 2.4rem 0 0.6rem; }}
  p {{ margin: 0.4rem 0 0.8rem; }}
  .meta, figcaption, footer {{ color: var(--muted); font-family: ui-sans-serif, system-ui, sans-serif; font-size: 0.85rem; }}
  .banner {{ background: #f3e2c2; border: 1px solid #e0c48a; padding: 0.75rem 0.9rem; border-radius: 6px; }}
  figure {{ margin: 1rem 0 1.4rem; background: var(--card); border: 1px solid var(--line); border-radius: 8px; padding: 0.6rem; }}
  img {{ width: 100%; height: auto; display: block; }}
  figcaption {{ margin-top: 0.4rem; }}
  .table-wrap {{ overflow-x: auto; border: 1px solid var(--line); border-radius: 8px; background: var(--card); }}
  table {{ border-collapse: collapse; width: 100%; font-family: ui-sans-serif, system-ui, sans-serif; font-size: 0.82rem; }}
  th, td {{ padding: 0.45rem 0.6rem; text-align: right; white-space: nowrap; border-bottom: 1px solid var(--line); }}
  th:first-child, td:first-child, th:nth-child(2), td:nth-child(2) {{ text-align: left; }}
  th {{ color: var(--muted); font-weight: 600; background: #faf7f1; }}
  td.up {{ color: var(--up); }} td.down {{ color: var(--down); }}
  td.Leading {{ color: var(--up); font-weight: 650; }}
  td.Improving {{ color: #1d4e89; font-weight: 650; }}
  td.Weakening {{ color: #8a5a10; font-weight: 650; }}
  td.Lagging {{ color: var(--down); font-weight: 650; }}
  ul {{ padding-left: 1.2rem; }}
  footer {{ margin-top: 2.5rem; border-top: 1px solid var(--line); padding-top: 0.8rem; }}
</style>
</head>
<body>
<main>
  <p class="meta">Intermarket rotation lab</p>
  <h1>Relative rotation versus {html.escape(benchmark.label)}</h1>
  <p class="meta">Generated {generated:%Y-%m-%d %H:%M} · data: {html.escape(source or "unknown")} · benchmark {html.escape(benchmark.ticker)}</p>
  {warning}
  <p>Each point is an asset’s relative strength against {html.escape(benchmark.label)} and whether that strength is speeding up or slowing down. The seasonal panels are a separate question: what this market has usually done, on its own, at this point in the calendar. The board crosses the two. This is a study, not a recommendation.</p>

  <h2>Board</h2>
  <p>Score = quadrant (Leading +2, Improving +1, Weakening −1, Lagging −2) plus the sign of the seasonal drift over the next {weeks_ahead} ISO weeks. Highest alignment is at the top.</p>
  {_html_table(board)}

  <h2>Relative Rotation Graph</h2>
  <p>RS-Ratio is 100 + {z_scale:g} times the {window}-day z-score of log(price / {html.escape(benchmark.ticker)}). RS-Momentum is the same transform of the change in RS-Ratio. Trails are the last {trail} weekly points. Heading is the angle of the last five sessions: 0° due east, 90° due north.</p>
  {figure("rrg", "Relative Rotation Graph")}
  {_html_table(snapshot)}

  <h2>Seasonality</h2>
  <p>Each completed year is rebased to 100 at its first ISO week and averaged. The band is the 10th–90th percentile of those years. The shaded window is the next {weeks_ahead} weeks, wrapping week 52 into week 1 of the following year. The year still in progress is not part of the sample. Histories here run {lo}–{hi} years.</p>
  {figure("seasonality", "Average-year seasonal composite")}
  <h2>Forward window</h2>
  {_html_table(outlook)}
  {figure("outlook", "Ranked seasonal drift")}

  <h2>Month of the year</h2>
  <p>Mean return, in percent. The month still in progress is left out.</p>
  {_html_table(monthly)}
  {figure("heatmap", "Monthly seasonality heatmap")}
  <p>Share of years that month was positive.</p>
  {_html_table(win_rate)}

  <h2>Caveats</h2>
  <ul>
    <li>A seasonal average is a tendency. A wide percentile band means the years do not agree.</li>
    <li>The rotation axes are a rolling z-score, so they describe the recent regime, not a fixed historical percentile.</li>
    <li>These are ETF and spot closes, not futures. Roll yield is not in the series.</li>
  </ul>
  <footer>Not investment advice. Prices are adjusted closes; crypto weekends are forward-filled onto the equity calendar.</footer>
</main>
</body>
</html>
"""


def _html_table(frame: pd.DataFrame) -> str:
    if frame is None or frame.empty:
        return "<p class='empty'>No data.</p>"
    df = frame.copy()
    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()
    head = "".join(f"<th>{html.escape(str(col))}</th>" for col in df.columns)
    rows: list[str] = []
    named = {"Leading", "Weakening", "Lagging", "Improving"}
    for _, row in df.iterrows():
        cells: list[str] = []
        for value in row:
            if isinstance(value, float):
                text = "" if np.isnan(value) else f"{value:,.2f}"
            else:
                text = str(value)
            classes: list[str] = []
            if text in named:
                classes.append(text)
            elif text.startswith("+"):
                classes.append("up")
            elif text.startswith("-") and text != "-":
                classes.append("down")
            attr = f" class='{' '.join(classes)}'" if classes else ""
            cells.append(f"<td{attr}>{html.escape(text)}</td>")
        rows.append("<tr>" + "".join(cells) + "</tr>")
    return (
        "<div class='table-wrap'><table><thead><tr>"
        + head
        + "</tr></thead><tbody>"
        + "".join(rows)
        + "</tbody></table></div>"
    )


def _md_table(frame: pd.DataFrame) -> str:
    """Render a DataFrame as a GitHub-flavoured markdown table (no deps)."""
    if frame is None or frame.empty:
        return "_No data._"
    df = frame.copy()
    if not isinstance(df.index, pd.RangeIndex):
        df = df.reset_index()
    header = "| " + " | ".join(str(c) for c in df.columns) + " |"
    sep = "|" + "|".join(["---"] * len(df.columns)) + "|"
    rows = []
    for _, row in df.iterrows():
        cells = []
        for value in row:
            if isinstance(value, float):
                cells.append("" if np.isnan(value) else f"{value:,.2f}")
            else:
                cells.append(str(value))
        rows.append("| " + " | ".join(cells) + " |")
    return "\n".join([header, sep, *rows])


def _format_board(board: pd.DataFrame) -> pd.DataFrame:
    if board is None or board.empty:
        return board
    out = board.copy()
    for column, digits in (
        ("rs_ratio", 2),
        ("rs_momentum", 2),
        ("expected_return_pct", 2),
        ("seasonal_z", 2),
    ):
        if column in out.columns:
            signed = column in {"expected_return_pct", "seasonal_z"}
            out[column] = out[column].map(lambda v, d=digits, s=signed: _signed(v, d) if s else ("-" if pd.isna(v) else f"{v:.2f}"))
    for column in ("win_rate_pct", "ytd_vs_history_pct", "heading_deg"):
        if column in out.columns:
            out[column] = out[column].map(lambda v: "-" if pd.isna(v) else f"{v:.0f}")
    show = [
        "label",
        "ticker",
        "quadrant",
        "stance",
        "score",
        "rs_ratio",
        "rs_momentum",
        "expected_return_pct",
        "win_rate_pct",
        "seasonal_z",
    ]
    return out[[c for c in show if c in out.columns]]


def _signed(value, digits: int) -> str:
    if pd.isna(value):
        return "-"
    return f"{float(value):+.{digits}f}"


def _month_name(month: int) -> str:
    return ("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")[
        int(month) - 1
    ]
