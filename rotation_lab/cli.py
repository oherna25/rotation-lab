"""Command line interface for rotation-lab.

Examples
--------
Full pipeline (RRG + seasonality + report)::

    rotation-lab run

Offline demo with synthetic data::

    rotation-lab run --offline --out output_demo

Custom universe, dark charts, focused highlight::

    rotation-lab run --preset crypto
    rotation-lab run --assets "GLD=Gold,TLT=Long Bonds" --theme dark --focus GLD

RRG / seasonality only::

    rotation-lab rrg --trail 30 --resample W-FRI
    rotation-lab season --outlook-weeks 8
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from . import __version__
from .config import (
    DEFAULT_BENCHMARK,
    DEFAULT_START,
    PRESETS,
    asset_index,
    benchmark_asset,
    preset_universe,
    universe_from_spec,
)
from .data import DEFAULT_CACHE_DIR, fetch_prices
from .plots import (
    plot_monthly_heatmap,
    plot_rrg,
    plot_seasonal_grid,
    plot_seasonal_outlook,
    save_figure,
)
from .report import write_report
from .rrg import RRGParams, compute_rrg
from .seasonality import build_monthly_seasonality, build_seasonal_profile, seasonal_outlook
from .signals import alignment_board

LOG = logging.getLogger("rotation_lab")

__all__ = ["main"]


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    _setup_logging(args.quiet)
    if not getattr(args, "command", None):
        args = parser.parse_args([*(argv or []), "run"] if argv is not None else ["run"])
    try:
        if args.command == "rrg":
            return _cmd_rrg(args)
        if args.command == "season":
            return _cmd_season(args)
        return _cmd_run(args)
    except KeyboardInterrupt:  # pragma: no cover
        print("\nInterrupted.", file=sys.stderr)
        return 130


# --------------------------------------------------------------------------- #
# commands
# --------------------------------------------------------------------------- #
def _cmd_rrg(args) -> int:
    universe, benchmark, prices, source, is_demo = _load(args)
    result = compute_rrg(prices, benchmark.ticker, RRGParams(window=args.window, z_scale=args.z_scale))
    snapshot = result.snapshot()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    fig = plot_rrg(
        result,
        universe,
        trail=args.trail,
        resample=_resample(args.resample),
        focus=_focus(args),
        theme=args.theme,
        source=source,
        title=f"Intermarket Relative Rotation Graph  ·  benchmark {benchmark.ticker} ({benchmark.label})",
    )
    path = save_figure(fig, out_dir / "rrg.png")
    snapshot.to_csv(out_dir / "rrg_latest.csv", index=False)
    result.long_frame().to_csv(out_dir / "rrg_history.csv", index=False)

    print(f"RRG written to {path}")
    print(snapshot.to_string(index=False))
    if is_demo:
        print("\nNOTE: synthetic demo data in use (no network / no cache).")
    return 0


def _cmd_season(args) -> int:
    universe, benchmark, prices, source, is_demo = _load(args)
    all_assets = [*universe, benchmark]
    meta = asset_index(all_assets)
    labels = {t: a.label for t, a in meta.items()}

    profile = build_seasonal_profile(prices)
    monthly = build_monthly_seasonality(prices)
    outlook = seasonal_outlook(profile, prices, weeks_ahead=args.outlook_weeks, labels=labels)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = plot_seasonal_grid(
        profile,
        prices,
        all_assets,
        outlook,
        theme=args.theme,
        source=source,
        weeks_ahead=args.outlook_weeks,
    )
    heat = plot_monthly_heatmap(monthly, all_assets, theme=args.theme, source=source)
    bars = plot_seasonal_outlook(outlook, all_assets, theme=args.theme, source=source, weeks_ahead=args.outlook_weeks)

    save_figure(grid, out_dir / "seasonality_grid.png")
    save_figure(heat, out_dir / "monthly_heatmap.png")
    save_figure(bars, out_dir / "seasonal_outlook.png")
    profile.weekly_stats.to_csv(out_dir / "seasonal_weekly_stats.csv", index=False)
    monthly.long().to_csv(out_dir / "monthly_seasonality.csv", index=False)
    outlook.table.to_csv(out_dir / "seasonal_outlook.csv", index=False)

    print(f"Seasonality charts + CSVs written to {out_dir}/")
    print(outlook.table.to_string(index=False))
    if is_demo:
        print("\nNOTE: synthetic demo data in use (no network / no cache).")
    return 0


def _cmd_run(args) -> int:
    universe, benchmark, prices, source, is_demo = _load(args)
    all_assets = [*universe, benchmark]

    params = RRGParams(window=args.window, z_scale=args.z_scale)
    result = compute_rrg(prices, benchmark.ticker, params)
    snapshot = result.snapshot()

    profile = build_seasonal_profile(prices)
    monthly = build_monthly_seasonality(prices)
    labels = {ticker: asset.label for ticker, asset in asset_index(all_assets).items()}
    outlook = seasonal_outlook(profile, prices, weeks_ahead=args.outlook_weeks, labels=labels)
    board = alignment_board(snapshot, outlook, labels=labels)

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rrg_fig = plot_rrg(
        result,
        universe,
        trail=args.trail,
        resample=_resample(args.resample),
        focus=_focus(args),
        theme=args.theme,
        source=source,
        title=f"Intermarket Relative Rotation Graph  ·  benchmark {benchmark.ticker} ({benchmark.label})",
    )
    save_figure(rrg_fig, out_dir / "rrg.png")

    grid = plot_seasonal_grid(
        profile, prices, all_assets, outlook, theme=args.theme, source=source, weeks_ahead=args.outlook_weeks
    )
    save_figure(grid, out_dir / "seasonality_grid.png")

    heat = plot_monthly_heatmap(monthly, all_assets, theme=args.theme, source=source)
    save_figure(heat, out_dir / "monthly_heatmap.png")

    bars = plot_seasonal_outlook(
        outlook, all_assets, theme=args.theme, source=source, weeks_ahead=args.outlook_weeks
    )
    save_figure(bars, out_dir / "seasonal_outlook.png")

    # ---- data exports ----
    prices.to_csv(out_dir / "prices.csv")
    result.long_frame().to_csv(out_dir / "rrg_history.csv", index=False)
    snapshot.to_csv(out_dir / "rrg_latest.csv", index=False)
    profile.weekly_stats.to_csv(out_dir / "seasonal_weekly_stats.csv", index=False)
    monthly.long().to_csv(out_dir / "monthly_seasonality.csv", index=False)
    outlook.table.to_csv(out_dir / "seasonal_outlook.csv", index=False)
    board.to_csv(out_dir / "alignment.csv", index=False)

    if not args.no_report:
        write_report(
            out_dir / "report.md",
            result=result,
            snapshot=snapshot,
            profile=profile,
            monthly=monthly,
            outlook=outlook,
            assets=universe,
            benchmark=benchmark,
            source=source,
            is_demo=is_demo,
            trail=args.trail,
            weeks_ahead=args.outlook_weeks,
            images={
                "rrg": "rrg.png",
                "seasonality": "seasonality_grid.png",
                "heatmap": "monthly_heatmap.png",
                "outlook": "seasonal_outlook.png",
            },
        )

    # ---- console summary ----
    print(f"Universe : {', '.join(a.ticker for a in universe)}  (benchmark {benchmark.ticker})")
    print(f"Data     : {source}  ·  {prices.index[0]:%Y-%m-%d} -> {prices.index[-1]:%Y-%m-%d}  ·  {len(prices)} rows")
    print()
    print("RRG quadrant snapshot")
    print(snapshot.to_string(index=False))
    print()
    print(f"Forward seasonal outlook (next {args.outlook_weeks} ISO weeks)")
    print(outlook.table.to_string(index=False))
    print()
    print("Rotation x seasonality")
    print(board.to_string(index=False))
    print()
    extra = "" if args.no_report else ", report.md, index.html"
    print(f"Artifacts written to {out_dir}/  (charts, CSVs{extra})")
    if is_demo:
        print("\nNOTE: synthetic demo data in use (no network / no cache). Charts are illustrative only.")
    return 0


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def _load(args):
    universe = universe_from_spec(args.assets) if args.assets else preset_universe(args.preset)
    benchmark = benchmark_asset(args.benchmark)
    universe = [asset for asset in universe if asset.ticker != benchmark.ticker]
    if not universe:
        raise RuntimeError("Universe is empty after removing the benchmark")
    tickers = [a.ticker for a in universe] + [benchmark.ticker]

    data = fetch_prices(
        tickers,
        start=args.start,
        end=args.end,
        cache_dir=Path(args.cache_dir) if args.cache_dir else DEFAULT_CACHE_DIR,
        refresh_days=args.cache_refresh_days,
        offline=args.offline,
    )
    prices = data.prices.dropna(axis=1, how="all")
    if benchmark.ticker not in prices.columns:
        raise RuntimeError(f"Benchmark {benchmark.ticker} has no data")

    missing = [t for t in tickers if t not in prices.columns]
    if missing:
        LOG.warning("Dropping assets with no data: %s", ", ".join(missing))
        universe = [a for a in universe if a.ticker in prices.columns]
    if not universe:
        raise RuntimeError("No assets left with data")

    return universe, benchmark, prices, data.source, data.is_demo


def _resample(value: str | None) -> str | None:
    if value is None:
        return None
    value = value.strip()
    if value.lower() in {"", "none", "daily", "d"}:
        return None
    return value


def _focus(args) -> str | None:
    return args.focus.strip().upper() if args.focus else None


def _setup_logging(quiet: bool) -> None:
    level = logging.WARNING if quiet else logging.INFO
    logging.basicConfig(level=level, format="%(levelname)s %(message)s")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="rotation-lab",
        description="Intermarket Relative Rotation Graph + Larry Williams-style seasonality.",
    )
    parser.add_argument("--version", action="version", version=f"rotation-lab {__version__}")
    sub = parser.add_subparsers(dest="command")

    def common(p: argparse.ArgumentParser) -> None:
        p.add_argument("--preset", default="intermarket", choices=sorted(PRESETS), help="Named universe, ignored when --assets is set (default: %(default)s)")
        p.add_argument("--assets", default=None, help='Replaces the preset, e.g. "GLD=Gold,TLT=Long Bonds"')
        p.add_argument("--benchmark", default=DEFAULT_BENCHMARK.ticker, help="Benchmark ticker (default: %(default)s)")
        p.add_argument("--start", default=DEFAULT_START, help="History start date YYYY-MM-DD (default: %(default)s)")
        p.add_argument("--end", default=None, help="History end date YYYY-MM-DD (default: today)")
        p.add_argument("--out", default="output", help="Output directory (default: %(default)s)")
        p.add_argument("--theme", default="light", choices=["light", "dark"], help="Chart theme (default: %(default)s)")
        p.add_argument("--offline", action="store_true", help="Never touch the network; use cache or synthetic data")
        p.add_argument("--cache-dir", default=str(DEFAULT_CACHE_DIR), help="Price cache directory (default: %(default)s)")
        p.add_argument("--cache-refresh-days", type=int, default=1, help="Refetch cache older than N days (default: %(default)s)")
        p.add_argument("--quiet", action="store_true", help="Only log warnings and errors")
        # shared analytics knobs
        p.add_argument("--window", type=int, default=63, help="RRG z-score window in trading days (default: %(default)s)")
        p.add_argument("--z-scale", type=float, default=5.0, help="RRG chart points per std dev (default: %(default)s)")
        p.add_argument("--trail", type=int, default=26, help="Trail length per asset (default: %(default)s)")
        p.add_argument("--resample", default="W-FRI", help='Trail thinning rule; "none" for daily (default: %(default)s)')
        p.add_argument("--focus", default=None, help="Highlight one ticker, e.g. --focus GLD")
        p.add_argument("--outlook-weeks", type=int, default=4, help="Seasonal outlook horizon in ISO weeks (default: %(default)s)")
        p.add_argument("--no-report", action="store_true", help="Skip markdown report generation")

    run = sub.add_parser("run", help="Full pipeline: RRG + seasonality + charts + CSVs + report")
    common(run)
    rrg = sub.add_parser("rrg", help="Relative Rotation Graph only")
    common(rrg)
    season = sub.add_parser("season", help="Seasonality charts and tables only")
    common(season)

    return parser


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
