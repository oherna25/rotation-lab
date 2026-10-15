"""Single-screen RRG and Larry Williams seasonality dashboard.

Run it with::

    pip install -e ".[interactive]"
    streamlit run streamlit_app.py

The screen matches a sector-rotation terminal: trails on the left, the
seasonal matrix on the right, and the selected name's annual cycle under
the matrix. Refresh Live Data downloads prices again instead of reusing a
fresh cache file.
"""

from __future__ import annotations

from datetime import timedelta

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "streamlit is required for the interactive app:\n"
        '  pip install -e ".[interactive]"\n'
        "  streamlit run streamlit_app.py"
    ) from exc

import pandas as pd

from rotation_lab.config import DEFAULT_BENCHMARK, DEFAULT_START, benchmark_asset, preset_universe

try:
    from rotation_lab.dashboard import QUADRANT_COLOR, rrg_figure, seasonal_board, seasonal_figure
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "plotly is required for the dashboard:\n"
        '  pip install -e ".[interactive]"\n'
        "  streamlit run streamlit_app.py"
    ) from exc

from rotation_lab.data import fetch_prices
from rotation_lab.rrg import RRGParams, compute_rrg

UNIVERSES: tuple[tuple[str, str], ...] = (
    ("11 S&P 500 Sectors (vs. SPY)", "sectors"),
    ("Intermarket (vs. SPY)", "intermarket"),
    ("Metals (vs. SPY)", "metals"),
    ("Rates & credit (vs. SPY)", "rates"),
    ("Crypto (vs. SPY)", "crypto"),
)
_PRETTY = {
    "sectors": "S&P 500 Sectors",
    "intermarket": "Intermarket",
    "metals": "Metals",
    "rates": "Rates & Credit",
    "crypto": "Crypto",
}


@st.cache_data(show_spinner=False, ttl=3600)
def _load_prices(tickers: tuple[str, ...], start: str, offline: bool, force_refresh: bool):
    data = fetch_prices(list(tickers), start=start, offline=offline, force_refresh=force_refresh)
    return data.prices, data.source, data.is_demo, data.missing


def _css() -> None:
    st.markdown(
        """
        <style>
          .stApp { background: #0b1220; }
          header[data-testid="stHeader"] { background: #0b1220; }
          #MainMenu, footer { visibility: hidden; }
          .block-container { padding-top: 0.7rem; padding-bottom: 1.2rem; max-width: 100%; }
          div[data-testid="stButton"] > button {
            background: #12382f; color: #c9f7e4; border: 1px solid #1f6f5b;
            border-radius: 6px; font-weight: 650; width: 100%;
          }
          div[data-testid="stButton"] > button:hover { border-color: #3DDC97; color: #ffffff; }
          div[data-testid="stDataFrame"] { border: 1px solid #243044; border-radius: 8px; }
          div[data-testid="stCaptionContainer"] { color: #8b98ab; }
        </style>
        """,
        unsafe_allow_html=True,
    )


def _style_matrix(frame: pd.DataFrame):
    def quadrant(value: str) -> str:
        return f"color: {QUADRANT_COLOR.get(str(value), '#d5dde8')}; font-weight: 650"

    def bias(value: str) -> str:
        text = str(value)
        if text.startswith("▲"):
            return "color: #3DDC97; font-weight: 600"
        if text.startswith("▼"):
            return "color: #E15B64; font-weight: 600"
        return "color: #d5dde8"

    styled = frame.style.map(quadrant, subset=["Quadrant"]).map(bias, subset=["30d Bias (Win%)"])
    return styled.format({"RS-Ratio": "{:.2f}", "RS-Mom": "{:.2f}", "Seas Idx": "{:.0f}"}, na_rep="—")


@st.fragment(run_every=timedelta(minutes=15))
def _auto_tick() -> None:
    """Wake every 15 minutes. With Auto on, ask the page to download again.

    ``fetched_at`` is bumped before the rerun so this tick does not schedule
    a second download on the run that actually fetches.
    """
    import time

    if not st.session_state.get("auto_15m"):
        return
    now = time.time()
    last = float(st.session_state.get("fetched_at", now))
    if now - last < 15 * 60:
        return
    st.session_state["fetched_at"] = now
    st.session_state["force_refresh"] = True
    st.rerun()


def _board() -> None:
    """Charts and the matrix."""
    import time

    label = st.session_state.get("universe_label", UNIVERSES[0][0])
    preset = next((key for text, key in UNIVERSES if text == label), "sectors")
    weekly = (st.session_state.get("freq") or "Weekly") != "Daily"
    trail = int(st.session_state.get("tail") or 7)
    show_seasonal = bool(st.session_state.get("show_seasonal", True))
    outright = bool(st.session_state.get("outright", True))
    relative = bool(st.session_state.get("relative", False))
    show_drift = bool(st.session_state.get("defend", False))
    offline = bool(st.session_state.get("offline", False))

    # Outright is the primary study unless the user asked only for relative.
    primary_relative = relative and not outright
    bench = benchmark_asset(DEFAULT_BENCHMARK.ticker)
    universe = [asset for asset in preset_universe(preset) if asset.ticker != bench.ticker]
    tickers = tuple(asset.ticker for asset in universe) + (bench.ticker,)

    force = bool(st.session_state.pop("force_refresh", False))
    if "fetched_at" not in st.session_state:
        st.session_state["fetched_at"] = time.time()
    if force:
        _load_prices.clear()

    try:
        if force:
            with st.spinner("Refreshing live data…"):
                prices, source, is_demo, missing = _load_prices(tickers, DEFAULT_START, offline, True)
        else:
            prices, source, is_demo, missing = _load_prices(tickers, DEFAULT_START, offline, False)
    except Exception as exc:
        st.error(f"Could not load data: {exc}")
        return
    if force:
        st.session_state["fetched_at"] = time.time()

    if missing:
        st.warning("No data, so these were dropped (not filled with synthetic prices): " + ", ".join(missing))
        universe = [asset for asset in universe if asset.ticker in prices.columns]
    if bench.ticker not in prices.columns:
        st.error(f"No data for benchmark {bench.ticker}.")
        return
    if not universe:
        st.error("No assets left with data.")
        return
    if is_demo:
        st.warning("Synthetic demo data. The pictures show the method, not the market.")

    try:
        result = compute_rrg(prices, bench.ticker, RRGParams())
    except Exception as exc:
        st.error(f"Could not compute the rotation graph: {exc}")
        return

    table, cycles = seasonal_board(
        result.snapshot(),
        universe,
        prices,
        benchmark=bench.ticker,
        relative=primary_relative,
    )
    stamp = time.strftime("%H:%M:%S", time.localtime(st.session_state["fetched_at"]))
    as_of = pd.Timestamp(prices.index[-1]).strftime("%Y-%m-%d")
    status, demo_toggle = st.columns([8, 1.15])
    with status:
        st.caption(f"updated {stamp} · {source} · as of {as_of} · benchmark {bench.ticker}")
    with demo_toggle:
        st.checkbox(
            "Offline",
            value=False,
            key="offline",
            help="Use the cache, or a labelled synthetic book if there is no cache.",
        )

    chart, side = st.columns([1.85, 1.0], gap="small")
    with chart:
        pretty = _PRETTY.get(preset, preset)
        st.plotly_chart(
            rrg_figure(
                result,
                trail=trail,
                weekly=weekly,
                title=f"1. {pretty} (vs. {bench.ticker}) | Center Benchmark (100, 100): {bench.ticker}",
            ),
            width="stretch",
            config={"displayModeBar": False},
        )
    with side:
        st.markdown("**RRG & Larry Williams Seasonal Matrix (Click Row to Plot Cycle)**")
        event = st.dataframe(
            _style_matrix(table),
            hide_index=True,
            width="stretch",
            height=418,
            on_select="rerun",
            selection_mode="single-row",
            key=f"matrix-{preset}-{int(primary_relative)}",
            column_config={
                "Ticker": st.column_config.TextColumn(width="small"),
                "Class": st.column_config.TextColumn(width="small"),
                "RS-Ratio": st.column_config.NumberColumn(format="%.2f", width="small"),
                "RS-Mom": st.column_config.NumberColumn("RS-Mom", format="%.2f", width="small"),
                "Quadrant": st.column_config.TextColumn(width="small"),
                "Seas Idx": st.column_config.NumberColumn(format="%d", width="small"),
                "30d Bias (Win%)": st.column_config.TextColumn(width="medium"),
            },
        )
        picked = list(getattr(event.selection, "rows", []) or [])
        if picked:
            st.session_state["selected_ticker"] = str(table.iloc[picked[0]]["Ticker"])
        selected = st.session_state.get("selected_ticker")
        if selected not in set(table["Ticker"]):
            selected = str(table.iloc[0]["Ticker"]) if len(table) else None
        if show_seasonal and selected:
            asset = next(item for item in universe if item.ticker == selected)
            cycle = cycles.get(selected)
            if cycle is None:
                st.info(f"{selected} does not have enough history for a seasonal cycle.")
            else:
                overlay = None
                overlay_name = None
                if outright and relative:
                    _, relative_cycles = seasonal_board(
                        result.snapshot(),
                        [asset],
                        prices,
                        benchmark=bench.ticker,
                        relative=True,
                    )
                    overlay = relative_cycles.get(selected)
                    overlay_name = f"{cycle.year} vs {bench.ticker} (scaled)"
                mode = "vs " + bench.ticker if primary_relative else "Outright"
                st.plotly_chart(
                    seasonal_figure(
                        cycle,
                        ticker=selected,
                        asset_label=asset.label,
                        mode_label=mode,
                        show_drift=show_drift,
                        overlay=overlay,
                        overlay_name=overlay_name,
                    ),
                    width="stretch",
                    config={"displayModeBar": False},
                )
    st.caption("Analytical study of relative rotation and calendar tendency. Not a trading system and not advice.")


def main() -> None:
    st.set_page_config(
        page_title="Top-Down RRG & Larry Williams True Seasonal Suite",
        page_icon="📈",
        layout="wide",
        initial_sidebar_state="collapsed",
    )
    _css()
    refresh, universe, freq, tail, seasonal, outright, relative, defend, auto = st.columns(
        [1.2, 2.2, 1.35, 0.62, 0.9, 1.15, 1.15, 1.05, 0.95]
    )
    with refresh:
        st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
        if st.button("Refresh Live Data", type="primary", icon=":material/refresh:"):
            st.session_state["force_refresh"] = True
            st.rerun()
    with universe:
        st.selectbox("Universe", [text for text, _key in UNIVERSES], key="universe_label")
    with freq:
        st.segmented_control("RRG", ["Weekly", "Daily"], default="Weekly", key="freq")
    with tail:
        st.number_input("Tail", min_value=2, max_value=60, value=7, step=1, key="tail")
    with seasonal:
        st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
        st.checkbox("Seasonal", value=True, key="show_seasonal")
    with outright:
        st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
        st.checkbox("Outright Price", value=True, key="outright")
    with relative:
        st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
        st.checkbox("Rel vs Bench", value=False, key="relative")
    with defend:
        st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
        st.checkbox("Defend Drift", value=False, key="defend")
    with auto:
        st.markdown("<div style='height:1.75rem'></div>", unsafe_allow_html=True)
        st.checkbox("Auto (15m)", value=False, key="auto_15m")
    _auto_tick()
    _board()


if __name__ == "__main__":
    main()
