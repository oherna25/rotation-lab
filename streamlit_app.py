"""Interactive dashboard for rotation-lab.

Run it with::

    pip install -e ".[interactive]"
    streamlit run streamlit_app.py

The CLI writes static PNGs and CSVs. This app is the same pipeline with
the window, trail and horizon as controls.
"""

from __future__ import annotations

try:
    import streamlit as st
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "streamlit is required for the interactive app:\n"
        "  pip install -e \".[interactive]\"\n"
        "  streamlit run streamlit_app.py"
    ) from exc

import matplotlib.pyplot as plt

from rotation_lab.config import (
    DEFAULT_BENCHMARK,
    DEFAULT_START,
    PRESETS,
    benchmark_asset,
    preset_universe,
    universe_from_spec,
    universe_spec,
)
from rotation_lab.data import fetch_prices
from rotation_lab.plots import (
    plot_monthly_heatmap,
    plot_rrg,
    plot_seasonal_grid,
    plot_seasonal_outlook,
)
from rotation_lab.rrg import QUADRANT_ORDER, RRGParams, compute_rrg
from rotation_lab.seasonality import (
    build_monthly_seasonality,
    build_seasonal_profile,
    seasonal_outlook,
)
from rotation_lab.signals import alignment_board


@st.cache_data(show_spinner=False, ttl=3600)
def _load_prices(tickers: tuple[str, ...], start: str, offline: bool):
    data = fetch_prices(list(tickers), start=start, offline=offline)
    return data.prices, data.source, data.is_demo, data.missing


def _render(fig) -> None:
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)


def main() -> None:
    st.set_page_config(page_title="Rotation Lab", page_icon="🧭", layout="wide")
    st.title("Intermarket Rotation Lab")
    st.caption(
        "Relative Rotation Graph versus a benchmark, plus Larry Williams-style seasonality. "
        "Educational. Not a recommendation to buy or sell anything."
    )

    with st.sidebar:
        st.header("Universe")
        preset = st.selectbox("Preset", list(PRESETS), index=0)
        if st.session_state.get("_preset") != preset:
            st.session_state["universe_spec"] = universe_spec(preset_universe(preset))
            st.session_state["_preset"] = preset
        spec = st.text_input("Tickers (TICKER=Label,...)", key="universe_spec")
        benchmark = st.text_input("Benchmark", value=DEFAULT_BENCHMARK.ticker).strip().upper()
        start = st.text_input("History start", value=DEFAULT_START)
        offline = st.toggle("Offline / synthetic demo", value=False)

        st.header("RRG")
        window = st.slider("Z-score window (trading days)", 20, 252, 63, 1)
        z_scale = st.slider("Z-scale (points per σ)", 1.0, 10.0, 5.0, 0.5)
        trail = st.slider("Trail length", 4, 78, 26, 1)
        resample = st.selectbox("Trail frequency", ["W-FRI", "2W-FRI", "ME", "none"], index=0)
        focus = st.text_input("Focus ticker (optional)", value="").strip().upper()

        st.header("Seasonality")
        weeks_ahead = st.slider("Outlook horizon (ISO weeks)", 1, 13, 4)
        theme = st.selectbox("Chart theme", ["light", "dark"])

    bench = benchmark_asset(benchmark or DEFAULT_BENCHMARK.ticker)
    try:
        universe = [asset for asset in universe_from_spec(spec) if asset.ticker != bench.ticker]
    except ValueError as exc:
        st.error(str(exc))
        st.stop()
    if not universe:
        st.error("Universe is empty after removing the benchmark.")
        st.stop()

    tickers = tuple(asset.ticker for asset in universe) + (bench.ticker,)
    try:
        prices, source, is_demo, missing = _load_prices(tickers, start, offline)
    except Exception as exc:
        st.error(f"Could not load data: {exc}")
        st.stop()

    if missing:
        st.warning("No data, so these were dropped (not filled with synthetic prices): " + ", ".join(missing))
        universe = [asset for asset in universe if asset.ticker in prices.columns]
    if bench.ticker not in prices.columns:
        st.error(f"No data for benchmark {bench.ticker}.")
        st.stop()
    if not universe:
        st.error("No assets left with data.")
        st.stop()
    if is_demo:
        st.warning("Synthetic demo data. The pictures show the method, not the market.")

    labels = {asset.ticker: asset.label for asset in [*universe, bench]}
    result = compute_rrg(prices, bench.ticker, RRGParams(window=int(window), z_scale=float(z_scale)))
    snapshot = result.snapshot()
    profile = build_seasonal_profile(prices)
    monthly = build_monthly_seasonality(prices)
    outlook = seasonal_outlook(profile, prices, weeks_ahead=int(weeks_ahead), labels=labels)
    board = alignment_board(snapshot, outlook, labels=labels)
    seasonal_assets = [*universe, bench]

    st.caption(f"Data: {source} · {prices.index[0]:%Y-%m-%d} → {prices.index[-1]:%Y-%m-%d} · benchmark {bench.ticker}")

    tabs = st.tabs(["Board", "RRG", "Seasonality", "Monthly", "Outlook", "Prices"])

    with tabs[0]:
        counts = snapshot["quadrant"].value_counts() if not snapshot.empty else {}
        columns = st.columns(len(QUADRANT_ORDER))
        for column, quadrant in zip(columns, QUADRANT_ORDER):
            column.metric(quadrant, int(counts.get(quadrant, 0)))
        st.dataframe(board, use_container_width=True, hide_index=True)
        with st.expander("How the score is built"):
            st.markdown(
                """
RRG asks whether the asset is strong **versus the benchmark**.
Seasonality asks whether the asset's **own history** was usually up over the next N ISO weeks
(week 52 wraps into week 1 of the next year).

| Quadrant | Score |
|---|---|
| Leading | +2 |
| Improving | +1 |
| Weakening | -1 |
| Lagging | -2 |

The seasonal drift then adds +1, 0 or -1. A Leading name with a positive seasonal window scores +3.
Heading on the RRG tab is the angle of the last five sessions: 0° is due east, 90° is due north.
                """
            )

    with tabs[1]:
        _render(
            plot_rrg(
                result,
                universe,
                trail=int(trail),
                resample=None if resample == "none" else resample,
                focus=focus or None,
                theme=theme,
                source=source,
                title=f"Intermarket RRG · benchmark {bench.ticker} ({bench.label})",
            )
        )
        st.dataframe(snapshot, use_container_width=True, hide_index=True)

    with tabs[2]:
        _render(
            plot_seasonal_grid(
                profile,
                prices,
                seasonal_assets,
                outlook,
                theme=theme,
                source=source,
                weeks_ahead=int(weeks_ahead),
            )
        )

    with tabs[3]:
        metric = st.radio("Metric", ["mean", "median", "win_rate"], horizontal=True)
        _render(plot_monthly_heatmap(monthly, seasonal_assets, theme=theme, source=source, metric=metric))
        st.dataframe(monthly.long(), use_container_width=True, hide_index=True)

    with tabs[4]:
        _render(
            plot_seasonal_outlook(
                outlook,
                seasonal_assets,
                theme=theme,
                source=source,
                weeks_ahead=int(weeks_ahead),
            )
        )
        st.dataframe(outlook.table, use_container_width=True, hide_index=True)

    with tabs[5]:
        st.dataframe(prices.tail(250).iloc[::-1], use_container_width=True)


if __name__ == "__main__":
    main()
