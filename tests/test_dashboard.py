"""Dashboard figures and the seasonal matrix that sits beside the RRG."""

import pandas as pd

from rotation_lab.config import Asset
from rotation_lab.dashboard import format_bias, rrg_figure, seasonal_board, seasonal_figure
from rotation_lab.data import synthetic_prices
from rotation_lab.rrg import compute_rrg


def test_bias_cell_uses_the_terminal_arrow():
    assert format_bias(4.9, 57) == "▲ +4.9 (57%)"
    assert format_bias(-1.25, 40).startswith("▼")


def test_matrix_sorts_by_rs_ratio_and_the_chart_labels_quadrants():
    prices = synthetic_prices(["SPY", "XLU", "XLE", "XLF"], "2018-01-02", "2024-06-28", seed=4)
    result = compute_rrg(prices, "SPY")
    assets = [
        Asset("XLU", "Utilities", "Defense"),
        Asset("XLE", "Energy", "Cyclical"),
        Asset("XLF", "Financials", "Cyclical"),
    ]
    table, cycles = seasonal_board(result.snapshot(), assets, prices, benchmark="SPY", relative=False)
    assert list(table["Ticker"]) == list(table.sort_values("RS-Ratio", ascending=False)["Ticker"])
    assert set(table["Class"]) == {"Defense", "Cyclical"}
    assert table["30d Bias (Win%)"].str.contains("%").all()
    assert set(cycles) == set(table["Ticker"])

    figure = rrg_figure(
        result,
        trail=7,
        weekly=True,
        title="1. S&P 500 Sectors (vs. SPY) | Center Benchmark (100, 100): SPY",
    )
    labels = [annotation.text for annotation in figure.layout.annotations]
    assert labels == ["IMPROVING", "LEADING", "LAGGING", "WEAKENING"]
    assert figure.layout.xaxis.title.text.startswith("JdK RS-Ratio")
    assert figure.layout.plot_bgcolor == "#0c1424"
    assert len(figure.data) == 3

    cycle = cycles[table.iloc[0]["Ticker"]]
    seasonal = seasonal_figure(
        cycle,
        ticker=cycle.current.name or table.iloc[0]["Ticker"],
        asset_label="Utilities",
        mode_label="Outright",
        show_drift=True,
    )
    names = [trace.name for trace in seasonal.data if trace.name]
    assert any("True Seasonal" in name for name in names)
    assert any("YTD Actual" in name for name in names)
    assert "Defended drift" in names
    assert "Outright" in seasonal.layout.title.text


def test_relative_board_uses_the_ratio_not_the_outright_path():
    idx = pd.bdate_range("2016-01-04", "2023-02-15")
    asset = pd.Series(100.0, index=idx)
    bench = pd.Series(100.0, index=idx)
    for ts in idx:
        if ts.month == 3:
            asset.loc[ts] = 100.0 + 20.0 * min((ts.day - 1) / 30.0, 1.0)
        elif ts.month > 3:
            asset.loc[ts] = 120.0
    prices = pd.DataFrame({"XLU": asset, "SPY": bench})
    # Flat benchmark, so the ratio matches the outright March rally.
    snapshot = pd.DataFrame(
        [{"ticker": "XLU", "rs_ratio": 103.5, "rs_momentum": 101.2, "quadrant": "Leading"}]
    )
    table, cycles = seasonal_board(
        snapshot,
        [Asset("XLU", "Utilities", "Defense")],
        prices,
        benchmark="SPY",
        relative=True,
    )
    assert table.iloc[0]["Seas Idx"] == 0
    assert cycles["XLU"].bias_pts > 5
