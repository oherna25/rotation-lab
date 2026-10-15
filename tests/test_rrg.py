"""RRG: quadrant boundaries and the relative-strength direction."""

import numpy as np
import pandas as pd
import pytest

from rotation_lab.rrg import RRGParams, RRGResult, compute_rrg, quadrant_name


def test_quadrant_boundaries():
    assert quadrant_name(100, 100) == "Leading"
    assert quadrant_name(100, 99) == "Weakening"
    assert quadrant_name(99, 100) == "Improving"
    assert quadrant_name(99, 99) == "Lagging"
    assert quadrant_name(float("nan"), 100) == ""


def test_relative_trend_pushes_rs_ratio():
    idx = pd.bdate_range("2018-01-01", periods=300)
    t = np.linspace(0, 1, len(idx))
    prices = pd.DataFrame(
        {
            "SPY": 100 * np.exp(0.2 * t),
            "WIN": 100 * np.exp(1.2 * t),
            "LOSE": 100 * np.exp(-0.4 * t),
        },
        index=idx,
    )
    result = compute_rrg(prices, "SPY", RRGParams(window=40, z_scale=5))
    assert "SPY" not in result.rs_ratio.columns
    snap = result.snapshot().set_index("ticker")
    assert snap.loc["WIN", "rs_ratio"] > 100
    assert snap.loc["LOSE", "rs_ratio"] < 100
    assert snap.loc["WIN", "quadrant"] in {"Leading", "Weakening"}
    assert snap.loc["LOSE", "quadrant"] in {"Lagging", "Improving"}


def test_heading_of_a_move_up_and_right():
    idx = pd.bdate_range("2024-01-01", periods=8)
    result = RRGResult(
        rs=pd.DataFrame({"A": np.linspace(100, 108, len(idx))}, index=idx),
        rs_ratio=pd.DataFrame({"A": np.linspace(100, 110, len(idx))}, index=idx),
        rs_momentum=pd.DataFrame({"A": np.linspace(100, 106, len(idx))}, index=idx),
        quadrants=pd.DataFrame({"A": ["Leading"] * len(idx)}, index=idx),
        benchmark="SPY",
        params=RRGParams(),
    )
    snap = result.snapshot()
    assert snap.loc[0, "quadrant"] == "Leading"
    assert 0 < snap.loc[0, "heading_deg"] < 90
    assert snap.loc[0, "rs"] == pytest.approx(108)


def test_rejects_short_window_and_missing_benchmark():
    idx = pd.bdate_range("2024-01-01", periods=30)
    prices = pd.DataFrame({"SPY": np.linspace(100, 110, len(idx)), "GLD": np.linspace(50, 60, len(idx))}, index=idx)
    with pytest.raises(KeyError):
        compute_rrg(prices, "QQQ")
    with pytest.raises(ValueError):
        compute_rrg(prices, "SPY", RRGParams(window=2))
