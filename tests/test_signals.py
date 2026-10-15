"""Alignment board: quadrant score plus the sign of the seasonal drift."""

import pandas as pd

from rotation_lab.signals import alignment_board


def test_leading_with_a_tailwind_outranks_a_lagging_headwind():
    snapshot = pd.DataFrame(
        [
            {"ticker": "GLD", "quadrant": "Leading", "rs_ratio": 105.0, "rs_momentum": 102.0},
            {"ticker": "TLT", "quadrant": "Lagging", "rs_ratio": 95.0, "rs_momentum": 97.0},
        ]
    )
    outlook = pd.DataFrame(
        [
            {"ticker": "GLD", "expected_return_pct": 1.5, "win_rate_pct": 70.0},
            {"ticker": "TLT", "expected_return_pct": -0.8, "win_rate_pct": 40.0},
        ]
    )
    board = alignment_board(snapshot, outlook, labels={"GLD": "Gold", "TLT": "Bonds"})
    assert list(board["ticker"]) == ["GLD", "TLT"]
    assert list(board["score"]) == [3, -3]
    assert board.loc[0, "stance"] == "Aligned: continuation"
    assert board.loc[1, "stance"] == "Avoid"
    assert board.loc[0, "label"] == "Gold"


def test_flat_seasonal_window_adds_nothing():
    snapshot = pd.DataFrame([{"ticker": "GLD", "quadrant": "Leading", "rs_ratio": 101.0, "rs_momentum": 100.0}])
    outlook = pd.DataFrame([{"ticker": "GLD", "expected_return_pct": 0.0}])
    board = alignment_board(snapshot, outlook)
    assert board.loc[0, "score"] == 2
    assert board.loc[0, "stance"] == "Flat seasonal window"
