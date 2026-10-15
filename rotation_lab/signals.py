"""Join the latest RRG quadrant with the forward seasonal drift.

Seasonality here is the asset's own calendar tendency (Larry Williams).
The RRG reading is relative to the benchmark. The board is the cross of
those two different questions: is it strong *versus the benchmark*, and
does its own history say the next few weeks are usually kind?
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .seasonality import SeasonalOutlook

__all__ = ["QUAD_SCORE", "STANCE", "alignment_board"]

# Leading is the strongest relative-strength state; Lagging the weakest.
# Improving outranks Weakening because the turn has started.
QUAD_SCORE: dict[str, int] = {"Leading": 2, "Improving": 1, "Weakening": -1, "Lagging": -2}

STANCE: dict[tuple[str, bool], str] = {
    ("Leading", True): "Aligned: continuation",
    ("Leading", False): "Late: seasonal headwind",
    ("Improving", True): "Early rotation, seasonal support",
    ("Improving", False): "Unconfirmed: seasonal headwind",
    ("Weakening", True): "Trim: fading RS, seasonal bid",
    ("Weakening", False): "Reduce: momentum and calendar down",
    ("Lagging", True): "Contrarian seasonal only",
    ("Lagging", False): "Avoid",
}


def alignment_board(
    snapshot: pd.DataFrame,
    outlook: SeasonalOutlook | pd.DataFrame,
    labels: dict[str, str] | None = None,
) -> pd.DataFrame:
    """One row per asset: quadrant, seasonal window, and a combined score.

    Score = quadrant score (-2..+2) + sign of the seasonal drift (-1, 0, +1).
    Leading into a positive seasonal window is +3; Lagging into a negative
    window is -3. Ties break on RS-Ratio.
    """
    labels = labels or {}
    if snapshot is None or snapshot.empty:
        return pd.DataFrame()

    out = outlook.table if isinstance(outlook, SeasonalOutlook) else outlook
    if out is None or out.empty or "ticker" not in getattr(out, "columns", []):
        out = pd.DataFrame(columns=["ticker"])

    keep = [
        c
        for c in ("ticker", "quadrant", "rs_ratio", "rs_momentum", "heading_deg", "rs", "as_of")
        if c in snapshot.columns
    ]
    left = snapshot[keep].copy()
    right_cols = [
        c
        for c in (
            "ticker",
            "expected_return_pct",
            "win_rate_pct",
            "seasonal_z",
            "ytd_vs_history_pct",
            "years",
            "current_week",
            "target_week",
        )
        if c in out.columns
    ]
    merged = left.merge(out[right_cols], on="ticker", how="left") if right_cols else left

    if "expected_return_pct" in merged.columns:
        expected = merged["expected_return_pct"]
    else:
        expected = pd.Series(np.nan, index=merged.index)
    seasonal_sign = np.sign(expected.fillna(0.0).to_numpy(dtype=float))
    merged["score"] = merged["quadrant"].map(QUAD_SCORE).fillna(0).astype(int) + seasonal_sign.astype(int)
    merged["stance"] = [_stance(quadrant, exp) for quadrant, exp in zip(merged["quadrant"], expected)]
    merged["label"] = merged["ticker"].map(lambda ticker: labels.get(ticker, ticker))
    sort_cols = ["score"] + (["rs_ratio"] if "rs_ratio" in merged.columns else [])
    ascending = [False] * len(sort_cols)
    merged = merged.sort_values(sort_cols, ascending=ascending).reset_index(drop=True)

    order = [
        "label",
        "ticker",
        "quadrant",
        "stance",
        "score",
        "rs_ratio",
        "rs_momentum",
        "heading_deg",
        "expected_return_pct",
        "win_rate_pct",
        "seasonal_z",
        "ytd_vs_history_pct",
        "years",
        "current_week",
        "target_week",
    ]
    return merged[[c for c in order if c in merged.columns]]


def _stance(quadrant: str, expected: float) -> str:
    if pd.isna(expected):
        return ""
    if expected > 0:
        return STANCE.get((quadrant, True), "")
    if expected < 0:
        return STANCE.get((quadrant, False), "")
    return "Flat seasonal window"
