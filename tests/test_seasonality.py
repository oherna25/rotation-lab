"""Seasonality: ISO-year bucketing, monthly bias, and the wrapping outlook."""

import numpy as np
import pandas as pd
import pytest

from rotation_lab.seasonality import (
    _weekly_log_returns,
    build_monthly_seasonality,
    build_seasonal_profile,
    seasonal_outlook,
)


def test_iso_year_keeps_late_december_with_week_1():
    # Monday 2019-12-30 is ISO week 1 of 2020, not week 1 of calendar 2019.
    idx = pd.to_datetime(["2019-12-30", "2019-12-31", "2020-01-02"])
    series = pd.Series([100.0, 110.0, 121.0], index=idx)
    out = _weekly_log_returns(series)
    assert list(out["year"]) == [2020]
    assert list(out["week"]) == [1]
    assert out["ret"].iloc[0] == pytest.approx(np.log(1.21))


def test_march_dominates_the_monthly_table():
    idx = pd.bdate_range("2012-01-02", "2024-06-14")
    rng = np.random.default_rng(1)
    rets = rng.normal(0.0, 0.001, len(idx))
    rets[idx.month.to_numpy() == 3] += 0.01
    prices = pd.DataFrame({"X": 100 * np.exp(np.cumsum(rets))}, index=idx)
    monthly = build_monthly_seasonality(prices)
    march = monthly.mean_pct.loc["X", 3]
    others = monthly.mean_pct.loc["X"].drop(labels=[3])
    assert march > others.max()


def test_forward_window_wraps_into_the_next_year():
    idx = pd.bdate_range("2015-01-05", "2024-12-31")
    iso = idx.isocalendar()
    end_mask = (iso.year.astype(int) == 2024) & (iso.week.astype(int) == 50)
    assert end_mask.any()
    idx = idx[idx <= idx[end_mask].max()]
    iso_week = idx.isocalendar().week.astype(int)
    rets = np.full(len(idx), -0.0005)
    rets[iso_week.isin([1, 2, 51, 52]).to_numpy()] = 0.012
    prices = pd.DataFrame({"GOLD": 100 * np.exp(np.cumsum(rets))}, index=idx)

    profile = build_seasonal_profile(prices, min_years=4)
    outlook = seasonal_outlook(profile, prices, weeks_ahead=4, labels={"GOLD": "Gold"})
    row = outlook.table.set_index("ticker").loc["GOLD"]
    assert int(row["current_week"]) == 50
    assert int(row["target_week"]) == 2
    assert row["expected_return_pct"] > 5
    assert row["win_rate_pct"] >= 80
    assert row["years"] >= 4
    assert row["label"] == "Gold"

    with pytest.raises(ValueError):
        seasonal_outlook(profile, prices, weeks_ahead=0)
