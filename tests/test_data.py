"""Price loading: deterministic demo data, and no synthetic fill next to real prices."""

import os
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from rotation_lab.data import fetch_prices, synthetic_prices

ROOT = Path(__file__).resolve().parents[1]


def test_synthetic_prices_do_not_depend_on_hash_randomisation():
    code = (
        "from rotation_lab.data import synthetic_prices\n"
        "df = synthetic_prices(['GLD', 'BTC-USD'], '2020-01-01', '2020-04-01', seed=11)\n"
        "print(f'{df.iloc[-1, 0]:.8f},{df.iloc[-1, 1]:.8f}')\n"
    )
    env = os.environ.copy()
    env.pop("PYTHONHASHSEED", None)
    env["PYTHONPATH"] = str(ROOT)
    runs = [
        subprocess.check_output([sys.executable, "-c", code], env=env, cwd=ROOT).decode().strip()
        for _ in range(2)
    ]
    assert runs[0] == runs[1]
    left, right = runs[0].split(",")
    assert left != right


def test_offline_with_no_cache_is_labelled_demo(tmp_path):
    data = fetch_prices(
        ["GLD", "TLT"],
        start="2021-01-01",
        end="2021-06-01",
        cache_dir=tmp_path,
        offline=True,
    )
    assert data.is_demo
    assert set(data.prices.columns) == {"GLD", "TLT"}
    assert data.source == "synthetic demo"
    assert data.missing == ()


def test_partial_cache_is_not_padded_with_synthetic_prices(tmp_path):
    idx = pd.bdate_range("2020-01-01", "2020-12-31")
    series = pd.Series(np.linspace(100, 110, len(idx)), index=idx)
    series.to_csv(tmp_path / "GLD.csv", header=["close"])
    data = fetch_prices(
        ["GLD", "TLT"],
        start="2020-01-01",
        end="2020-12-31",
        cache_dir=tmp_path,
        offline=True,
    )
    assert list(data.prices.columns) == ["GLD"]
    assert data.is_demo is False
    assert data.missing == ("TLT",)
    assert data.prices["GLD"].iloc[0] == pytest.approx(100)
    assert data.prices["GLD"].iloc[-1] == pytest.approx(110)
