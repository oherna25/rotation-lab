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


def test_fresh_cache_skips_the_network(tmp_path, monkeypatch):
    idx = pd.bdate_range("2020-01-01", "2020-06-30")
    series = pd.Series(np.linspace(100, 120, len(idx)), index=idx, name="GLD")
    series.to_csv(tmp_path / "GLD.csv", header=["close"])

    def _boom(*_args, **_kwargs):
        raise AssertionError("fresh cache should not download")

    monkeypatch.setattr("rotation_lab.data._download", _boom)
    data = fetch_prices(
        ["GLD"],
        start="2020-01-01",
        end="2020-06-30",
        cache_dir=tmp_path,
        refresh_days=10,
    )
    assert data.prices["GLD"].iloc[-1] == pytest.approx(120)
    assert data.source == "cache"


def test_force_refresh_downloads_even_when_the_cache_is_fresh(tmp_path, monkeypatch):
    idx = pd.bdate_range("2020-01-01", "2020-06-30")
    series = pd.Series(np.linspace(100, 120, len(idx)), index=idx, name="GLD")
    series.to_csv(tmp_path / "GLD.csv", header=["close"])
    calls: dict[str, list[str]] = {}

    def _fake(tickers, start, end, cache_dir):
        del start, end, cache_dir
        calls["tickers"] = list(tickers)
        refreshed = series.copy()
        refreshed.iloc[-1] = 999.0
        return {"GLD": refreshed}

    monkeypatch.setattr("rotation_lab.data._download", _fake)
    data = fetch_prices(
        ["GLD"],
        start="2020-01-01",
        end="2020-06-30",
        cache_dir=tmp_path,
        refresh_days=10,
        force_refresh=True,
    )
    assert calls["tickers"] == ["GLD"]
    assert data.prices["GLD"].iloc[-1] == pytest.approx(999)
    assert "yfinance" in data.source


def test_force_refresh_keeps_the_cache_when_the_download_fails(tmp_path, monkeypatch):
    idx = pd.bdate_range("2020-01-01", "2020-06-30")
    series = pd.Series(np.linspace(100, 120, len(idx)), index=idx, name="GLD")
    series.to_csv(tmp_path / "GLD.csv", header=["close"])
    monkeypatch.setattr("rotation_lab.data._download", lambda *_args, **_kwargs: {})
    data = fetch_prices(
        ["GLD"],
        start="2020-01-01",
        end="2020-06-30",
        cache_dir=tmp_path,
        refresh_days=10,
        force_refresh=True,
    )
    assert data.prices["GLD"].iloc[-1] == pytest.approx(120)
    assert data.source == "cache"
    assert data.is_demo is False


def test_a_hung_download_is_abandoned(tmp_path, monkeypatch):
    import time

    import yfinance

    monkeypatch.setattr("rotation_lab.data._TICKER_TIMEOUT_S", 0.2)
    monkeypatch.setattr("rotation_lab.data._DOWNLOAD_BUDGET_S", 0.8)

    def _hang(*_args, **_kwargs):
        time.sleep(5)
        raise AssertionError("the hung download should have been abandoned")

    monkeypatch.setattr(yfinance, "download", _hang)
    started = time.monotonic()
    data = fetch_prices(
        ["GLD", "TLT"],
        start="2020-01-01",
        end="2020-06-01",
        cache_dir=tmp_path,
        refresh_days=0,
        force_refresh=True,
    )
    assert time.monotonic() - started < 3
    assert data.is_demo
    assert data.source == "synthetic demo"


def test_cache_write_error_does_not_drop_the_download(tmp_path, monkeypatch):
    idx = pd.bdate_range("2020-01-01", "2020-04-01")
    frame = pd.DataFrame({"Close": np.linspace(100, 110, len(idx))}, index=idx)

    import yfinance

    monkeypatch.setattr(yfinance, "download", lambda *_args, **_kwargs: frame)

    def _unwritable(*_args, **_kwargs):
        raise OSError("read-only file system")

    monkeypatch.setattr("rotation_lab.data._write_cache", _unwritable)
    data = fetch_prices(
        ["GLD"],
        start="2020-01-01",
        end="2020-04-01",
        cache_dir=tmp_path,
        refresh_days=0,
        force_refresh=True,
    )
    assert data.is_demo is False
    assert data.prices["GLD"].iloc[0] == pytest.approx(100)
    assert data.prices["GLD"].iloc[-1] == pytest.approx(110)


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
