"""Offline pipeline: charts, the alignment board, and a report that names the assets."""

from pathlib import Path

import pandas as pd

from rotation_lab.cli import main


def test_offline_run_writes_the_intermarket_book(tmp_path: Path):
    code = main(
        [
            "run",
            "--offline",
            "--quiet",
            "--theme",
            "dark",
            "--preset",
            "intermarket",
            "--start",
            "2018-01-01",
            "--end",
            "2024-12-31",
            "--out",
            str(tmp_path),
        ]
    )
    assert code == 0
    for name in (
        "rrg.png",
        "seasonality_grid.png",
        "monthly_heatmap.png",
        "seasonal_outlook.png",
        "alignment.csv",
        "report.md",
        "index.html",
    ):
        file = tmp_path / name
        assert file.stat().st_size > 100
    assert (tmp_path / "rrg.png").read_bytes().startswith(b"\x89PNG")

    report = (tmp_path / "report.md").read_text(encoding="utf-8")
    assert "Gold" in report
    assert "Silver" in report
    assert "synthetic demo" in report.lower()
    assert "| asset | Jan |" in report
    assert "| stance |" in report

    page = (tmp_path / "index.html").read_text(encoding="utf-8")
    assert "<img src='rrg.png'" in page
    assert "Gold" in page
    assert (tmp_path / ".nojekyll").is_file()

    board = pd.read_csv(tmp_path / "alignment.csv")
    tickers = set(board["ticker"])
    assert "BTC-USD" in tickers
    assert "ETH-USD" not in tickers
    assert "SPY" not in tickers
    assert board["score"].between(-3, 3).all()


def test_short_history_still_writes_charts(tmp_path: Path):
    code = main(
        [
            "run",
            "--offline",
            "--quiet",
            "--no-report",
            "--assets",
            "GLD=Gold",
            "--start",
            "2024-11-01",
            "--end",
            "2024-12-15",
            "--out",
            str(tmp_path),
        ]
    )
    assert code == 0
    for name in ("rrg.png", "seasonality_grid.png", "monthly_heatmap.png", "seasonal_outlook.png"):
        assert (tmp_path / name).stat().st_size > 100
