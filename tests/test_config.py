"""Universe presets and the CLI asset spec."""

import pytest

from rotation_lab.config import (
    DEFAULT_UNIVERSE,
    benchmark_asset,
    preset_universe,
    universe_from_spec,
)


def test_spec_parses_labels_and_bare_tickers():
    assets = universe_from_spec("gld=Gold, tlt")
    assert [asset.ticker for asset in assets] == ["GLD", "TLT"]
    assert assets[0].label == "Gold"
    assert assets[1].label == "TLT"
    assert assets[0].color != assets[1].color


def test_empty_spec_is_the_intermarket_book():
    assert [asset.ticker for asset in universe_from_spec(None)] == [asset.ticker for asset in DEFAULT_UNIVERSE]
    assert "SLV" in {asset.ticker for asset in DEFAULT_UNIVERSE}
    assert "EEM" in {asset.ticker for asset in DEFAULT_UNIVERSE}


def test_presets_and_benchmark_label():
    assert {asset.ticker for asset in preset_universe("crypto")} == {"BTC-USD", "ETH-USD"}
    assert benchmark_asset("spy").label == "S&P 500"
    assert benchmark_asset("QQQ").ticker == "QQQ"
    with pytest.raises(KeyError):
        preset_universe("nope")
    with pytest.raises(ValueError):
        universe_from_spec(",")
