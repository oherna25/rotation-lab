"""Static configuration for rotation-lab.

Defines the default *intermarket* universe (currencies, metals, energy, rates,
crypto), the 11 S&P 500 sector ETFs, the benchmark, a shared colour palette
and helper constructors.

The default book rotates through the macro asset classes that transmit
liquidity between markets:

    US Dollar -> Rates -> Commodities -> Equities -> Crypto

``sectors`` is the classic relative-rotation book: the 11 SPDR sector ETFs
versus SPY. Everything is overridable from the CLI
(``--assets "GLD=Gold,TLT=Long Bonds"``) or programmatically
(see :func:`universe_from_spec`).
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = [
    "Asset",
    "DEFAULT_BENCHMARK",
    "DEFAULT_UNIVERSE",
    "DEFAULT_START",
    "PALETTE",
    "PRESETS",
    "asset_index",
    "benchmark_asset",
    "default_universe",
    "preset_universe",
    "universe_from_spec",
    "universe_spec",
]


@dataclass(frozen=True)
class Asset:
    """A tradable instrument tracked by the app."""

    ticker: str
    label: str
    group: str = "Other"
    color: str | None = None


#: Benchmark everything is measured against (relative rotation is always *relative*).
DEFAULT_BENCHMARK = Asset("SPY", "S&P 500", "Equities", "#C44E52")

#: Default intermarket universe.  ETFs are used instead of futures so the
#: series are back-adjustment free and have long, clean histories.
DEFAULT_UNIVERSE: tuple[Asset, ...] = (
    Asset("UUP", "US Dollar", "Currencies", "#4C72B0"),
    Asset("FXE", "Euro", "Currencies", "#8172B2"),
    Asset("GLD", "Gold", "Metals", "#D4AF37"),
    Asset("SLV", "Silver", "Metals", "#8E9AAB"),
    Asset("CPER", "Copper", "Metals", "#8C564B"),
    Asset("USO", "Crude Oil", "Energy", "#4D4D4D"),
    Asset("TLT", "Long Bonds", "Rates", "#55A868"),
    Asset("EEM", "EM Equities", "Equities", "#DD8452"),
    Asset("BTC-USD", "Bitcoin", "Crypto", "#F7931A"),
)

#: SPDR sector ETFs, the classic RRG universe. ``group`` is the cycle
#: bucket shown in the seasonal matrix (defensive, cyclical, growth, rates).
SECTOR_UNIVERSE: tuple[Asset, ...] = (
    Asset("XLB", "Materials", "Cyclical", "#C4A35A"),
    Asset("XLC", "Comm. Services", "Growth", "#5B8DEF"),
    Asset("XLE", "Energy", "Cyclical", "#E07A3D"),
    Asset("XLF", "Financials", "Cyclical", "#4C9AFF"),
    Asset("XLI", "Industrials", "Cyclical", "#7EB6FF"),
    Asset("XLK", "Technology", "Growth", "#8F9BFF"),
    Asset("XLP", "Staples", "Defense", "#6FCF97"),
    Asset("XLRE", "Real Estate", "Rate-Sens", "#E0A106"),
    Asset("XLU", "Utilities", "Defense", "#3DDC97"),
    Asset("XLV", "Health Care", "Defensive", "#3CB7A0"),
    Asset("XLY", "Discretionary", "Cyclical", "#5AA9FF"),
)

#: Named books. ``--assets`` replaces the selected preset entirely.
PRESETS: dict[str, tuple[Asset, ...]] = {
    "intermarket": DEFAULT_UNIVERSE,
    "sectors": SECTOR_UNIVERSE,
    "metals": (
        Asset("GLD", "Gold", "Metals", "#D4AF37"),
        Asset("SLV", "Silver", "Metals", "#8E9AAB"),
        Asset("CPER", "Copper", "Metals", "#8C564B"),
        Asset("PPLT", "Platinum", "Metals", "#7E8AA2"),
        Asset("USO", "Crude Oil", "Energy", "#4D4D4D"),
    ),
    "rates": (
        Asset("TLT", "20+ Year Bonds", "Rates", "#55A868"),
        Asset("IEF", "7-10 Year Bonds", "Rates", "#2A9D8F"),
        Asset("SHY", "1-3 Year Bonds", "Rates", "#8FBC8F"),
        Asset("HYG", "High Yield", "Credit", "#C44E52"),
        Asset("LQD", "IG Credit", "Credit", "#8172B2"),
    ),
    "crypto": (
        Asset("BTC-USD", "Bitcoin", "Crypto", "#F7931A"),
        Asset("ETH-USD", "Ether", "Crypto", "#627EEA"),
    ),
}

#: Colours assigned to custom assets added via ``--assets``.
PALETTE: tuple[str, ...] = (
    "#4C72B0",
    "#DD8452",
    "#55A868",
    "#C44E52",
    "#8172B2",
    "#937860",
    "#DA8BC3",
    "#8C8C8C",
    "#CCB974",
    "#64B5CD",
)

#: Sensible default history start.  Long enough for a meaningful seasonal
#: composite (>= 10 calendar years) while keeping every default ETF alive.
DEFAULT_START = "2013-01-01"


def default_universe() -> list[Asset]:
    """Return a mutable copy of the default intermarket universe."""
    return list(DEFAULT_UNIVERSE)


def preset_universe(name: str) -> list[Asset]:
    """Return a mutable copy of a named universe."""
    key = (name or "intermarket").strip().lower()
    try:
        return list(PRESETS[key])
    except KeyError as exc:
        known = ", ".join(sorted(PRESETS))
        raise KeyError(f"Unknown preset {name!r}; choose from {known}") from exc


def universe_spec(assets: list[Asset] | tuple[Asset, ...]) -> str:
    """Serialise assets to the ``TICKER=Label,...`` form accepted by the CLI."""
    return ",".join(f"{a.ticker}={a.label}" for a in assets)


def benchmark_asset(ticker: str) -> Asset:
    """Benchmark metadata. The default ticker keeps its label and colour."""
    ticker = ticker.strip().upper()
    if ticker == DEFAULT_BENCHMARK.ticker:
        return DEFAULT_BENCHMARK
    return Asset(ticker, ticker, "Benchmark", "#C44E52")


def universe_from_spec(spec: str | None) -> list[Asset]:
    """Parse a CLI spec like ``"GLD=Gold,TLT=Long Bonds,BTC-USD=Bitcoin"``.

    Each item is ``TICKER`` or ``TICKER=Label``.  Missing labels fall back to
    the ticker itself and colours are cycled from :data:`PALETTE`.
    Passing ``None`` or an empty string returns the default universe.
    """
    if spec is None or not spec.strip():
        return default_universe()

    assets: list[Asset] = []
    for i, raw in enumerate(spec.split(",")):
        raw = raw.strip()
        if not raw:
            continue
        ticker, _, label = raw.partition("=")
        ticker = ticker.strip().upper()
        label = label.strip() or ticker
        assets.append(Asset(ticker, label, group="Custom", color=PALETTE[i % len(PALETTE)]))
    if not assets:
        raise ValueError(f"Could not parse any assets from spec: {spec!r}")
    return assets


def asset_index(assets: list[Asset]) -> dict[str, Asset]:
    """Map ``ticker -> Asset`` for quick lookups in plots and reports."""
    return {a.ticker: a for a in assets}

