"""rotation-lab: intermarket Relative Rotation Graph + Larry Williams-style seasonality.

Quick start::

    from rotation_lab import compute_rrg, RRGParams, fetch_prices, build_seasonal_profile

    data = fetch_prices(["SPY", "GLD", "TLT"], start="2013-01-01")
    result = compute_rrg(data.prices, benchmark="SPY", params=RRGParams(window=63))
    print(result.snapshot())

See ``rotation_lab.cli`` for the command line interface and ``README.md`` for
the full methodology.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .config import (
    PRESETS,
    Asset,
    DEFAULT_BENCHMARK,
    DEFAULT_UNIVERSE,
    benchmark_asset,
    preset_universe,
    universe_from_spec,
    universe_spec,
)
from .data import MarketData, fetch_prices, synthetic_prices
from .rrg import QUADRANT_ORDER, RRGParams, RRGResult, compute_rrg, quadrant_name
from .seasonality import (
    MonthlySeasonality,
    SeasonalOutlook,
    SeasonalProfile,
    build_monthly_seasonality,
    build_seasonal_profile,
    seasonal_outlook,
)
from .signals import alignment_board

__all__ = [
    "Asset",
    "DEFAULT_BENCHMARK",
    "DEFAULT_UNIVERSE",
    "PRESETS",
    "alignment_board",
    "benchmark_asset",
    "preset_universe",
    "universe_spec",
    "MarketData",
    "MonthlySeasonality",
    "QUADRANT_ORDER",
    "RRGParams",
    "RRGResult",
    "SeasonalOutlook",
    "SeasonalProfile",
    "__version__",
    "build_monthly_seasonality",
    "build_seasonal_profile",
    "compute_rrg",
    "fetch_prices",
    "quadrant_name",
    "seasonal_outlook",
    "synthetic_prices",
    "universe_from_spec",
]
