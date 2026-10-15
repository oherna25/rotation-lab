# rotation-lab

Intermarket **Relative Rotation Graph** plus **Larry Williams-style seasonality**.

The RRG asks which market is strong or weak *versus a benchmark* (default: SPY), and whether that relative strength is accelerating. Seasonality asks a different question: in this calendar week, what has this market itself usually done. The alignment board crosses the two.

This is an analytical tool. It is not a trading system and it is not advice.

The published page is [oherna25.github.io/rotation-lab](https://oherna25.github.io/rotation-lab/). GitHub Actions rebuilds it on every push to `main`, and on weekday evenings after the US cash close. If Yahoo Finance does not answer, the page is labelled synthetic and still deploys.

## Install

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev,interactive]"
```

Python 3.10+. Runtime dependencies are numpy, pandas, matplotlib and yfinance. Streamlit is optional (`interactive`).

## Run

Live data (Yahoo Finance, cached under `.cache/prices`):

```bash
rotation-lab run
```

Offline, when the network is down or you only want to see the method:

```bash
rotation-lab run --offline --out output_demo
```

Other entry points:

```bash
rotation-lab rrg --trail 30 --focus GLD
rotation-lab season --outlook-weeks 8
rotation-lab run --preset crypto
rotation-lab run --preset rates --benchmark SPY
rotation-lab run --assets "GLD=Gold,TLT=Long Bonds,USO=Crude Oil" --theme dark
python -m rotation_lab run --offline
```

Interactive controls (same calculations):

```bash
streamlit run streamlit_app.py
```

## Universe

Default preset `intermarket`, all versus SPY:

| Ticker | Market |
|---|---|
| UUP | US Dollar |
| FXE | Euro |
| GLD | Gold |
| SLV | Silver |
| CPER | Copper |
| USO | Crude oil |
| TLT | Long bonds |
| EEM | Emerging-market equities |
| BTC-USD | Bitcoin |

Other presets: `metals` (GLD, SLV, CPER, PPLT, USO), `rates` (TLT, IEF, SHY, HYG, LQD), `crypto` (BTC-USD, ETH-USD). `--assets` replaces the preset. The benchmark is removed from the universe if you include it in both.

History starts at 2013-01-01 so the seasonal composite has a decade to work with. Pass `--start` to change it.

## What you get

`rotation-lab run` writes into `--out` (default `output/`):

| File | Contents |
|---|---|
| `rrg.png` | RRG with weekly trails, arrowheads, quadrant shading |
| `seasonality_grid.png` | Average-year composite, 10th–90th percentile band, this year, outlook window shaded |
| `monthly_heatmap.png` | Month-of-year mean return |
| `seasonal_outlook.png` | Ranked drift over the next N ISO weeks |
| `alignment.csv` | Quadrant × seasonal stance, one row per asset |
| `rrg_latest.csv` / `rrg_history.csv` | Latest reading and the full RRG history |
| `seasonal_outlook.csv` / `seasonal_weekly_stats.csv` / `monthly_seasonality.csv` | Seasonal tables |
| `prices.csv` | Adjusted closes actually used |
| `report.md` | The same numbers, with the charts embedded |
| `index.html` | The same report as a static page (what GitHub Pages serves) |

## Method

### Relative Rotation Graph

Julius de Kempenaer / StockCharts layout. For each asset other than the benchmark:

```
RS        = P_asset / P_benchmark
RS-Ratio  = 100 + z_scale * zscore(log RS, window)
RS-Momentum = 100 + z_scale * zscore(Δ RS-Ratio, window)
```

Default `window` is 63 trading days, `z_scale` is 5. 100 is neutral on both axes. The z-score uses a full window (`min_periods=window`), so the momentum series needs about two windows of warm-up.

|  | Momentum ≥ 100 | Momentum < 100 |
|---|---|---|
| RS-Ratio ≥ 100 | Leading | Weakening |
| RS-Ratio < 100 | Improving | Lagging |

The usual cycle is clockwise: Improving → Leading → Weakening → Lagging. `heading_deg` is the angle of the last five sessions on that plane (0° due east, 90° due north).

Trails default to the last 26 weekly (Friday) points.

### Seasonality

Weekly log returns are summed by **ISO year** and **ISO week**. Week 53 is folded into week 52. Using the calendar year here is wrong: the last days of December are often ISO week 1 of the next year.

Each complete year (one that has data by ISO week 4) is cumulated and rebased to 100 at its first week. The composite is the mean of those paths, excluding the in-progress year when at least four completed years exist. The band is the 10th–90th percentile across years. A wide band means the average is not a tendency.

The forward outlook does **not** divide two points on that composite. It compounds the next N ISO weeks of log return in each historical year. Week 52 wraps into week 1 of the *following* year, so a December window includes January, and January is not borrowed from the same year (that would be look-ahead). The in-progress year is not a sample. `expected_return_pct` is the equal-weighted mean of the completed windows; `win_rate_pct` is the fraction that were positive. `seasonal_z` ranks that drift inside the universe. `ytd_vs_history_pct` is where this year's path sits versus prior years at the same week.

Monthly stats are the same idea on calendar months. The month still in progress is dropped.

### Alignment score

Seasonality is absolute. The RRG is relative. The score adds them anyway, on purpose: they are two votes, not one number.

| Piece | Contribution |
|---|---|
| Leading | +2 |
| Improving | +1 |
| Weakening | −1 |
| Lagging | −2 |
| Seasonal drift | sign(expected), so −1, 0 or +1 |

Sorted high to low. Ties break on RS-Ratio. The stance column is the English reading of that cell (aligned continuation, early rotation, trim, avoid, and so on).

## Data

1. Per-ticker CSV cache in `.cache/prices`.
2. Refresh anything older than `--cache-refresh-days` (default 1) from Yahoo Finance.
3. If the refresh fails, keep the stale cache.
4. If **nothing** real is available, build a deterministic synthetic book and label every chart `synthetic demo`.

A ticker that fails on its own is dropped. It is not replaced with a random walk sitting next to real prices; that would invent a rotation.

Adjusted closes. The panel is put on one calendar: union of the observed dates and business days, then forward-filled, so a Friday close is what an equity ETF shows over a weekend when crypto traded.

## Tests

```bash
pytest
```

The tests check the quadrant map, that a market which trends up against the benchmark lands on the right of the RRG, that a March-only drift shows up in March, that a week-50 outlook wraps to week 2, and that the offline CLI writes the charts and the report.

## Caveats

- A seasonal edge is a tilt. The samples are short (about ten years on the default book; less for Bitcoin).
- RRG normalises by a rolling z-score, so the axes describe the *recent* regime, not a fixed historical percentile.
- ETF closes are not futures. Roll yield is absent. Crypto trades through equity holidays; the forward-fill hides that on the equity columns.
- Nothing here sizes a position or picks a stop.
