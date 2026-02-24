# Kalshi Sports Prediction Markets — Open vs Close Regime Analysis

## The One-Sentence Version

This tool measures **how much a sports betting market learns between when it opens and when it closes**, and whether that learning is statistically significant or just noise.

---

## What Is Kalshi?

Kalshi is a regulated prediction market. For sports, each game gets a binary contract — for example, *"Will the Lakers beat the Celtics?"* — that trades between $0.00 and $1.00. A price of $0.65 means the market thinks there's a 65% chance the Lakers win. When the game ends, the contract settles at $1.00 (yes) or $0.00 (no).

Each game produces **two contracts**: one for each team. If Lakers YES is at $0.65, Celtics YES is at $0.35. They always sum to $1.00.

---

## What This Project Does

You pick a sport and run **`regime.py`**. It pulls every settled (finished) game market for that sport over the last 30 days from the Kalshi API, grabs the **opening price** (first candle of the trading day) and the **closing price** (last candle before resolution), cleans the data, and runs the analysis.

### The Pipeline

**Step 1 — Fetch settled markets** from the Kalshi API for the selected sport and date range. Results are cached to disk as JSON so re-runs skip the API call.

**Step 2 — Fetch hourly candlestick data** for each market. Each candle has price data (open, close, high, low) and a **volume** field (contracts traded in that hour). The script extracts the opening price from the first candle, the closing price from the last candle, and sums volume across all candles to get total contracts traded per market. This runs in parallel across 4 threads.

**Step 3 — Compute features.** From `p_open`, `p_close`, volume, and the binary outcome `y` (0 or 1):

| Feature | Formula | Meaning |
|---|---|---|
| `drift` | `p_close − p_open` | How much the price moved |
| `abs_drift` | `\|drift\|` | Magnitude of movement |
| `edge_open` | `y − p_open` | Forecast error of the opening price |
| `edge_close` | `y − p_close` | Forecast error of the closing price |
| `favorite` | `1 if p_open > 0.5` | Did this contract open as the favorite? |
| `underdog_win` | `1 if p_open < 0.5 and y = 1` | Did an underdog win? |
| `volume` | sum of hourly candle volumes | Total contracts traded for this market |
| `clv_raw` | `p_close − p_open` | Raw closing line value |
| `clv_correct` | `1 if line moved toward outcome` | Did the market get more accurate? |
| `pnl_buy_open` | `y − p_open` | PnL of buying YES at open |
| `dow_name` | day of week from close time (ET) | Monday, Tuesday, etc. |

**Step 4 — Deduplicate paired contracts.** Each game has two mirror contracts (Team A, Team B). The script strips the team suffix from the ticker (e.g. `KXNBAGAME-25NOV30HOUUTA-HOU` → `KXNBAGAME-25NOV30HOUUTA`), then keeps only the side with the higher `p_open` (the favorite). This cuts the dataset in half but gives independent observations.

**Step 5 — Filter live-settled contracts.** Contracts where `p_close ≤ 0.03` or `p_close ≥ 0.97` are removed. These are games where the outcome was already effectively known at market close — the "closing price" is just the known result, not a prediction.

**Step 6 — Sanity checks.** Per-series summary stats (count, mean, std, etc.) for `p_open`, `p_close`, `drift`, `abs_drift`, `volume`.

**Step 7 — Run the analysis:** drift regime (with bootstrap), day-of-week regime, volume summary (including volume-weighted Brier), closing line value, and calibration.

---

## Key Concepts

### Brier Score

The average of `(outcome − forecast)²` across all contracts. Lower is better.

- **0.00** = you predicted the exact right probability every time
- **0.25** = no better than always guessing 50%

"Brier improvement" = `brier_open − brier_close`. A positive number means the closing price was a more accurate forecast than the opening price.

### Drift

`p_close − p_open`. How much the price moved during the trading day.

- **High absolute drift** = the market moved a lot (new information arrived)
- **Low absolute drift** = the opening price barely changed

### Regime

A way of grouping data by a shared characteristic and comparing market behavior across groups. The script uses two regimes:

- **Drift regime**: Days classified as High drift (top 30% by daily mean absolute drift), Low drift (bottom 30%), or Mid (middle 40%), using within-series quantiles. Bootstrap significance testing is run on the Brier improvement gap between High and Low.
- **Day-of-week regime**: Contracts grouped by the weekday they closed (Monday through Sunday, ET timezone). No quantile binning — just raw grouping by day name. Shows games, volume, Brier scores, CLV correct rate, and improvement per weekday.

### Volume

Total contracts traded for a market during its trading window. Extracted by summing the `volume` (or `volume_fp`) field across all hourly candlesticks. The script reports volume at every level: per contract, per day, per regime bucket, per calibration bucket, and computes **volume-weighted Brier scores** — these weight high-volume markets more heavily, reflecting where the money actually was.

### Live-Settled

A contract where the market was still technically open while the game was already in progress or finished. The closing price reflects the known (or nearly known) outcome, not a prediction. These inflate Brier close scores if left in.

### Closing Line Value (CLV)

The closing price is considered the "sharpest" estimate because it incorporates all available information. CLV measures whether the opening price was systematically better or worse than the close.

The script computes:
- `clv_raw`: `p_close − p_open` (positive = price went up)
- `clv_correct`: 1 if the price moved toward the actual outcome (up when `y=1`, down when `y=0`)
- `pnl_buy_open`: `y − p_open` (what you'd earn/lose buying YES at the opening price)

### Calibration

When the market prices a contract at 70%, does that team actually win ~70% of the time? Contracts are binned by opening price into decile buckets, and the implied probability is compared to the actual win rate. Calibration buckets now also show mean volume per contract.

---

## NBA Results (example run)

The following reflects a typical run of `regime.py --sport nba` (default 30-day lookback). Pipeline steps [1/7]–[7/7] match the script output.

### Data Flow

| Stage | Count | What Happened |
|---|---|---|
| Settled markets from API | (varies) | All NBA settled markets in the lookback window |
| After candlestick extraction | 364 | Markets with no valid open/close dropped |
| After dedup | 182 | 182 mirror pairs removed (kept favorite side only) |
| After live-settled filter | **84** | 98 contracts removed (53.8%) where `p_close ≤ 0.03` or `≥ 0.97` |

Final dataset: **84 contracts across 84 games over 24 days** (98 live-settled excluded).

### Distribution of Clean Data (sanity check output)

The script prints `[6/7] Sanity checks...` then a per-series summary. Example for 84 contracts:

| Stat | p_open | p_close | drift | abs_drift | volume |
|---|---|---|---|---|---|---|
| mean | 0.658 | 0.592 | −0.066 | 0.246 | 3,089,403 |
| std | 0.108 | 0.293 | 0.298 | 0.180 | 1,789,096 |
| min | 0.41 | 0.04 | −0.71 | 0.00 | 633,274 |
| 50% | 0.635 | 0.625 | −0.01 | 0.225 | 2,572,738 |
| max | 0.90 | 0.96 | 0.44 | 0.71 | 7,959,462 |

The data skews toward favorites (`p_open` mean = 0.658) because dedup keeps the favorite side. Mean drift is slightly negative (−0.066), meaning prices drifted down on average.

### Drift Regime Results

| Regime | Days | Avg Games/Day | Mean \|Drift\| | Brier Open | Brier Close | Brier Improvement |
|---|---|---|---|---|---|---|
| High drift | 7 | 2.86 | 0.399 | 0.332 | 0.149 | **0.184** |
| Low drift | 7 | 3.00 | 0.141 | 0.210 | 0.186 | **0.023** |
| Mid | 10 | 4.30 | 0.240 | 0.254 | 0.146 | **0.108** |

On high-drift days the close price was 0.184 Brier points more accurate than the open. On low-drift days the improvement was only 0.023.

**Bootstrap test**: The difference between high-drift and low-drift Brier improvement is +0.1605.
- 95% CI: [+0.0557, +0.2663]
- p-value: 0.0004
- **Statistically significant at α=0.05.** The CI doesn't cross zero.

This means: on days when the market moves a lot, the closing price is genuinely a much better forecast than the opening price. This isn't noise.

### Day-of-Week Regime

Contracts are grouped by weekday (close time ET). The script prints a table with: `dow_name`, `games`, `total_volume`, `mean_volume`, `mean_abs_drift`, `mean_drift`, `brier_open`, `brier_close`, `clv_correct_rate`, `favorite_rate`, `brier_improvement`. Example excerpt:

| Day | Games | Mean Volume | Mean \|Drift\| | Brier Open | Brier Close | CLV Correct | Brier Improvement |
|---|---|---|---|---|---|---|---|
| Monday | 8 | 3.85M | 0.188 | 0.194 | 0.175 | 62.5% | 0.019 |
| Tuesday | 7 | 3.14M | 0.380 | 0.290 | 0.157 | 85.7% | 0.133 |
| Sunday | 17 | 3.29M | 0.261 | 0.274 | 0.118 | 76.5% | **0.156** |

This regime replaces the old upset regime (which had no underdog contracts left after dedup).

### Closing Line Value

| Metric | Value |
|---|---|
| Contracts analyzed | 84 |
| Mean CLV (close − open) | −0.066 |
| Line moved in correct direction | 78.6% of the time |
| Mean \|CLV\| | 0.246 |
| Mean PnL (buy YES at open) | −0.087 |

**By opening price bucket:**

| Bucket | n | Mean CLV | Line Correct Rate | Mean PnL |
|---|---|---|---|---|
| Slight underdog (30–50%) | 1 | +0.440 | 100.0% | +0.590 |
| Slight favorite (50–70%) | 57 | −0.028 | 82.5% | −0.056 |
| Heavy favorite (70–100%) | 26 | −0.169 | 69.2% | −0.179 |

The line moved in the correct direction 78.6% of the time overall — the market consistently gets more accurate from open to close. The negative mean PnL (−0.087) means blindly buying YES at open is a losing strategy in this sample. Heavy favorites saw the biggest price drops (mean CLV = −0.169), and the line was correct 69.2% of the time for them.

### Calibration

Contracts are binned by opening price (`pbin`); the script prints `n`, `mean_volume`, `implied_open`, `implied_close`, `realized`, `brier_open`, `brier_close`, `mean_abs_drift`. Summary:

| Opening Price Bucket | n | Implied (Open) | Implied (Close) | Realized Win % | Brier Open | Brier Close |
|---|---|---|---|---|---|---|
| 40–50% | 1 | 41.0% | 85.0% | 100.0% | 0.348 | 0.023 |
| 50–60% | 32 | 55.9% | 55.4% | 53.1% | 0.253 | 0.132 |
| 60–70% | 25 | 65.2% | 59.6% | 56.0% | 0.235 | 0.135 |
| 70–80% | 16 | 76.2% | 59.4% | 56.3% | 0.291 | 0.232 |
| 80–90% | 10 | 84.7% | 67.5% | 70.0% | 0.232 | 0.089 |

The 40–50% bucket has only 1 contract — ignore it. In the 50–60% range, calibration is reasonable (55.9% implied vs 53.1% actual). In the 70–80% range, the opening price implies 76.2% while the actual win rate is 56.3%; the closing price partially corrects (59.4% implied). The 80–90% bucket shows 84.7% implied vs 70.0% actual.

### Volume Summary

The script reports aggregate volume and **volume-weighted Brier scores** (high-volume contracts count more). Example:

| Metric | Value |
|---|---|
| Total volume across all contracts | 259,509,874 |
| Mean volume per contract | 3,089,403 |
| Median volume per contract | 2,572,738 |
| Volume-weighted Brier (open) | 0.2594 |
| Volume-weighted Brier (close) | 0.1410 |
| Volume-weighted improvement | 0.1184 |

---

## How to Use It

Run `regime.py` from the project directory:

```bash
python regime.py                        # interactive sport picker
python regime.py --sport nba            # NBA, last 30 days
python regime.py --sport nhl --days 14  # NHL, last 14 days
python regime.py --sport mlb --no-cache # force re-fetch
python regime.py --list                 # show available sports
```

| Flag | What It Does |
|---|---|
| `--sport nba` | Pick sport without interactive menu |
| `--days 14` | Change lookback window (default: 30) |
| `--no-cache` | Force re-fetch from API, ignore disk cache |
| `--workers 8` | More parallel threads for candlestick fetches (default: 4) |
| `--keep-live-settled` | Don't filter out contracts where outcome was already known |
| `--keep-pairs` | Keep both sides of each game (don't deduplicate) |

Available sports: `nba`, `nfl`, `nhl`, `mlb`, `ncaaf`, `ncaamb`, `ncaawb`, `atp`, `wta`, `atpc`, `epl`, `ucl`, `ufc`

---

## Requirements

- Python 3.10+
- `pip install requests numpy pandas matplotlib`


---

## Caveats

**Small sample.** 84 contracts over 24 days. Drift regime buckets have 7 days each. The bootstrap test helps, but validate over longer windows and multiple sports before acting on any finding.

**Favorite-side bias.** Dedup keeps the favorite side of each game, so nearly every contract has `p_open > 0.5`. This is why the old upset regime couldn't run (almost no underdog contracts remain after dedup) and why calibration buckets below 50% are nearly empty. The day-of-week regime replaces the upset regime and works on contract-level data regardless of side.

**Live-settled threshold is a judgment call.** The default is 0.03 — contracts with `p_close ≤ 0.03` or `≥ 0.97` are filtered. In this NBA run, 53.8% of contracts were filtered. A different threshold would change the sample size and results.

**Volume is per-contract, not per-game.** After dedup, volume reflects only the favorite-side contract. The underdog-side volume is discarded. Total game liquidity is roughly 2× what the script reports.
