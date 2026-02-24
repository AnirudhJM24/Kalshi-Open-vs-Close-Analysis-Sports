# Kalshi Sports Prediction Markets — Open vs Close Regime Analysis

## The One-Sentence Version

This tool measures **how much a sports betting market learns between when it opens and when it closes**, and whether that learning is statistically significant or just noise.

---

## What Is Kalshi?

Kalshi is a regulated prediction market. For sports, each game gets a binary contract — for example, *"Will the Lakers beat the Celtics?"* — that trades between $0.00 and $1.00. A price of $0.65 means the market thinks there's a 65% chance the Lakers win. When the game ends, the contract settles at $1.00 (yes) or $0.00 (no).

Each game produces **two contracts**: one for each team. If Lakers YES is at $0.65, Celtics YES is at $0.35. They always sum to $1.00.

---

## What This Project Does

You pick a sport. The script pulls every settled (finished) game market for that sport over the last 30 days from the Kalshi API, grabs the **opening price** (first candle of the trading day) and the **closing price** (last candle before resolution), cleans the data, and runs four analyses.

### The Pipeline

**Step 1 — Fetch settled markets** from the Kalshi API for the selected sport and date range. Results are cached to disk as JSON so re-runs skip the API call.

**Step 2 — Fetch hourly candlestick data** for each market to extract the opening and closing prices. The trading window is the same calendar day in New York time (midnight ET minus a 1-hour buffer through the market's close time). This runs in parallel across 4 threads because each market needs its own API call.

**Step 3 — Compute features.** From `p_open`, `p_close`, and the binary outcome `y` (0 or 1):

| Feature | Formula | Meaning |
|---|---|---|
| `drift` | `p_close − p_open` | How much the price moved |
| `abs_drift` | `\|drift\|` | Magnitude of movement |
| `edge_open` | `y − p_open` | Forecast error of the opening price |
| `edge_close` | `y − p_close` | Forecast error of the closing price |
| `favorite` | `1 if p_open > 0.5` | Did this contract open as the favorite? |
| `underdog_win` | `1 if p_open < 0.5 and y = 1` | Did an underdog win? |

**Step 4 — Deduplicate paired contracts.** Each game has two mirror contracts (Team A, Team B). The script strips the team suffix from the ticker (e.g. `KXNBAGAME-25NOV30HOUUTA-HOU` → `KXNBAGAME-25NOV30HOUUTA`), then keeps only the side with the higher `p_open` (the favorite). This cuts the dataset in half but gives independent observations.

**Step 5 — Filter live-settled contracts.** Contracts where `p_close ≤ 0.03` or `p_close ≥ 0.97` are removed. These are games where the outcome was already effectively known at market close — the "closing price" is just the known result, not a prediction.

**Step 6 — Run four analyses:** drift regimes, upset regimes, closing line value, and calibration.

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

A way of grouping trading days by a shared characteristic. The script uses two:

- **Drift regime**: Days are classified as High drift (top 30% by mean absolute drift), Low drift (bottom 30%), or Mid (middle 40%), using within-series quantiles.
- **Upset regime**: Same approach, but classifying by the daily underdog win rate — Upset-heavy (top 30%), Favorite-dominant (bottom 30%), or Mid.

### Live-Settled

A contract where the market was still technically open while the game was already in progress or finished. The closing price reflects the known (or nearly known) outcome, not a prediction. These inflate Brier close scores if left in.

### Closing Line Value (CLV)

The closing price is considered the "sharpest" estimate because it incorporates all available information. CLV measures whether the opening price was systematically better or worse than the close.

The script computes:
- `clv_raw`: `p_close − p_open` (positive = price went up)
- `clv_correct`: 1 if the price moved toward the actual outcome (up when `y=1`, down when `y=0`)
- `pnl_buy_open`: `y − p_open` (what you'd earn/lose buying YES at the opening price)

### Calibration

When the market prices a contract at 70%, does that team actually win ~70% of the time? Contracts are binned by opening price into decile buckets, and the implied probability is compared to the actual win rate.

---

## NBA Results (Jan 25 – Feb 24, 2026)

### Data Flow

| Stage | Count | What Happened |
|---|---|---|
| Settled markets from API | 368 | All NBA settled markets in the 30-day window |
| After candlestick extraction | 364 | 4 markets had no valid open/close price data |
| After dedup | 182 | 182 mirror pairs removed (kept favorite side only) |
| After live-settled filter | **84** | 98 contracts removed (53.8%) where `p_close ≤ 0.03` or `≥ 0.97` |

Final dataset: **84 contracts across 84 games over 24 days.**

### Distribution of Clean Data

| Stat | p_open | p_close | drift | abs_drift |
|---|---|---|---|---|
| mean | 0.658 | 0.592 | −0.066 | 0.246 |
| std | 0.108 | 0.293 | 0.298 | 0.180 |
| min | 0.41 | 0.04 | −0.71 | 0.00 |
| median | 0.635 | 0.625 | −0.01 | 0.225 |
| max | 0.90 | 0.96 | 0.44 | 0.71 |

The data skews toward favorites (`p_open` mean = 0.658) because dedup keeps the favorite side. Mean drift is slightly negative (−0.066), meaning prices drifted down on average.

### Drift Regime Results

| Regime | Days | Avg Games/Day | Mean \|Drift\| | Brier Open | Brier Close | Brier Improvement |
|---|---|---|---|---|---|---|
| High drift | 7 | 2.86 | 0.399 | 0.332 | 0.149 | **0.184** |
| Low drift | 7 | 3.00 | 0.141 | 0.210 | 0.186 | **0.023** |
| Mid | 10 | 4.30 | 0.240 | 0.254 | 0.146 | **0.108** |

On high-drift days the close price was 0.184 Brier points more accurate than the open. On low-drift days the improvement was only 0.023.

**Bootstrap test**: The difference between high-drift and low-drift Brier improvement is +0.161.
- 95% CI: [+0.056, +0.266]
- p-value: 0.0004
- **Statistically significant at α=0.05.** The CI doesn't cross zero.

This means: on days when the market moves a lot, the closing price is genuinely a much better forecast than the opening price. This isn't noise.

### Upset Regime Results

Not enough data. After dedup and live-settled filtering, the `underdog_win_rate` column had fewer than 3 unique values across the 24 days. Because dedup keeps the favorite side, almost no contracts have `p_open < 0.5`, so `underdog_win` is 0 on nearly every day. This regime needs the full paired dataset (`--keep-pairs`) or a longer time window to produce results.

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

| Opening Price Bucket | n | Implied (Open) | Implied (Close) | Realized Win % | Brier Open | Brier Close |
|---|---|---|---|---|---|---|
| 40–50% | 1 | 41.0% | 85.0% | 100.0% | 0.348 | 0.023 |
| 50–60% | 32 | 55.9% | 55.4% | 53.1% | 0.253 | 0.132 |
| 60–70% | 25 | 65.2% | 59.6% | 56.0% | 0.235 | 0.135 |
| 70–80% | 16 | 76.2% | 59.4% | 56.3% | 0.291 | 0.232 |
| 80–90% | 10 | 84.7% | 67.5% | 70.0% | 0.232 | 0.089 |

The 40–50% bucket has only 1 contract — ignore it. In the 50–60% range, calibration is reasonable (55.9% implied vs 53.1% actual). In the 70–80% range, the opening price implies 76.2% while the actual win rate is 56.3%. The closing price partially corrects this (59.4% implied). The 80–90% bucket (10 contracts) shows 84.7% implied vs 70.0% actual.

---

## How to Use It

```bash
python kalshi_regimes_v3.py                        # interactive sport picker
python kalshi_regimes_v3.py --sport nba            # NBA, last 30 days
python kalshi_regimes_v3.py --sport nhl --days 14  # NHL, last 14 days
python kalshi_regimes_v3.py --sport mlb --no-cache # force re-fetch
python kalshi_regimes_v3.py --list                 # show available sports
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
- Kalshi API access (set `KALSHI_AUTH` env var, or edit the `headers` dict in the Config class)

---

## Caveats

**Small sample.** 84 contracts over 24 days. Regime buckets have 7 days each. The bootstrap test helps, but validate over longer windows and multiple sports before acting on any finding.

**Favorite-side bias.** Dedup keeps the favorite side of each game, so nearly every contract has `p_open > 0.5`. This is why the upset regime couldn't run (almost no underdog contracts remain) and why calibration buckets below 50% are nearly empty. This is a tradeoff — you get independent observations but lose the underdog perspective.

**Live-settled threshold is a judgment call.** The default is 0.03 — contracts with `p_close ≤ 0.03` or `≥ 0.97` are filtered. In this NBA run, 53.8% of contracts were filtered. A different threshold would change the sample size and results.

**No volume filter.** A contract with 3 trades and one with 500 trades are weighted equally.

