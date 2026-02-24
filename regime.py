"""
Kalshi Sports — Open vs Close Regime Analysis (v4)
====================================================
Single-sport analysis over the last N days of settled markets.

Changes from v3:
  - Removed upset regime (broken after dedup, uses retroactive labels)
  - Added day-of-week regime (observable before games start)
  - Added per-contract volume from candlestick data
  - Volume-weighted Brier scores
  - Volume in calibration, CLV, and sanity outputs

Usage:
  python kalshi_regimes_v4.py                     # interactive sport picker
  python kalshi_regimes_v4.py --sport nba          # NBA, last 30 days
  python kalshi_regimes_v4.py --sport nhl --days 14 --no-cache
  python kalshi_regimes_v4.py --list               # show available sports
"""

# ── Imports ──────────────────────────────────────────────────────────────────
import os
import re
import json
import time
import random
import argparse
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Optional

from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import requests
from requests.adapters import HTTPAdapter

pd.set_option("display.max_columns", 200)


# ── Sport Registry ──────────────────────────────────────────────────────────
SPORT_MENU = {
    1:  ("KXNBAGAME",            "nba",      "NBA (Pro Basketball)"),
    2:  ("KXNFLGAME",            "nfl",      "NFL (Pro Football)"),
    3:  ("KXNHLGAME",            "nhl",      "NHL (Hockey)"),
    4:  ("KXMLBGAME",            "mlb",      "MLB (Baseball)"),
    5:  ("KXNCAAFGAME",          "ncaaf",    "NCAA Football"),
    6:  ("KXNCAAMBGAME",         "ncaamb",   "NCAA Men's Basketball"),
    7:  ("KXNCAAWBGAME",         "ncaawb",   "NCAA Women's Basketball"),
    8:  ("KXATPMATCH",           "atp",      "ATP Tennis"),
    9:  ("KXWTAMATCH",           "wta",      "WTA Tennis"),
    10: ("KXATPCHALLENGERMATCH",  "atpc",    "ATP Challenger Tennis"),
    11: ("KXEPLGAME",            "epl",      "English Premier League"),
    12: ("KXUCLGAME",            "ucl",      "UEFA Champions League"),
    13: ("KXUFCFIGHT",           "ufc",      "UFC / MMA"),
}

# Build lookup: short_name → (ticker, display_name)
_SPORT_BY_SHORT = {short: (ticker, display) for _, (ticker, short, display) in SPORT_MENU.items()}
_SPORT_BY_TICKER = {ticker: (ticker, display) for _, (ticker, _, display) in SPORT_MENU.items()}


def resolve_sport(name: str) -> tuple[str, str]:
    """Resolve a CLI sport name (short or ticker) to (series_ticker, display_name)."""
    key = name.strip().lower()
    if key in _SPORT_BY_SHORT:
        return _SPORT_BY_SHORT[key]
    upper = name.strip().upper()
    if upper in _SPORT_BY_TICKER:
        return _SPORT_BY_TICKER[upper]
    raise ValueError(
        f"Unknown sport '{name}'. Use --list to see options. "
        f"Valid short names: {', '.join(sorted(_SPORT_BY_SHORT.keys()))}"
    )


def pick_sport_interactive() -> tuple[str, str]:
    """Interactive menu — returns (series_ticker, display_name)."""
    print("\n╔══════════════════════════════════════════╗")
    print("║       Choose a Sport to Analyze          ║")
    print("╠══════════════════════════════════════════╣")
    for num, (_, short, name) in sorted(SPORT_MENU.items()):
        print(f"║  {num:>2}.  {name:<35s} ║")
    print("╚══════════════════════════════════════════╝")

    while True:
        try:
            choice = int(input("\nEnter number (1-13): ").strip())
            if choice in SPORT_MENU:
                ticker, _, name = SPORT_MENU[choice]
                print(f"\n✓ Selected: {name} ({ticker})")
                return ticker, name
        except (ValueError, EOFError):
            pass
        print("  Invalid choice. Try again.")


# ── Configuration ────────────────────────────────────────────────────────────
@dataclass
class Config:
    """All tunable parameters in one place."""
    base_url: str = os.environ.get(
        "KALSHI_BASE_URL",
        "https://api.elections.kalshi.com/trade-api/v2",
    )
    headers: dict = field(default_factory=dict)
    sleep_between_calls: float = 0.05

    # Set dynamically at runtime
    target_series: str = ""
    target_name: str = ""
    min_close_date: str = ""
    max_close_date: str = ""

    # Candlestick settings
    day_tz: str = "America/New_York"
    day_start_buffer_hours: int = 1
    period_interval_minutes: int = 60
    max_workers: int = 4

    # Regime thresholds
    regime_lo_q: float = 0.30
    regime_hi_q: float = 0.70

    # Cache
    cache_dir: str = "kalshi_cache"
    no_cache: bool = False

    # Lookback
    lookback_days: int = 30

    # Live-settled filter threshold
    live_settled_threshold: float = 0.03  # p_close <= this or >= 1-this

    # Bootstrap
    bootstrap_n: int = 5000
    bootstrap_ci: float = 0.95

    def set_sport(self, ticker: str, name: str):
        self.target_series = ticker
        self.target_name = name
        today = datetime.now(timezone.utc).date()
        start = today - timedelta(days=self.lookback_days)
        self.max_close_date = today.isoformat()
        self.min_close_date = start.isoformat()


CFG = Config()


# ── HTTP Session ─────────────────────────────────────────────────────────────
_session = requests.Session()
_session.mount("https://", HTTPAdapter(pool_connections=50, pool_maxsize=50))
_session.mount("http://", HTTPAdapter(pool_connections=50, pool_maxsize=50))


def kalshi_get(
    url: str,
    params: Optional[dict] = None,
    max_retries: int = 4,
    timeout: tuple = (5, 25),
) -> dict:
    """GET JSON with connection pooling, 429 backoff, and transient-error retries."""
    for attempt in range(max_retries):
        r = _session.get(url, params=params, headers=CFG.headers, timeout=timeout)
        if r.status_code == 429:
            retry_after = r.headers.get("Retry-After")
            wait = float(retry_after) if retry_after else min(8.0, 1.5 * 2**attempt)
            time.sleep(wait * (0.7 + 0.6 * random.random()))
            continue
        if r.status_code in (500, 502, 503, 504) and attempt < max_retries - 1:
            time.sleep(min(4.0, 0.5 * 2**attempt))
            continue
        if r.status_code != 200:
            raise RuntimeError(f"HTTP {r.status_code}: {r.text[:400]}")
        return r.json()
    raise RuntimeError("Too many retries (rate limited / transient errors).")


# ── Step 1: Pull Settled Markets ─────────────────────────────────────────────
def get_settled_markets(series_ticker: str, min_ts: int, max_ts: int) -> list[dict]:
    """Paginate settled markets for a single series."""
    markets, cursor = [], None
    while True:
        params = {
            "series_ticker": series_ticker,
            "status": "settled",
            "min_close_ts": min_ts,
            "max_close_ts": max_ts,
            "limit": 1000,
        }
        if cursor:
            params["cursor"] = cursor
        data = kalshi_get(f"{CFG.base_url}/markets", params=params)
        if not isinstance(data, dict):
            break
        markets.extend(data.get("markets") or [])
        cursor = data.get("cursor")
        if not cursor:
            break
        time.sleep(CFG.sleep_between_calls)
    return markets


def fetch_settled_markets_single(series_ticker: str) -> dict[str, list[dict]]:
    """Fetch settled markets for a single series, with disk cache."""
    min_ts = int(pd.to_datetime(CFG.min_close_date, utc=True).timestamp())
    max_ts = int(pd.to_datetime(CFG.max_close_date, utc=True).timestamp())

    os.makedirs(CFG.cache_dir, exist_ok=True)
    cache_path = os.path.join(
        CFG.cache_dir,
        f"settled_{series_ticker}_{CFG.min_close_date}_{CFG.max_close_date}.json",
    )

    if not CFG.no_cache and os.path.exists(cache_path):
        try:
            with open(cache_path, "r") as f:
                cached = json.load(f)
            print(f"  Loaded cache: {cache_path}")
            return {series_ticker: cached}
        except (json.JSONDecodeError, IOError) as e:
            print(f"  ⚠ Bad cache ({e}), re-fetching.")

    mkts = get_settled_markets(series_ticker, min_ts, max_ts)
    print(f"  {series_ticker}: {len(mkts)} settled markets")

    try:
        with open(cache_path, "w") as f:
            json.dump(mkts, f)
        print(f"  Saved cache: {cache_path}")
    except IOError as e:
        print(f"  ⚠ Cache write failed: {e}")

    return {series_ticker: mkts}


# ── Step 2: Candlestick → Open/Close Extraction ─────────────────────────────
def _norm_str(v) -> Optional[str]:
    return v.strip().upper() if isinstance(v, str) else None


_YES_VALS = frozenset({"YES", "Y", "TRUE", "T", "WIN", "W", "1"})
_NO_VALS = frozenset({"NO", "N", "FALSE", "F", "LOSE", "L", "0"})


def extract_binary_outcome(m: dict) -> Optional[int]:
    """Extract binary outcome (0 or 1) from a settled market dict."""
    for k in ("result", "resolution", "resolved_outcome", "settlement",
              "settlement_value", "outcome", "settled_outcome", "winning_outcome"):
        v = m.get(k)
        if v is None:
            continue
        if isinstance(v, (int, float)) and v in (0, 1):
            return int(v)
        s = _norm_str(v)
        if s in _YES_VALS:
            return 1
        if s in _NO_VALS:
            return 0
        if isinstance(v, dict):
            for sub_k in ("outcome", "result", "resolution", "value"):
                sv = v.get(sub_k)
                if isinstance(sv, (int, float)) and sv in (0, 1):
                    return int(sv)
                ss = _norm_str(sv)
                if ss in _YES_VALS:
                    return 1
                if ss in _NO_VALS:
                    return 0
    return None


def get_close_dt_utc(m: dict) -> Optional[pd.Timestamp]:
    """Extract close datetime in UTC."""
    for k in ("close_ts", "close_time_ts"):
        v = m.get(k)
        if isinstance(v, (int, float)) and v > 0:
            return pd.to_datetime(int(v), unit="s", utc=True)
    for k in ("close_time", "closeTime"):
        v = m.get(k)
        if isinstance(v, str) and v.strip():
            dt = pd.to_datetime(v, utc=True, errors="coerce")
            if not pd.isna(dt):
                return dt
    return None


def _parse_price(v) -> Optional[float]:
    """Normalize a price to [0, 1] range."""
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v / 100.0 if 1.5 < v <= 100 else v


def fetch_candlesticks(
    series_ticker: str, market_ticker: str, start_ts: int, end_ts: int
) -> list[dict]:
    """Fetch candlesticks, trying multiple endpoint paths."""
    params = {
        "start_ts": int(start_ts),
        "end_ts": int(end_ts),
        "period_interval": CFG.period_interval_minutes,
        "include_latest_before_start": False,
    }
    endpoints = [
        f"{CFG.base_url}/markets/{market_ticker}/candlesticks",
        f"{CFG.base_url}/series/{series_ticker}/markets/{market_ticker}/candlesticks",
        f"{CFG.base_url}/historical/markets/{market_ticker}/candlesticks",
    ]
    last_err = None
    for url in endpoints:
        try:
            data = kalshi_get(url, params=params)
            candles = data.get("candlesticks", [])
            if isinstance(candles, list):
                return candles
        except Exception as e:
            last_err = e
    if last_err:
        raise last_err
    return []


def _extract_candle_price(candle: dict, side: str) -> Optional[float]:
    """Extract 'open' or 'close' price from a candlestick dict."""
    suffixes = (f"{side}_dollars", side, f"{side}_price", f"{side}_cents")
    p = candle.get("price")
    if isinstance(p, dict):
        for s in suffixes:
            v = _parse_price(p.get(s))
            if v is not None:
                return v
    for s in suffixes:
        v = _parse_price(candle.get(s))
        if v is not None:
            return v
    return None


def open_close_from_candles(candles: list[dict]) -> tuple[Optional[float], Optional[float]]:
    """O(1) open/close extraction with fallback scan."""
    if not candles:
        return None, None
    p_open = _extract_candle_price(candles[0], "open")
    p_close = _extract_candle_price(candles[-1], "close")
    if p_open is None:
        for c in candles:
            p_open = _extract_candle_price(c, "open")
            if p_open is not None:
                break
    if p_close is None:
        for c in reversed(candles):
            p_close = _extract_candle_price(c, "close")
            if p_close is not None:
                break
    if p_open is None or p_close is None:
        return None, None
    return float(p_open), float(p_close)


def _daily_window(close_dt_utc: pd.Timestamp) -> tuple[int, int]:
    """Compute candlestick fetch window [start_ts, end_ts] for a daily market."""
    close_et = close_dt_utc.tz_convert(CFG.day_tz)
    day_start_et = close_et.floor("D") - pd.Timedelta(hours=CFG.day_start_buffer_hours)
    start_ts = int(day_start_et.tz_convert("UTC").timestamp())
    end_ts = int(close_dt_utc.timestamp())
    if start_ts >= end_ts:
        start_ts = max(0, end_ts - 6 * 3600)
    return start_ts, end_ts


def _extract_candle_volume(candles: list[dict]) -> int:
    """Sum volume across all candles. Tries volume_fp (string) then volume (int)."""
    total = 0
    for c in candles:
        # Prefer volume_fp (fixed-point string), fall back to volume (int)
        vfp = c.get("volume_fp")
        if vfp is not None:
            try:
                total += int(float(vfp))
                continue
            except (TypeError, ValueError):
                pass
        v = c.get("volume")
        if isinstance(v, (int, float)):
            total += int(v)
    return total


def _process_market(series_ticker: str, m: dict) -> Optional[dict]:
    """Process a single market: fetch candles, extract open/close + volume, return row dict."""
    ticker = m.get("ticker") or m.get("market_ticker") or m.get("id")
    if not ticker:
        return None
    close_dt = get_close_dt_utc(m)
    if close_dt is None:
        return None
    y = extract_binary_outcome(m)
    if y is None:
        return None
    start_ts, end_ts = _daily_window(close_dt)
    candles = fetch_candlesticks(series_ticker, ticker, start_ts, end_ts)
    p_open, p_close = open_close_from_candles(candles)
    if p_open is None or p_close is None:
        return None

    # Volume: sum across hourly candles for this market's trading window
    volume = _extract_candle_volume(candles)

    # Day of week from close time in ET (0=Mon, 6=Sun)
    close_et = close_dt.tz_convert(CFG.day_tz)
    dow = close_et.dayofweek  # 0=Mon ... 6=Sun
    dow_name = close_et.strftime("%A")  # "Monday", "Tuesday", etc.

    return {
        "series": series_ticker,
        "market": ticker,
        "close_dt_utc": close_dt,
        "day": close_et.floor("D").tz_convert("UTC"),
        "y": int(y),
        "p_open": float(p_open),
        "p_close": float(p_close),
        "volume": volume,
        "dow": dow,
        "dow_name": dow_name,
    }


def build_contract_df(markets_by_series: dict[str, list[dict]]) -> pd.DataFrame:
    """Parallel-fetch candlesticks for all markets and build contract DataFrame."""
    tasks = [
        (series, m)
        for series, markets in markets_by_series.items()
        for m in markets
    ]
    total = len(tasks)
    print(
        f"  Fetching candles: {total} markets "
        f"(tz={CFG.day_tz}, interval={CFG.period_interval_minutes}m, "
        f"workers={CFG.max_workers})"
    )
    rows = []
    done_count = 0
    report_every = max(1, total // 20)

    with ThreadPoolExecutor(max_workers=CFG.max_workers) as ex:
        futures = {ex.submit(_process_market, s, m): (s, m) for s, m in tasks}
        for f in concurrent.futures.as_completed(futures):
            done_count += 1
            try:
                r = f.result()
            except Exception:
                r = None
            if r is not None:
                rows.append(r)
            if done_count % report_every == 0 or done_count == total:
                print(f"    {done_count}/{total} ({done_count/total:.0%}) — {len(rows)} rows", flush=True)

    df = pd.DataFrame(rows)
    print(f"  Raw rows with open+close+outcome: {len(df)}")
    return df


# ── Step 3: Feature Engineering ──────────────────────────────────────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add drift, edge, favorite, underdog_win, and CLV columns (vectorized)."""
    df = df.copy()
    df["drift"] = df["p_close"] - df["p_open"]
    df["abs_drift"] = df["drift"].abs()
    df["edge_open"] = df["y"] - df["p_open"]
    df["edge_close"] = df["y"] - df["p_close"]
    df["favorite"] = (df["p_open"] > 0.5).astype(int)
    df["underdog_win"] = ((df["p_open"] < 0.5) & (df["y"] == 1)).astype(int)
    # CLV features (also used by DOW regime)
    df["clv_raw"] = df["p_close"] - df["p_open"]
    df["clv_correct"] = (((df["y"] == 1) & (df["clv_raw"] > 0)) |
                         ((df["y"] == 0) & (df["clv_raw"] < 0))).astype(int)
    df["pnl_buy_open"] = df["y"] - df["p_open"]
    df["clv_edge"] = df["p_close"] - df["p_open"]
    return df


# ── NEW: Paired-Contract Deduplication ───────────────────────────────────────
def _extract_game_id(ticker: str) -> str:
    """Extract game ID from a Kalshi market ticker by stripping the team suffix.

    Examples:
      KXNBAGAME-25NOV30HOUUTA-HOU  →  KXNBAGAME-25NOV30HOUUTA
      KXNFLGAME-25DEC01DALNYG-DAL  →  KXNFLGAME-25DEC01DALNYG
      KXMLBGAME-25JUL31ATLCIN-CIN  →  KXMLBGAME-25JUL31ATLCIN

    For tickers that don't match the pattern, returns the full ticker (no dedup).
    """
    # Pattern: SERIES-DATECODETEAMS-TEAM  → strip last -TEAM
    parts = ticker.rsplit("-", 1)
    if len(parts) == 2 and len(parts[1]) <= 5 and parts[1].isalpha():
        return parts[0]
    return ticker


def deduplicate_paired_contracts(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one contract per game (the favorite side, or first alphabetically).

    Each Kalshi game produces two mirror contracts (Team A wins, Team B wins).
    We keep only the side with p_open > 0.5 (the favorite). If both are exactly
    0.5, we keep the first alphabetically. This halves the dataset but removes
    the double-counting that inflates apparent sample size.
    """
    before = len(df)
    df = df.copy()
    df["game_id"] = df["market"].apply(_extract_game_id)

    # Within each game, keep the contract with the higher opening price (favorite side)
    # Ties broken by ticker name (alphabetical)
    df = df.sort_values(["game_id", "p_open", "market"], ascending=[True, False, True])
    df = df.drop_duplicates(subset="game_id", keep="first")

    after = len(df)
    dropped = before - after
    print(f"  Deduplication: {before} → {after} contracts ({dropped} mirror pairs removed)")
    return df


# ── NEW: Live-Settled Filtering ──────────────────────────────────────────────
def filter_live_settled(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into pre-game-close (clean) and live-settled (contaminated).

    A contract is "live-settled" if p_close is near 0 or 1, meaning the game
    was already effectively decided when the market closed. These inflate
    Brier close scores because the "close price" is just the known outcome.

    Returns: (df_clean, df_live_settled)
    """
    threshold = CFG.live_settled_threshold
    is_live = (df["p_close"] <= threshold) | (df["p_close"] >= 1.0 - threshold)

    df_clean = df[~is_live].copy()
    df_live = df[is_live].copy()

    pct_live = len(df_live) / len(df) * 100 if len(df) > 0 else 0
    print(f"  Live-settled filter (threshold={threshold}):")
    print(f"    Clean contracts: {len(df_clean)}")
    print(f"    Live-settled (removed): {len(df_live)} ({pct_live:.1f}%)")
    print(f"    These had p_close ≤ {threshold} or ≥ {1-threshold} — outcome already known")

    return df_clean, df_live


# ── Regime Analysis ──────────────────────────────────────────────────────────
def _brier(x: pd.Series) -> float:
    """Mean squared error (Brier score)."""
    return float(np.mean(np.square(x)))


def _aggregate_daily(df_series: pd.DataFrame) -> pd.DataFrame:
    """Aggregate contract-level data to daily stats."""
    return (
        df_series.groupby("day")
        .agg(
            n=("market", "size"),
            total_volume=("volume", "sum"),
            mean_volume=("volume", "mean"),
            mean_abs_drift=("abs_drift", "mean"),
            mean_drift=("drift", "mean"),
            brier_open=("edge_open", _brier),
            brier_close=("edge_close", _brier),
            underdog_win_rate=("underdog_win", "mean"),
            favorite_rate=("favorite", "mean"),
        )
        .reset_index()
        .sort_values("day")
    )


def _classify_regime(
    values: pd.Series, lo_label: str, hi_label: str, mid_label: str = "Mid"
) -> pd.Series:
    """Classify values into three bins based on config quantiles."""
    q_lo = values.quantile(CFG.regime_lo_q)
    q_hi = values.quantile(CFG.regime_hi_q)
    return pd.Series(
        np.where(values <= q_lo, lo_label, np.where(values >= q_hi, hi_label, mid_label)),
        index=values.index,
    )


# ── NEW: Bootstrap Significance ──────────────────────────────────────────────
def _bootstrap_brier_improvement(
    daily: pd.DataFrame,
    regime_col: str,
    label_hi: str,
    label_lo: str,
    n_boot: int = None,
    ci: float = None,
) -> dict:
    """Bootstrap the Brier improvement difference between two regimes.

    Returns dict with point estimate, CI bounds, and p-value for the null
    hypothesis that the improvement difference is zero.
    """
    n_boot = n_boot or CFG.bootstrap_n
    ci = ci or CFG.bootstrap_ci

    hi_days = daily.loc[daily[regime_col] == label_hi, "brier_open"].values - \
              daily.loc[daily[regime_col] == label_hi, "brier_close"].values
    lo_days = daily.loc[daily[regime_col] == label_lo, "brier_open"].values - \
              daily.loc[daily[regime_col] == label_lo, "brier_close"].values

    if len(hi_days) < 3 or len(lo_days) < 3:
        return {"point_est": np.nan, "ci_lo": np.nan, "ci_hi": np.nan,
                "p_value": np.nan, "n_hi": len(hi_days), "n_lo": len(lo_days)}

    observed_diff = hi_days.mean() - lo_days.mean()

    rng = np.random.default_rng(42)
    boot_diffs = np.empty(n_boot)
    for i in range(n_boot):
        hi_sample = rng.choice(hi_days, size=len(hi_days), replace=True)
        lo_sample = rng.choice(lo_days, size=len(lo_days), replace=True)
        boot_diffs[i] = hi_sample.mean() - lo_sample.mean()

    alpha = 1.0 - ci
    ci_lo = float(np.percentile(boot_diffs, 100 * alpha / 2))
    ci_hi = float(np.percentile(boot_diffs, 100 * (1 - alpha / 2)))

    # Two-sided p-value: fraction of bootstrap samples on the other side of zero
    p_value = float(np.mean(boot_diffs <= 0)) if observed_diff > 0 else float(np.mean(boot_diffs >= 0))

    return {
        "point_est": float(observed_diff),
        "ci_lo": ci_lo,
        "ci_hi": ci_hi,
        "p_value": p_value,
        "n_hi": len(hi_days),
        "n_lo": len(lo_days),
    }


def drift_regime_analysis(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Regime 1: High vs Low Drift days (information-flow regime).

    Returns (summary_df, bootstrap_results).
    """
    results = []
    boot_results = {}

    for series, df_s in df.groupby("series"):
        daily = _aggregate_daily(df_s)
        if len(daily) < 6:
            continue
        daily["drift_regime"] = _classify_regime(
            daily["mean_abs_drift"], "Low drift", "High drift"
        )
        summary = (
            daily.groupby("drift_regime")
            .agg(
                days=("day", "size"),
                avg_n=("n", "mean"),
                mean_abs_drift=("mean_abs_drift", "mean"),
                mean_drift=("mean_drift", "mean"),
                brier_open=("brier_open", "mean"),
                brier_close=("brier_close", "mean"),
                underdog_win_rate=("underdog_win_rate", "mean"),
                favorite_rate=("favorite_rate", "mean"),
            )
            .reset_index()
        )
        summary["brier_improvement"] = summary["brier_open"] - summary["brier_close"]
        summary["series"] = series
        results.append(summary)

        # Bootstrap
        boot = _bootstrap_brier_improvement(daily, "drift_regime", "High drift", "Low drift")
        boot_results[series] = boot

    summary_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    return summary_df, boot_results


def dow_regime_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Regime 2: Day-of-week — observable before games start.

    Groups contracts by the day of week they closed (ET), then compares
    Brier scores, drift, volume, and CLV across weekdays.
    """
    results = []
    for series, df_s in df.groupby("series"):
        if len(df_s) < 10:
            continue

        agg = (
            df_s.groupby("dow_name")
            .agg(
                games=("market", "size"),
                total_volume=("volume", "sum"),
                mean_volume=("volume", "mean"),
                mean_abs_drift=("abs_drift", "mean"),
                mean_drift=("drift", "mean"),
                brier_open=("edge_open", _brier),
                brier_close=("edge_close", _brier),
                clv_correct_rate=("clv_correct", "mean"),
                favorite_rate=("favorite", "mean"),
            )
            .reset_index()
        )
        agg["brier_improvement"] = agg["brier_open"] - agg["brier_close"]
        agg["series"] = series

        # Sort by day of week order
        dow_order = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                     "Friday": 4, "Saturday": 5, "Sunday": 6}
        agg["_sort"] = agg["dow_name"].map(dow_order)
        agg = agg.sort_values("_sort").drop(columns="_sort")

        results.append(agg)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def _print_bootstrap_results(boot_results: dict, regime_name: str):
    """Pretty-print bootstrap significance results."""
    ci_pct = int(CFG.bootstrap_ci * 100)
    for series, b in boot_results.items():
        sig = "✓ SIGNIFICANT" if b["p_value"] < 0.05 else "✗ not significant"
        print(f"  {series}: Δ Brier improvement = {b['point_est']:+.4f}")
        print(f"    {ci_pct}% CI: [{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}]")
        print(f"    p-value: {b['p_value']:.4f}  ({sig} at α=0.05)")
        print(f"    n_hi={b['n_hi']}, n_lo={b['n_lo']}")


# ── NEW: Closing Line Value (CLV) Analysis ──────────────────────────────────
def clv_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Closing Line Value analysis — did the open price have exploitable edge?

    CLV features are pre-computed in add_features():
      - clv_raw: p_close - p_open
      - clv_correct: 1 if the line moved toward the actual outcome
      - pnl_buy_open: y - p_open
      - clv_edge: p_close - p_open (for YES buyers)
    """
    df = df.copy()

    # Aggregate by regime
    overall = {
        "total_contracts": len(df),
        "mean_clv_raw": df["clv_raw"].mean(),
        "clv_correct_rate": df["clv_correct"].mean(),
        "mean_pnl_buy_open": df["pnl_buy_open"].mean(),
        "mean_abs_clv": df["clv_raw"].abs().mean(),
    }

    # CLV by opening-price bucket
    df["pbin"] = pd.cut(df["p_open"], bins=[0, 0.3, 0.5, 0.7, 1.0],
                        labels=["Heavy underdog (0-30)", "Slight underdog (30-50)",
                                "Slight favorite (50-70)", "Heavy favorite (70-100)"])

    clv_by_bucket = (
        df.groupby("pbin", observed=True)
        .agg(
            n=("market", "size"),
            mean_volume=("volume", "mean"),
            mean_clv=("clv_raw", "mean"),
            clv_correct_rate=("clv_correct", "mean"),
            mean_pnl=("pnl_buy_open", "mean"),
            mean_abs_clv=("clv_edge", lambda x: x.abs().mean()),
        )
        .reset_index()
    )

    print("\n  ── Overall CLV Summary ──")
    print(f"    Contracts analyzed: {overall['total_contracts']}")
    print(f"    Mean CLV (close − open): {overall['mean_clv_raw']:+.4f}")
    print(f"    Line moved correctly: {overall['clv_correct_rate']:.1%} of the time")
    print(f"    Mean |CLV|: {overall['mean_abs_clv']:.4f}")
    print(f"    Mean PnL (buy YES at open): {overall['mean_pnl_buy_open']:+.4f}")

    print("\n  ── CLV by Opening Price Bucket ──")
    print(clv_by_bucket.to_string(index=False))

    # Plot
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    # Left: CLV distribution
    axes[0].hist(df["clv_raw"].values, bins=50, alpha=0.7, color="steelblue", edgecolor="white")
    axes[0].axvline(0, color="red", linestyle="--", alpha=0.7)
    axes[0].axvline(df["clv_raw"].mean(), color="orange", linestyle="-", linewidth=2,
                    label=f'Mean CLV: {df["clv_raw"].mean():+.3f}')
    axes[0].set_xlabel("CLV (close − open)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Closing Line Value Distribution")
    axes[0].legend()

    # Right: CLV correct rate by bucket
    if not clv_by_bucket.empty:
        x = range(len(clv_by_bucket))
        axes[1].bar(x, clv_by_bucket["clv_correct_rate"], color="teal", alpha=0.7, edgecolor="white")
        axes[1].axhline(0.5, color="red", linestyle="--", alpha=0.5, label="50% (random)")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(clv_by_bucket["pbin"].astype(str), rotation=15, ha="right")
        axes[1].set_ylabel("Fraction line moved correctly")
        axes[1].set_title("Line Accuracy by Opening Price Bucket")
        axes[1].set_ylim(0, 1)
        axes[1].legend()

    plt.tight_layout()
    plt.show()

    return clv_by_bucket


# ── Calibration ──────────────────────────────────────────────────────────────
def calibration_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Calibration by opening-price decile bucket."""
    df = df.copy()
    df["pbin"] = pd.cut(df["p_open"], bins=np.linspace(0, 1, 11))
    bucket = (
        df.groupby("pbin", observed=True)
        .agg(
            n=("market", "size"),
            mean_volume=("volume", "mean"),
            implied_open=("p_open", "mean"),
            implied_close=("p_close", "mean"),
            realized=("y", "mean"),
            brier_open=("edge_open", _brier),
            brier_close=("edge_close", _brier),
            mean_abs_drift=("abs_drift", "mean"),
        )
        .dropna()
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(bucket["implied_open"], bucket["realized"], "o-", label="Open price", linewidth=2)
    ax.plot(bucket["implied_close"], bucket["realized"], "s-", label="Close price", linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.6, label="Perfect calibration")
    ax.set_xlabel("Implied probability")
    ax.set_ylabel("Realized win rate")
    ax.set_title(f"Calibration: Open vs Close — {CFG.target_name}")
    ax.legend()
    plt.tight_layout()
    plt.show()

    return bucket


# ── Sanity Plots ─────────────────────────────────────────────────────────────
def sanity_plots(df: pd.DataFrame):
    """Quick distribution checks."""
    for series, grp in df.groupby("series"):
        print(f"\n  {series}: {len(grp)} contracts")
        print(grp[["p_open", "p_close", "drift", "abs_drift", "volume"]].describe().round(4).to_string())

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        grp["p_open"].hist(bins=50, ax=axes[0], alpha=0.7)
        axes[0].set_title(f"Opening Price — {CFG.target_name}")
        axes[0].set_xlabel("p_open")

        grp["drift"].hist(bins=60, ax=axes[1], alpha=0.7, color="orange")
        axes[1].set_title(f"Drift (close − open) — {CFG.target_name}")
        axes[1].set_xlabel("drift")

        grp["volume"].hist(bins=40, ax=axes[2], alpha=0.7, color="green")
        axes[2].set_title(f"Volume per Contract — {CFG.target_name}")
        axes[2].set_xlabel("volume (contracts traded)")

        plt.tight_layout()
        plt.show()


# ── CLI ──────────────────────────────────────────────────────────────────────
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Kalshi Sports — Open vs Close Regime Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="Examples:\n"
               "  %(prog)s                          # interactive\n"
               "  %(prog)s --sport nba               # NBA, last 30 days\n"
               "  %(prog)s --sport nhl --days 14     # NHL, last 14 days\n"
               "  %(prog)s --sport mlb --no-cache    # force re-fetch\n"
               "  %(prog)s --list                    # show available sports",
    )
    p.add_argument("--sport", "-s", type=str, default=None,
                   help="Sport short name (nba, nfl, nhl, mlb, ncaaf, ncaamb, atp, etc.)")
    p.add_argument("--days", "-d", type=int, default=30,
                   help="Lookback window in days (default: 30)")
    p.add_argument("--no-cache", action="store_true",
                   help="Force re-fetch, ignore disk cache")
    p.add_argument("--list", action="store_true",
                   help="List available sports and exit")
    p.add_argument("--workers", type=int, default=4,
                   help="Parallel workers for candlestick fetches (default: 4)")
    p.add_argument("--keep-live-settled", action="store_true",
                   help="Don't filter out live-settled contracts")
    p.add_argument("--keep-pairs", action="store_true",
                   help="Don't deduplicate paired contracts (keep both sides)")
    return p


# ── Main ─────────────────────────────────────────────────────────────────────
def main():
    parser = build_parser()
    args = parser.parse_args()

    # --list: print sports and exit
    if args.list:
        print("\nAvailable sports:")
        for _, (ticker, short, name) in sorted(SPORT_MENU.items()):
            print(f"  {short:<8s}  {ticker:<22s}  {name}")
        return

    # Apply CLI args to config
    CFG.lookback_days = args.days
    CFG.max_workers = args.workers
    CFG.no_cache = args.no_cache

    # Resolve sport: CLI flag or interactive
    if args.sport:
        ticker, name = resolve_sport(args.sport)
    else:
        ticker, name = pick_sport_interactive()

    CFG.set_sport(ticker, name)

    print("\n" + "=" * 60)
    print(f"  {name} ({ticker})")
    print(f"  {CFG.min_close_date}  →  {CFG.max_close_date}  ({CFG.lookback_days} days)")
    if not args.keep_live_settled:
        print(f"  Live-settled filter: ON (threshold={CFG.live_settled_threshold})")
    if not args.keep_pairs:
        print(f"  Paired-contract dedup: ON")
    print("=" * 60)

    # ── 1. Fetch markets ──
    print(f"\n[1/7] Fetching settled markets...")
    markets_by_series = fetch_settled_markets_single(ticker)
    n_markets = len(markets_by_series.get(ticker, []))
    if n_markets == 0:
        print(f"\n⚠ No settled markets for {name} in the last {CFG.lookback_days} days.")
        print("  This sport may be out of season. Try --sport nba or increase --days.")
        return
    print(f"  → {n_markets} settled markets")

    # ── 2. Build contract DataFrame ──
    print(f"\n[2/7] Building contract dataset...")
    df = build_contract_df(markets_by_series)
    if df.empty:
        print("⚠ No data after candlestick extraction. Check API auth.")
        return

    # ── 3. Feature engineering ──
    print(f"\n[3/7] Computing features...")
    df = add_features(df)

    # ── 4. Deduplicate paired contracts ──
    if not args.keep_pairs:
        print(f"\n[4/7] Deduplicating paired contracts...")
        df = deduplicate_paired_contracts(df)
    else:
        print(f"\n[4/7] Skipping dedup (--keep-pairs)")

    # ── 5. Filter live-settled ──
    df_live = pd.DataFrame()
    if not args.keep_live_settled:
        print(f"\n[5/7] Filtering live-settled contracts...")
        df, df_live = filter_live_settled(df)
        if df.empty:
            print("⚠ All contracts were live-settled! Try --keep-live-settled to analyze anyway.")
            return
    else:
        print(f"\n[5/7] Skipping live-settled filter (--keep-live-settled)")

    # ── 6. Sanity checks ──
    print(f"\n[6/7] Sanity checks...")
    sanity_plots(df)

    # ── 7. Analysis ──
    print(f"\n[7/7] Analysis for {name} ({len(df)} clean contracts)...")

    # Drift regime
    print("\n" + "─" * 50)
    print("  DRIFT REGIME (Information Flow)")
    print("─" * 50)
    drift_df, drift_boot = drift_regime_analysis(df)
    if drift_df.empty:
        print("  Not enough data for drift regime analysis.")
    else:
        print(drift_df.to_string(index=False))
        print("\n  Bootstrap significance (High drift improvement vs Low drift improvement):")
        _print_bootstrap_results(drift_boot, "drift")

    # Day-of-week regime
    print("\n" + "─" * 50)
    print("  DAY-OF-WEEK REGIME")
    print("─" * 50)
    dow_df = dow_regime_analysis(df)
    if dow_df.empty:
        print("  Not enough data for day-of-week analysis.")
    else:
        print(dow_df.to_string(index=False))

    # Volume summary
    print("\n" + "─" * 50)
    print("  VOLUME SUMMARY")
    print("─" * 50)
    total_vol = df["volume"].sum()
    mean_vol = df["volume"].mean()
    median_vol = df["volume"].median()
    print(f"  Total volume across all contracts: {total_vol:,}")
    print(f"  Mean volume per contract: {mean_vol:,.0f}")
    print(f"  Median volume per contract: {median_vol:,.0f}")
    # Volume-weighted Brier
    if total_vol > 0:
        vw_brier_open = np.average(df["edge_open"] ** 2, weights=df["volume"]) if df["volume"].sum() > 0 else np.nan
        vw_brier_close = np.average(df["edge_close"] ** 2, weights=df["volume"]) if df["volume"].sum() > 0 else np.nan
        print(f"  Volume-weighted Brier (open):  {vw_brier_open:.4f}")
        print(f"  Volume-weighted Brier (close): {vw_brier_close:.4f}")
        print(f"  Volume-weighted improvement:   {vw_brier_open - vw_brier_close:.4f}")

    # CLV
    print("\n" + "─" * 50)
    print("  CLOSING LINE VALUE (CLV)")
    print("─" * 50)
    clv_df = clv_analysis(df)

    # Calibration
    print("\n" + "─" * 50)
    print("  CALIBRATION")
    print("─" * 50)
    cal_df = calibration_analysis(df)
    print(cal_df.to_string())

    # Summary
    n_games = df["game_id"].nunique() if "game_id" in df.columns else len(df)
    n_days = df["day"].nunique()
    print(f"\n{'='*60}")
    print(f"  ✓ Done: {len(df)} contracts across {n_games} games over {n_days} days")
    if len(df_live) > 0:
        print(f"    ({len(df_live)} live-settled contracts excluded)")
    print(f"{'='*60}")

    return {
        "df": df,
        "df_live_settled": df_live,
        "drift_summary": drift_df,
        "drift_bootstrap": drift_boot,
        "dow_summary": dow_df,
        "clv": clv_df,
        "calibration": cal_df,
    }


if __name__ == "__main__":
    result = main()