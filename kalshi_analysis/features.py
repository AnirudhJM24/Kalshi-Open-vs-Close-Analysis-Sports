"""
Contract DataFrame building, feature engineering, deduplication, and filtering.
"""

import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from typing import Optional

import numpy as np
import pandas as pd

from .config import CFG
from .api import fetch_candlesticks


# ── Outcome & Price Extraction ───────────────────────────────────────────────
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
    """Sum volume across all candles."""
    total = 0
    for c in candles:
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
    """Process a single market: fetch candles, extract open/close + volume."""
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

    volume = _extract_candle_volume(candles)

    close_et = close_dt.tz_convert(CFG.day_tz)
    dow = close_et.dayofweek
    dow_name = close_et.strftime("%A")

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
                print(f"    {done_count}/{total} ({done_count/total:.0%}) -- {len(rows)} rows", flush=True)

    df = pd.DataFrame(rows)
    print(f"  Raw rows with open+close+outcome: {len(df)}")
    return df


# ── Feature Engineering ──────────────────────────────────────────────────────
def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add drift, edge, favorite, underdog_win, and CLV columns (vectorized)."""
    df = df.copy()
    df["drift"] = df["p_close"] - df["p_open"]
    df["abs_drift"] = df["drift"].abs()
    df["edge_open"] = df["y"] - df["p_open"]
    df["edge_close"] = df["y"] - df["p_close"]
    df["favorite"] = (df["p_open"] > 0.5).astype(int)
    df["underdog_win"] = ((df["p_open"] < 0.5) & (df["y"] == 1)).astype(int)
    df["clv_raw"] = df["p_close"] - df["p_open"]
    df["clv_correct"] = (((df["y"] == 1) & (df["clv_raw"] > 0)) |
                         ((df["y"] == 0) & (df["clv_raw"] < 0))).astype(int)
    df["pnl_buy_open"] = df["y"] - df["p_open"]
    df["clv_edge"] = df["p_close"] - df["p_open"]
    return df


# ── Paired-Contract Deduplication ────────────────────────────────────────────
def _extract_game_id(ticker: str) -> str:
    """Extract game ID from a Kalshi market ticker by stripping the team suffix."""
    parts = ticker.rsplit("-", 1)
    if len(parts) == 2 and len(parts[1]) <= 5 and parts[1].isalpha():
        return parts[0]
    return ticker


def deduplicate_paired_contracts(df: pd.DataFrame) -> pd.DataFrame:
    """Keep one contract per game (the favorite side, or first alphabetically)."""
    before = len(df)
    df = df.copy()
    df["game_id"] = df["market"].apply(_extract_game_id)
    df = df.sort_values(["game_id", "p_open", "market"], ascending=[True, False, True])
    df = df.drop_duplicates(subset="game_id", keep="first")
    after = len(df)
    dropped = before - after
    print(f"  Deduplication: {before} -> {after} contracts ({dropped} mirror pairs removed)")
    return df


# ── Live-Settled Filtering ───────────────────────────────────────────────────
def filter_live_settled(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split DataFrame into pre-game-close (clean) and live-settled (contaminated)."""
    threshold = CFG.live_settled_threshold
    is_live = (df["p_close"] <= threshold) | (df["p_close"] >= 1.0 - threshold)
    df_clean = df[~is_live].copy()
    df_live = df[is_live].copy()
    pct_live = len(df_live) / len(df) * 100 if len(df) > 0 else 0
    print(f"  Live-settled filter (threshold={threshold}):")
    print(f"    Clean contracts: {len(df_clean)}")
    print(f"    Live-settled (removed): {len(df_live)} ({pct_live:.1f}%)")
    print(f"    These had p_close <= {threshold} or >= {1-threshold} -- outcome already known")
    return df_clean, df_live
