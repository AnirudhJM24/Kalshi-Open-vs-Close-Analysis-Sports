"""
HTTP session, Kalshi API helpers, and market/candlestick fetching.
"""

import os
import json
import time
import random
from typing import Optional

import pandas as pd
import requests
from requests.adapters import HTTPAdapter

from .config import CFG


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


# ── Settled Markets ──────────────────────────────────────────────────────────
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
            print(f"  Bad cache ({e}), re-fetching.")

    mkts = get_settled_markets(series_ticker, min_ts, max_ts)
    print(f"  {series_ticker}: {len(mkts)} settled markets")

    try:
        with open(cache_path, "w") as f:
            json.dump(mkts, f)
        print(f"  Saved cache: {cache_path}")
    except IOError as e:
        print(f"  Cache write failed: {e}")

    return {series_ticker: mkts}


# ── Candlesticks ─────────────────────────────────────────────────────────────
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
