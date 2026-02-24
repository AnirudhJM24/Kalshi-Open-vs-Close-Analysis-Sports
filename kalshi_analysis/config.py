"""
Configuration, sport registry, and resolution helpers.
"""

import os
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone


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
_SPORT_BY_SHORT = {
    short: (ticker, display)
    for _, (ticker, short, display) in SPORT_MENU.items()
}
_SPORT_BY_TICKER = {
    ticker: (ticker, display)
    for _, (ticker, _, display) in SPORT_MENU.items()
}


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
                print(f"\n> Selected: {name} ({ticker})")
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
