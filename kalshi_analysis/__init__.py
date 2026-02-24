"""
Kalshi Sports -- Open vs Close Regime Analysis

Public API re-exports for use by the Streamlit app and CLI.
"""

# Force non-interactive matplotlib backend before anything imports pyplot.
# This prevents crashes on headless Linux servers.
import matplotlib
matplotlib.use("Agg")

import pandas as pd

pd.set_option("display.max_columns", 200)

from .config import (
    CFG,
    SPORT_MENU,
    resolve_sport,
    pick_sport_interactive,
)
from .api import (
    fetch_settled_markets_single,
    fetch_candlesticks,
)
from .features import (
    build_contract_df,
    add_features,
    deduplicate_paired_contracts,
    filter_live_settled,
)
from .analysis import (
    drift_regime_analysis,
    dow_regime_analysis,
    clv_analysis,
    calibration_analysis,
    print_bootstrap_results,
)

__all__ = [
    "CFG",
    "SPORT_MENU",
    "resolve_sport",
    "pick_sport_interactive",
    "fetch_settled_markets_single",
    "fetch_candlesticks",
    "build_contract_df",
    "add_features",
    "deduplicate_paired_contracts",
    "filter_live_settled",
    "drift_regime_analysis",
    "dow_regime_analysis",
    "clv_analysis",
    "calibration_analysis",
    "print_bootstrap_results",
]
