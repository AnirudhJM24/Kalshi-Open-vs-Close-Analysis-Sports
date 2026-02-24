"""
Regime analysis, CLV, calibration, and bootstrap significance testing.
"""

import numpy as np
import pandas as pd

from .config import CFG


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


# ── Bootstrap Significance ──────────────────────────────────────────────────
def _bootstrap_brier_improvement(
    daily: pd.DataFrame,
    regime_col: str,
    label_hi: str,
    label_lo: str,
    n_boot: int = None,
    ci: float = None,
) -> dict:
    """Bootstrap the Brier improvement difference between two regimes."""
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
    """Regime 1: High vs Low Drift days (information-flow regime)."""
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

        boot = _bootstrap_brier_improvement(daily, "drift_regime", "High drift", "Low drift")
        boot_results[series] = boot

    summary_df = pd.concat(results, ignore_index=True) if results else pd.DataFrame()
    return summary_df, boot_results


def dow_regime_analysis(df: pd.DataFrame) -> pd.DataFrame:
    """Regime 2: Day-of-week -- observable before games start."""
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

        dow_order = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                     "Friday": 4, "Saturday": 5, "Sunday": 6}
        agg["_sort"] = agg["dow_name"].map(dow_order)
        agg = agg.sort_values("_sort").drop(columns="_sort")

        results.append(agg)

    return pd.concat(results, ignore_index=True) if results else pd.DataFrame()


def clv_analysis(df: pd.DataFrame) -> tuple[dict, pd.DataFrame]:
    """Closing Line Value analysis. Returns (overall_dict, clv_by_bucket_df)."""
    df = df.copy()

    overall = {
        "total_contracts": len(df),
        "mean_clv_raw": float(df["clv_raw"].mean()),
        "clv_correct_rate": float(df["clv_correct"].mean()),
        "mean_pnl_buy_open": float(df["pnl_buy_open"].mean()),
        "mean_abs_clv": float(df["clv_raw"].abs().mean()),
    }

    df["pbin"] = pd.cut(
        df["p_open"],
        bins=[0, 0.3, 0.5, 0.7, 1.0],
        labels=["Heavy underdog (0-30)", "Slight underdog (30-50)",
                "Slight favorite (50-70)", "Heavy favorite (70-100)"],
    )

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

    return overall, clv_by_bucket


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
            brier_open=("edge_open", lambda x: float(np.mean(np.square(x)))),
            brier_close=("edge_close", lambda x: float(np.mean(np.square(x)))),
            mean_abs_drift=("abs_drift", "mean"),
        )
        .dropna()
    )
    return bucket.reset_index()


def print_bootstrap_results(boot_results: dict, regime_name: str):
    """Pretty-print bootstrap significance results."""
    ci_pct = int(CFG.bootstrap_ci * 100)
    for series, b in boot_results.items():
        sig = "SIGNIFICANT" if b["p_value"] < 0.05 else "not significant"
        print(f"  {series}: delta Brier improvement = {b['point_est']:+.4f}")
        print(f"    {ci_pct}% CI: [{b['ci_lo']:+.4f}, {b['ci_hi']:+.4f}]")
        print(f"    p-value: {b['p_value']:.4f}  ({sig} at alpha=0.05)")
        print(f"    n_hi={b['n_hi']}, n_lo={b['n_lo']}")
