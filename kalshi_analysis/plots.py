"""
Matplotlib plots for CLI usage.

Uses the non-interactive 'Agg' backend so plots work on headless Linux servers.
Figures are saved or shown depending on the environment.
"""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd

from .config import CFG


def sanity_plots(df: pd.DataFrame):
    """Quick distribution checks (CLI mode -- prints stats, saves/shows figures)."""
    for series, grp in df.groupby("series"):
        print(f"\n  {series}: {len(grp)} contracts")
        print(grp[["p_open", "p_close", "drift", "abs_drift", "volume"]].describe().round(4).to_string())

        fig, axes = plt.subplots(1, 3, figsize=(16, 4))
        grp["p_open"].hist(bins=50, ax=axes[0], alpha=0.7)
        axes[0].set_title(f"Opening Price -- {CFG.target_name}")
        axes[0].set_xlabel("p_open")

        grp["drift"].hist(bins=60, ax=axes[1], alpha=0.7, color="orange")
        axes[1].set_title(f"Drift (close - open) -- {CFG.target_name}")
        axes[1].set_xlabel("drift")

        grp["volume"].hist(bins=40, ax=axes[2], alpha=0.7, color="green")
        axes[2].set_title(f"Volume per Contract -- {CFG.target_name}")
        axes[2].set_xlabel("volume (contracts traded)")

        plt.tight_layout()
        plt.show()
        plt.close(fig)


def clv_plots(df: pd.DataFrame, clv_by_bucket: pd.DataFrame):
    """CLV distribution and accuracy bar chart (CLI mode)."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(df["clv_raw"].values, bins=50, alpha=0.7, color="steelblue", edgecolor="white")
    axes[0].axvline(0, color="red", linestyle="--", alpha=0.7)
    mean_clv = df["clv_raw"].mean()
    axes[0].axvline(mean_clv, color="orange", linestyle="-", linewidth=2,
                    label=f"Mean CLV: {mean_clv:+.3f}")
    axes[0].set_xlabel("CLV (close - open)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Closing Line Value Distribution")
    axes[0].legend()

    if not clv_by_bucket.empty:
        x = range(len(clv_by_bucket))
        axes[1].bar(x, clv_by_bucket["clv_correct_rate"], color="teal", alpha=0.7, edgecolor="white")
        axes[1].axhline(0.5, color="red", linestyle="--", alpha=0.5, label="50% (random)")
        axes[1].set_xticks(list(x))
        axes[1].set_xticklabels(clv_by_bucket["pbin"].astype(str), rotation=15, ha="right")
        axes[1].set_ylabel("Fraction line moved correctly")
        axes[1].set_title("Line Accuracy by Opening Price Bucket")
        axes[1].set_ylim(0, 1)
        axes[1].legend()

    plt.tight_layout()
    plt.show()
    plt.close(fig)


def calibration_plot(bucket: pd.DataFrame):
    """Calibration scatter: implied vs realized (CLI mode)."""
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(bucket["implied_open"], bucket["realized"], "o-", label="Open price", linewidth=2)
    ax.plot(bucket["implied_close"], bucket["realized"], "s-", label="Close price", linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.6, label="Perfect calibration")
    ax.set_xlabel("Implied probability")
    ax.set_ylabel("Realized win rate")
    ax.set_title(f"Calibration: Open vs Close -- {CFG.target_name}")
    ax.legend()
    plt.tight_layout()
    plt.show()
    plt.close(fig)
