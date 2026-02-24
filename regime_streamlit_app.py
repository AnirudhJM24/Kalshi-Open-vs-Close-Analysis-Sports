import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st

import regime


st.set_page_config(
    page_title="Kalshi Sports — Open vs Close Regime (Streamlit)",
    layout="wide",
)


def _get_sport_choices() -> list[tuple[str, str]]:
    """Return list of (short, display_name) from regime.SPORT_MENU."""
    return [
        (short, display)
        for _, (_, short, display) in sorted(regime.SPORT_MENU.items())
    ]


@st.cache_data(show_spinner=False)
def run_regime_pipeline(
    sport_short: str,
    days: int,
    keep_pairs: bool,
    keep_live_settled: bool,
    workers: int,
    no_cache: bool,
):
    """Run the core regime pipeline for given settings and return dataframes."""
    ticker, name = regime.resolve_sport(sport_short)

    # Configure global CFG used by regime.py
    regime.CFG.lookback_days = days
    regime.CFG.max_workers = workers
    regime.CFG.no_cache = no_cache
    regime.CFG.set_sport(ticker, name)

    markets_by_series = regime.fetch_settled_markets_single(ticker)
    markets = markets_by_series.get(ticker, [])
    if len(markets) == 0:
        return {
            "sport_ticker": ticker,
            "sport_name": name,
            "df": pd.DataFrame(),
            "df_live": pd.DataFrame(),
            "drift_df": pd.DataFrame(),
            "drift_boot": {},
            "dow_df": pd.DataFrame(),
        }

    df = regime.build_contract_df(markets_by_series)
    if df.empty:
        return {
            "sport_ticker": ticker,
            "sport_name": name,
            "df": df,
            "df_live": pd.DataFrame(),
            "drift_df": pd.DataFrame(),
            "drift_boot": {},
            "dow_df": pd.DataFrame(),
        }

    df = regime.add_features(df)

    if not keep_pairs:
        df = regime.deduplicate_paired_contracts(df)

    df_live = pd.DataFrame()
    if not keep_live_settled:
        df, df_live = regime.filter_live_settled(df)
        if df.empty:
            return {
                "sport_ticker": ticker,
                "sport_name": name,
                "df": df,
                "df_live": df_live,
                "drift_df": pd.DataFrame(),
                "drift_boot": {},
                "dow_df": pd.DataFrame(),
            }

    drift_df, drift_boot = regime.drift_regime_analysis(df)
    dow_df = regime.dow_regime_analysis(df)

    return {
        "sport_ticker": ticker,
        "sport_name": name,
        "df": df,
        "df_live": df_live,
        "drift_df": drift_df,
        "drift_boot": drift_boot,
        "dow_df": dow_df,
    }


def plot_sanity(df: pd.DataFrame, target_name: str):
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))

    df["p_open"].hist(bins=50, ax=axes[0], alpha=0.7)
    axes[0].set_title(f"Opening Price — {target_name}")
    axes[0].set_xlabel("p_open")

    df["drift"].hist(bins=60, ax=axes[1], alpha=0.7, color="orange")
    axes[1].set_title(f"Drift (close − open) — {target_name}")
    axes[1].set_xlabel("drift")

    df["volume"].hist(bins=40, ax=axes[2], alpha=0.7, color="green")
    axes[2].set_title(f"Volume per Contract — {target_name}")
    axes[2].set_xlabel("volume (contracts traded)")

    plt.tight_layout()
    st.pyplot(fig)


def compute_clv(df: pd.DataFrame):
    """Compute CLV summary and bucketed stats (mirrors regime.clv_analysis logic)."""
    df = df.copy()

    overall = {
        "total_contracts": int(len(df)),
        "mean_clv_raw": float(df["clv_raw"].mean()),
        "clv_correct_rate": float(df["clv_correct"].mean()),
        "mean_pnl_buy_open": float(df["pnl_buy_open"].mean()),
        "mean_abs_clv": float(df["clv_raw"].abs().mean()),
    }

    df["pbin"] = pd.cut(
        df["p_open"],
        bins=[0, 0.3, 0.5, 0.7, 1.0],
        labels=[
            "Heavy underdog (0-30)",
            "Slight underdog (30-50)",
            "Slight favorite (50-70)",
            "Heavy favorite (70-100)",
        ],
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

    return overall, clv_by_bucket, df


def plot_clv(clv_df: pd.DataFrame, clv_by_bucket: pd.DataFrame):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    axes[0].hist(clv_df["clv_raw"].values, bins=50, alpha=0.7, color="steelblue", edgecolor="white")
    axes[0].axvline(0, color="red", linestyle="--", alpha=0.7)
    mean_clv = clv_df["clv_raw"].mean()
    axes[0].axvline(mean_clv, color="orange", linestyle="-", linewidth=2, label=f"Mean CLV: {mean_clv:+.3f}")
    axes[0].set_xlabel("CLV (close − open)")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Closing Line Value Distribution")
    axes[0].legend()

    if not clv_by_bucket.empty:
        x = np.arange(len(clv_by_bucket))
        axes[1].bar(
            x,
            clv_by_bucket["clv_correct_rate"],
            color="teal",
            alpha=0.7,
            edgecolor="white",
        )
        axes[1].axhline(0.5, color="red", linestyle="--", alpha=0.5, label="50% (random)")
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(clv_by_bucket["pbin"].astype(str), rotation=15, ha="right")
        axes[1].set_ylabel("Fraction line moved correctly")
        axes[1].set_title("Line Accuracy by Opening Price Bucket")
        axes[1].set_ylim(0, 1)
        axes[1].legend()

    plt.tight_layout()
    st.pyplot(fig)


def compute_calibration(df: pd.DataFrame, target_name: str):
    """Compute calibration bucket stats and return (bucket_df, fig)."""
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

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(bucket["implied_open"], bucket["realized"], "o-", label="Open price", linewidth=2)
    ax.plot(bucket["implied_close"], bucket["realized"], "s-", label="Close price", linewidth=2)
    ax.plot([0, 1], [0, 1], "--", color="gray", alpha=0.6, label="Perfect calibration")
    ax.set_xlabel("Implied probability")
    ax.set_ylabel("Realized win rate")
    ax.set_title(f"Calibration: Open vs Close — {target_name}")
    ax.legend()
    plt.tight_layout()

    return bucket.reset_index(), fig


def plot_drift_regime(drift_df: pd.DataFrame):
    if drift_df.empty:
        st.info("Not enough data for drift regime analysis.")
        return

    # Aggregate across series (usually just one)
    agg = (
        drift_df.groupby("drift_regime")
        .agg(
            days=("days", "sum"),
            mean_abs_drift=("mean_abs_drift", "mean"),
            brier_open=("brier_open", "mean"),
            brier_close=("brier_close", "mean"),
            brier_improvement=("brier_improvement", "mean"),
        )
        .reset_index()
    )

    st.subheader("Drift Regime Summary")
    st.dataframe(agg)

    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(agg))
    ax.bar(x, agg["brier_improvement"], color="steelblue", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(agg["drift_regime"], rotation=0)
    ax.set_ylabel("Brier improvement (open − close)")
    ax.set_title("Brier Improvement by Drift Regime")
    plt.tight_layout()
    st.pyplot(fig)


def plot_dow_regime(dow_df: pd.DataFrame):
    if dow_df.empty:
        st.info("Not enough data for day-of-week analysis.")
        return

    st.subheader("Day-of-Week Regime Summary")
    st.dataframe(dow_df)

    fig, ax = plt.subplots(figsize=(8, 4))
    x = np.arange(len(dow_df))
    ax.bar(x, dow_df["brier_improvement"], color="purple", alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(dow_df["dow_name"], rotation=15, ha="right")
    ax.set_ylabel("Brier improvement (open − close)")
    ax.set_title("Brier Improvement by Day of Week")
    plt.tight_layout()
    st.pyplot(fig)


def main():
    st.title("Kalshi Sports — Open vs Close Regime (Streamlit)")
    st.markdown(
        "Single-sport analysis of Kalshi sports markets over the last N days, "
        "visualized in Streamlit. This wraps the `regime.py` pipeline."
    )

    # Sidebar controls
    sport_choices = _get_sport_choices()
    short_to_name = {short: display for short, display in sport_choices}

    default_short = "nba" if "nba" in short_to_name else sport_choices[0][0]
    sport_short = st.sidebar.selectbox(
        "Sport",
        options=[s for s, _ in sport_choices],
        index=[s for s, _ in sport_choices].index(default_short),
        format_func=lambda s: f"{s.upper()} — {short_to_name[s]}",
    )

    days = st.sidebar.slider("Lookback window (days)", min_value=7, max_value=90, value=30, step=1)
    workers = st.sidebar.slider("Parallel workers", min_value=1, max_value=16, value=4, step=1)
    no_cache = st.sidebar.checkbox("Force re-fetch (ignore disk cache)", value=False)
    keep_live_settled = st.sidebar.checkbox("Keep live-settled contracts (no filter)", value=False)
    keep_pairs = st.sidebar.checkbox("Keep both sides of each game (no dedup)", value=False)

    st.sidebar.markdown("---")
    run_clicked = st.sidebar.button("Run analysis", type="primary")
    st.sidebar.markdown("Run this app with:\n\n`streamlit run regime_streamlit_app.py`")

    if not run_clicked:
        st.info("Choose a sport and options in the sidebar, then click **Run analysis** to fetch data and run the regime analysis.")
        return

    with st.spinner("Fetching data and running regime analysis..."):
        result = run_regime_pipeline(
            sport_short=sport_short,
            days=days,
            keep_pairs=keep_pairs,
            keep_live_settled=keep_live_settled,
            workers=workers,
            no_cache=no_cache,
        )

    df = result["df"]
    df_live = result["df_live"]
    drift_df = result["drift_df"]
    dow_df = result["dow_df"]
    sport_name = result["sport_name"]

    if df.empty:
        st.warning(
            f"No clean contracts found for {sport_name} over the last {days} days "
            "(after candlestick extraction / filtering). "
            "Try increasing the lookback window or choosing another sport."
        )
        return

    n_games = df["game_id"].nunique() if "game_id" in df.columns else len(df)
    n_days = df["day"].nunique()

    total_vol = df["volume"].sum()
    mean_vol = df["volume"].mean()
    median_vol = df["volume"].median()

    if total_vol > 0:
        vw_brier_open = float(np.average(df["edge_open"] ** 2, weights=df["volume"]))
        vw_brier_close = float(np.average(df["edge_close"] ** 2, weights=df["volume"]))
        vw_improvement = vw_brier_open - vw_brier_close
    else:
        vw_brier_open = vw_brier_close = vw_improvement = float("nan")

    top_cols = st.columns(4)
    top_cols[0].metric("Clean contracts", f"{len(df):,}")
    top_cols[1].metric("Games", f"{n_games:,}")
    top_cols[2].metric("Days", f"{n_days:,}")
    if not df_live.empty:
        top_cols[3].metric("Live-settled excluded", f"{len(df_live):,}")
    else:
        top_cols[3].metric("Live-settled excluded", "0")

    vol_cols = st.columns(3)
    vol_cols[0].metric("Total volume", f"{int(total_vol):,}")
    vol_cols[1].metric("Mean volume / contract", f"{mean_vol:,.0f}")
    vol_cols[2].metric("Median volume / contract", f"{median_vol:,.0f}")

    brier_cols = st.columns(3)
    brier_cols[0].metric("VW Brier (open)", f"{vw_brier_open:.4f}")
    brier_cols[1].metric("VW Brier (close)", f"{vw_brier_close:.4f}")
    brier_cols[2].metric("VW improvement", f"{vw_improvement:+.4f}")

    tabs = st.tabs(
        [
            "Sanity",
            "Drift Regime",
            "Day-of-Week Regime",
            "CLV",
            "Calibration",
            "Raw Contracts",
        ]
    )

    with tabs[0]:
        st.header("Sanity Checks")
        st.markdown("Distributions of opening prices, drift, and volume.")
        plot_sanity(df, sport_name)

    with tabs[1]:
        st.header("Drift Regime (Information Flow)")
        plot_drift_regime(drift_df)

    with tabs[2]:
        st.header("Day-of-Week Regime")
        plot_dow_regime(dow_df)

    with tabs[3]:
        st.header("Closing Line Value (CLV)")
        overall_clv, clv_by_bucket, clv_df = compute_clv(df)

        clv_metrics = st.columns(4)
        clv_metrics[0].metric("Contracts analyzed", f"{overall_clv['total_contracts']:,}")
        clv_metrics[1].metric("Mean CLV (close − open)", f"{overall_clv['mean_clv_raw']:+.4f}")
        clv_metrics[2].metric("Line moved correctly", f"{overall_clv['clv_correct_rate']:.1%}")
        clv_metrics[3].metric("Mean PnL (buy YES at open)", f"{overall_clv['mean_pnl_buy_open']:+.4f}")

        st.subheader("CLV by Opening Price Bucket")
        st.dataframe(clv_by_bucket)
        plot_clv(clv_df, clv_by_bucket)

    with tabs[4]:
        st.header("Calibration: Open vs Close")
        cal_bucket, cal_fig = compute_calibration(df, sport_name)
        st.subheader("Calibration Buckets")
        st.dataframe(cal_bucket)
        st.subheader("Calibration Plot")
        st.pyplot(cal_fig)

    with tabs[5]:
        st.header("Raw Contract-Level Data")
        st.dataframe(df)


if __name__ == "__main__":
    main()

