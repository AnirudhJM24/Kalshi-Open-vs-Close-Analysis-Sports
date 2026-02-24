"""
Kalshi Sports -- Open vs Close Regime Analysis (CLI)

Usage:
  python cli.py                          # interactive sport picker
  python cli.py --sport nba              # NBA, last 30 days
  python cli.py --sport nhl --days 14    # NHL, last 14 days
  python cli.py --sport mlb --no-cache   # force re-fetch
  python cli.py --list                   # show available sports
"""

import argparse

import numpy as np
import pandas as pd

import kalshi_analysis as ka
from kalshi_analysis.plots import sanity_plots, clv_plots, calibration_plot


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Kalshi Sports -- Open vs Close Regime Analysis",
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


def main():
    parser = build_parser()
    args = parser.parse_args()

    if args.list:
        print("\nAvailable sports:")
        for _, (ticker, short, name) in sorted(ka.SPORT_MENU.items()):
            print(f"  {short:<8s}  {ticker:<22s}  {name}")
        return

    ka.CFG.lookback_days = args.days
    ka.CFG.max_workers = args.workers
    ka.CFG.no_cache = args.no_cache

    if args.sport:
        ticker, name = ka.resolve_sport(args.sport)
    else:
        ticker, name = ka.pick_sport_interactive()

    ka.CFG.set_sport(ticker, name)

    print("\n" + "=" * 60)
    print(f"  {name} ({ticker})")
    print(f"  {ka.CFG.min_close_date}  ->  {ka.CFG.max_close_date}  ({ka.CFG.lookback_days} days)")
    if not args.keep_live_settled:
        print(f"  Live-settled filter: ON (threshold={ka.CFG.live_settled_threshold})")
    if not args.keep_pairs:
        print(f"  Paired-contract dedup: ON")
    print("=" * 60)

    # 1. Fetch markets
    print(f"\n[1/7] Fetching settled markets...")
    markets_by_series = ka.fetch_settled_markets_single(ticker)
    n_markets = len(markets_by_series.get(ticker, []))
    if n_markets == 0:
        print(f"\nNo settled markets for {name} in the last {ka.CFG.lookback_days} days.")
        print("  This sport may be out of season. Try --sport nba or increase --days.")
        return
    print(f"  -> {n_markets} settled markets")

    # 2. Build contract DataFrame
    print(f"\n[2/7] Building contract dataset...")
    df = ka.build_contract_df(markets_by_series)
    if df.empty:
        print("No data after candlestick extraction. Check API auth.")
        return

    # 3. Feature engineering
    print(f"\n[3/7] Computing features...")
    df = ka.add_features(df)

    # 4. Deduplicate paired contracts
    if not args.keep_pairs:
        print(f"\n[4/7] Deduplicating paired contracts...")
        df = ka.deduplicate_paired_contracts(df)
    else:
        print(f"\n[4/7] Skipping dedup (--keep-pairs)")

    # 5. Filter live-settled
    df_live = pd.DataFrame()
    if not args.keep_live_settled:
        print(f"\n[5/7] Filtering live-settled contracts...")
        df, df_live = ka.filter_live_settled(df)
        if df.empty:
            print("All contracts were live-settled! Try --keep-live-settled to analyze anyway.")
            return
    else:
        print(f"\n[5/7] Skipping live-settled filter (--keep-live-settled)")

    # 6. Sanity checks
    print(f"\n[6/7] Sanity checks...")
    sanity_plots(df)

    # 7. Analysis
    print(f"\n[7/7] Analysis for {name} ({len(df)} clean contracts)...")

    # Drift regime
    print("\n" + "-" * 50)
    print("  DRIFT REGIME (Information Flow)")
    print("-" * 50)
    drift_df, drift_boot = ka.drift_regime_analysis(df)
    if drift_df.empty:
        print("  Not enough data for drift regime analysis.")
    else:
        print(drift_df.to_string(index=False))
        print("\n  Bootstrap significance (High drift improvement vs Low drift improvement):")
        ka.print_bootstrap_results(drift_boot, "drift")

    # Day-of-week regime
    print("\n" + "-" * 50)
    print("  DAY-OF-WEEK REGIME")
    print("-" * 50)
    dow_df = ka.dow_regime_analysis(df)
    if dow_df.empty:
        print("  Not enough data for day-of-week analysis.")
    else:
        print(dow_df.to_string(index=False))

    # Volume summary
    print("\n" + "-" * 50)
    print("  VOLUME SUMMARY")
    print("-" * 50)
    total_vol = df["volume"].sum()
    mean_vol = df["volume"].mean()
    median_vol = df["volume"].median()
    print(f"  Total volume across all contracts: {total_vol:,}")
    print(f"  Mean volume per contract: {mean_vol:,.0f}")
    print(f"  Median volume per contract: {median_vol:,.0f}")
    if total_vol > 0:
        vw_brier_open = np.average(df["edge_open"] ** 2, weights=df["volume"]) if df["volume"].sum() > 0 else np.nan
        vw_brier_close = np.average(df["edge_close"] ** 2, weights=df["volume"]) if df["volume"].sum() > 0 else np.nan
        print(f"  Volume-weighted Brier (open):  {vw_brier_open:.4f}")
        print(f"  Volume-weighted Brier (close): {vw_brier_close:.4f}")
        print(f"  Volume-weighted improvement:   {vw_brier_open - vw_brier_close:.4f}")

    # CLV
    print("\n" + "-" * 50)
    print("  CLOSING LINE VALUE (CLV)")
    print("-" * 50)
    overall_clv, clv_by_bucket = ka.clv_analysis(df)
    print(f"    Contracts analyzed: {overall_clv['total_contracts']}")
    print(f"    Mean CLV (close - open): {overall_clv['mean_clv_raw']:+.4f}")
    print(f"    Line moved correctly: {overall_clv['clv_correct_rate']:.1%} of the time")
    print(f"    Mean |CLV|: {overall_clv['mean_abs_clv']:.4f}")
    print(f"    Mean PnL (buy YES at open): {overall_clv['mean_pnl_buy_open']:+.4f}")
    print("\n  CLV by Opening Price Bucket:")
    print(clv_by_bucket.to_string(index=False))
    clv_plots(df, clv_by_bucket)

    # Calibration
    print("\n" + "-" * 50)
    print("  CALIBRATION")
    print("-" * 50)
    cal_df = ka.calibration_analysis(df)
    print(cal_df.to_string())
    calibration_plot(cal_df)

    # Summary
    n_games = df["game_id"].nunique() if "game_id" in df.columns else len(df)
    n_days = df["day"].nunique()
    print(f"\n{'='*60}")
    print(f"  Done: {len(df)} contracts across {n_games} games over {n_days} days")
    if len(df_live) > 0:
        print(f"    ({len(df_live)} live-settled contracts excluded)")
    print(f"{'='*60}")

    return {
        "df": df,
        "df_live_settled": df_live,
        "drift_summary": drift_df,
        "drift_bootstrap": drift_boot,
        "dow_summary": dow_df,
        "clv": clv_by_bucket,
        "calibration": cal_df,
    }


if __name__ == "__main__":
    result = main()
