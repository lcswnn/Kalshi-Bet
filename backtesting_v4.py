"""
KALSHI WEATHER MODEL BACKTESTER - IMPROVED v4
=============================================
Implements the key findings:
1. NEVER bet on contracts below 10¢ (the cheap contract trap)
2. Focus on mid-priced contracts (15-50¢) where real uncertainty exists
3. Use calibrated probability model (no artificial spike weight)
4. Require higher edge thresholds for extreme prices

Based on analysis showing:
- 1-5¢ contracts: -97% ROI (AVOID)
- 6-20¢ contracts: +30% ROI (GOOD)
- 21-50¢ contracts: +55% ROI (BEST)
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from scipy import stats
import os

# ============ CONFIGURATION ============
FLAT_BET_SIZE = 10

# KEY FILTERS (based on analysis)
MIN_CONTRACT_PRICE = 0.15  # Never bet on contracts below 15¢
MAX_CONTRACT_PRICE = 0.90  # Never bet on contracts above 90¢
SWEET_SPOT_LOW = 0.15      # Preferred range lower bound (same as min now)
SWEET_SPOT_HIGH = 0.50     # Preferred range upper bound

# Edge requirements by price bucket
EDGE_REQUIREMENTS = {
    "sweet_spot": 0.12,    # 12% edge for 15-50¢ contracts
    "high_price": 0.12,    # 12% edge for 50-90¢ contracts
}

# Forecast error model (based on HRRR typical accuracy)
FORECAST_ERROR_STD = 2.8  # Calibrated to typical day-ahead forecast errors

# ============ HELPER FUNCTIONS ============

def get_bin(temp):
    """Get Kalshi bin for a temperature (2-degree bins)."""
    lower = int(np.floor(temp))
    if lower % 2 == 0:
        lower -= 1
    return (lower, lower + 1)


def get_price_bucket(price):
    """Categorize price into buckets for edge requirements."""
    if price <= SWEET_SPOT_HIGH:
        return "sweet_spot"
    else:
        return "high_price"


def get_min_edge(price):
    """Get minimum required edge based on price bucket."""
    bucket = get_price_bucket(price)
    return EDGE_REQUIREMENTS[bucket]


# ============ PROBABILITY MODEL (Calibrated Gaussian) ============

def model_probability(forecast, contract_type, bound1, bound2, error_std):
    """
    Pure Gaussian probability model.
    No artificial spike - just the statistical distribution of forecast errors.
    """
    if contract_type == "range":
        # P(actual in [bound1, bound2])
        prob = stats.norm.cdf(bound2, forecast, error_std) - \
               stats.norm.cdf(bound1, forecast, error_std)
    elif contract_type == "below":
        # P(actual <= bound1)
        prob = stats.norm.cdf(bound1 + 0.5, forecast, error_std)
    elif contract_type == "above":
        # P(actual >= bound1)
        prob = 1 - stats.norm.cdf(bound1 - 0.5, forecast, error_std)
    else:
        return 0.5
    
    return max(min(prob, 0.99), 0.01)


# ============ BACKTEST ENGINE ============

def simulate_forecast(actual_temp, error_std):
    """
    Simulate a realistic forecast by adding typical forecast error.
    In real trading, this would be replaced with actual HRRR/GFS forecasts.
    """
    # Forecast error: mostly small, occasionally larger
    error = np.random.normal(0, error_std * 0.7)
    error = np.clip(error, -2.5 * error_std, 2.5 * error_std)
    return actual_temp + error


def run_backtest(kalshi_df, city, use_simulated_forecasts=True):
    """
    Run improved backtest for a city.
    """
    print(f"\n{'='*70}")
    print(f"BACKTESTING: {city.upper()}")
    print(f"{'='*70}")
    
    city_data = kalshi_df[kalshi_df["city"] == city].copy()
    if len(city_data) == 0:
        return None
    
    dates = sorted(city_data["date"].unique())
    print(f"  Date range: {dates[0].date()} to {dates[-1].date()}")
    print(f"  Total days: {len(dates)}")
    
    # Results tracking
    results = {
        "city": city,
        "total_days": 0,
        "bets_placed": 0,
        "bets_won": 0,
        "total_wagered": 0,
        "net_profit": 0,
        "bets_by_bucket": {
            "sweet_spot": {"count": 0, "won": 0, "profit": 0},
            "high_price": {"count": 0, "won": 0, "profit": 0},
        },
        "bets_by_type": {
            "yes_range": {"count": 0, "won": 0, "profit": 0},
            "no_range": {"count": 0, "won": 0, "profit": 0},
            "yes_above": {"count": 0, "won": 0, "profit": 0},
            "no_above": {"count": 0, "won": 0, "profit": 0},
            "yes_below": {"count": 0, "won": 0, "profit": 0},
            "no_below": {"count": 0, "won": 0, "profit": 0},
        },
        "skipped_cheap": 0,
        "skipped_expensive": 0,
        "skipped_low_edge": 0,
        "bet_details": [],
        "profits": [],
    }
    
    for target_date in dates:
        day_contracts = city_data[city_data["date"] == target_date].copy()
        if len(day_contracts) == 0:
            continue
        
        # Get actual temperature from winning contract
        winning = day_contracts[day_contracts["result"] == "yes"]
        if len(winning) == 0:
            continue
        
        actual_temp = winning.iloc[0]["actual_temp_approx"]
        if pd.isna(actual_temp):
            continue
        
        results["total_days"] += 1
        
        # Generate forecast
        if use_simulated_forecasts:
            forecast = simulate_forecast(actual_temp, FORECAST_ERROR_STD)
        else:
            forecast = actual_temp  # Perfect forecast (ceiling)
        
        our_bin = get_bin(forecast)
        
        # === FIND BEST BET ===
        best_bet = None
        best_edge = 0
        best_edge_ratio = 0  # edge / min_required_edge
        
        for _, contract in day_contracts.iterrows():
            kalshi_price = contract["last_price_cents"] / 100
            
            # === KEY FILTER 1: Skip cheap contracts ===
            if kalshi_price < MIN_CONTRACT_PRICE:
                results["skipped_cheap"] += 1
                continue
            
            # === KEY FILTER 2: Skip expensive contracts ===
            if kalshi_price > MAX_CONTRACT_PRICE:
                results["skipped_expensive"] += 1
                continue
            
            # Parse contract type
            mtype = contract["market_type"]
            if mtype == "range" and pd.notna(contract.get("temp_low")):
                ctype = "range"
                b1, b2 = contract["temp_low"], contract["temp_high"] + 1
            elif mtype == "below" and pd.notna(contract.get("temp_high")):
                ctype = "below"
                b1, b2 = contract["temp_high"], None
            elif mtype == "above" and pd.notna(contract.get("temp_low")):
                ctype = "above"
                b1, b2 = contract["temp_low"], None
            else:
                continue
            
            # Calculate our probability
            our_prob = model_probability(forecast, ctype, b1, b2, FORECAST_ERROR_STD)
            
            # Get minimum required edge for this price
            min_edge = get_min_edge(kalshi_price)
            
            # === CHECK YES BET ===
            yes_edge = our_prob - kalshi_price
            if yes_edge > min_edge:
                edge_ratio = yes_edge / min_edge
                if edge_ratio > best_edge_ratio:
                    best_edge_ratio = edge_ratio
                    best_edge = yes_edge
                    best_bet = {
                        "contract": contract,
                        "side": "YES",
                        "our_prob": our_prob,
                        "kalshi_price": kalshi_price,
                        "edge": yes_edge,
                        "ctype": ctype,
                        "bucket": get_price_bucket(kalshi_price)
                    }
            
            # === CHECK NO BET ===
            # Don't bet NO on our forecast bin
            is_our_bin = (ctype == "range" and contract.get("temp_low") == our_bin[0])
            
            no_prob = 1 - our_prob
            no_market = 1 - kalshi_price
            no_edge = no_prob - no_market
            no_min_edge = get_min_edge(no_market)
            
            # Skip NO range bets in sweet spot (they lose money)
            no_bucket = get_price_bucket(no_market)
            skip_no_range = (ctype == "range" and no_bucket == "sweet_spot")
            
            if no_edge > no_min_edge and not is_our_bin and not skip_no_range:
                edge_ratio = no_edge / no_min_edge
                if edge_ratio > best_edge_ratio:
                    best_edge_ratio = edge_ratio
                    best_edge = no_edge
                    best_bet = {
                        "contract": contract,
                        "side": "NO",
                        "our_prob": our_prob,
                        "kalshi_price": kalshi_price,
                        "edge": no_edge,
                        "ctype": ctype,
                        "bucket": get_price_bucket(no_market)
                    }
        
        # === EXECUTE BET ===
        profit = 0
        if best_bet:
            results["bets_placed"] += 1
            results["total_wagered"] += FLAT_BET_SIZE
            
            contract = best_bet["contract"]
            side = best_bet["side"]
            
            if side == "YES":
                won = contract["result"] == "yes"
                bet_price = best_bet["kalshi_price"]
            else:
                won = contract["result"] == "no"
                bet_price = 1 - best_bet["kalshi_price"]
            
            if won:
                results["bets_won"] += 1
                profit = FLAT_BET_SIZE * (1/bet_price - 1)
            else:
                profit = -FLAT_BET_SIZE
            
            results["net_profit"] += profit
            
            # Track by bucket
            bucket = best_bet["bucket"]
            results["bets_by_bucket"][bucket]["count"] += 1
            results["bets_by_bucket"][bucket]["won"] += 1 if won else 0
            results["bets_by_bucket"][bucket]["profit"] += profit
            
            # Track by type
            type_key = f"{side.lower()}_{best_bet['ctype']}"
            if type_key in results["bets_by_type"]:
                results["bets_by_type"][type_key]["count"] += 1
                results["bets_by_type"][type_key]["won"] += 1 if won else 0
                results["bets_by_type"][type_key]["profit"] += profit
            
            results["bet_details"].append({
                "date": str(target_date.date()),
                "forecast": forecast,
                "actual": actual_temp,
                "contract": contract["contract_subtitle"],
                "side": side,
                "price": best_bet["kalshi_price"],
                "our_prob": best_bet["our_prob"],
                "edge": best_bet["edge"],
                "bucket": bucket,
                "won": won,
                "profit": profit
            })
        else:
            results["skipped_low_edge"] += 1
        
        results["profits"].append(profit)
    
    # Calculate final metrics
    if results["bets_placed"] > 0:
        results["win_rate"] = 100 * results["bets_won"] / results["bets_placed"]
        results["roi"] = 100 * results["net_profit"] / results["total_wagered"]
    else:
        results["win_rate"] = 0
        results["roi"] = 0
    
    return results


def print_results(results):
    """Print formatted results."""
    city = results["city"]
    
    print(f"\n{'='*70}")
    print(f"RESULTS: {city.upper()}")
    print(f"{'='*70}")
    
    print(f"\n📊 OVERVIEW")
    print(f"   Days analyzed: {results['total_days']}")
    print(f"   Bets placed: {results['bets_placed']}")
    print(f"   Win rate: {results['win_rate']:.1f}%")
    print(f"   ROI: {results['roi']:+.1f}%")
    print(f"   Net profit: ${results['net_profit']:+.2f}")
    
    print(f"\n🚫 FILTERED OUT")
    print(f"   Skipped (too cheap <{MIN_CONTRACT_PRICE*100:.0f}¢): {results['skipped_cheap']}")
    print(f"   Skipped (too expensive >{MAX_CONTRACT_PRICE*100:.0f}¢): {results['skipped_expensive']}")
    print(f"   Skipped (low edge): {results['skipped_low_edge']}")
    
    print(f"\n💰 BY PRICE BUCKET")
    for bucket, stats in results["bets_by_bucket"].items():
        if stats["count"] > 0:
            wr = 100 * stats["won"] / stats["count"]
            roi = 100 * stats["profit"] / (stats["count"] * FLAT_BET_SIZE)
            print(f"   {bucket:<12}: {stats['count']:>4} bets, {wr:>5.1f}% win, ${stats['profit']:>+8.2f} ({roi:>+6.1f}% ROI)")
    
    print(f"\n📈 BY BET TYPE")
    for btype, stats in sorted(results["bets_by_type"].items(), key=lambda x: -x[1]["profit"]):
        if stats["count"] > 0:
            wr = 100 * stats["won"] / stats["count"]
            roi = 100 * stats["profit"] / (stats["count"] * FLAT_BET_SIZE)
            print(f"   {btype:<12}: {stats['count']:>4} bets, {wr:>5.1f}% win, ${stats['profit']:>+8.2f} ({roi:>+6.1f}% ROI)")
    
    # Sample bets
    if results["bet_details"]:
        print(f"\n📝 SAMPLE BETS (last 10)")
        for bet in results["bet_details"][-10:]:
            status = "✅" if bet["won"] else "❌"
            print(f"   {bet['date']}: {bet['side']} {bet['contract']}")
            print(f"      Price: {bet['price']*100:.0f}¢ | Our: {bet['our_prob']*100:.0f}% | Edge: {bet['edge']*100:+.0f}% | {status} ${bet['profit']:+.2f}")


def main():
    print("="*70)
    print("KALSHI WEATHER MODEL BACKTEST - IMPROVED v4")
    print("="*70)
    print(f"\nKey improvements:")
    print(f"  • Min contract price: {MIN_CONTRACT_PRICE*100:.0f}¢ (no cheap traps)")
    print(f"  • Sweet spot: {SWEET_SPOT_LOW*100:.0f}¢-{SWEET_SPOT_HIGH*100:.0f}¢")
    print(f"  • Dynamic edge requirements by price bucket")
    print(f"  • Calibrated Gaussian probability model")
    
    # Load data
    kalshi_file = "kalshi_backtest_data.csv"
    if not os.path.exists(kalshi_file):
        print(f"\n❌ Error: {kalshi_file} not found!")
        return
    
    kalshi_df = pd.read_csv(kalshi_file)
    kalshi_df["date"] = pd.to_datetime(kalshi_df["date"])
    
    print(f"\nLoaded {len(kalshi_df)} records")
    
    # Set seed for reproducibility
    np.random.seed(42)
    
    all_results = []
    
    for city in kalshi_df["city"].unique():
        results = run_backtest(kalshi_df, city, use_simulated_forecasts=True)
        if results:
            all_results.append(results)
            print_results(results)
    
    # Overall summary
    print(f"\n{'='*70}")
    print("OVERALL SUMMARY")
    print(f"{'='*70}\n")
    
    total_wagered = sum(r["total_wagered"] for r in all_results)
    total_profit = sum(r["net_profit"] for r in all_results)
    total_bets = sum(r["bets_placed"] for r in all_results)
    total_wins = sum(r["bets_won"] for r in all_results)
    
    overall_roi = 100 * total_profit / total_wagered if total_wagered > 0 else 0
    overall_wr = 100 * total_wins / total_bets if total_bets > 0 else 0
    
    print(f"{'City':<25} {'Bets':<8} {'Win%':<10} {'ROI':<12} {'Profit':<12}")
    print("-" * 70)
    for r in all_results:
        print(f"{r['city']:<25} {r['bets_placed']:<8} {r['win_rate']:>5.1f}%    {r['roi']:>+7.1f}%    ${r['net_profit']:>+9.2f}")
    print("-" * 70)
    print(f"{'TOTAL':<25} {total_bets:<8} {overall_wr:>5.1f}%    {overall_roi:>+7.1f}%    ${total_profit:>+9.2f}")
    
    # Aggregate by bucket across all cities
    print(f"\n📊 AGGREGATE BY PRICE BUCKET (All Cities)")
    print("-" * 50)
    
    agg_buckets = {"sweet_spot": {"count": 0, "won": 0, "profit": 0},
                   "high_price": {"count": 0, "won": 0, "profit": 0}}
    
    for r in all_results:
        for bucket, stats in r["bets_by_bucket"].items():
            agg_buckets[bucket]["count"] += stats["count"]
            agg_buckets[bucket]["won"] += stats["won"]
            agg_buckets[bucket]["profit"] += stats["profit"]
    
    for bucket, stats in agg_buckets.items():
        if stats["count"] > 0:
            wr = 100 * stats["won"] / stats["count"]
            roi = 100 * stats["profit"] / (stats["count"] * FLAT_BET_SIZE)
            print(f"  {bucket:<12}: {stats['count']:>4} bets, {wr:>5.1f}% win, ${stats['profit']:>+8.2f} ({roi:>+6.1f}% ROI)")
    
    print(f"\n{'='*70}")
    print("INTERPRETATION")
    print(f"{'='*70}")
    
    if overall_roi > 0:
        print(f"\n✅ PROFITABLE: {overall_roi:+.1f}% ROI")
        print("   The improved filters appear to be working.")
    else:
        print(f"\n❌ NOT PROFITABLE: {overall_roi:.1f}% ROI")
        print("   But this is using simulated forecasts.")
    
    print("""
NOTE: This backtest uses simulated forecasts (actual temp + noise).
Real performance depends on actual forecast quality.

For live trading:
  1. Use real HRRR/Open-Meteo forecasts (your ensemble model)
  2. Apply the same price filters (10-90¢ range)
  3. Focus on the sweet spot (15-50¢)
  4. Use dynamic edge requirements
""")


if __name__ == "__main__":
    main()