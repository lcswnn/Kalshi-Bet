import requests
from datetime import datetime, timedelta
import json
import os

# ============ CONFIGURATION ============
cities = {
    "chicago": {
        "name": "Chicago",
        "kalshi_series": "KXHIGHCHI"
    },
    "nyc": {
        "name": "New York City",
        "kalshi_series": "KXHIGHNY"
    },
    "la": {
        "name": "Los Angeles",
        "kalshi_series": "KXHIGHLA"
    }
}

# File to store our bets
BETS_LOG_FILE = "bets_log.json"

# ============ FUNCTIONS ============
def load_bets_log():
    """Load the bets log from file."""
    if os.path.exists(BETS_LOG_FILE):
        with open(BETS_LOG_FILE, 'r') as f:
            return json.load(f)
    return {"bets": []}


def save_bets_log(log):
    """Save the bets log to file."""
    with open(BETS_LOG_FILE, 'w') as f:
        json.dump(log, f, indent=2)


def log_bet(city, date, contract, bet_side, price, amount, model_prob, edge):
    """Log a bet we're placing."""
    log = load_bets_log()
    
    bet = {
        "city": city,
        "date": date,
        "contract": contract,
        "bet_side": bet_side,
        "price": price,
        "amount": amount,
        "model_prob": model_prob,
        "edge": edge,
        "logged_at": datetime.now().isoformat(),
        "result": None,
        "profit": None
    }
    
    log["bets"].append(bet)
    save_bets_log(log)
    print(f"✅ Logged bet: {city} {contract} {bet_side} at {price*100:.0f}¢")


def get_settled_markets(series_ticker, target_date):
    """Get settled markets for a specific date."""
    BASE_URL = "https://api.elections.kalshi.com/trade-api/v2"
    
    # Format date for Kalshi ticker (e.g., "25DEC03")
    target_kalshi = target_date.strftime("%y%b%d").upper()
    
    # Get markets (including settled ones)
    params = {
        "series_ticker": series_ticker,
        "status": "settled",
        "limit": 100
    }
    
    response = requests.get(f"{BASE_URL}/markets", params=params)
    
    if response.status_code != 200:
        print(f"Error fetching markets: {response.status_code}")
        return []
    
    markets = response.json().get("markets", [])
    
    # Filter for target date
    target_markets = [m for m in markets if target_kalshi in m["ticker"]]
    
    return target_markets


def find_winning_contract(markets):
    """Find which contract won (result = 'yes')."""
    for m in markets:
        if m.get("result") == "yes":
            return m
    return None


def check_yesterday_results():
    """Check results for yesterday's markets."""
    yesterday = datetime.now().date() - timedelta(days=1)
    yesterday_str = yesterday.strftime("%Y-%m-%d")
    
    print("=" * 60)
    print("YESTERDAY'S RESULTS")
    print(f"Date: {yesterday_str}")
    print("=" * 60)
    
    results = []
    
    for city_key, city in cities.items():
        print(f"\n{'='*40}")
        print(f"{city['name'].upper()}")
        print(f"{'='*40}")
        
        markets = get_settled_markets(city["kalshi_series"], yesterday)
        
        if not markets:
            print(f"No settled markets found for {yesterday_str}")
            continue
        
        # Find winning contract
        winner = find_winning_contract(markets)
        
        if winner:
            print(f"🏆 WINNING CONTRACT: {winner['subtitle']}")
            print(f"   Settled at: {winner.get('last_price', 'N/A')}¢")
            
            # Extract actual temperature from winning contract
            subtitle = winner['subtitle']
            if "to" in subtitle:
                # e.g., "18° to 19°" -> actual was 18-19
                print(f"   Actual high: {subtitle}")
            elif "or below" in subtitle:
                print(f"   Actual high: {subtitle}")
            elif "or above" in subtitle:
                print(f"   Actual high: {subtitle}")
            
            results.append({
                "city": city["name"],
                "date": yesterday_str,
                "winning_contract": winner["subtitle"],
                "ticker": winner["ticker"]
            })
        else:
            print("No winning contract found (market may not be settled yet)")
        
        # Show all contracts and their results
        print(f"\nAll contracts:")
        for m in sorted(markets, key=lambda x: x.get("last_price", 0), reverse=True):
            result_marker = "✅" if m.get("result") == "yes" else "❌"
            print(f"   {result_marker} {m['subtitle']:<20} Result: {m.get('result', 'N/A')}")
    
    return results


def check_our_bets_performance():
    """Check how our logged bets performed."""
    log = load_bets_log()
    
    if not log["bets"]:
        print("\nNo bets logged yet.")
        return
    
    yesterday = datetime.now().date() - timedelta(days=1)
    yesterday_str = yesterday.strftime("%Y-%m-%d")
    
    print(f"\n{'='*60}")
    print("OUR BETS PERFORMANCE")
    print(f"{'='*60}")
    
    # Find bets for yesterday
    yesterday_bets = [b for b in log["bets"] if b["date"] == yesterday_str]
    
    if not yesterday_bets:
        print(f"No bets logged for {yesterday_str}")
        return
    
    total_wagered = 0
    total_profit = 0
    wins = 0
    losses = 0
    
    for bet in yesterday_bets:
        city_key = None
        for key, city in cities.items():
            if city["name"] == bet["city"]:
                city_key = key
                break
        
        if not city_key:
            continue
        
        # Get settled markets for this city
        markets = get_settled_markets(cities[city_key]["kalshi_series"], yesterday)
        winner = find_winning_contract(markets)
        
        if not winner:
            print(f"⏳ {bet['city']}: {bet['contract']} - Not settled yet")
            continue
        
        # Check if our bet won
        winning_contract = winner["subtitle"]
        bet_contract = bet["contract"]
        bet_side = bet["bet_side"]
        
        # Determine if we won
        if bet_side == "YES":
            won = (bet_contract == winning_contract)
        else:  # NO bet
            won = (bet_contract != winning_contract)
        
        # Calculate profit/loss
        if won:
            if bet_side == "YES":
                profit = bet["amount"] * ((1 / bet["price"]) - 1)
            else:
                profit = bet["amount"] * (bet["price"] / (1 - bet["price"]))
            wins += 1
        else:
            profit = -bet["amount"]
            losses += 1
        
        total_wagered += bet["amount"]
        total_profit += profit
        
        result_marker = "✅ WON" if won else "❌ LOST"
        print(f"\n{bet['city']}: {bet['contract']} {bet['bet_side']}")
        print(f"   {result_marker}")
        print(f"   Wagered: ${bet['amount']:.2f}")
        print(f"   Profit: ${profit:+.2f}")
        print(f"   Actual winner: {winning_contract}")
    
    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total bets: {len(yesterday_bets)}")
    print(f"Wins: {wins} | Losses: {losses}")
    print(f"Win rate: {wins/(wins+losses)*100:.1f}%" if (wins+losses) > 0 else "N/A")
    print(f"Total wagered: ${total_wagered:.2f}")
    print(f"Total profit: ${total_profit:+.2f}")
    print(f"ROI: {(total_profit/total_wagered)*100:+.1f}%" if total_wagered > 0 else "N/A")


def check_historical_performance():
    """Check overall historical performance of all logged bets."""
    log = load_bets_log()
    
    if not log["bets"]:
        print("\nNo bets logged yet.")
        return
    
    print(f"\n{'='*60}")
    print("HISTORICAL PERFORMANCE (ALL TIME)")
    print(f"{'='*60}")
    
    total_wagered = 0
    total_profit = 0
    wins = 0
    losses = 0
    pending = 0
    
    by_city = {}
    
    for bet in log["bets"]:
        bet_date = datetime.strptime(bet["date"], "%Y-%m-%d").date()
        
        # Skip future bets
        if bet_date >= datetime.now().date():
            pending += 1
            continue
        
        city_key = None
        for key, city in cities.items():
            if city["name"] == bet["city"]:
                city_key = key
                break
        
        if not city_key:
            continue
        
        # Get settled markets
        markets = get_settled_markets(cities[city_key]["kalshi_series"], bet_date)
        winner = find_winning_contract(markets)
        
        if not winner:
            pending += 1
            continue
        
        # Check if we won
        winning_contract = winner["subtitle"]
        bet_side = bet["bet_side"]
        
        if bet_side == "YES":
            won = (bet["contract"] == winning_contract)
        else:
            won = (bet["contract"] != winning_contract)
        
        # Calculate profit
        if won:
            if bet_side == "YES":
                profit = bet["amount"] * ((1 / bet["price"]) - 1)
            else:
                profit = bet["amount"] * (bet["price"] / (1 - bet["price"]))
            wins += 1
        else:
            profit = -bet["amount"]
            losses += 1
        
        total_wagered += bet["amount"]
        total_profit += profit
        
        # Track by city
        if bet["city"] not in by_city:
            by_city[bet["city"]] = {"wins": 0, "losses": 0, "profit": 0, "wagered": 0}
        
        by_city[bet["city"]]["wagered"] += bet["amount"]
        by_city[bet["city"]]["profit"] += profit
        if won:
            by_city[bet["city"]]["wins"] += 1
        else:
            by_city[bet["city"]]["losses"] += 1
    
    # Overall summary
    print(f"\nOverall:")
    print(f"   Total bets: {wins + losses} settled, {pending} pending")
    print(f"   Wins: {wins} | Losses: {losses}")
    if (wins + losses) > 0:
        print(f"   Win rate: {wins/(wins+losses)*100:.1f}%")
        print(f"   Total wagered: ${total_wagered:.2f}")
        print(f"   Total profit: ${total_profit:+.2f}")
        print(f"   ROI: {(total_profit/total_wagered)*100:+.1f}%")
    
    # By city
    print(f"\nBy City:")
    for city, stats in by_city.items():
        total_bets = stats["wins"] + stats["losses"]
        if total_bets > 0:
            win_rate = stats["wins"] / total_bets * 100
            roi = (stats["profit"] / stats["wagered"]) * 100 if stats["wagered"] > 0 else 0
            print(f"   {city}: {stats['wins']}W/{stats['losses']}L ({win_rate:.0f}%) | ${stats['profit']:+.2f} ({roi:+.1f}% ROI)")


def manual_log_bet():
    """Manually log a bet from user input."""
    print("\n" + "=" * 60)
    print("LOG A NEW BET")
    print("=" * 60)
    
    print("\nCities:")
    print("  1. Chicago")
    print("  2. New York City")
    print("  3. Los Angeles")
    
    city_choice = input("Select city (1/2/3): ").strip()
    city_map = {"1": "Chicago", "2": "New York City", "3": "Los Angeles"}
    city = city_map.get(city_choice, "Chicago")
    
    date = input("Date (YYYY-MM-DD, or press Enter for tomorrow): ").strip()
    if not date:
        date = (datetime.now().date() + timedelta(days=1)).strftime("%Y-%m-%d")
    
    contract = input("Contract (e.g., '18° to 19°'): ").strip()
    bet_side = input("Bet side (YES/NO): ").strip().upper()
    price = float(input("Price (e.g., 0.41 for 41¢): ").strip())
    amount = float(input("Amount wagered ($): ").strip())
    model_prob = float(input("Model probability (e.g., 0.58 for 58%): ").strip())
    edge = float(input("Edge (e.g., 0.17 for +17%): ").strip())
    
    log_bet(city, date, contract, bet_side, price, amount, model_prob, edge)


# ============ MAIN ============
if __name__ == "__main__":
    print("=" * 60)
    print("KALSHI WEATHER BETTING - PERFORMANCE TRACKER")
    print("=" * 60)
    print("\nOptions:")
    print("  1. Check yesterday's results")
    print("  2. Check our bets performance")
    print("  3. Check historical performance (all time)")
    print("  4. Log a new bet")
    print("  5. Run all checks")
    
    choice = input("\nSelect option (1-5): ").strip()
    
    if choice == "1":
        check_yesterday_results()
    elif choice == "2":
        check_our_bets_performance()
    elif choice == "3":
        check_historical_performance()
    elif choice == "4":
        manual_log_bet()
    elif choice == "5":
        check_yesterday_results()
        check_our_bets_performance()
        check_historical_performance()
    else:
        print("Invalid choice, running all checks...")
        check_yesterday_results()
        check_our_bets_performance()
        check_historical_performance()