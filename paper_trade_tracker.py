#!/usr/bin/env python3
"""
PAPER TRADING TRACKER
=====================
Daily paper trading system that:
1. Fetches today's NBA games from Kalshi
2. Gets model predictions
3. Logs recommended bets to CSV
4. Tracks performance over time
5. Shows running statistics

Usage:
    python paper_trade_tracker.py
    
    Run this once per day (ideally in the morning before games start)
    
Output files:
    - paper_trades.csv: All recommended bets with outcomes
    - paper_trades_summary.json: Current performance stats
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json
from pathlib import Path
import sys

# Import from betting model
sys.path.append(str(Path(__file__).parent))
from sports_betting_model import (
    EloRatingSystem, 
    KalshiClient,
    parse_game_from_market,
    analyze_betting_opportunity,
    get_eastern_now,
    MODEL_SAVE_PATH,
    STARTING_BANKROLL
)

# ============ CONFIGURATION ============

PAPER_TRADES_CSV = "paper_trades.csv"
SUMMARY_JSON = "paper_trades_summary.json"
STARTING_PAPER_BANKROLL = 1000.0

# CSV Columns
CSV_COLUMNS = [
    'date',
    'trade_id',
    'game_date',
    'game_time',
    'home_team',
    'away_team',
    'team_betting_on',
    'is_home',
    'model_prob',
    'kalshi_price',
    'edge',
    'edge_pct',
    'recommended_bet',
    'kalshi_ticker',
    'status',
    'actual_result',
    'profit_loss',
    'bankroll_after'
]


# ============ PAPER TRADING SYSTEM ============

class PaperTradingTracker:
    """
    Track paper trades over time.
    """
    
    def __init__(self, csv_path=PAPER_TRADES_CSV, initial_bankroll=STARTING_PAPER_BANKROLL):
        self.csv_path = csv_path
        self.initial_bankroll = initial_bankroll
        self.current_bankroll = initial_bankroll
        
        # Load existing trades if file exists
        if Path(csv_path).exists():
            self.trades_df = pd.read_csv(csv_path)
            self.trades_df['date'] = pd.to_datetime(self.trades_df['date'], format='mixed')
            self.trades_df['game_date'] = pd.to_datetime(self.trades_df['game_date'], format='mixed')
            
            # Calculate current bankroll from completed trades
            completed = self.trades_df[self.trades_df['status'] == 'completed']
            if len(completed) > 0:
                self.current_bankroll = completed.iloc[-1]['bankroll_after']
        else:
            # Create new DataFrame
            self.trades_df = pd.DataFrame(columns=CSV_COLUMNS)
        
        print(f"📊 Paper Trading Tracker Initialized")
        print(f"   CSV: {csv_path}")
        print(f"   Current Bankroll: ${self.current_bankroll:,.2f}")
        print(f"   Total Trades Logged: {len(self.trades_df)}")
    
    def get_next_trade_id(self):
        """Generate next trade ID."""
        if len(self.trades_df) == 0:
            return "PT001"
        else:
            last_id = self.trades_df['trade_id'].iloc[-1]
            num = int(last_id[2:]) + 1
            return f"PT{num:03d}"
    
    def log_trade(self, bet_info):
        """
        Log a paper trade recommendation.
        
        Args:
            bet_info: Dictionary with bet details from analyze_betting_opportunity
        """
        # Check if this trade already exists (same ticker and date)
        ticker = bet_info.get('market_ticker', 'N/A')
        today = datetime.now().date()
        
        existing = self.trades_df[
            (self.trades_df['kalshi_ticker'] == ticker) &
            (pd.to_datetime(self.trades_df['date']).dt.date == today)
        ]
        
        if len(existing) > 0:
            print(f"⚠️  Trade already logged today: {ticker}")
            return existing.iloc[0]['trade_id']
        
        trade_id = self.get_next_trade_id()
        
        # Parse game date/time from market info if available
        game_date = bet_info.get('game_date', datetime.now())
        if isinstance(game_date, str):
            game_date = pd.to_datetime(game_date)
        
        trade_record = {
            'date': datetime.now(),
            'trade_id': trade_id,
            'game_date': game_date.date() if hasattr(game_date, 'date') else game_date,
            'game_time': game_date.strftime('%H:%M') if hasattr(game_date, 'strftime') else 'TBD',
            'home_team': bet_info['team_a'] if bet_info.get('team_a_home') else bet_info['team_b'],
            'away_team': bet_info['team_b'] if bet_info.get('team_a_home') else bet_info['team_a'],
            'team_betting_on': bet_info['team_a'],
            'is_home': bet_info.get('team_a_home', True),
            'model_prob': bet_info['our_prob'],
            'kalshi_price': bet_info['bet_price'],
            'edge': bet_info['edge'],
            'edge_pct': bet_info['edge_pct'],
            'recommended_bet': bet_info['bet_size'],
            'kalshi_ticker': bet_info.get('market_ticker', 'N/A'),
            'status': 'pending',
            'actual_result': None,
            'profit_loss': None,
            'bankroll_after': self.current_bankroll
        }
        
        # Add to DataFrame
        self.trades_df = pd.concat([self.trades_df, pd.DataFrame([trade_record])], ignore_index=True)
        
        # Save to CSV
        self.trades_df.to_csv(self.csv_path, index=False)
        
        return trade_id
    
    def update_trade_result(self, trade_id, won, actual_price=None):
        """
        Update a trade with actual result.
        
        Args:
            trade_id: Trade ID to update
            won: Boolean - did the bet win?
            actual_price: Actual price paid (if different from recommendation)
        """
        idx = self.trades_df[self.trades_df['trade_id'] == trade_id].index
        
        if len(idx) == 0:
            print(f"❌ Trade {trade_id} not found")
            return False
        
        idx = idx[0]
        
        # Get trade details
        bet_size = self.trades_df.loc[idx, 'recommended_bet']
        price = actual_price if actual_price else self.trades_df.loc[idx, 'kalshi_price']
        
        # Calculate profit/loss
        if won:
            profit = bet_size * (1 / price - 1)
        else:
            profit = -bet_size
        
        # Update bankroll
        new_bankroll = self.current_bankroll + profit
        
        # Update record
        self.trades_df.loc[idx, 'actual_result'] = 'won' if won else 'lost'
        self.trades_df.loc[idx, 'profit_loss'] = profit
        self.trades_df.loc[idx, 'bankroll_after'] = new_bankroll
        self.trades_df.loc[idx, 'status'] = 'completed'
        
        self.current_bankroll = new_bankroll
        
        # Save
        self.trades_df.to_csv(self.csv_path, index=False)
        
        print(f"✅ Updated {trade_id}: {'WON' if won else 'LOST'} | P/L: ${profit:+.2f} | Bankroll: ${new_bankroll:,.2f}")
        
        return True
    
    def get_pending_trades(self):
        """Get all pending trades that need results."""
        pending = self.trades_df[self.trades_df['status'] == 'pending'].copy()
        return pending
    
    def get_statistics(self):
        """Calculate current performance statistics."""
        completed = self.trades_df[self.trades_df['status'] == 'completed']
        
        if len(completed) == 0:
            return {
                'total_trades': 0,
                'completed_trades': 0,
                'pending_trades': len(self.trades_df),
                'wins': 0,
                'losses': 0,
                'win_rate': 0,
                'total_profit': 0,
                'total_wagered': 0,
                'roi': 0,
                'current_bankroll': self.current_bankroll,
                'initial_bankroll': self.initial_bankroll,
                'return_pct': 0
            }
        
        wins = (completed['actual_result'] == 'won').sum()
        total_completed = len(completed)
        win_rate = (wins / total_completed * 100) if total_completed > 0 else 0
        
        total_profit = completed['profit_loss'].sum()
        total_wagered = completed['recommended_bet'].sum()
        roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0
        
        return {
            'total_trades': len(self.trades_df),
            'completed_trades': total_completed,
            'pending_trades': len(self.trades_df[self.trades_df['status'] == 'pending']),
            'wins': wins,
            'losses': total_completed - wins,
            'win_rate': win_rate,
            'total_profit': total_profit,
            'total_wagered': total_wagered,
            'roi': roi,
            'current_bankroll': self.current_bankroll,
            'initial_bankroll': self.initial_bankroll,
            'return_pct': ((self.current_bankroll - self.initial_bankroll) / self.initial_bankroll * 100)
        }
    
    def print_summary(self):
        """Print current performance summary."""
        stats = self.get_statistics()
        
        print(f"\n{'='*70}")
        print("📊 PAPER TRADING PERFORMANCE SUMMARY")
        print(f"{'='*70}")
        print(f"Initial Bankroll:     ${stats['initial_bankroll']:>12,.2f}")
        print(f"Current Bankroll:     ${stats['current_bankroll']:>12,.2f}")
        print(f"Total Return:         ${stats['total_profit']:>12,.2f}")
        print(f"Return %:              {stats['return_pct']:>12.2f}%")
        print(f"{'-'*70}")
        print(f"Total Trades:          {stats['total_trades']:>12,}")
        print(f"Completed:             {stats['completed_trades']:>12,}")
        print(f"Pending:               {stats['pending_trades']:>12,}")
        print(f"{'-'*70}")
        
        if stats['completed_trades'] > 0:
            print(f"Wins:                  {stats['wins']:>12,}")
            print(f"Losses:                {stats['losses']:>12,}")
            print(f"Win Rate:              {stats['win_rate']:>12.1f}%")
            print(f"ROI:                   {stats['roi']:>12.2f}%")
        
        print(f"{'='*70}\n")
        
        # Save summary to JSON
        with open(SUMMARY_JSON, 'w') as f:
            json.dump(stats, f, indent=2, default=str)


# ============ DAILY RUNNER ============

def run_daily_paper_trading(demo_mode=False):
    """
    Main function to run daily paper trading.
    
    1. Fetch today's markets from Kalshi
    2. Get model recommendations
    3. Log trades to CSV
    4. Show summary
    """
    
    print(f"\n{'='*70}")
    print("🎯 DAILY PAPER TRADING RUN")
    print(f"{'='*70}")
    print(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Mode: {'DEMO' if demo_mode else 'PRODUCTION'}")
    print(f"{'='*70}\n")
    
    # Load Elo system
    if Path(MODEL_SAVE_PATH).exists():
        print("📈 Loading Elo ratings...")
        elo_system = EloRatingSystem.load(MODEL_SAVE_PATH)
        print(f"   Loaded ratings for {len(elo_system.ratings)} teams")
    else:
        print("⚠️  No Elo ratings found, starting fresh")
        print("   Recommendation: Train the model first with historical data")
        elo_system = EloRatingSystem()
    
    # Initialize tracker
    tracker = PaperTradingTracker()
    
    # Fetch Kalshi markets
    print(f"\n🔍 Fetching NBA markets from Kalshi...")
    client = KalshiClient(use_demo=demo_mode)
    markets = client.get_sports_markets(league="NBA")
    
    if not markets:
        print("❌ No markets found on Kalshi")
        print("   This is normal if:")
        print("   - No NBA games today")
        print("   - Markets not posted yet (check back later)")
        print("   - Using demo mode (demo has limited markets)")
        return
    
    print(f"   Found {len(markets)} potential markets\n")
    
    # Get today's date
    eastern_now = get_eastern_now()
    today = eastern_now.date()
    
    # Analyze each market
    recommendations = []
    
    for market in markets:
        game_info = parse_game_from_market(market)
        
        if not game_info:
            continue
        
        # Filter to today's games only
        game_date = game_info.get('game_date')
        if game_date:
            game_day = game_date.date()
            if game_day != today:
                continue
        
        # Get betting recommendation
        bet = analyze_betting_opportunity(elo_system, game_info, tracker.current_bankroll)
        
        if bet:
            recommendations.append(bet)
    
    # Sort by edge (highest first)
    recommendations.sort(key=lambda x: x['edge'], reverse=True)
    
    # Log recommendations
    if recommendations:
        print(f"{'='*70}")
        print(f"🎯 FOUND {len(recommendations)} BETTING OPPORTUNITIES TODAY")
        print(f"{'='*70}\n")
        
        for i, bet in enumerate(recommendations, 1):
            trade_id = tracker.log_trade(bet)
            
            home_away = "🏠 HOME" if bet['team_a_home'] else "✈️  AWAY"
            
            print(f"#{i}. Trade ID: {trade_id}")
            print(f"   {bet['team_a']} vs {bet['team_b']} ({home_away})")
            print(f"   Model: {bet['our_prob']*100:.1f}% | Kalshi: {bet['bet_price']*100:.0f}¢ | Edge: {bet['edge_pct']:+.1f}%")
            print(f"   Recommended Bet: ${bet['bet_size']:.2f}")
            print(f"   Potential Profit: ${bet['potential_profit']:.2f}")
            print(f"   Ticker: {bet.get('market_ticker', 'N/A')}")
            print()
        
        print(f"{'='*70}")
        print(f"💾 Logged {len(recommendations)} trades to {PAPER_TRADES_CSV}")
        print(f"{'='*70}\n")
        
    else:
        print(f"{'='*70}")
        print(f"❌ NO BETTING OPPORTUNITIES TODAY")
        print(f"{'='*70}")
        print(f"   Reason: All games are priced too efficiently")
        print(f"   (No edges meet the minimum threshold)")
        print()
    
    # Show current performance
    tracker.print_summary()
    
    # Show pending trades that need results
    pending = tracker.get_pending_trades()
    
    if len(pending) > 0:
        print(f"{'='*70}")
        print(f"⏳ PENDING TRADES NEEDING RESULTS: {len(pending)}")
        print(f"{'='*70}\n")
        
        for _, trade in pending.head(10).iterrows():
            print(f"Trade {trade['trade_id']}: {trade['team_betting_on']} vs opponent")
            print(f"   Game Date: {trade['game_date']}")
            print(f"   Bet: ${trade['recommended_bet']:.2f} @ {trade['kalshi_price']*100:.0f}¢")
            print(f"   To update: python paper_trade_tracker.py --update {trade['trade_id']} --result [won/lost]")
            print()
        
        if len(pending) > 10:
            print(f"   ... and {len(pending) - 10} more pending trades\n")


def update_trade_result(trade_id, result):
    """
    Update a trade with its result.
    
    Args:
        trade_id: Trade ID (e.g., "PT001")
        result: "won" or "lost"
    """
    tracker = PaperTradingTracker()
    
    if result.lower() not in ['won', 'lost']:
        print(f"❌ Invalid result: {result}")
        print(f"   Must be 'won' or 'lost'")
        return
    
    won = result.lower() == 'won'
    success = tracker.update_trade_result(trade_id, won)
    
    if success:
        tracker.print_summary()


# ============ COMMAND LINE INTERFACE ============

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='Paper Trading Tracker')
    parser.add_argument('--demo', action='store_true',
                       help='Use Kalshi demo API instead of production')
    parser.add_argument('--update', type=str,
                       help='Update a trade result (provide trade ID)')
    parser.add_argument('--result', type=str, choices=['won', 'lost'],
                       help='Trade result (won or lost)')
    parser.add_argument('--summary', action='store_true',
                       help='Show summary only (no new trades)')
    parser.add_argument('--pending', action='store_true',
                       help='Show pending trades only')
    
    args = parser.parse_args()
    
    # Update existing trade
    if args.update:
        if not args.result:
            print("❌ Must provide --result [won/lost] when updating a trade")
            return
        update_trade_result(args.update, args.result)
        return
    
    # Show summary only
    if args.summary:
        tracker = PaperTradingTracker()
        tracker.print_summary()
        return
    
    # Show pending only
    if args.pending:
        tracker = PaperTradingTracker()
        pending = tracker.get_pending_trades()
        
        if len(pending) == 0:
            print("✅ No pending trades")
        else:
            print(f"\n{'='*70}")
            print(f"⏳ PENDING TRADES: {len(pending)}")
            print(f"{'='*70}\n")
            
            for _, trade in pending.iterrows():
                print(f"Trade {trade['trade_id']}: {trade['team_betting_on']}")
                print(f"   Game: {trade['game_date']} at {trade['game_time']}")
                print(f"   Bet: ${trade['recommended_bet']:.2f} @ {trade['kalshi_price']*100:.0f}¢")
                print(f"   Edge: {trade['edge_pct']:+.1f}%")
                print()
        return
    
    # Run daily paper trading
    run_daily_paper_trading(demo_mode=args.demo)


if __name__ == "__main__":
    main()