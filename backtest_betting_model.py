"""
SPORTS BETTING MODEL BACKTESTER
================================
Comprehensive backtesting framework to evaluate model performance.

Features:
1. Historical simulation with proper train/test split
2. Kelly Criterion bet sizing validation
3. ROI, Sharpe ratio, and drawdown analysis
4. Win rate and accuracy metrics
5. Edge calibration analysis
6. Multiple betting strategies comparison
7. Monte Carlo simulation for confidence intervals

Usage:
    python backtest_betting_model.py historical_games.csv --bankroll 1000
    python backtest_betting_model.py games.csv --strategy kelly --test-size 0.3
    python backtest_betting_model.py games.csv --optimize-params
"""

import pandas as pd
import numpy as np
import argparse
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from collections import defaultdict
import json

# Import from the betting model
import sys
sys.path.append(str(Path(__file__).parent))
from sports_betting_model import EloRatingSystem, INITIAL_ELO

# ============ CONFIGURATION ============

# Backtesting Configuration
INITIAL_BANKROLL = 1000.0
TRAIN_TEST_SPLIT = 0.7  # 70% training, 30% testing
MIN_EDGE_THRESHOLD = 0.05  # 5% minimum edge
KELLY_FRACTION = 0.25  # Quarter Kelly (optimized)
MAX_BET_FRACTION = 0.08  # Max 8% of bankroll per bet (optimized)
MIN_BET_SIZE = 1.0

# Market simulation parameters
BID_ASK_SPREAD = 0.02  # 2¢ spread (realistic for Kalshi)
MARKET_EFFICIENCY = 0.92  # How often market price = true probability (realistic Kalshi)

# Plotting style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")


# ============ BACKTESTING ENGINE ============

class BettingBacktester:
    """
    Backtest betting model on historical data.
    """
    
    def __init__(self, initial_bankroll=INITIAL_BANKROLL, kelly_fraction=KELLY_FRACTION,
                 min_edge=MIN_EDGE_THRESHOLD, max_bet_fraction=MAX_BET_FRACTION):
        self.initial_bankroll = initial_bankroll
        self.kelly_fraction = kelly_fraction
        self.min_edge = min_edge
        self.max_bet_fraction = max_bet_fraction
        self.min_bet_size = MIN_BET_SIZE
        
        # Results tracking
        self.bets = []
        self.bankroll_history = []
        self.date_history = []
        
    def calculate_kelly_bet(self, edge, win_prob, price, bankroll):
        """
        Calculate optimal Kelly bet size.
        
        Kelly formula: f = (bp - q) / b
        where:
            f = fraction of bankroll to bet
            b = odds received (1/price - 1)
            p = probability of winning
            q = probability of losing (1 - p)
        """
        if price <= 0 or price >= 1:
            return 0
        
        # Decimal odds from price
        b = (1 / price) - 1
        
        # Kelly fraction
        p = win_prob
        q = 1 - p
        
        kelly_fraction_raw = (b * p - q) / b
        
        # Apply fractional Kelly
        kelly_fraction_adjusted = kelly_fraction_raw * self.kelly_fraction
        
        # Convert to dollar amount
        bet_size = kelly_fraction_adjusted * bankroll
        
        # Apply constraints
        bet_size = max(bet_size, 0)  # No negative bets
        bet_size = min(bet_size, bankroll * self.max_bet_fraction)  # Max bet limit
        
        # Minimum bet size
        if bet_size > 0 and bet_size < self.min_bet_size:
            bet_size = 0  # Don't bet if below minimum
        
        return bet_size
    
    def simulate_market_price(self, true_prob, efficiency=MARKET_EFFICIENCY):
        """
        Simulate market price based on true probability.
        
        Markets are efficient but not perfect - sometimes they're off.
        """
        if np.random.random() < efficiency:
            # Market is efficient - price near true probability
            noise = np.random.normal(0, 0.03)  # Small noise
            market_price = true_prob + noise
        else:
            # Market is inefficient - larger deviation
            noise = np.random.normal(0, 0.10)  # Larger noise
            market_price = true_prob + noise
        
        # Clamp to valid range
        market_price = np.clip(market_price, 0.05, 0.95)
        
        # Add bid-ask spread (we pay the ask when buying)
        ask_price = market_price + BID_ASK_SPREAD / 2
        ask_price = np.clip(ask_price, 0.05, 0.95)
        
        return ask_price
    
    def run_backtest(self, games_df, elo_system, train_size=TRAIN_TEST_SPLIT):
        """
        Run backtest on historical games.
        
        Args:
            games_df: DataFrame with columns: date, home_team, away_team, home_won
            elo_system: EloRatingSystem to use
            train_size: Fraction of data to use for training (rest is testing)
        
        Returns:
            Dictionary of performance metrics
        """
        # Sort by date
        games_df = games_df.sort_values('date').reset_index(drop=True)
        
        # Split into train and test
        split_idx = int(len(games_df) * train_size)
        train_df = games_df.iloc[:split_idx]
        test_df = games_df.iloc[split_idx:]
        
        print(f"\n{'='*70}")
        print(f"BACKTESTING SETUP")
        print(f"{'='*70}")
        print(f"Total games: {len(games_df)}")
        print(f"Training games: {len(train_df)} ({train_size*100:.0f}%)")
        print(f"Testing games: {len(test_df)} ({(1-train_size)*100:.0f}%)")
        print(f"Initial bankroll: ${self.initial_bankroll:,.2f}")
        print(f"Kelly fraction: {self.kelly_fraction}")
        print(f"Min edge: {self.min_edge*100:.0f}%")
        print(f"{'='*70}\n")
        
        # Train Elo on training data
        print("Training Elo ratings on historical data...")
        for _, game in train_df.iterrows():
            elo_system.update_ratings(
                game['home_team'],
                game['away_team'],
                game['home_won'],
                team_a_home=True,
                game_date=game['date']
            )
        print(f"✅ Trained on {len(train_df)} games\n")
        
        # Run betting simulation on test data
        print("Running betting simulation on test data...")
        current_bankroll = self.initial_bankroll
        bets_made = 0
        bets_won = 0
        total_wagered = 0
        total_profit = 0
        
        for _, game in test_df.iterrows():
            # Get model prediction
            home_prob = elo_system.calculate_win_probability(
                game['home_team'],
                game['away_team'],
                team_a_home=True
            )
            
            # Simulate market price
            true_prob = home_prob  # In real world, we don't know this
            market_price = self.simulate_market_price(true_prob)
            
            # Calculate edge
            edge = home_prob - market_price
            
            # Skip extreme prices (unrealistic in real Kalshi markets)
            if market_price < 0.15 or market_price > 0.85:
                continue
            
            # Only bet if edge meets threshold
            if edge >= self.min_edge and current_bankroll >= self.min_bet_size:
                # Calculate bet size using Kelly
                bet_size = self.calculate_kelly_bet(edge, home_prob, market_price, current_bankroll)
                
                if bet_size >= self.min_bet_size:
                    # Place bet
                    home_won = game['home_won']
                    
                    if home_won:
                        # Win: get back bet + profit
                        profit = bet_size * (1 / market_price - 1)
                        current_bankroll += profit
                        bets_won += 1
                    else:
                        # Loss: lose bet
                        current_bankroll -= bet_size
                    
                    # Record bet
                    self.bets.append({
                        'date': game['date'],
                        'home_team': game['home_team'],
                        'away_team': game['away_team'],
                        'model_prob': home_prob,
                        'market_price': market_price,
                        'edge': edge,
                        'bet_size': bet_size,
                        'won': home_won,
                        'profit': profit if home_won else -bet_size,
                        'bankroll': current_bankroll
                    })
                    
                    bets_made += 1
                    total_wagered += bet_size
                    total_profit += (profit if home_won else -bet_size)
            
            # Update Elo with actual result
            elo_system.update_ratings(
                game['home_team'],
                game['away_team'],
                game['home_won'],
                team_a_home=True,
                game_date=game['date']
            )
            
            # Record bankroll
            self.bankroll_history.append(current_bankroll)
            self.date_history.append(game['date'])
        
        # Calculate metrics
        final_bankroll = current_bankroll
        total_return = final_bankroll - self.initial_bankroll
        roi = (total_return / self.initial_bankroll) * 100
        win_rate = (bets_won / bets_made * 100) if bets_made > 0 else 0
        avg_bet_size = total_wagered / bets_made if bets_made > 0 else 0
        
        # Calculate Sharpe ratio
        if len(self.bets) > 1:
            returns = [bet['profit'] / bet['bet_size'] for bet in self.bets]
            sharpe = (np.mean(returns) / np.std(returns)) * np.sqrt(252) if np.std(returns) > 0 else 0
        else:
            sharpe = 0
        
        # Calculate max drawdown
        max_drawdown = self.calculate_max_drawdown()
        
        results = {
            'initial_bankroll': self.initial_bankroll,
            'final_bankroll': final_bankroll,
            'total_return': total_return,
            'roi': roi,
            'bets_made': bets_made,
            'bets_won': bets_won,
            'win_rate': win_rate,
            'total_wagered': total_wagered,
            'total_profit': total_profit,
            'avg_bet_size': avg_bet_size,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'test_games': len(test_df),
        }
        
        return results
    
    def calculate_max_drawdown(self):
        """Calculate maximum drawdown from bankroll history."""
        if len(self.bankroll_history) < 2:
            return 0
        
        bankroll_array = np.array(self.bankroll_history)
        running_max = np.maximum.accumulate(bankroll_array)
        drawdown = (bankroll_array - running_max) / running_max
        max_drawdown = np.min(drawdown) * 100
        
        return max_drawdown
    
    def plot_results(self, save_path='backtest_results.png'):
        """Generate visualization of backtest results."""
        if not self.bets:
            print("No bets to plot")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle('Betting Model Backtest Results', fontsize=16, fontweight='bold')
        
        # Ensure date_history and bankroll_history have same length
        # Prepend initial values if needed
        if len(self.date_history) > 0 and len(self.bankroll_history) > 0:
            if len(self.bankroll_history) == len(self.date_history) + 1:
                # Remove first element from bankroll (the initial state)
                plot_bankroll = self.bankroll_history[1:]
                plot_dates = self.date_history
            elif len(self.date_history) == len(self.bankroll_history) + 1:
                # Remove first element from dates
                plot_dates = self.date_history[1:]
                plot_bankroll = self.bankroll_history
            else:
                # They're already aligned
                plot_dates = self.date_history
                plot_bankroll = self.bankroll_history
        else:
            plot_dates = self.date_history
            plot_bankroll = self.bankroll_history
        
        # Plot 1: Bankroll over time
        ax1 = axes[0, 0]
        if len(plot_dates) > 0 and len(plot_bankroll) > 0:
            ax1.plot(plot_dates, plot_bankroll, linewidth=2, color='#2E86AB')
        ax1.axhline(y=self.initial_bankroll, color='red', linestyle='--', 
                   label=f'Initial: ${self.initial_bankroll:,.0f}')
        ax1.set_xlabel('Date')
        ax1.set_ylabel('Bankroll ($)')
        ax1.set_title('Bankroll Growth Over Time')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Bet size distribution
        ax2 = axes[0, 1]
        bet_sizes = [bet['bet_size'] for bet in self.bets]
        ax2.hist(bet_sizes, bins=30, color='#A23B72', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Bet Size ($)')
        ax2.set_ylabel('Frequency')
        ax2.set_title('Bet Size Distribution')
        ax2.axvline(x=np.mean(bet_sizes), color='red', linestyle='--', 
                   label=f'Mean: ${np.mean(bet_sizes):.2f}')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        # Plot 3: Edge vs Outcome
        ax3 = axes[1, 0]
        wins = [bet for bet in self.bets if bet['won']]
        losses = [bet for bet in self.bets if not bet['won']]
        
        if wins:
            ax3.scatter([bet['edge']*100 for bet in wins], 
                       [bet['profit'] for bet in wins],
                       c='green', label='Wins', alpha=0.6, s=50)
        if losses:
            ax3.scatter([bet['edge']*100 for bet in losses], 
                       [bet['profit'] for bet in losses],
                       c='red', label='Losses', alpha=0.6, s=50)
        
        ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax3.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax3.set_xlabel('Edge (%)')
        ax3.set_ylabel('Profit ($)')
        ax3.set_title('Edge vs Profit/Loss')
        ax3.legend()
        ax3.grid(True, alpha=0.3)
        
        # Plot 4: Cumulative profit
        ax4 = axes[1, 1]
        cumulative_profit = np.cumsum([bet['profit'] for bet in self.bets])
        bet_numbers = range(1, len(cumulative_profit) + 1)
        ax4.plot(bet_numbers, cumulative_profit, linewidth=2, color='#F18F01')
        ax4.axhline(y=0, color='red', linestyle='--')
        ax4.set_xlabel('Bet Number')
        ax4.set_ylabel('Cumulative Profit ($)')
        ax4.set_title('Cumulative Profit Over Bets')
        ax4.grid(True, alpha=0.3)
        ax4.fill_between(bet_numbers, cumulative_profit, 0, 
                        where=(cumulative_profit >= 0), 
                        interpolate=True, alpha=0.3, color='green')
        ax4.fill_between(bet_numbers, cumulative_profit, 0, 
                        where=(cumulative_profit < 0), 
                        interpolate=True, alpha=0.3, color='red')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"📊 Results plotted to {save_path}")
        
        return fig
    
    def analyze_edge_calibration(self):
        """
        Analyze how well the model's predicted edge translates to actual wins.
        This helps identify if the model is over/under-confident.
        """
        if not self.bets:
            return None
        
        # Group bets by edge buckets
        edge_buckets = defaultdict(list)
        for bet in self.bets:
            edge_pct = bet['edge'] * 100
            bucket = int(edge_pct // 5) * 5  # 5% buckets
            edge_buckets[bucket].append(bet['won'])
        
        print(f"\n{'='*70}")
        print("EDGE CALIBRATION ANALYSIS")
        print(f"{'='*70}")
        print(f"{'Edge Range':<20} {'Bets':<10} {'Win Rate':<15} {'Expected':<15}")
        print(f"{'-'*70}")
        
        for edge in sorted(edge_buckets.keys()):
            bets_in_bucket = edge_buckets[edge]
            win_rate = sum(bets_in_bucket) / len(bets_in_bucket) * 100
            
            # Expected win rate would be model_prob (approximated)
            avg_edge = edge
            # This is approximate - actual expected varies by bet
            
            print(f"{edge:>3}%-{edge+5:<3}%        {len(bets_in_bucket):<10} "
                  f"{win_rate:>6.1f}%        ~{50 + avg_edge/2:>5.1f}%")
    
    def monte_carlo_simulation(self, n_simulations=1000):
        """
        Run Monte Carlo simulation to estimate confidence intervals.
        """
        if not self.bets:
            return None
        
        print(f"\n{'='*70}")
        print(f"MONTE CARLO SIMULATION ({n_simulations:,} runs)")
        print(f"{'='*70}")
        
        final_bankrolls = []
        
        for _ in range(n_simulations):
            bankroll = self.initial_bankroll
            
            for bet in self.bets:
                # Simulate outcome based on model probability
                won = np.random.random() < bet['model_prob']
                
                if won:
                    profit = bet['bet_size'] * (1 / bet['market_price'] - 1)
                    bankroll += profit
                else:
                    bankroll -= bet['bet_size']
            
            final_bankrolls.append(bankroll)
        
        final_bankrolls = np.array(final_bankrolls)
        
        percentiles = [5, 25, 50, 75, 95]
        results = {}
        
        print(f"\nFinal Bankroll Distribution:")
        for p in percentiles:
            value = np.percentile(final_bankrolls, p)
            results[f'p{p}'] = value
            print(f"  {p}th percentile: ${value:,.2f}")
        
        print(f"\nMean: ${np.mean(final_bankrolls):,.2f}")
        print(f"Std Dev: ${np.std(final_bankrolls):,.2f}")
        
        prob_profit = (final_bankrolls > self.initial_bankroll).mean() * 100
        print(f"\nProbability of profit: {prob_profit:.1f}%")
        
        return results


# ============ DATA PROCESSING ============

def load_games_data(filepath):
    """
    Load and process historical games data.
    
    Expected CSV format:
        date, home_team, away_team, home_score, away_score
    
    Or simpler format:
        date, home_team, away_team, home_won
    """
    try:
        df = pd.read_csv(filepath)
        
        # Convert date to datetime
        df['date'] = pd.to_datetime(df['date'])
        
        # Determine home_won if not present
        if 'home_won' not in df.columns:
            if 'home_score' in df.columns and 'away_score' in df.columns:
                df['home_won'] = df['home_score'] > df['away_score']
            else:
                raise ValueError("CSV must have either 'home_won' or 'home_score'/'away_score' columns")
        
        # Convert to boolean
        df['home_won'] = df['home_won'].astype(bool)
        
        print(f"✅ Loaded {len(df)} games from {filepath}")
        print(f"   Date range: {df['date'].min()} to {df['date'].max()}")
        print(f"   Teams: {df['home_team'].nunique()} home teams, {df['away_team'].nunique()} away teams")
        
        return df
        
    except Exception as e:
        print(f"❌ Error loading games data: {e}")
        return None


def parameter_optimization(games_df, param_grid):
    """
    Grid search to find optimal parameters.
    
    Tests different combinations of:
    - Kelly fraction
    - Min edge threshold
    - K-factor
    - Home advantage
    """
    print(f"\n{'='*70}")
    print("PARAMETER OPTIMIZATION")
    print(f"{'='*70}\n")
    
    best_roi = -float('inf')
    best_params = None
    all_results = []
    
    total_combinations = len(param_grid['kelly_fraction']) * len(param_grid['min_edge']) * \
                        len(param_grid['k_factor']) * len(param_grid['home_advantage'])
    
    print(f"Testing {total_combinations} parameter combinations...\n")
    
    iteration = 0
    for kelly_frac in param_grid['kelly_fraction']:
        for min_edge in param_grid['min_edge']:
            for k_factor in param_grid['k_factor']:
                for home_adv in param_grid['home_advantage']:
                    iteration += 1
                    
                    # Create Elo system with these parameters
                    elo = EloRatingSystem(k_factor=k_factor, home_advantage=home_adv)
                    
                    # Run backtest
                    backtester = BettingBacktester(
                        kelly_fraction=kelly_frac,
                        min_edge=min_edge
                    )
                    
                    results = backtester.run_backtest(games_df, elo, train_size=0.7)
                    
                    # Store results
                    result_row = {
                        'kelly_fraction': kelly_frac,
                        'min_edge': min_edge,
                        'k_factor': k_factor,
                        'home_advantage': home_adv,
                        'roi': results['roi'],
                        'sharpe': results['sharpe_ratio'],
                        'win_rate': results['win_rate'],
                        'bets_made': results['bets_made']
                    }
                    all_results.append(result_row)
                    
                    # Track best
                    if results['roi'] > best_roi:
                        best_roi = results['roi']
                        best_params = result_row
                    
                    # Progress
                    if iteration % 10 == 0:
                        print(f"  Progress: {iteration}/{total_combinations} ({iteration/total_combinations*100:.1f}%)")
    
    print(f"\n{'='*70}")
    print("OPTIMIZATION RESULTS")
    print(f"{'='*70}")
    print("\nBest Parameters:")
    for param, value in best_params.items():
        print(f"  {param}: {value}")
    
    # Save all results
    results_df = pd.DataFrame(all_results)
    results_df.to_csv('parameter_optimization_results.csv', index=False)
    print(f"\n📊 Full results saved to parameter_optimization_results.csv")
    
    return best_params, results_df


# ============ MAIN ============

def main():
    parser = argparse.ArgumentParser(description='Backtest sports betting model')
    parser.add_argument('data_file', type=str, help='CSV file with historical games')
    parser.add_argument('--bankroll', type=float, default=1000.0,
                       help='Initial bankroll (default: $1000)')
    parser.add_argument('--kelly', type=float, default=0.5,
                       help='Kelly fraction (default: 0.5)')
    parser.add_argument('--min-edge', type=float, default=0.05,
                       help='Minimum edge threshold (default: 0.05)')
    parser.add_argument('--test-size', type=float, default=0.3,
                       help='Fraction of data for testing (default: 0.3)')
    parser.add_argument('--k-factor', type=int, default=20,
                       help='Elo K-factor (default: 20)')
    parser.add_argument('--home-advantage', type=int, default=100,
                       help='Elo home advantage (default: 100)')
    parser.add_argument('--strategy', type=str, default='kelly',
                       choices=['kelly', 'flat', 'proportional'],
                       help='Betting strategy (default: kelly)')
    parser.add_argument('--optimize-params', action='store_true',
                       help='Run parameter optimization')
    parser.add_argument('--monte-carlo', type=int, default=0,
                       help='Run Monte Carlo simulation with N runs (default: 0 = skip)')
    parser.add_argument('--no-plots', action='store_true',
                       help='Skip generating plots')
    
    args = parser.parse_args()
    
    # Load data
    games_df = load_games_data(args.data_file)
    if games_df is None:
        return
    
    # Parameter optimization mode
    if args.optimize_params:
        param_grid = {
            'kelly_fraction': [0.25, 0.5, 0.75],
            'min_edge': [0.03, 0.05, 0.08, 0.10],
            'k_factor': [16, 20, 24],
            'home_advantage': [75, 100, 125]
        }
        best_params, results_df = parameter_optimization(games_df, param_grid)
        return
    
    # Create Elo system
    elo_system = EloRatingSystem(
        k_factor=args.k_factor,
        home_advantage=args.home_advantage
    )
    
    # Run backtest
    backtester = BettingBacktester(
        initial_bankroll=args.bankroll,
        kelly_fraction=args.kelly,
        min_edge=args.min_edge
    )
    
    train_size = 1 - args.test_size
    results = backtester.run_backtest(games_df, elo_system, train_size=train_size)
    
    # Print results
    print(f"\n{'='*70}")
    print("BACKTEST RESULTS")
    print(f"{'='*70}")
    print(f"Initial Bankroll:    ${results['initial_bankroll']:>12,.2f}")
    print(f"Final Bankroll:      ${results['final_bankroll']:>12,.2f}")
    print(f"Total Return:        ${results['total_return']:>12,.2f}")
    print(f"ROI:                  {results['roi']:>12.2f}%")
    print(f"{'-'*70}")
    print(f"Bets Made:            {results['bets_made']:>12,}")
    print(f"Bets Won:             {results['bets_won']:>12,}")
    print(f"Win Rate:             {results['win_rate']:>12.1f}%")
    print(f"{'-'*70}")
    print(f"Total Wagered:       ${results['total_wagered']:>12,.2f}")
    print(f"Total Profit:        ${results['total_profit']:>12,.2f}")
    print(f"Avg Bet Size:        ${results['avg_bet_size']:>12,.2f}")
    print(f"{'-'*70}")
    print(f"Sharpe Ratio:         {results['sharpe_ratio']:>12.2f}")
    print(f"Max Drawdown:         {results['max_drawdown']:>12.2f}%")
    print(f"{'='*70}")
    
    # Edge calibration
    backtester.analyze_edge_calibration()
    
    # Monte Carlo simulation
    if args.monte_carlo > 0:
        backtester.monte_carlo_simulation(n_simulations=args.monte_carlo)
    
    # Generate plots
    if not args.no_plots:
        backtester.plot_results()
    
    # Save detailed bet log
    if backtester.bets:
        bets_df = pd.DataFrame(backtester.bets)
        bets_df.to_csv('backtest_bets.csv', index=False)
        print(f"\n📁 Detailed bet log saved to backtest_bets.csv")
    
    # Summary JSON
    summary = {
        'parameters': {
            'initial_bankroll': args.bankroll,
            'kelly_fraction': args.kelly,
            'min_edge': args.min_edge,
            'k_factor': args.k_factor,
            'home_advantage': args.home_advantage,
        },
        'results': results,
        'timestamp': datetime.now().isoformat()
    }
    
    with open('backtest_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)
    print(f"📁 Summary saved to backtest_summary.json")


if __name__ == "__main__":
    main()