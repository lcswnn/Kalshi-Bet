#!/usr/bin/env python3
"""
MLB Advanced Features Generator (Memory-Efficient Version)
Adds park factors, rest days, ELO ratings, and other advanced features
to the base matchup dataset.
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import gc

# ============================================================================
# PARK FACTORS (2019-2024 averages)
# ============================================================================

PARK_FACTORS = {
    'ARI': 104, 'ATL': 102, 'BAL': 103, 'BOS': 106, 'CHC': 102,
    'CWS': 103, 'CIN': 107, 'CLE': 97,  'COL': 115, 'DET': 98,
    'HOU': 100, 'KC': 99,   'LAA': 98,  'LAD': 96,  'MIA': 93,
    'MIL': 103, 'MIN': 101, 'NYM': 96,  'NYY': 105, 'OAK': 96,
    'PHI': 103, 'PIT': 97,  'SD': 94,   'SF': 93,   'SEA': 95,
    'STL': 98,  'TB': 95,   'TEX': 104, 'TOR': 102, 'WSH': 100
}

DIVISIONS = {
    'ARI': 'NLW', 'ATL': 'NLE', 'BAL': 'ALE', 'BOS': 'ALE', 'CHC': 'NLC',
    'CWS': 'ALC', 'CIN': 'NLC', 'CLE': 'ALC', 'COL': 'NLW', 'DET': 'ALC',
    'HOU': 'ALW', 'KC': 'ALC',  'LAA': 'ALW', 'LAD': 'NLW', 'MIA': 'NLE',
    'MIL': 'NLC', 'MIN': 'ALC', 'NYM': 'NLE', 'NYY': 'ALE', 'OAK': 'ALW',
    'PHI': 'NLE', 'PIT': 'NLC', 'SD': 'NLW',  'SF': 'NLW',  'SEA': 'ALW',
    'STL': 'NLC', 'TB': 'ALE',  'TEX': 'ALW', 'TOR': 'ALE', 'WSH': 'NLE'
}

# ============================================================================
# ELO RATING SYSTEM (Vectorized where possible)
# ============================================================================

def calculate_elo_ratings(df: pd.DataFrame, k_factor: float = 4.0, 
                          home_advantage: float = 24) -> pd.DataFrame:
    """Calculate ELO ratings efficiently using numpy arrays."""
    
    # Sort by date and reset index
    df = df.sort_values('game_date').reset_index(drop=True)
    n_games = len(df)
    
    # Pre-allocate output arrays
    home_elo = np.zeros(n_games)
    away_elo = np.zeros(n_games)
    home_win_prob = np.zeros(n_games)
    
    # Team ratings dictionary
    ratings = {}
    
    # Convert to numpy for speed
    home_teams = df['home_team'].values
    away_teams = df['away_team'].values
    home_scores = df['home_score'].values
    away_scores = df['away_score'].values
    seasons = df['season'].values
    
    for i in range(n_games):
        home_team = home_teams[i]
        away_team = away_teams[i]
        season = seasons[i]
        
        # Get or initialize ratings
        home_key = (home_team, season)
        away_key = (away_team, season)
        
        if home_key not in ratings:
            prev_key = (home_team, season - 1)
            if prev_key in ratings:
                ratings[home_key] = 1500 + (ratings[prev_key] - 1500) * 0.67
            else:
                ratings[home_key] = 1500.0
        
        if away_key not in ratings:
            prev_key = (away_team, season - 1)
            if prev_key in ratings:
                ratings[away_key] = 1500 + (ratings[prev_key] - 1500) * 0.67
            else:
                ratings[away_key] = 1500.0
        
        hr = ratings[home_key]
        ar = ratings[away_key]
        
        # Store pre-game ratings
        home_elo[i] = hr
        away_elo[i] = ar
        
        # Expected score
        diff = hr - ar + home_advantage
        exp_home = 1 / (1 + 10 ** (-diff / 400))
        home_win_prob[i] = exp_home
        
        # Update ratings
        home_score = home_scores[i]
        away_score = away_scores[i]
        
        if home_score > away_score:
            home_result = 1
        elif home_score < away_score:
            home_result = 0
        else:
            home_result = 0.5
        
        # Margin of victory multiplier
        mov = abs(home_score - away_score)
        mov_mult = min(np.log(mov + 1) * 0.7 + 0.6, 2.5)
        
        delta = k_factor * mov_mult * (home_result - exp_home)
        ratings[home_key] = hr + delta
        ratings[away_key] = ar - delta
        
        # Progress
        if (i + 1) % 5000 == 0:
            print(f"  ELO: Processed {i+1}/{n_games} games...")
    
    df['home_elo'] = home_elo
    df['away_elo'] = away_elo
    df['home_win_prob'] = home_win_prob
    df['elo_diff'] = home_elo - away_elo
    
    return df


# ============================================================================
# FEATURE FUNCTIONS (All memory-efficient)
# ============================================================================

def add_rest_days(df: pd.DataFrame) -> pd.DataFrame:
    """Add rest days for each team."""
    print("  Building team schedule...")
    
    df['game_date'] = pd.to_datetime(df['game_date'])
    
    # Create minimal team schedule
    home = df[['game_id', 'game_date', 'home_team']].rename(columns={'home_team': 'team'})
    home['is_home'] = True
    away = df[['game_id', 'game_date', 'away_team']].rename(columns={'away_team': 'team'})
    away['is_home'] = False
    
    schedule = pd.concat([home, away], ignore_index=True)
    schedule = schedule.sort_values(['team', 'game_date'])
    
    # Calculate rest days
    schedule['prev_date'] = schedule.groupby('team')['game_date'].shift(1)
    schedule['rest_days'] = (schedule['game_date'] - schedule['prev_date']).dt.days
    schedule['rest_days'] = schedule['rest_days'].fillna(3).clip(upper=7)
    
    # Split and merge back
    home_rest = schedule[schedule['is_home']][['game_id', 'rest_days']]
    home_rest.columns = ['game_id', 'home_rest_days']
    
    away_rest = schedule[~schedule['is_home']][['game_id', 'rest_days']]
    away_rest.columns = ['game_id', 'away_rest_days']
    
    df = df.merge(home_rest, on='game_id', how='left')
    df = df.merge(away_rest, on='game_id', how='left')
    df['rest_advantage'] = df['home_rest_days'] - df['away_rest_days']
    
    del schedule, home, away, home_rest, away_rest
    gc.collect()
    
    return df


def add_streak_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add winning/losing streak features - ultra memory efficient."""
    print("  Calculating streaks directly...")
    
    df = df.sort_values('game_date').reset_index(drop=True)
    n = len(df)
    
    # Pre-allocate output
    home_streaks = np.zeros(n)
    away_streaks = np.zeros(n)
    
    # Track current streak per team
    current_streak = {}
    
    # Get arrays for speed
    home_teams = df['home_team'].values
    away_teams = df['away_team'].values
    home_wins = df['home_win'].values
    
    for i in range(n):
        ht = home_teams[i]
        at = away_teams[i]
        hw = home_wins[i]
        
        # Get pre-game streaks
        home_streaks[i] = current_streak.get(ht, 0)
        away_streaks[i] = current_streak.get(at, 0)
        
        # Update home team streak
        prev_h = current_streak.get(ht, 0)
        if hw == 1:  # home won
            current_streak[ht] = max(1, prev_h + 1) if prev_h >= 0 else 1
        else:  # home lost
            current_streak[ht] = min(-1, prev_h - 1) if prev_h <= 0 else -1
        
        # Update away team streak
        prev_a = current_streak.get(at, 0)
        if hw == 0:  # away won
            current_streak[at] = max(1, prev_a + 1) if prev_a >= 0 else 1
        else:  # away lost
            current_streak[at] = min(-1, prev_a - 1) if prev_a <= 0 else -1
        
        if (i + 1) % 5000 == 0:
            print(f"  Streaks: Processed {i+1}/{n} games...")
    
    df['home_streak'] = home_streaks
    df['away_streak'] = away_streaks
    df['streak_diff'] = home_streaks - away_streaks
    
    return df


def add_simple_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add simple features that don't require iteration."""
    
    df['game_date'] = pd.to_datetime(df['game_date'])
    
    # Park factors
    df['park_factor'] = df['home_team'].map(PARK_FACTORS).fillna(100)
    df['park_factor_adj'] = (df['park_factor'] - 100) / 100
    
    # Schedule features
    df['day_of_week'] = df['game_date'].dt.dayofweek
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
    df['month'] = df['game_date'].dt.month
    
    # Days into season
    season_starts = df.groupby('season')['game_date'].transform('min')
    df['days_into_season'] = (df['game_date'] - season_starts).dt.days
    
    # Game number
    df = df.sort_values(['home_team', 'season', 'game_date'])
    df['home_game_num'] = df.groupby(['home_team', 'season']).cumcount() + 1
    df = df.sort_values(['away_team', 'season', 'game_date'])
    df['away_game_num'] = df.groupby(['away_team', 'season']).cumcount() + 1
    
    # Division features
    df['home_division'] = df['home_team'].map(DIVISIONS)
    df['away_division'] = df['away_team'].map(DIVISIONS)
    df['same_division'] = (df['home_division'] == df['away_division']).astype(int)
    df['interleague'] = (df['home_division'].str[0] != df['away_division'].str[0]).astype(int)
    
    # Re-sort by date
    df = df.sort_values('game_date').reset_index(drop=True)
    
    return df


def add_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add composite/derived features."""
    
    # Strength metric (if we have the components)
    if 'home_elo' in df.columns:
        if 'home_pyth_win_pct' in df.columns:
            df['home_strength'] = (
                (df['home_elo'] - 1500) / 100 * 0.5 +
                (df['home_pyth_win_pct'] - 0.5) * 2 * 0.5
            )
            df['away_strength'] = (
                (df['away_elo'] - 1500) / 100 * 0.5 +
                (df['away_pyth_win_pct'] - 0.5) * 2 * 0.5
            )
            df['strength_diff'] = df['home_strength'] - df['away_strength']
        else:
            # Just use ELO-based strength
            df['home_strength'] = (df['home_elo'] - 1500) / 100
            df['away_strength'] = (df['away_elo'] - 1500) / 100
            df['strength_diff'] = df['home_strength'] - df['away_strength']
    
    # Adjusted expected total
    if 'expected_total' in df.columns and 'park_factor' in df.columns:
        df['adj_expected_total'] = df['expected_total'] * (df['park_factor'] / 100)
    
    return df


# ============================================================================
# MAIN FUNCTION
# ============================================================================

def add_advanced_features(input_file: str, output_file: str = None) -> pd.DataFrame:
    """Add all advanced features to the dataset."""
    
    print("=" * 60)
    print("ADDING ADVANCED FEATURES (Memory-Efficient)")
    print("=" * 60)
    
    # Load data
    print(f"\nLoading data from {input_file}...")
    df = pd.read_csv(input_file)
    print(f"Loaded {len(df)} games")
    
    # Add ELO ratings (must be first - needs chronological order)
    print("\n[1/5] Calculating ELO ratings...")
    df = calculate_elo_ratings(df)
    gc.collect()
    
    # Add rest days
    print("\n[2/5] Adding rest days...")
    df = add_rest_days(df)
    gc.collect()
    
    # Add simple features (park factors, schedule, divisions)
    print("\n[3/5] Adding simple features...")
    df = add_simple_features(df)
    gc.collect()
    
    # Add streak features
    print("\n[4/5] Adding streak features...")
    df = add_streak_features(df)
    gc.collect()
    
    # Add composite features
    print("\n[5/5] Adding composite features...")
    df = add_composite_features(df)
    
    # Save
    if output_file is None:
        output_file = input_file.replace('.csv', '_advanced.csv')
    
    print(f"\nSaving to {output_file}...")
    df.to_csv(output_file, index=False)
    
    # Summary
    print("\n" + "=" * 60)
    print("COMPLETE!")
    print("=" * 60)
    print(f"Total games: {len(df)}")
    print(f"Total features: {len(df.columns)}")
    
    new_features = [
        'home_elo', 'away_elo', 'elo_diff', 'home_win_prob',
        'home_rest_days', 'away_rest_days', 'rest_advantage',
        'park_factor', 'park_factor_adj',
        'day_of_week', 'is_weekend', 'month', 'days_into_season',
        'home_streak', 'away_streak', 'streak_diff',
        'same_division', 'interleague',
        'home_strength', 'away_strength', 'strength_diff'
    ]
    
    print("\nNew features added:")
    for feat in new_features:
        if feat in df.columns:
            print(f"  ✓ {feat}")
    
    return df


if __name__ == "__main__":
    import sys
    
    input_file = sys.argv[1] if len(sys.argv) > 1 else "mlb_historical_data/mlb_matchups_features.csv"
    output_file = sys.argv[2] if len(sys.argv) > 2 else None
    
    df = add_advanced_features(input_file, output_file)