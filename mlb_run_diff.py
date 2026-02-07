#!/usr/bin/env python3
"""
MLB Historical Data Fetcher
Fetches game results, team stats, and pitcher data from 2019-2025
for building prediction models for Kalshi MLB betting markets.

Data Sources:
- MLB Stats API (free, official)
- Baseball Reference (via pybaseball)
"""

import requests
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time
import json
import os
from typing import Dict, List, Optional, Tuple
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# CONFIGURATION
# ============================================================================

SEASONS = [2019, 2020, 2021, 2022, 2023, 2024, 2025]
OUTPUT_DIR = "mlb_historical_data"
MLB_API_BASE = "https://statsapi.mlb.com/api/v1"

# Team ID mapping for MLB Stats API
TEAM_IDS = {
    'ARI': 109, 'ATL': 144, 'BAL': 110, 'BOS': 111, 'CHC': 112,
    'CWS': 145, 'CIN': 113, 'CLE': 114, 'COL': 115, 'DET': 116,
    'HOU': 117, 'KC': 118, 'LAA': 108, 'LAD': 119, 'MIA': 146,
    'MIL': 158, 'MIN': 142, 'NYM': 121, 'NYY': 147, 'OAK': 133,
    'PHI': 143, 'PIT': 134, 'SD': 135, 'SF': 137, 'SEA': 136,
    'STL': 138, 'TB': 139, 'TEX': 140, 'TOR': 141, 'WSH': 120
}

# Reverse mapping
ID_TO_TEAM = {v: k for k, v in TEAM_IDS.items()}

# ============================================================================
# MLB STATS API FUNCTIONS
# ============================================================================

def get_schedule(season: int, start_date: str = None, end_date: str = None) -> List[Dict]:
    """Fetch game schedule for a season or date range."""
    
    if start_date and end_date:
        url = f"{MLB_API_BASE}/schedule"
        params = {
            'sportId': 1,
            'startDate': start_date,
            'endDate': end_date,
            'gameType': 'R,P',  # Regular season and playoffs
            'hydrate': 'team,linescore,decisions,probablePitcher'
        }
    else:
        url = f"{MLB_API_BASE}/schedule"
        params = {
            'sportId': 1,
            'season': season,
            'gameType': 'R,P',
            'hydrate': 'team,linescore,decisions,probablePitcher'
        }
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        print(f"Error fetching schedule: {response.status_code}")
        return []
    
    data = response.json()
    games = []
    
    for date_entry in data.get('dates', []):
        for game in date_entry.get('games', []):
            games.append(game)
    
    return games


def parse_game_data(game: Dict) -> Optional[Dict]:
    """Parse a single game into a flat dictionary."""
    
    try:
        # Skip games that haven't been played
        if game.get('status', {}).get('abstractGameState') != 'Final':
            return None
        
        game_id = game.get('gamePk')
        game_date = game.get('gameDate', '')[:10]
        
        # Teams
        away_team = game.get('teams', {}).get('away', {})
        home_team = game.get('teams', {}).get('home', {})
        
        away_id = away_team.get('team', {}).get('id')
        home_id = home_team.get('team', {}).get('id')
        
        away_abbrev = ID_TO_TEAM.get(away_id, 'UNK')
        home_abbrev = ID_TO_TEAM.get(home_id, 'UNK')
        
        # Scores
        away_score = away_team.get('score', 0)
        home_score = home_team.get('score', 0)
        
        # Linescore details
        linescore = game.get('linescore', {})
        innings = linescore.get('currentInning', 9)
        
        # Probable/Starting pitchers
        away_pitcher = away_team.get('probablePitcher', {})
        home_pitcher = home_team.get('probablePitcher', {})
        
        # Decisions (winner/loser)
        decisions = game.get('decisions', {})
        winner = decisions.get('winner', {})
        loser = decisions.get('loser', {})
        
        return {
            'game_id': game_id,
            'game_date': game_date,
            'season': int(game_date[:4]),
            'away_team': away_abbrev,
            'home_team': home_abbrev,
            'away_team_id': away_id,
            'home_team_id': home_id,
            'away_score': away_score,
            'home_score': home_score,
            'total_runs': away_score + home_score,
            'home_win': 1 if home_score > away_score else 0,
            'innings': innings,
            'extra_innings': 1 if innings > 9 else 0,
            'away_starter_id': away_pitcher.get('id'),
            'away_starter_name': away_pitcher.get('fullName'),
            'home_starter_id': home_pitcher.get('id'),
            'home_starter_name': home_pitcher.get('fullName'),
            'winning_pitcher_id': winner.get('id'),
            'losing_pitcher_id': loser.get('id'),
            'venue_id': game.get('venue', {}).get('id'),
            'venue_name': game.get('venue', {}).get('name'),
            'day_night': game.get('dayNight'),
            'game_type': game.get('gameType')  # R=regular, P=playoff
        }
    except Exception as e:
        print(f"Error parsing game: {e}")
        return None


def get_team_stats(team_id: int, season: int, stat_type: str = 'season') -> Dict:
    """Fetch team statistics for a season."""
    
    url = f"{MLB_API_BASE}/teams/{team_id}/stats"
    params = {
        'stats': stat_type,
        'season': season,
        'group': 'hitting,pitching,fielding'
    }
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        return {}
    
    data = response.json()
    stats = {}
    
    for stat_group in data.get('stats', []):
        group_name = stat_group.get('group', {}).get('displayName', '')
        for split in stat_group.get('splits', []):
            stat_data = split.get('stat', {})
            for key, value in stat_data.items():
                stats[f"{group_name.lower()}_{key}"] = value
    
    return stats


def get_standings(season: int, date: str = None) -> pd.DataFrame:
    """Fetch standings for a season or specific date."""
    
    url = f"{MLB_API_BASE}/standings"
    params = {
        'leagueId': '103,104',  # AL and NL
        'season': season,
        'standingsTypes': 'regularSeason'
    }
    
    if date:
        params['date'] = date
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        return pd.DataFrame()
    
    data = response.json()
    standings_list = []
    
    for record in data.get('records', []):
        division = record.get('division', {}).get('nameShort', '')
        for team_record in record.get('teamRecords', []):
            team_id = team_record.get('team', {}).get('id')
            standings_list.append({
                'team_id': team_id,
                'team_abbrev': ID_TO_TEAM.get(team_id, 'UNK'),
                'division': division,
                'wins': team_record.get('wins', 0),
                'losses': team_record.get('losses', 0),
                'win_pct': team_record.get('winningPercentage', '.000'),
                'games_back': team_record.get('gamesBack', '-'),
                'streak': team_record.get('streak', {}).get('streakCode', ''),
                'runs_scored': team_record.get('runsScored', 0),
                'runs_allowed': team_record.get('runsAllowed', 0),
                'run_diff': team_record.get('runDifferential', 0),
                'home_record': team_record.get('records', {}).get('splitRecords', [{}])[0].get('wins', 0),
                'away_record': team_record.get('records', {}).get('splitRecords', [{}])[1].get('wins', 0) if len(team_record.get('records', {}).get('splitRecords', [])) > 1 else 0,
                'last_10': team_record.get('records', {}).get('splitRecords', [{}])[-1].get('wins', 0) if team_record.get('records', {}).get('splitRecords', []) else 0
            })
    
    return pd.DataFrame(standings_list)


def get_pitcher_stats(pitcher_id: int, season: int) -> Dict:
    """Fetch pitcher statistics for a season."""
    
    url = f"{MLB_API_BASE}/people/{pitcher_id}/stats"
    params = {
        'stats': 'season',
        'season': season,
        'group': 'pitching'
    }
    
    response = requests.get(url, params=params)
    if response.status_code != 200:
        return {}
    
    data = response.json()
    
    for stat_group in data.get('stats', []):
        for split in stat_group.get('splits', []):
            return split.get('stat', {})
    
    return {}


def get_game_boxscore(game_id: int) -> Dict:
    """Fetch detailed boxscore for a game."""
    
    url = f"{MLB_API_BASE}/game/{game_id}/boxscore"
    
    response = requests.get(url)
    if response.status_code != 200:
        return {}
    
    return response.json()


# ============================================================================
# FEATURE ENGINEERING
# ============================================================================

def calculate_rolling_stats(df: pd.DataFrame, team_col: str, 
                           windows: List[int] = [5, 10, 20]) -> pd.DataFrame:
    """Calculate rolling statistics for a team."""
    
    df = df.sort_values('game_date')
    
    for window in windows:
        # Runs scored
        df[f'runs_scored_L{window}'] = df.groupby(team_col)['runs_scored'].transform(
            lambda x: x.rolling(window, min_periods=1).mean().shift(1)
        )
        
        # Runs allowed
        df[f'runs_allowed_L{window}'] = df.groupby(team_col)['runs_allowed'].transform(
            lambda x: x.rolling(window, min_periods=1).mean().shift(1)
        )
        
        # Win percentage
        df[f'win_pct_L{window}'] = df.groupby(team_col)['win'].transform(
            lambda x: x.rolling(window, min_periods=1).mean().shift(1)
        )
    
    return df


def create_team_game_log(games_df: pd.DataFrame) -> pd.DataFrame:
    """Create a team-level game log from game results."""
    
    # Create home team rows
    home_df = games_df.copy()
    home_df['team'] = home_df['home_team']
    home_df['opponent'] = home_df['away_team']
    home_df['runs_scored'] = home_df['home_score']
    home_df['runs_allowed'] = home_df['away_score']
    home_df['win'] = home_df['home_win']
    home_df['is_home'] = 1
    
    # Create away team rows
    away_df = games_df.copy()
    away_df['team'] = away_df['away_team']
    away_df['opponent'] = away_df['home_team']
    away_df['runs_scored'] = away_df['away_score']
    away_df['runs_allowed'] = away_df['home_score']
    away_df['win'] = 1 - away_df['home_win']
    away_df['is_home'] = 0
    
    # Combine
    team_log = pd.concat([home_df, away_df], ignore_index=True)
    team_log = team_log.sort_values(['team', 'game_date'])
    
    return team_log


def add_pythagorean_expectation(df: pd.DataFrame) -> pd.DataFrame:
    """Add Pythagorean win expectation based on runs."""
    
    # Calculate cumulative runs
    df['cum_runs_scored'] = df.groupby('team')['runs_scored'].cumsum().shift(1)
    df['cum_runs_allowed'] = df.groupby('team')['runs_allowed'].cumsum().shift(1)
    
    # Pythagorean expectation (exponent of 1.83 is common for baseball)
    exp = 1.83
    df['pyth_win_pct'] = (
        df['cum_runs_scored'] ** exp / 
        (df['cum_runs_scored'] ** exp + df['cum_runs_allowed'] ** exp)
    )
    
    return df


def create_matchup_features(games_df: pd.DataFrame, team_log: pd.DataFrame) -> pd.DataFrame:
    """Create matchup-level features for prediction."""
    
    # Get the latest stats for each team before each game
    team_stats = team_log.groupby('team').apply(
        lambda x: x.sort_values('game_date')
    ).reset_index(drop=True)
    
    # Calculate rolling features
    team_stats = calculate_rolling_stats(team_stats, 'team')
    team_stats = add_pythagorean_expectation(team_stats)
    
    # Merge back to games
    # Home team features
    home_features = team_stats[team_stats['is_home'] == 1][[
        'game_id', 'team', 'runs_scored_L5', 'runs_scored_L10', 'runs_scored_L20',
        'runs_allowed_L5', 'runs_allowed_L10', 'runs_allowed_L20',
        'win_pct_L5', 'win_pct_L10', 'win_pct_L20', 'pyth_win_pct'
    ]].rename(columns=lambda x: f'home_{x}' if x not in ['game_id', 'team'] else x)
    
    # Away team features  
    away_features = team_stats[team_stats['is_home'] == 0][[
        'game_id', 'team', 'runs_scored_L5', 'runs_scored_L10', 'runs_scored_L20',
        'runs_allowed_L5', 'runs_allowed_L10', 'runs_allowed_L20',
        'win_pct_L5', 'win_pct_L10', 'win_pct_L20', 'pyth_win_pct'
    ]].rename(columns=lambda x: f'away_{x}' if x not in ['game_id', 'team'] else x)
    
    # Merge features to games
    matchups = games_df.merge(
        home_features, left_on=['game_id', 'home_team'], right_on=['game_id', 'team'], how='left'
    ).drop(columns=['team'])
    
    matchups = matchups.merge(
        away_features, left_on=['game_id', 'away_team'], right_on=['game_id', 'team'], how='left'
    ).drop(columns=['team'])
    
    return matchups


# ============================================================================
# MAIN DATA COLLECTION
# ============================================================================

def fetch_all_games(seasons: List[int] = SEASONS, 
                    save_progress: bool = True) -> pd.DataFrame:
    """Fetch all games for specified seasons."""
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    all_games = []
    
    for season in seasons:
        print(f"\n{'='*50}")
        print(f"Fetching {season} season...")
        print(f"{'='*50}")
        
        # Check for cached data
        cache_file = f"{OUTPUT_DIR}/games_{season}.csv"
        if os.path.exists(cache_file):
            print(f"Loading cached data for {season}...")
            season_df = pd.read_csv(cache_file)
            all_games.append(season_df)
            continue
        
        # Fetch from API
        games = get_schedule(season)
        print(f"Found {len(games)} games for {season}")
        
        parsed_games = []
        for i, game in enumerate(games):
            parsed = parse_game_data(game)
            if parsed:
                parsed_games.append(parsed)
            
            if (i + 1) % 500 == 0:
                print(f"  Processed {i+1}/{len(games)} games...")
        
        if parsed_games:
            season_df = pd.DataFrame(parsed_games)
            print(f"Successfully parsed {len(season_df)} completed games")
            
            if save_progress:
                season_df.to_csv(cache_file, index=False)
                print(f"Saved to {cache_file}")
            
            all_games.append(season_df)
        
        # Be nice to the API
        time.sleep(1)
    
    # Combine all seasons
    if all_games:
        full_df = pd.concat(all_games, ignore_index=True)
        full_df = full_df.sort_values(['game_date', 'game_id'])
        return full_df
    
    return pd.DataFrame()


def build_prediction_dataset(games_df: pd.DataFrame) -> pd.DataFrame:
    """Build the full prediction dataset with all features."""
    
    print("\nBuilding prediction dataset...")
    
    # Create team game log
    print("  Creating team game log...")
    team_log = create_team_game_log(games_df)
    
    # Create matchup features
    print("  Creating matchup features...")
    matchups = create_matchup_features(games_df, team_log)
    
    # Add derived features
    print("  Adding derived features...")
    
    # Run differential features
    matchups['home_run_diff_L10'] = matchups['home_runs_scored_L10'] - matchups['home_runs_allowed_L10']
    matchups['away_run_diff_L10'] = matchups['away_runs_scored_L10'] - matchups['away_runs_allowed_L10']
    matchups['run_diff_advantage'] = matchups['home_run_diff_L10'] - matchups['away_run_diff_L10']
    
    # Win percentage differential
    matchups['win_pct_diff_L10'] = matchups['home_win_pct_L10'] - matchups['away_win_pct_L10']
    
    # Pythagorean differential
    matchups['pyth_diff'] = matchups['home_pyth_win_pct'] - matchups['away_pyth_win_pct']
    
    # Expected total runs
    matchups['expected_total'] = (
        matchups['home_runs_scored_L10'] + matchups['away_runs_scored_L10']
    )
    
    return matchups


def fetch_pitcher_season_stats(games_df: pd.DataFrame) -> pd.DataFrame:
    """Fetch season stats for all pitchers in the dataset."""
    
    print("\nFetching pitcher statistics...")
    
    # Get unique pitcher IDs
    home_pitchers = games_df[['home_starter_id', 'season']].dropna().drop_duplicates()
    home_pitchers.columns = ['pitcher_id', 'season']
    
    away_pitchers = games_df[['away_starter_id', 'season']].dropna().drop_duplicates()
    away_pitchers.columns = ['pitcher_id', 'season']
    
    all_pitchers = pd.concat([home_pitchers, away_pitchers]).drop_duplicates()
    all_pitchers['pitcher_id'] = all_pitchers['pitcher_id'].astype(int)
    
    print(f"  Found {len(all_pitchers)} unique pitcher-seasons")
    
    # Check for cache
    cache_file = f"{OUTPUT_DIR}/pitcher_stats.csv"
    if os.path.exists(cache_file):
        print("  Loading cached pitcher stats...")
        return pd.read_csv(cache_file)
    
    # Fetch stats
    pitcher_stats = []
    total_pitchers = len(all_pitchers)
    for i, (_, row) in enumerate(all_pitchers.iterrows()):
        stats = get_pitcher_stats(int(row['pitcher_id']), int(row['season']))
        if stats:
            stats['pitcher_id'] = int(row['pitcher_id'])
            stats['season'] = int(row['season'])
            pitcher_stats.append(stats)
        
        if (i + 1) % 100 == 0:
            print(f"  Processed {i+1}/{total_pitchers} pitchers...")
        
        time.sleep(0.1)  # Rate limiting
    
    if pitcher_stats:
        pitcher_df = pd.DataFrame(pitcher_stats)
        pitcher_df.to_csv(cache_file, index=False)
        return pitcher_df
    
    return pd.DataFrame()


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def main():
    """Main execution function."""
    
    print("="*60)
    print("MLB HISTORICAL DATA FETCHER")
    print("="*60)
    print(f"Seasons: {SEASONS}")
    print(f"Output directory: {OUTPUT_DIR}")
    print()
    
    # Fetch all games
    games_df = fetch_all_games(SEASONS)
    
    if games_df.empty:
        print("No games fetched. Exiting.")
        return
    
    print(f"\nTotal games fetched: {len(games_df)}")
    print(f"Date range: {games_df['game_date'].min()} to {games_df['game_date'].max()}")
    
    # Save raw games
    games_df.to_csv(f"{OUTPUT_DIR}/all_games_raw.csv", index=False)
    print(f"\nSaved raw games to {OUTPUT_DIR}/all_games_raw.csv")
    
    # Build prediction dataset
    prediction_df = build_prediction_dataset(games_df)
    
    # Fetch pitcher stats
    pitcher_df = fetch_pitcher_season_stats(games_df)
    
    if not pitcher_df.empty:
        # Merge pitcher stats
        print("\nMerging pitcher statistics...")
        
        # Home pitcher
        home_pitcher_cols = ['pitcher_id', 'season', 'era', 'whip', 'strikeOuts', 
                           'inningsPitched', 'wins', 'losses', 'homeRuns']
        if all(col in pitcher_df.columns for col in home_pitcher_cols):
            home_pitcher_stats = pitcher_df[home_pitcher_cols].rename(
                columns=lambda x: f'home_starter_{x}' if x not in ['pitcher_id', 'season'] else x
            )
            prediction_df = prediction_df.merge(
                home_pitcher_stats,
                left_on=['home_starter_id', 'season'],
                right_on=['pitcher_id', 'season'],
                how='left'
            ).drop(columns=['pitcher_id'], errors='ignore')
        
        # Away pitcher
        if all(col in pitcher_df.columns for col in home_pitcher_cols):
            away_pitcher_stats = pitcher_df[home_pitcher_cols].rename(
                columns=lambda x: f'away_starter_{x}' if x not in ['pitcher_id', 'season'] else x
            )
            prediction_df = prediction_df.merge(
                away_pitcher_stats,
                left_on=['away_starter_id', 'season'],
                right_on=['pitcher_id', 'season'],
                how='left'
            ).drop(columns=['pitcher_id'], errors='ignore')
    
    # Save final dataset
    output_file = f"{OUTPUT_DIR}/mlb_matchups_features.csv"
    prediction_df.to_csv(output_file, index=False)
    print(f"\nSaved prediction dataset to {output_file}")
    
    # Print summary stats
    print("\n" + "="*60)
    print("DATASET SUMMARY")
    print("="*60)
    print(f"Total games: {len(prediction_df)}")
    print(f"Seasons: {prediction_df['season'].unique()}")
    print(f"Features: {len(prediction_df.columns)}")
    print(f"\nTarget distribution (home_win):")
    print(prediction_df['home_win'].value_counts(normalize=True))
    print(f"\nMean total runs: {prediction_df['total_runs'].mean():.2f}")
    print(f"Games with >8.5 runs: {(prediction_df['total_runs'] > 8.5).mean()*100:.1f}%")
    
    # Show feature list
    print(f"\nFeature columns:")
    for col in sorted(prediction_df.columns):
        print(f"  - {col}")
    
    return prediction_df


if __name__ == "__main__":
    df = main()