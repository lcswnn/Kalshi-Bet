"""
SAMPLE DATA GENERATOR
====================
Generate synthetic NBA game data for testing the backtesting framework.

This creates realistic game data with:
- Team strength variations
- Home court advantage
- Seasonal trends
- Realistic score distributions
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import random

# NBA Teams (2024-25 season)
NBA_TEAMS = [
    "Boston Celtics", "Milwaukee Bucks", "Philadelphia 76ers", "Miami Heat",
    "Cleveland Cavaliers", "New York Knicks", "Brooklyn Nets", "Atlanta Hawks",
    "Toronto Raptors", "Chicago Bulls", "Indiana Pacers", "Charlotte Hornets",
    "Washington Wizards", "Orlando Magic", "Detroit Pistons",
    "Denver Nuggets", "Minnesota Timberwolves", "Oklahoma City Thunder", 
    "Phoenix Suns", "LA Clippers", "Dallas Mavericks", "Sacramento Kings",
    "Golden State Warriors", "LA Lakers", "Houston Rockets", "Memphis Grizzlies",
    "New Orleans Pelicans", "San Antonio Spurs", "Utah Jazz", "Portland Trail Blazers"
]

# Team strength tiers (1=best, 5=worst)
TEAM_TIERS = {
    "Boston Celtics": 1, "Milwaukee Bucks": 1, "Denver Nuggets": 1,
    "Phoenix Suns": 2, "LA Clippers": 2, "Philadelphia 76ers": 2,
    "Miami Heat": 2, "Dallas Mavericks": 2, "Minnesota Timberwolves": 2,
    "Golden State Warriors": 3, "LA Lakers": 3, "Cleveland Cavaliers": 3,
    "Sacramento Kings": 3, "New York Knicks": 3, "Oklahoma City Thunder": 1,
    "Memphis Grizzlies": 3, "Atlanta Hawks": 4, "Brooklyn Nets": 4,
    "Toronto Raptors": 4, "Chicago Bulls": 4, "Indiana Pacers": 4,
    "Houston Rockets": 4, "New Orleans Pelicans": 4, "Charlotte Hornets": 5,
    "Washington Wizards": 5, "Orlando Magic": 4, "Detroit Pistons": 5,
    "San Antonio Spurs": 5, "Utah Jazz": 5, "Portland Trail Blazers": 5
}

def generate_team_strength():
    """Generate base strength for each team."""
    strengths = {}
    for team, tier in TEAM_TIERS.items():
        # Tier 1: 1550-1650, Tier 2: 1500-1600, etc.
        base = 1700 - (tier * 100)
        variation = 50
        strengths[team] = np.random.uniform(base - variation, base + variation)
    return strengths

def simulate_game(home_team, away_team, team_strengths, home_advantage=100):
    """
    Simulate a single game outcome.
    
    Returns:
        tuple: (home_won, home_score, away_score)
    """
    # Calculate win probabilities using Elo-like formula
    home_strength = team_strengths[home_team] + home_advantage
    away_strength = team_strengths[away_team]
    
    expected_home = 1 / (1 + 10 ** ((away_strength - home_strength) / 400))
    
    # Determine winner
    home_won = np.random.random() < expected_home
    
    # Generate realistic scores
    # NBA average: ~110 points per team
    base_score = 110
    
    if home_won:
        home_score = np.random.normal(115, 10)
        away_score = np.random.normal(105, 10)
    else:
        home_score = np.random.normal(105, 10)
        away_score = np.random.normal(115, 10)
    
    # Ensure scores are reasonable
    home_score = max(85, min(145, int(home_score)))
    away_score = max(85, min(145, int(away_score)))
    
    # Ensure winner has higher score
    if home_won and home_score <= away_score:
        home_score = away_score + np.random.randint(1, 10)
    elif not home_won and away_score <= home_score:
        away_score = home_score + np.random.randint(1, 10)
    
    return home_won, home_score, away_score

def generate_season_schedule(teams, games_per_team=82, season_start_date=datetime(2023, 10, 15)):
    """
    Generate a realistic NBA season schedule.
    
    Args:
        teams: List of team names
        games_per_team: Number of games per team (default: 82 for NBA)
        season_start_date: Start date of season
    
    Returns:
        List of games as dictionaries
    """
    games = []
    team_game_count = {team: 0 for team in teams}
    current_date = season_start_date
    
    # Generate games over ~6 months (180 days)
    season_days = 180
    
    while min(team_game_count.values()) < games_per_team:
        # Each day, schedule multiple games
        daily_games = np.random.randint(5, 15)  # 5-15 games per day
        
        available_teams = [t for t in teams if team_game_count[t] < games_per_team]
        
        if len(available_teams) < 2:
            break
        
        for _ in range(daily_games):
            if len(available_teams) < 2:
                break
            
            # Select home and away teams
            home_team = np.random.choice(available_teams)
            available_away = [t for t in available_teams if t != home_team]
            
            if not available_away:
                break
            
            away_team = np.random.choice(available_away)
            
            # Schedule game
            games.append({
                'date': current_date,
                'home_team': home_team,
                'away_team': away_team
            })
            
            # Update counts
            team_game_count[home_team] += 1
            team_game_count[away_team] += 1
            
            # Remove teams if they've hit game limit
            if team_game_count[home_team] >= games_per_team:
                available_teams.remove(home_team)
            if away_team in available_teams and team_game_count[away_team] >= games_per_team:
                available_teams.remove(away_team)
        
        # Move to next day
        current_date += timedelta(days=1)
    
    return games

def generate_nba_season_data(output_file='sample_nba_games.csv', 
                             seasons=3, 
                             games_per_team=82):
    """
    Generate multiple seasons of NBA game data.
    
    Args:
        output_file: CSV file to save data
        seasons: Number of seasons to generate
        games_per_team: Games per team per season
    """
    print(f"Generating {seasons} seasons of NBA data...")
    print(f"Teams: {len(NBA_TEAMS)}")
    print(f"Games per team per season: {games_per_team}")
    
    all_games = []
    team_strengths = generate_team_strength()
    
    for season in range(seasons):
        season_start = datetime(2021 + season, 10, 15)
        print(f"\nSeason {season + 1}: {season_start.year}-{season_start.year + 1}")
        
        # Generate schedule
        schedule = generate_season_schedule(NBA_TEAMS, games_per_team, season_start)
        print(f"  Scheduled {len(schedule)} games")
        
        # Simulate games
        for game in schedule:
            home_won, home_score, away_score = simulate_game(
                game['home_team'],
                game['away_team'],
                team_strengths
            )
            
            all_games.append({
                'date': game['date'],
                'home_team': game['home_team'],
                'away_team': game['away_team'],
                'home_score': home_score,
                'away_score': away_score,
                'home_won': home_won
            })
        
        # Add some randomness to team strengths for next season (teams improve/decline)
        for team in team_strengths:
            change = np.random.normal(0, 30)  # Teams can improve/decline
            team_strengths[team] = max(1200, min(1800, team_strengths[team] + change))
    
    # Create DataFrame
    df = pd.DataFrame(all_games)
    df = df.sort_values('date').reset_index(drop=True)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    
    print(f"\n✅ Generated {len(df)} total games")
    print(f"📁 Saved to {output_file}")
    print(f"\nData summary:")
    print(f"  Date range: {df['date'].min()} to {df['date'].max()}")
    print(f"  Home win rate: {df['home_won'].mean()*100:.1f}%")
    print(f"  Avg home score: {df['home_score'].mean():.1f}")
    print(f"  Avg away score: {df['away_score'].mean():.1f}")
    
    return df

if __name__ == "__main__":
    # Generate sample data
    generate_nba_season_data(
        output_file='sample_nba_games.csv',
        seasons=3,
        games_per_team=82
    )