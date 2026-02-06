from pybaseball import schedule_and_record, team_batting, team_pitching, playerid_lookup, statcast_pitcher

# Get team schedule/results for a season
games_2024 = schedule_and_record(2024, 'CHC')  # Cubs example

# Get team batting stats
batting_2024 = team_batting(2024)

# Get team pitching stats
pitching_2024 = team_pitching(2024)