import pandas as pd
from nba_api.stats.endpoints import LeagueDashPlayerStats, LeagueDashTeamStats, LeagueGameLog
import time
import os
from nba_api.stats.library.http import NBAStatsHTTP
from requests import Session

# Create a custom session with browser-like headers
custom_session = Session()
custom_session.headers.update({
    "Host": "stats.nba.com",
    "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                  "AppleWebKit/537.36 (KHTML, like Gecko) "
                  "Chrome/120.0.0.0 Safari/537.36",
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Referer": "https://www.nba.com/",
    "Origin": "https://www.nba.com",
    "Connection": "keep-alive",
})

# Inject it into nba_api globally
NBAStatsHTTP._session = custom_session

os.makedirs("data/raw", exist_ok=True)


SEASONS = [
    '2015-16','2016-17','2017-18','2018-19','2019-20',
    '2020-21','2021-22','2022-23','2023-24','2024-25'
]

import random
import time
from requests.exceptions import ConnectionError

def safe_api_call(api_callable, max_retries=5):
    """
    Safely executes an nba_api endpoint with retry logic.
    """
    for attempt in range(max_retries):
        try:
            result = api_callable()
            time.sleep(random.uniform(5, 9))
            return result

        except ConnectionError as e:
            wait = 2 ** attempt  # exponential backoff
            print(f"Connection failed. Retry {attempt+1}/{max_retries} in {wait}s...")
            time.sleep(wait)

    raise Exception("Max retries exceeded. NBA API likely blocking requests.")


def fetch_player_stats(seasons, per_mode='PerGame',season_type='Playoffs',measure_type='Base'):
    """Fetches NBA player statistics for the specified seasons.

    Retrieves player statistics for either regular season or playoffs
    from the NBA Stats API using the LeagueDashPlayerStats endpoint.
    Can fetch standard per-game stats ('Base') or advanced metrics ('Advanced').

    Args:
        seasons: A sequence of season strings to fetch, e.g., ['2015-16', '2016-17'].
        per_mode: How to normalize stats. Options include 'PerGame', 'Totals', etc.
        season_type: Type of season to fetch. Options include 'Regular Season', 'Playoffs'.
        measure_type: Type of stats to fetch. 'Base' for standard stats, 'Advanced' for advanced metrics.

    Returns:
        A pandas DataFrame containing all player stats for the requested seasons.
        Includes a 'SEASON' column indicating the corresponding season.

    Raises:
        Exception: If the NBA API request fails for a given season.
    """
    all_playoffs = []

    for season in seasons:
        df = safe_api_call(lambda: LeagueDashPlayerStats(
            season=season,
            season_type_all_star=season_type,
            per_mode_detailed=per_mode,
            measure_type_detailed_defense=measure_type
        ).get_data_frames()[0])
        df['SEASON'] = season
        all_playoffs.append(df)
        time.sleep(1)
    player_stats = pd.concat(all_playoffs,ignore_index=True)

    return player_stats

def fetch_team_stats(seasons,per_mode='PerGame',season_type='Playoffs'):
    """Fetches NBA team statistics for the specified seasons.

    Retrieves team statistics for either regular season or playoffs
    from the NBA Stats API using the LeagueDashTeamStats endpoint.

    Args:
        seasons: A sequence of season strings to fetch, e.g., ['2015-16', '2016-17'].
        per_mode: How to normalize stats. Options include 'PerGame', 'Totals', etc.
        season_type: Type of season to fetch. Options include 'Regular Season', 'Playoffs'.

    Returns:
        A pandas DataFrame containing all team stats for the requested seasons.
        Includes a 'SEASON' column indicating the corresponding season.

    Raises:
        Exception: If the NBA API request fails for a given season.
    """
    all_playoffs = []

    for season in seasons:
        df = safe_api_call(lambda: LeagueDashTeamStats(
            season=season,
            season_type_all_star=season_type,
            per_mode_detailed=per_mode
        ).get_data_frames()[0])
        df['SEASON'] = season
        all_playoffs.append(df)
        time.sleep(1)
    team_stats = pd.concat(all_playoffs,ignore_index=True)

    return team_stats


def fetch_game_logs(seasons, player_or_team='P', season_type='Playoffs'):
    """Fetches NBA game logs for players or teams for the specified seasons.

    Retrieves game-by-game statistics for either players ('P') or teams ('T')
    from the NBA Stats API using the LeagueGameLog endpoint. Supports
    both regular season and playoffs.

    Args:
        seasons: A sequence of season strings to fetch, e.g., ['2015-16', '2016-17'].
        player_or_team: 'P' for player game logs, 'T' for team game logs.
        season_type: Type of season to fetch. Options include 'Regular Season' or 'Playoffs'.

    Returns:
        A pandas DataFrame containing all game logs for the requested seasons.
        Includes a 'SEASON' column indicating the corresponding season.

    Raises:
        Exception: If the NBA API request fails for a given season.
    """
    all_playoffs = []

    for season in seasons:
        df = safe_api_call(lambda: LeagueGameLog(
            season=season,
            season_type_all_star=season_type,
            player_or_team_abbreviation=player_or_team
        ).get_data_frames()[0])
        df['SEASON'] = season
        all_playoffs.append(df)
        time.sleep(1)
    log_stats = pd.concat(all_playoffs,ignore_index=True)

    return log_stats

def main():
    """Main function to fetch and save NBA playoff data.

    This function orchestrates the data collection pipeline for NBA playoffs
    from 2015-16 through 2024-25. It performs the following tasks:

    1. Fetches player per-game statistics and saves them to CSV.
    2. Fetches player advanced statistics and saves them to CSV.
    3. Fetches team statistics and saves them to CSV.
    4. Fetches player game logs and saves them to CSV.

    Each dataset is saved to the "data/raw/" folder with descriptive filenames.

    Notes:
        - Requires an internet connection to fetch data from the NBA Stats API.
        - Includes a 1-second pause between API calls to avoid rate limits.

    Returns:
        None

    Raises:
        Exception: If any of the API requests fail or if CSV saving fails.
    """
    player_pergame = fetch_player_stats(SEASONS,per_mode='PerGame')
    player_pergame.to_csv("data/raw/player_per_game_playoffs_2015_2025.csv",index=False)

    player_advanced = fetch_player_stats(SEASONS,measure_type='Advanced',per_mode='PerGame')
    player_advanced.to_csv("data/raw/player_advanced_playoffs_2015_2025.csv",index=False)

    playoff_team_stats = fetch_team_stats(SEASONS)
    playoff_team_stats.to_csv("data/raw/playoff_team_stats_2015_2025.csv",index=False)

    regular_team_stats = fetch_team_stats(SEASONS,season_type="Regular Season")
    regular_team_stats.to_csv("data/raw/regular_team_stats_2015_2025.csv",index=False)


    game_logs = fetch_game_logs(SEASONS,player_or_team='P')
    game_logs.to_csv("data/raw/game_logs_2015_2025.csv",index=False)

if __name__ == "__main__":
    main()




