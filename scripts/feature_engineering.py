import pandas as pd
import os
from process_data import main as run_processing



def create_playoff_stage(df):
    """Creates multiclass playoff stage label based on playoff wins.

    Takes in team stats from 2015-2025, and depending on how many wins they can get in a season,
    determines their placement in the playoffs. i.e whether a team makes it to the finals or makes
    a first round exit.

    Args:
        df: A dataframe that contains all team stats for every season from 2015-2025

    Returns:
        A df with the "playoff_stage" column added to the input.
        NaN values (teams with no playoff wins recorded) are replaced with None.
    """

    labels = [
        "First Round",
        "Second Round",
        "Conference Finals",
        "Finals",
        "Champion"
    ]

    df["playoff_stage"] = pd.cut(
        df['w'],
        bins=[-1,3,7,11,15,20],
        labels=labels
    )

    df["playoff_stage"] = df["playoff_stage"].astype(str).replace("nan", None)

    return df

def engineer_team_features(df):
    """Creates interpretable team-level features

    Derives shooting profile metrics not directly available from the API.

    Args:
        df: a dataframe consisting 'fg3a', 'fga', 'fta' columns.

    Returns:
        df with 'three_point_rate' and 'free_throw_rate' columns added
    """
    df = df.copy()

    # Shooting profile metrics
    df["three_point_rate"] = df["fg3a"] / df["fga"]
    df["free_throw_rate"] = df["fta"] / df["fga"]


    return df

def engineer_best_player(df):
    """Selects the top player per team per season by PIE and returns their stats.

    For each (team_id, season) group, ranks players by PIE (Player Impact
    Estimate) and keeps only the highest-ranked player. Their stats are
    prefixed with 'top1_' to distinguish them from team-level features.

    Args:
        df: DataFrame of player stats containing 'team_id', 'team_abbreviation',
            'season', 'pie', and the stat columns listed in agg_cols.

    Returns:
        DataFrame with one row per (team_id, season) containing the top
        player's stats, prefixed with 'top1_'.
    """
    df = df.copy()

    agg_cols = [
        "pie",
        "ts_pct",
        "efg_pct",
        "usg_pct",
        "ast_to",
        "stl",
        "blk",
        "def_rating",
        "net_rating",
        "oreb_pct",
        "dreb_pct"
    ]

    df = df.dropna(subset=["pie"]).copy()

    df["pie_rank"] = (
        df.groupby(["team_id","season"])["pie"]
        .rank(method="first",ascending=False)
    )

    top_players = df[df["pie_rank"] <= 1]

    best = (
        top_players
        .groupby(["team_id", "team_abbreviation", "season"])[agg_cols]
        .first()
        .reset_index()
        .rename(columns={col: f"top1_{col}" for col in agg_cols})
    )

    return best


def main():
    """Main function to create dataset that will be used for modeling.

    This function orchestrates the feature engineering pipeline. It performs the following tasks:
    
    1. Loads all cleaned files from process_data.
    2. Creates the playoff_stage label from playoff win totals.
    3. Engineers team-level features (shooting rates etc.).
    4. Selects the top player features (by PIE) from both playoff and
       regular season player data and saves them as separate files.
    5. Saves all outputs to data/features/.

    This script assumes that process_data has been run, and there are existing files in the processed_path
    directory. 

    Notes:
        - Designed to be run independently of other pipeline stages.
        - Each dataset is processed independently for modularity.

    Returns:
        None

    Raises:
        FileNotFoundError: If the raw data directory does not exist.
        Exception: If a CSV file cannot be read or written.
    """
    feature_path = "data/features"

    os.makedirs(feature_path,exist_ok=True)

    files = run_processing()

    playoff_team_data = files['playoff_team_stats']
    reg_team_data = files['reg_team_stats']
    playoff_player_data = files['playoff_player_stats']
    reg_player_data = files['reg_player_stats']



    # Labels for playoffs
    playoff_team_data = create_playoff_stage(playoff_team_data)

    # Team-level features
    playoff_team_data = engineer_team_features(playoff_team_data)
    reg_team_data = engineer_team_features(reg_team_data)

    # Best player features
    playoff_top1 = engineer_best_player(playoff_player_data)
    reg_top1 = engineer_best_player(reg_player_data)


    team_columns = [
        "team_id",
        "team_name",
        "season",

        # API-provided
        "efg_pct",
        "ts_pct",
        "off_rating",
        "def_rating",
        "net_rating",
        "pace",
        "pie",
        "oreb_pct",
        "dreb_pct",
        "tm_tov_pct",
        "ast_to",

        # Engineered
        "three_point_rate",
        "free_throw_rate",
        
    ]
       
    player_columns = [
        "player_id",
        "player_name",
        "team_id",
        "team_abbreviation",
        "season",
        "ts_pct",
        "efg_pct",
        "usg_pct",
        "ast",
        "ast_to",
        "stl",
        "blk",
        "def_rating",
        "net_rating",
        "pie",          
        "oreb_pct",
        "dreb_pct",
    ]

    
    playoff_team_data[team_columns + ["playoff_stage"]].to_csv(
        os.path.join(feature_path,"playoff_team_features.csv"),index=False)

    reg_team_data[team_columns].to_csv(
        os.path.join(feature_path,"reg_team_features.csv"),index=False)


    #playoff_player_data[player_columns].to_csv(os.path.join(feature_path,"playoff_player_features.csv"),index=False)


    #reg_player_data[player_columns].to_csv(os.path.join(feature_path,"reg_player_features.csv"),index=False)

    playoff_top1.to_csv(os.path.join(feature_path,"playoff_top1_features.csv"),index=False)

    reg_top1.to_csv(os.path.join(feature_path,"reg_top1_features.csv"),index=False)

    



if __name__ == "__main__":
    main()