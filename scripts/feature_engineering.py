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
    """
    conditions = [
        df['w']<= 3,                     #first round exit
        df['w'].between(4,7),            #second round exit
        df['w'].between(8,11),           #conference finals
        df['w'].between(12,15),          #finals
        df['w'] >= 16                    #champion
    ]

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

    return df

def engineer_team_features(df):
    """Creates intereptable team-level features

    creates new columns for extra stats that aren't already covered in the API to compare and 
    contrast with other teams.

    Args:
        df: a dataframe consisting of team stats

    Returns:
        the input df with the new stat columns added
    """
    df = df.copy()

    # SHOOTING PROFILE METRICS
    df["three_point_rate"] = df["fg3a"] / df["fga"]
    df["free_throw_rate"] = df["fta"] / df["fga"]


    return df

def engineer_top3_players(df,top_n = 3):
    """Aggregates top-N players per team per season into team-level features.
    
    For each (team_id, season) group, selects the top N players by PIE
    (Player Impact Estimate) and averages their stats. The resulting columns
    are prefixed with 'top3_' to distinguish them from team-level features.

    Called separately on playoff player data and regular season player data.


    Args:
        player_df: DataFrame of player stats. Must contain:
                   team_id, season, pie, and the stat columns listed below.
        top_n: Number of top players per team to aggregate (default: 3).

    Returns:
        DataFrame with one row per (team_id, season) with averaged top-N
        player stats, prefixed with 'top3_'.
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

    top_players = df[df["pie_rank"] <= top_n]

    top3_agg = (
        top_players
        .groupby(["team_id", "team_abbreviation", "season"])[agg_cols]
        .mean()
        .reset_index()
        .rename(columns={col: f"top3_{col}" for col in agg_cols})
    )

    return top3_agg

def main():
    """Main function to create dataset that will be used for modeling.

    This function orchestrates the feature engineering pipeline. It performs the following tasks:
    
    1. Loads all cleaned files from process_data.
    2. Creates the playoff_stage label from playoff win totals.
    3. Engineers team-level features (shooting rates etc.).
    4. Aggregates top-3 player features (by PIE) from both playoff and
       regular season player data and saves them as separate files.
       No merging is done here — modeling.py handles that at runtime,
       exactly as it does for team features.
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


    playoff_team_data = create_playoff_stage(playoff_team_data)

    playoff_team_data = engineer_team_features(playoff_team_data)
    reg_team_data = engineer_team_features(reg_team_data)

    playoff_top3 = engineer_top3_players(playoff_player_data)
    reg_top3 = engineer_top3_players(reg_player_data)


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
    #    "usage_efficiency"
]

    #numeric_cols = playoff_team_data[team_columns + ["playoff_stage"]].select_dtypes(include="number").columns
    #corr_matrix = playoff_team_data[numeric_cols].corr().round(2)
    #corr_path = os.path.join(feature_path,"correlation.csv")
    #corr_matrix.to_csv(corr_path,index=False)

    
        

    playoff_team_model = playoff_team_data[team_columns + ["playoff_stage"]]
    playoff_team_model.to_csv(os.path.join(feature_path,"playoff_team_features.csv"),index=False)

    reg_team_model = reg_team_data[team_columns]
    reg_team_model.to_csv(os.path.join(feature_path,"reg_team_features.csv"),index=False)

    playoff_player_model = playoff_player_data[player_columns]
    playoff_player_model.to_csv(os.path.join(feature_path,"playoff_player_features.csv"),index=False)

    reg_player_model = reg_player_data[player_columns]
    reg_player_model.to_csv(os.path.join(feature_path,"reg_player_features.csv"),index=False)

    playoff_top3.to_csv(os.path.join(feature_path,"playoff_top3_player_features.csv"),index=False)
    reg_top3.to_csv(os.path.join(feature_path,"reg_top3_player_features.csv"),index=False)

    



if __name__ == "__main__":
    main()