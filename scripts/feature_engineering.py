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

def engineer_player_features(df):
    """
    """
    df = df.copy

    



    return df

def main():
    """Main function to create dataset that will be used for modeling.

    This function orchestrates the feature engineering pipeline. It performs the following tasks:
    
    1. Assigns all files a name in a dictionary to be called later with the year removed for ease
    2. Creates the playoff class to describe a team's performance within the playoffs and how far 
        they proceed
    3. Creates intereptable team-level features
    4. As a result takes the large, messy team_stats file and cleans it and only keeps necessary data
        for modeling. 

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


    playoff_team_data = create_playoff_stage(playoff_team_data)
    playoff_team_data = engineer_team_features(playoff_team_data)


    reg_team_data = engineer_team_features(reg_team_data)

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
    

    numeric_cols = playoff_team_data[team_columns + ["playoff_stage"]].select_dtypes(include="number").columns
    corr_matrix = playoff_team_data[numeric_cols].corr().round(2)
    corr_path = os.path.join(feature_path,"correlation.csv")
    corr_matrix.to_csv(corr_path,index=False)
        

    playoff_team_model = playoff_team_data[team_columns + ["playoff_stage"]]
    playoff_output_path = os.path.join(feature_path,"playoff_team_modeling_dataset.csv")
    playoff_team_model.to_csv(playoff_output_path,index=False)

    reg_team_model = reg_team_data[team_columns]
    reg_output_path = os.path.join(feature_path,"reg_team_modeling_dataset.csv")
    reg_team_model.to_csv(reg_output_path,index=False)






if __name__ == "__main__":
    main()