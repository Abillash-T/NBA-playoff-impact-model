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

def engineer_features(df):
    """Creates intereptable team-level features

    creates new columns for more detailed stats that can be used to describe team performance
    or compare and contrast with other teams.

    Args:
        df: a dataframe consisting of team stats

    Returns:
        the input df with the new stat columns added
    """
    df = df.copy()



    # EFFICENCY METRICS
    df["effective_field_goal_pct"] =(df["fgm"] + (0.5 * df["fg3m"]))/ df["fga"]
    df["turnover_ratio"] = (
        df["tov"] * 100) / (df["fga"] + (df["fta"] * 0.44) + df["ast"] + df["tov"])
    df["offensive_rebound_rate"] = df["oreb"] / df["reb"] 

    df["assist_turnover_ratio"] = df["ast"] / df["tov"]


    # SHOOTING PROFILE METRICS
    df["three_point_rate"] = df["fg3a"] / df["fga"]
    df["free_throw_rate"] = df["fta"] / df["fga"]

    # TEAM CONTROL METRICS
    df["rebound_rate"] = df["reb"] / df["min"]
    df["defensive_rebound_rate"] = df["dreb"] / df["reb"]

    # DEFFENSIVE
    df["net_rating_proxy"] = df["plus_minus"]


    return df

def clean_columns(df):
    """Remove "_rank" from team stats as not relevant for analysis that will be performed later.

    Finds all column names that end in "_rank", if they exist, they are removed from the df

    Args:
        df: dataframe input
    
    Returns:
        df with columns removed
    """
    df = df.loc[:, ~df.columns.str.endswith("_rank")]

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
    processed_path = "data/processed"
    feature_path = "data/features"

    os.makedirs(feature_path,exist_ok=True)

    files = run_processing()

    team_data = files['playoff_team_stats']


    team_data = clean_columns(team_data)
    team_data = create_playoff_stage(team_data)
    team_data = engineer_features(team_data)

    selected_columns = [
        "team_id",
        "team_name",
        "season",
        "playoff_stage",
        "effective_field_goal_pct",
        "turnover_ratio",
        "offensive_rebound_rate",
        "assist_turnover_ratio",
        #"fg3_pct",
        "three_point_rate",
        "free_throw_rate",
        "rebound_rate",
        "defensive_rebound_rate",
        "net_rating_proxy",
        #"w_pct"
    ]

    numeric_cols = team_data[selected_columns].select_dtypes(include="number").columns
    corr_matrix = team_data[numeric_cols].corr().round(2)
    corr_path = os.path.join(feature_path,"correlation.csv")
    corr_matrix.to_csv(corr_path,index=False)


        

    team_model = team_data[selected_columns]


    output_path = os.path.join(feature_path,"team_modeling_dataset.csv")

    team_model.to_csv(output_path,index=False)




if __name__ == "__main__":
    main()