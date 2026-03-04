import pandas as pd
import os 


def clean_column_names(raw_path,processed_path):
    """Cleans and standardizes column names for all CSV files in a directory.

    Iterates through each CSV file in the specified raw data folder,
    converts column names to lowercase, removes leading/trailing
    whitespace, and replaces spaces with underscores. The cleaned
    DataFrames are stored in a dictionary keyed by filename to allow
    easy access in later stages of the pipeline.

    Args:
        raw_path: folder that includes base CSV files to be cleaned
        processed_path: output directory after cleaning files.

    Returns:
        A dictionary mapping each filename to its cleaned pandas DataFrame

    Raises:
        AttributeError: If the input is not a valid pandas DataFrame.
    """
    os.makedirs(processed_path, exist_ok=True)

    processed_files={}

    for filename in os.listdir(raw_path):
        if filename.endswith(".csv"):
            raw_file_path = os.path.join(raw_path,filename)

            df = pd.read_csv(raw_file_path)

            df.columns = (
                df.columns
                .str.lower()
                .str.strip()
                .str.replace(" ","_",regex=True)
            )

            clean_name = filename.replace('_2015_2025','')

            processed_file_path = os.path.join(processed_path,clean_name)
            df.to_csv(processed_file_path,index=False)

            clean_name = filename.replace('_2015_2025.csv', '')
            processed_files[clean_name] = df
    
    return processed_files

def merge_player_stats(box_stats, advanced_stats):
    """Merges Player per game and player advanced stats into one dataframe

    Merges the box_stats and advanced_stats csv's using the player id, team id, and season as keys
    to ensure no information is lost. Also keeps original files to ensure they can be used later.

    Args: 
        box_stats: file that includes player per game stats for the season
        advanced_stats: file that includes the advnaced player stats for the season

    Returns:
        The merged dataframe
    """
    merge_keys = ["player_id","team_id","season"]

    advanced_stats = advanced_stats.loc[
        :,~advanced_stats.columns.isin(box_stats.columns)
        | advanced_stats.columns.isin(merge_keys)
    ]

    df = pd.merge(
        box_stats,
        advanced_stats,
        on=merge_keys,
        how="inner"
    )

    return df



def main():
    """Main function to clean and standardize raw NBA datasets.

    This function orchestrates the data cleaning pipeline. It performs
    the following tasks:

    1. Creates the "data/processed/" directory if it does not exist.
    2. Iterates through all CSV files in "data/raw/".
    3. Standardizes column names using `clean_column_names`.
    4. Saves the cleaned dataset to "data/processed/" 
    5. It then merges the two player specific files together as a csv and saves it to 
       "data/processed/"

    This script assumes raw data has already been collected
    using `data_collection.py`.

    Notes:
        - Designed to be run independently of other pipeline stages.
        - Each dataset is processed independently for modularity.

    Returns:
        None

    Raises:
        FileNotFoundError: If the raw data directory does not exist.
        Exception: If a CSV file cannot be read or written.
    """
    raw_path = "data/raw"
    processed_path = "data/processed"


    files = clean_column_names(raw_path, processed_path) 


    combine = merge_player_stats(
        files['player_per_game_playoffs'],   # no year, no .csv
        files['player_advanced_playoffs']
    )

    combine_path = os.path.join(processed_path, "player_stats.csv")
    combine.to_csv(combine_path, index=False)

    files['player_stats'] = combine  # add clean key before returning
    return files    
    

if __name__ == "__main__":
    main()