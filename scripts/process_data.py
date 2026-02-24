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
                .str.replace(" ","_")
            )

            processed_file_path = os.path.join(processed_path,filename)
            df.to_csv(processed_file_path,index=False)

            processed_files[filename] = df
    
    return processed_files

def merge_player_stats(box_stats, advanced_stats):
    pass


def main():
    """Main function to clean and standardize raw NBA datasets.

    This function orchestrates the data cleaning pipeline. It performs
    the following tasks:

    1. Creates the "data/processed/" directory if it does not exist.
    2. Iterates through all CSV files in "data/raw/".
    3. Standardizes column names using `clean_column_names`.
    4. Saves the cleaned dataset to "data/processed/" with a 
       'clean_' prefix added to the filename.

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

    files = clean_column_names(raw_path,processed_path)

    merge_player_stats(files['player_per_game_playoffs_2015_2025.csv'],
                       files['player_advanced_playoffs_2015_2025.csv'])

    # print(files['game_logs_2015_2025.csv'].head())

   


if __name__ == "__main__":
    main()