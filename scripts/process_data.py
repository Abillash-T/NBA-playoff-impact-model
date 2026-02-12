import pandas as pd
import os 


def clean_column_names(df):
    """Standardizes column names in a DataFrame.

    Converts all column names to lowercase, removes leading/trailing
    whitespace, and replaces spaces with underscores. This ensures
    consistency across datasets and prevents merge/key errors caused
    by inconsistent naming.

    Args:
        df: A pandas DataFrame whose columns need to be standardized.

    Returns:
        A pandas DataFrame with cleaned column names.

    Raises:
        AttributeError: If the input is not a valid pandas DataFrame.
    """
    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
        .str.replace(" ","_")
    )

    return df

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
        - Does not perform feature engineering or dataset merging.
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

    os.makedirs(processed_path, exist_ok=True)

    for filename in os.listdir(raw_path):
        if filename.endswith(".csv"):
            raw_file_path = os.path.join(raw_path, filename)

            df = pd.read_csv(raw_file_path)

            df = clean_column_names(df)

            new_filename = f"clean_{filename}"
            processed_file_path = os.path.join(processed_path, new_filename)

            df.to_csv(processed_file_path, index=False)


if __name__ == "__main__":
    main()
