import pandas as pd
import os 

raw_path = "data/raw"
processed_path = "data/processed"

def clean_column_names(df):
    df.columns = (
        df.columns
        .str.lower()
        .str.strip()
        .str.replace(" ","_")
    )

    return df

def main():
    os.makedirs(processed_path, exist_ok=True)

    for filename in os.listdir(raw_path):
        if filename.endswith(".csv"):
            raw_file_path = os.path.join(raw_path, filename)

            df = pd.read_csv(raw_file_path)

            df = clean_column_names(df)

            new_filename = f"processed_{filename}"
            processed_file_path = os.path.join(processed_path, new_filename)

            df.to_csv(processed_file_path, index=False)


if __name__ == "__main__":
    main()

