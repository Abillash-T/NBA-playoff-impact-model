import pandas as pd
import os 

def load_processed_data(processed_path):
    dfs = {}

    for filename in os.listdir(processed_path):
        if filename.__contains__("clean_"):
            name = filename.replace("clean_", "")
            dfs[name] = pd.read_csv(
                os.path.join(processed_path, filename)
            )
    print(dfs)
    return dfs


def main():
    processed_path = "data/processed"
    load_processed_data(processed_path)


if __name__ == '__main__':
    main()