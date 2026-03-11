import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score



def predict_latest_season(df):
    """Based on a random forest classifier, generate likely finals contenders.

    Train a random forest classifier on all historical seasons, and generate
    deep playoff probability predictions for the most recent season.
    A deep playoff run is defined as reaching the Conference Finals, Finals, or
    winning the championship.

    Args:
        df: dataframe containing:
            - season
            - team_name (str)
            - playoff_stage (str)
            - engineered team efficiency features
            - engineered top player features

    Returns:
        DataFrame for the most recent season containing:
            - team_name
            - predicted_probability (probability of deep playoff run)
            - actual (binary ground truth)

    Notes:
        The model is trained on all seasons except the most recent one.
        The most recent season's regular season stats are appended with a
        fake season label ("2024-25-reg") and playoff_stage="Unknown"
        (which maps to deep_playoff_run=0) so they enrich training without
        ever becoming the prediction target.
    """
    df = df.dropna().copy()
    df["deep_playoff_run"] = (
        df["playoff_stage"].isin(["Conference Finals", "Finals", "Champion"])
    ).astype(int)

    team_features = [
    # Team-level: API-provided
    "efg_pct",
    "ts_pct",
    "off_rating",
    "def_rating",
    "pace",
    "pie",
    "oreb_pct",
    "dreb_pct",
    "tm_tov_pct",
    "ast_to",

    # Team-level Engineered
    "three_point_rate",
    "free_throw_rate",

    # Top player aggregates
    "top1_pie",
    "top1_def_rating",
    "top1_net_rating",
    "top1_efg_pct",
    "top1_usg_pct"
    ]


    # Exclude fake regular season rows from being the prediction target
    latest_season = sorted(
        [s for s in df["season"].unique() if "-reg" not in s]
    )[-1]

    print(f"\nPredicting Season: {latest_season}")

    train_idx = df["season"] != latest_season
    test_idx = df["season"] == latest_season

    X_train = df.loc[train_idx, team_features]
    y_train = df.loc[train_idx, "deep_playoff_run"]
    X_test = df.loc[test_idx, team_features]
    y_test = df.loc[test_idx, "deep_playoff_run"]

    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        random_state=1
    )
    model.fit(X_train, y_train)
    probs = model.predict_proba(X_test)[:, 1]

    results = df.loc[test_idx, ["team_name"]].copy()
    results["predicted_probability"] = probs
    results["actual"] = y_test.values
    results = results.sort_values("predicted_probability", ascending=False)

    print("\nPredicted Conference Finals Teams (Top 4):\n")
    print(results.head(4))

    print("\nActual Conference Finals Teams:\n")
    print(results[results["actual"] == 1])

    auc = roc_auc_score(y_test, probs)
    print(f"\nROC-AUC for {latest_season}: {auc:.3f}")



    return results


def main():
    """Main function to predict behaviours based on modeling dataset.

    Loads engineered team and top player feature datasets, merges them,
    appends the latest regular season stats with a fake season label for
    enriched training, generates predictions for the most recent NBA season,
    and saves results to disk.

    Output:
        data/results/latest_season_predictions.csv
    """
    playoff_df = pd.read_csv("data/features/playoff_team_features.csv")
    reg_df = pd.read_csv("data/features/reg_team_features.csv")
    playoff_top1 = pd.read_csv("data/features/playoff_top1_features.csv")
    reg_top1 = pd.read_csv("data/features/reg_top1_features.csv")

    # Merge historic playoff top1 player features into playoff team rows
    playoff_df = playoff_df.merge(playoff_top1, on=["team_id","season"],how="left")


    # Build latest regular sesason rows
    latest_reg = reg_df[reg_df["season"] == reg_df["season"].max()].copy()
    latest_reg["playoff_stage"] = "Unknown"   # maps to deep_playoff_run = 0
    latest_reg["season"] = "2024-25-reg"      # excluded from latest_season filter

    # Merge current reg season top1 player features into latest_reg rows
    latest_reg_top1 = reg_top1[reg_top1["season"] == reg_top1["season"].max()].copy()
    latest_reg_top1["season"] = "2024-25-reg" 
    latest_reg = latest_reg.merge(latest_reg_top1,on=["team_id","season"],how="left")


    combined_df = pd.concat([playoff_df, latest_reg], ignore_index=True)


    result_path = "data/results"
    os.makedirs(result_path, exist_ok=True)

    team_pred = predict_latest_season(combined_df)

    output_path = os.path.join(result_path, "latest_season_predictions.csv")
    team_pred.to_csv(output_path, index=False)

    


if __name__ == "__main__":
    main()