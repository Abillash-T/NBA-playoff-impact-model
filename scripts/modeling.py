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

    Returns:
        DataFrame for the most recent season containing:
            - team_name
            - predicted_probability (probability of deep playoff run)
            - actual (binary ground truth)

    Notes:
        The model is trained using all seasons except the most recent one to simulate a
        real-world forecasting scenario. The most recent season's regular season stats
        are appended to the training set with a fake season label to ensure they are
        included in training without being treated as the prediction target.
    """
    df = df.dropna().copy()
    df["deep_playoff_run"] = (
        df["playoff_stage"].isin(["Conference Finals", "Finals", "Champion"])
    ).astype(int)

    team_features = [
        "effective_field_goal_pct",
        "turnover_ratio",
        "offensive_rebound_rate",
        "assist_turnover_ratio",
        "free_throw_rate",
        "net_rating_proxy"
    ]

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

    Load engineered team feature datasets, append the latest regular season
    stats to the playoff dataset for enriched training, generate predictions
    for the most recent NBA season, and save results to disk.

    Output:
        data/results/latest_season_predictions.csv
    """
    playoff_df = pd.read_csv("data/features/playoff_team_modeling_dataset.csv")
    reg_df = pd.read_csv("data/features/reg_team_modeling_dataset.csv")

    # Get latest regular season stats for all 30 teams
    # Assign fake playoff_stage so dropna() keeps them, and fake season
    # so they're never treated as the prediction target
    latest_reg = reg_df[reg_df["season"] == reg_df["season"].max()].copy()
    latest_reg["playoff_stage"] = "Unknown"   # maps to deep_playoff_run = 0
    latest_reg["season"] = "2024-25-reg"      # excluded from latest_season filter

    # Combine with playoff dataset
    combined_df = pd.concat([playoff_df, latest_reg], ignore_index=True)

    result_path = "data/results"
    os.makedirs(result_path, exist_ok=True)

    team_pred = predict_latest_season(combined_df)

    output_path = os.path.join(result_path, "latest_season_predictions.csv")
    team_pred.to_csv(output_path, index=False)


if __name__ == "__main__":
    main()