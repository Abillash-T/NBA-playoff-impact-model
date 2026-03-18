import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


TEAM_FEATURES = [
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


def predict_latest_season(df):
    """Runs LOSO validation on historical seasons and predicts the latest season.

    For each historical season, trains on all other seasons and prints the
    ROC-AUC. For the latest season, trains on all historical data and prints
    the top 4 predicted contenders, actual Conference Finals teams, ROC-AUC,
    and feature importances.

    A deep playoff run is defined as reaching the Conference Finals, Finals,
    or winning the championship. Synthetic regular season rows (season labels
    containing '-reg') are used to enrich training for the latest season
    prediction but are excluded from LOSO folds and as prediction targets.

    Args:
        df: DataFrame containing TEAM_FEATURES, 'season', 'team_name',
            'playoff_stage', and 'deep_playoff_run'.

    Returns:
        Tuple of:
            - loso_df: DataFrame with columns ['season', 'roc_auc'] for each fold
            - results: DataFrame for the latest season with 'team_name',
                       'predicted_probability', and 'actual'
    """
    df = df.dropna(subset=TEAM_FEATURES).copy()

    historical = df[~df['season'].str.contains('-reg')].copy()
    seasons = sorted(historical["season"].unique())
    latest_season = seasons[-1]

    print("\n=== Leave-One-Season-Out Validation ===")
    loso_results = []

    for held_out in seasons:
        train_idx = historical[historical["season"] != held_out].dropna(subset=TEAM_FEATURES + ["deep_playoff_run"])
        test_idx =  historical[historical["season"] == held_out].dropna(subset=TEAM_FEATURES + ["deep_playoff_run"])

        if test_idx["deep_playoff_run"].nunique() < 2:
            print(f"  {held_out}: skipped (only one class in test set)")
            continue

        X_train = train_idx[TEAM_FEATURES]
        y_train = train_idx["deep_playoff_run"]
        
        X_test = test_idx[TEAM_FEATURES]
        y_test = test_idx["deep_playoff_run"]

        model = RandomForestClassifier(
            n_estimators=100,
            class_weight="balanced",
            random_state=1
        )
        model.fit(X_train,y_train)
        probs = model.predict_proba(X_test)[:, 1]

        auc = roc_auc_score(y_test, probs)

        if held_out == latest_season:

            # Latest season: full output including predictions and feature importances
            results = test_idx[["team_name"]].copy()
            results["predicted_probability"] = probs
            results["actual"] = y_test.values
            results = results.sort_values("predicted_probability", ascending=False)

            feature_importances = pd.DataFrame({
                "feature": TEAM_FEATURES,
                "importance": model.feature_importances_
            }).sort_values("importance", ascending=False)

            print(f"\n=== Predicting Season: {latest_season} ===")
            print("\nPredicted Conference Finals Teams (Top 4):\n")
            print(results.head(4))

            print("\nActual Conference Finals Teams:\n")
            print(results[results["actual"] == 1])

            print(f"\nROC-AUC for {latest_season}: {auc:.3f}")
            
        else:
            print(f"  {held_out}: ROC-AUC = {auc:.3f}")

        loso_results.append({"season":held_out,"roc_auc":auc})

    loso_df = pd.DataFrame(loso_results)
    print(f"\n  Mean ROC-AUC: {loso_df['roc_auc'].mean():.3f}")

    return loso_df,results,feature_importances




def main():
    """Main function to predict behaviours based on modeling dataset.

    Loads engineered team and top player feature datasets, merges them,
    appends the latest regular season stats with a fake season label for
    enriched training, then runs predict_latest_season which handles both LOSO validation
    and latest season prediction in a single pass. Saves all results to disk.
    

    Output:
        data/results/latest_season_predictions.csv
    """
    playoff_df = pd.read_csv("data/features/playoff_team_features.csv")
    reg_df = pd.read_csv("data/features/reg_team_features.csv")
    playoff_top1 = pd.read_csv("data/features/playoff_top1_features.csv")
    reg_top1 = pd.read_csv("data/features/reg_top1_features.csv")

    # Merge historic playoff top1 player features into playoff team rows
    playoff_df = playoff_df.merge(playoff_top1, on=["team_id","season"],how="left")

    # Label deep playoff runs
    playoff_df["deep_playoff_run"] = (
        playoff_df["playoff_stage"].isin(["Conference Finals", "Finals", "Champion"])
    ).astype(int)


    # Build latest regular sesason rows
    latest_reg = reg_df[reg_df["season"] == reg_df["season"].max()].copy()
    latest_reg["playoff_stage"] = "Unknown"   # maps to deep_playoff_run = 0
    latest_reg["deep_playoff_run"] = 0
    latest_reg["season"] = "2024-25-reg"      # excluded from latest_season filter

    # Merge current reg season top1 player features into latest_reg rows
    latest_reg_top1 = reg_top1[reg_top1["season"] == reg_top1["season"].max()].copy()
    latest_reg_top1["season"] = "2024-25-reg" 
    latest_reg = latest_reg.merge(latest_reg_top1,on=["team_id","season"],how="left")


    combined_df = pd.concat([playoff_df, latest_reg], ignore_index=True)


    result_path = "data/results"
    os.makedirs(result_path, exist_ok=True)

    # Run LOSO validation and predict the latest season
    loso_results, team_pred, feature_importances = predict_latest_season(combined_df)
    loso_results.to_csv(os.path.join(result_path, "loso_validation.csv"), index=False)
    team_pred.to_csv(os.path.join(result_path, "latest_season_predictions.csv"), index=False)
    feature_importances.to_csv(os.path.join(result_path, "feature_importances.csv"), index=False)





if __name__ == "__main__":
    main()





