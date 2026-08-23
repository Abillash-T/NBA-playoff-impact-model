import os

from feature_analysis import split_by_season

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from sklearn.ensemble import RandomForestClassifier


from sklearn.model_selection import RandomizedSearchCV, TimeSeriesSplit
from scipy.stats import randint

SELECTED_FEATURES = [
    "net_rating",
    "ts_pct",
    "top1_oreb_pct",
    "top1_pie",
    "top1_usg_pct",
    "pie",
    "top1_net_rating"
]

TEAM_CONFERENCES = {
    "ATL": "East",
    "BOS": "East",
    "BKN": "East",
    "CHA": "East",
    "CHI": "East",
    "CLE": "East",
    "DET": "East",
    "IND": "East",
    "MIA": "East",
    "MIL": "East",
    "NYK": "East",
    "ORL": "East",
    "PHI": "East",
    "TOR": "East",
    "WAS": "East",

    "DAL": "West",
    "DEN": "West",
    "GSW": "West",
    "HOU": "West",
    "LAC": "West",
    "LAL": "West",
    "MEM": "West",
    "MIN": "West",
    "NOP": "West",
    "OKC": "West",
    "PHX": "West",
    "POR": "West",
    "SAC": "West",
    "SAS": "West",
    "UTA": "West"
}

def train_random_forest(X,y):
    """Trains a random forest classifier using randomized hyperparameter search.

    Uses RandomizedSearchCV to select random forest hyperparameters based on
    ROC-AUC. Class weights are balanced to account for the relatively small
    number of teams that reach the Conference Finals.

    Args:
        X: Training feature data.
        y: Training target values indicating whether a team made the
            Conference Finals.

    Returns:
        A random forest classifier fitted using the best hyperparameters
        found during randomized search.
    """
    param_dist = {
        'n_estimators': randint(100,500),
        'max_depth': randint(3,15),
        'min_samples_split': randint(2,10),
        'min_samples_leaf': randint(1,5)
    }

    rf = RandomForestClassifier(random_state=42,
                                class_weight="balanced",
                                n_jobs=-1)
    

    search = RandomizedSearchCV(
        rf,param_distributions=param_dist,
        n_iter=10,cv=5,scoring="roc_auc",
        random_state=42,n_jobs=-1
    )

    search.fit(X,y)


    return search.best_estimator_

def train_logistic_regression(X,y):
    """Trains a standardized logistic regression classifier.

    Standardizes the input features before fitting logistic regression.
    Class weights are balanced to account for the relatively small number
    of teams that reach the Conference Finals.

    Args:
        X: Training feature data.
        y: Training target values indicating whether a team made the
            Conference Finals.

    Returns:
        A fitted pipeline containing a StandardScaler and logistic
        regression classifier.
    """
    model = Pipeline([
        ("scaler", StandardScaler()),
        ("logistic", LogisticRegression(
            class_weight="balanced",
            max_iter=1000,
            random_state=42
        ))
    ])

    model.fit(X,y)

    return model


def walk_forward_validation(model_df,train_model,min_train_seasons=3):
    """Evaluates a model using chronological walk-forward validation.

    Trains the model using all seasons available before each test season
    and evaluates its ROC-AUC on the following season. This prevents
    future seasons from being used when predicting earlier seasons.

    Args:
        model_df: A dataframe containing the selected features, season,
            and "made_conf_finals" target variable.
        train_model: A function that takes training features and labels
            and returns a fitted classification model.
        min_train_seasons: Minimum number of seasons to use for the first
            training period.

    Returns:
        A dataframe containing the test season and corresponding ROC-AUC
        for each walk-forward validation period.
    """
    seasons = sorted(model_df["season"].unique())
    results = []

    for i in range(min_train_seasons, len(seasons)):
        train_seasons = seasons[:i]
        test_season = seasons[i]

        train_df = model_df[model_df["season"].isin(train_seasons)]

        test_df = model_df[model_df["season"] == test_season]

        X_train = train_df[SELECTED_FEATURES]
        y_train = train_df["made_conf_finals"]

        X_test = test_df[SELECTED_FEATURES]
        y_test = test_df["made_conf_finals"]

        model = train_model(X_train,y_train)

        probabilities = model.predict_proba(X_test)[:,1]

        auc = roc_auc_score(y_test,probabilities)


        results.append({
            "test_season": test_season,
            "roc_auc": auc
        })

    results_df = pd.DataFrame(results)


    return results_df
    

def evaluate_model(model, X_test, y_test, model_name):
    """Evaluates a fitted classification model using ROC-AUC.

    Generates predicted probabilities for the positive class and calculates
    ROC-AUC on the historical test data.

    Args:
        model: A fitted classification model with a predict_proba method.
        X_test: Testing feature data.
        y_test: Testing target values.
        model_name: Name of the model used when displaying the evaluation
            result.

    Returns:
        The ROC-AUC score calculated on the test data.
    """
    probabilities = model.predict_proba(X_test)[:,1]

    test_auc = roc_auc_score(y_test,probabilities)
    print(
        f"{model_name} 80/20 current season test ROC-AUC: "
        f"{test_auc:.3f}"
    )

    return test_auc

def forecast_current_season(model,current_df):
    """Generates Conference Finals probabilities for the current season.

    Uses a fitted classification model to estimate the probability that
    each team reaches the Conference Finals. Teams are ranked by probability
    and the top two teams from each conference are selected as predicted
    Conference Finals teams.

    Args:
        model: A fitted classification model with a predict_proba method.
        current_df: A dataframe containing current-season team features,
            team IDs, and team abbreviations.

    Returns:
        A tuple containing:
            predictions: A dataframe containing all teams, conferences,
                and Conference Finals probabilities, sorted by probability.
            predicted_finalists: A dataframe containing the two highest-
                probability teams from each conference.
    """
    X_current = current_df[SELECTED_FEATURES]
    probabilities = model.predict_proba(X_current)[:,1]

    predictions = current_df[["team_id","team_abbreviation"]].copy()

    
    predictions["conference"] = (predictions["team_abbreviation"].map(TEAM_CONFERENCES))
    predictions["conf_finals_probability"] = probabilities

    predictions = predictions.sort_values("conf_finals_probability", ascending = False)

    predicted_finalists = (
        predictions.sort_values(["conference","conf_finals_probability"],
                                ascending=[True,False]).groupby("conference").head(2))

    return predictions, predicted_finalists

def save_feature_importance(logistic_model, rf_model, features, results_path):
    """Saves feature importance for both models to a single CSV.

    Extracts standardized logistic regression coefficients and random forest
    feature importances, combining them into one dataframe indexed by
    feature name for side-by-side comparison.

    Args:
        logistic_model: A fitted Pipeline containing a StandardScaler and
            LogisticRegression step.
        rf_model: A fitted RandomForestClassifier.
        features: The list of feature names, in the same order used to
            train both models.
        results_path: Directory to save the output CSV to.
    """
    logistic_coefs = logistic_model.named_steps["logistic"].coef_[0]
    rf_importances = rf_model.feature_importances_

    importance_df = pd.DataFrame({
        "feature": features,
        "logistic_coefficient": logistic_coefs,
        "logistic_abs_coefficient": abs(logistic_coefs),
        "rf_importance": rf_importances
    })

    importance_df = importance_df.sort_values("rf_importance", ascending=False)

    importance_df.to_csv(
        os.path.join(results_path, "feature_importance.csv"),
        index=False
    )

    return importance_df

def main():
    """Trains classification models and forecasts the current NBA season.

    Loads the historical modeling data and current regular-season features,
    evaluates logistic regression and random forest models using both
    walk-forward validation and a chronological 80/20 split, and then
    retrains the models using all historical seasons. The final models are
    used to estimate each current-season team's probability of reaching the
    Conference Finals.

    Walk-forward validation measures performance across individual seasons,
    while the chronological 80/20 split provides a separate historical
    evaluation using later seasons as the test set.

    The resulting team probabilities and walk-forward validation results
    are saved as CSV files in the results directory.
    """
    feature_path = "data/features"
    results_path = "data/results"

    model_df = pd.read_csv(os.path.join(feature_path, "modeling_features.csv"))
    reg_team_data = pd.read_csv(os.path.join(feature_path, "reg_team_features.csv"))

    forecast_season = "2025-26"

    model_df = model_df.dropna(subset=SELECTED_FEATURES)

    historical_df = model_df[model_df["season"] != forecast_season].copy()

    # Walk-forward validation
    logistic_results = walk_forward_validation(historical_df,train_logistic_regression)
    rf_results = walk_forward_validation(historical_df,train_random_forest)

    print(f"Logistic Regression mean walk-forward {logistic_results['roc_auc'].mean():.3f}")
    print(f"Random Forest mean walk-forward {rf_results['roc_auc'].mean():.3f}")


    # Historical evaluation
    X_train, X_test, y_train, y_test = split_by_season(model_df,SELECTED_FEATURES)

  
    logistic_model = train_logistic_regression(X_train,y_train)
    rf_model = train_random_forest(X_train,y_train)


    #Evaluate on historical test seasons
    logistic_auc = evaluate_model(
            logistic_model,
            X_test,
            y_test,
            "Logistic Regression"
        )
    
    rf_auc = evaluate_model(
        rf_model,
        X_test,
        y_test,
        "Random Forest"
    )

    
    # Train final models on all historical seasons
    X = historical_df[SELECTED_FEATURES]
    y = historical_df["made_conf_finals"]

    final_rf = train_random_forest(X,y)
    final_logistic = train_logistic_regression(X,y)

    importance_df = save_feature_importance(
        final_logistic, final_rf, SELECTED_FEATURES, results_path
    )


    # Current season
    current_df = reg_team_data[reg_team_data["season"] == forecast_season].copy()

    rf_predictions, rf_finalists = forecast_current_season(final_rf,current_df)
    logistic_predictions, logistic_finalists = forecast_current_season(final_logistic,current_df)


    print("\nRandom Forest predicted Conference Finals:")
    print(rf_finalists[["conference","team_abbreviation","conf_finals_probability"]])

    print("\nLogistic Regression predicted Conference Finals:")
    print(logistic_finalists[["conference","team_abbreviation","conf_finals_probability"]])


    rf_predictions.to_csv(os.path.join(results_path,"random_forest_predictions.csv"),
                          index=False)

    logistic_predictions.to_csv(os.path.join(results_path,"logistic_regression_predictions.csv"),
                                index=False)

    rf_results.to_csv(os.path.join(results_path,"random_forest_walk_forward.csv"),
                          index=False)


    logistic_results.to_csv(os.path.join(results_path,"logistic_walk_forward.csv"),
                            index=False)
    

if __name__ == "__main__":
    main()
