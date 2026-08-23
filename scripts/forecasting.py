import os

from feature_analysis import split_by_season

import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

from sklearn.ensemble import (
    RandomForestClassifier,
    GradientBoostingClassifier
)

from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint, uniform

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

def train_gradient_boosting(X,y):

    param_dist = {
        'n_estimators': randint(50,300),
        'max_depth': randint(1,5),
        'learning_rate': uniform(0.01, 0.15),
        'min_samples_leaf': randint(5,20)
    }

    gb = GradientBoostingClassifier(random_state=42)

    search = RandomizedSearchCV(
        gb,param_distributions=param_dist,
        n_iter=10,cv=5,scoring="roc_auc",
        random_state=42,n_jobs=-1
    )

    search.fit(X,y)

    return search.best_estimator_

def evaluate_model(model, X_test, y_test, model_name):
    probabilities = model.predict_proba(X_test)[:,1]

    test_auc = roc_auc_score(y_test,probabilities)
    print(
        f"{model_name} historical test ROC-AUC: "
        f"{test_auc:.3f}"
    )

    return test_auc

def forecast_current_season(model,current_df):

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

def main():

    feature_path = "data/features"
    results_path = "data/results"

    model_df = pd.read_csv(os.path.join(feature_path, "modeling_features.csv"))

    reg_team_data = pd.read_csv(os.path.join(feature_path, "reg_team_features.csv"))

    forecast_season = "2025-26"

    model_df = model_df.dropna(subset=SELECTED_FEATURES)

    # Historical evaluation
    X_train, X_test, y_train, y_test = split_by_season(
        model_df,
        SELECTED_FEATURES
    )


    rf_model = train_random_forest(X_train,y_train)
    logistic_model = train_logistic_regression(X_train,y_train)
    gb_model = train_gradient_boosting(X_train,y_train)


    #Evaluate on historical test seasons
    rf_auc = evaluate_model(
        rf_model,
        X_test,
        y_test,
        "Random Forest"
    )

    logistic_auc = evaluate_model(
        logistic_model,
        X_test,
        y_test,
        "Logistic Regression"
    )

    gb_auc = evaluate_model(
        gb_model,
        X_test,
        y_test,
        "Gradient Boosting"
    )

    # Train final models on all historical seasons
    X = model_df[SELECTED_FEATURES]
    y = model_df["made_conf_finals"]

    final_rf = train_random_forest(X,y)
    final_logistic = train_logistic_regression(X,y)
    final_gb = train_gradient_boosting(X,y)


    # Current season
    current_df = reg_team_data[reg_team_data["season"] == forecast_season].copy()

    rf_predictions, rf_finalists = forecast_current_season(final_rf,current_df)
    logistic_predictions, logistic_finalists = forecast_current_season(final_logistic,current_df)
    gb_predictions, gb_finalists = forecast_current_season(final_gb,current_df)

    print("\nRandom Forest predicted Conference Finals:")
    print(
        rf_finalists[
            [
                "conference",
                "team_abbreviation",
                "conf_finals_probability"
            ]
        ]
    )

    print("\nLogistic Regression predicted Conference Finals:")
    print(
        logistic_finalists[
            [
                "conference",
                "team_abbreviation",
                "conf_finals_probability"
            ]
        ]
    )

    print("\nGradient Boosting predicted Conference Finals:")
    print(
        gb_finalists[
            [
                "conference",
                "team_abbreviation",
                "conf_finals_probability"
            ]
        ]
    )
    rf_predictions.to_csv(os.path.join(results_path,"random_forest_predictions.csv"),
                          index=False)

    logistic_predictions.to_csv(os.path.join(results_path,"logistic_regression_predictions.csv"),
                                index=False)

    gb_predictions.to_csv(os.path.join(results_path,"gradient_boosting_predictions.csv"),
                          index=False)


if __name__ == "__main__":
    main()
