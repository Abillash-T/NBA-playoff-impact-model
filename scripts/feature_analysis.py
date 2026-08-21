import os
from process_data import main as run_processing

import pandas as pd
import numpy as np


from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LassoCV
from sklearn.tree import DecisionTreeClassifier


CANDIDATE_FEATURES = [
    #"w_pct",
    "off_rating", "def_rating", "net_rating",
    "efg_pct", "ts_pct",
    "ast_pct", "ast_to", "ast_ratio",
    "oreb_pct", "dreb_pct", "reb_pct",
    "tm_tov_pct",
    "pace",
    "pie",
    "fg3_pct", "ft_pct",
    "three_point_rate", "free_throw_rate",
    # top-1 player (by PIE) features
    "top1_pie", "top1_ts_pct", "top1_efg_pct", "top1_usg_pct",
    "top1_ast_to", "top1_def_rating", "top1_net_rating",
    "top1_oreb_pct", "top1_dreb_pct",
]

def build_label(playoff_team_df):
    """Creates a binary label indicating whether a team reached the Conference Finals.

    Uses playoff wins to determine whether a team reached at least the Conference
    Finals. Teams that reached the Conference Finals, Finals, or won the
    championship are labeled 1, while all other teams are labeled 0.

    Args:
        playoff_team_df: A dataframe containing team playoff statistics for
            each season.

    Returns:
        A dataframe containing "team_id", "season", and "made_conf_finals",
        where "made_conf_finals" is 1 if the team reached the Conference
        Finals and 0 otherwise.
    """
    labels = ["First Round", "Second Round", "Conference Finals", "Finals", "Champion"]
    df = playoff_team_df.copy()

    df["playoff_stage"] = pd.cut(
        df["w"], 
        bins = [-1,3,7,11,15,20], 
        labels=labels
        )
    
    df["made_conf_finals"] = (
            df["playoff_stage"]
            .isin(["Conference Finals", "Finals", "Champion"])
            .astype(int)
        )

    return df[["team_id", "season", "playoff_stage", "made_conf_finals"]].drop_duplicates()

def engineer_team_features(df):
    """Creates interpretable team-level features

    Derives shooting profile metrics not directly available from the API.

    Args:
        df: a dataframe consisting 'fg3a', 'fga', 'fta' columns.

    Returns:
        df with 'three_point_rate' and 'free_throw_rate' columns added
    """
    df = df.copy()

    # Shooting profile metrics
    df["three_point_rate"] = df["fg3a"] / df["fga"]
    df["free_throw_rate"] = df["fta"] / df["fga"]


    return df


def engineer_best_player(df):
    """Selects the top player per team per season by PIE and returns their stats.

    For each (team_id, season) group, ranks players by PIE (Player Impact
    Estimate) and keeps only the highest-ranked player. Their stats are
    prefixed with 'top1_' to distinguish them from team-level features.

    Args:
        df: DataFrame of player stats containing 'team_id', 'team_abbreviation',
            'season', 'pie', and the stat columns listed in agg_cols.

    Returns:
        DataFrame with one row per (team_id, season) containing the top
        player's stats, prefixed with 'top1_'.
    """
    df = df.copy()

    agg_cols = [
        "pie",
        "ts_pct",
        "efg_pct",
        "usg_pct",
        "ast_to",
        "stl",
        "blk",
        "def_rating",
        "net_rating",
        "oreb_pct",
        "dreb_pct"
    ]

    df = df.dropna(subset=["pie"]).copy()

    df["pie_rank"] = (
        df.groupby(["team_id","season"])["pie"]
        .rank(method="first",ascending=False)
    )

    top_players = df[df["pie_rank"] <= 1]

    best = (
        top_players
        .groupby(["team_id", "team_abbreviation", "season"])[agg_cols]
        .first()
        .reset_index()
        .rename(columns={col: f"top1_{col}" for col in agg_cols})
    )

    return best

def build_modeling_frame(reg_team,playoff_team,reg_player):
    """Combines regular-season team and player features with playoff outcomes.

    Merges regular-season team statistics with the top player's statistics for
    each team and season. The resulting dataframe also contains a binary label
    indicating whether the team reached the Conference Finals.

    Args:
        reg_team: A dataframe containing regular-season team statistics.
        playoff_team: A dataframe containing playoff team statistics used to
            create the Conference Finals label.
        reg_player: A dataframe containing regular-season player statistics.

    Returns:
        A dataframe containing one row per team and season with regular-season
        team features, top-player features, and the "made_conf_finals" label.
    """
    label = build_label(playoff_team)
    reg_team = engineer_team_features(reg_team)
    top1 = engineer_best_player(reg_player)

    df = reg_team.merge(top1, on=["team_id","season"], how="left")
    df = df.merge(label, on=["team_id","season"], how="left")
    df["made_conf_finals"] = df["made_conf_finals"].fillna(0).astype(int)

    return df

def split_by_season(model_df, features):
    """Splits the data into training and testing sets by season.

    Sorts the available seasons chronologically and uses the first 80% of
    seasons for training and the remaining 20% for testing. This ensures that 
    later seasons are not used to train models that are evaluated on those 
    later seasons.

    Args:
        model_df: A dataframe containing the features, season, and target
            variable.
        features: A list of feature names to use as predictors.

    Returns:
        X_train: Training data containing the selected features.
        X_test: Testing data containing the selected features.
        y_train: Training labels indicating whether a team made the
            Conference Finals.
        y_test: Testing labels indicating whether a team made the
            Conference Finals.
    """
    seasons = sorted(model_df["season"].unique())
    split_idx = int(len(seasons) * 0.8)
    
    train_seasons = seasons[:split_idx]
    test_seasons = seasons[split_idx:]
    
    train_df = model_df[model_df["season"].isin(train_seasons)].copy()
    test_df = model_df[model_df["season"].isin(test_seasons)].copy()
    
    X_train = train_df[features]
    y_train = train_df["made_conf_finals"]
    
    X_test = test_df[features]
    y_test = test_df["made_conf_finals"]

    return X_train, X_test, y_train, y_test


def run_lasso(X_train, y_train, X_test, y_test):
    """Performs LASSO regression for feature selection.

    Standardizes the training and testing features, fits a cross-validated
    LASSO model, and selects features with non-zero coefficients. Training
    and testing accuracy values and the estimated coefficients are also displayed.

    Args:
        X_train: Training feature data.
        y_train: Training target values.
        X_test: Testing feature data.
        y_test: Testing target values.

    Returns:
        A list containing the names of features with non-zero LASSO
        coefficients.
    """
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    lasso = LassoCV(cv=5,random_state=0,max_iter=5000)

    lasso.fit(X_train_scaled, y_train)

    lasso_coefs = pd.Series(lasso.coef_,index=CANDIDATE_FEATURES).sort_values()

    selected_features = lasso_coefs[lasso_coefs != 0].index.tolist()

    print("LASSO train R²:", lasso.score(X_train_scaled, y_train))
    print("LASSO test R²:", lasso.score(X_test_scaled, y_test))

    print("LASSO coefficients:")
    print(lasso_coefs)

    print("\nSelected features:")
    print(selected_features)

    print("\nNumber of selected features:", len(selected_features))

    return selected_features


def run_classification_tree(X_train, y_train,X_test,y_test):
    """Uses a classification tree for feature selection.

    Fits a decision tree classifier to the training data and calculates
    training and testing accuracy. Feature importance values are used to
    identify features that contribute to the tree's predictions.

    Args:
        X_train: Training feature data.
        y_train: Training target values.
        X_test: Testing feature data.
        y_test: Testing target values.

    Returns:
        A list containing the names of features with non-zero feature
        importance in the classification tree.
    """
    tree = DecisionTreeClassifier(
        max_depth=4,
        min_samples_leaf=10,
        class_weight="balanced",
        random_state=42
    )

    tree.fit(X_train, y_train)
    print("Tree train accuracy:", tree.score(X_train, y_train))
    print("Tree test accuracy:", tree.score(X_test, y_test))

    importances = pd.Series(tree.feature_importances_,index=X_train.columns).sort_values(ascending=False)

    print("Classification tree feature importance:")
    print(importances)

    selected_features = importances[importances > 0].index.tolist()

    print("\nSelected features:")
    print(selected_features)

    print("\nNumber of selected features:", len(selected_features))

    return selected_features



def main():
    """Processes NBA data, performs feature selection, and saves selected features.

    Loads and processes the NBA team and player data, constructs the modeling
    dataset, splits the data chronologically into training and testing sets,
    and performs LASSO and classification tree feature selection. The
    features selected by the classification tree are then saved as CSV files
    for use in subsequent forecasting models.
    """
    feature_path = "data/features"
    
    os.makedirs(feature_path,exist_ok=True)
    
    files = run_processing()
    
    playoff_team_data = files['playoff_team_stats']
    reg_team_data = files['reg_team_stats']
    playoff_player_data = files['playoff_player_stats']
    reg_player_data = files['reg_player_stats']

    playoff_top1 = engineer_best_player(playoff_player_data)
    reg_top1 = engineer_best_player(reg_player_data)

    df = build_modeling_frame(reg_team_data,playoff_team_data,reg_player_data)

    features = list(CANDIDATE_FEATURES)

    model_df = df.dropna(subset=features + ["made_conf_finals"]).copy()

    X_train, X_test, y_train, y_test = split_by_season(model_df,features)


    #lasso
    lasso_features = run_lasso(X_train,y_train,X_test,y_test)

    #Classification trees
    class_tree_features = run_classification_tree(X_train,y_train,X_test,y_test)

    team_selected = [ 
        "net_rating",
        "ts_pct",
        "pie"
    ] 

    top1_selected = [
        "top1_oreb_pct",
        "top1_pie",
        "top1_usg_pct",
        "top1_net_rating"

    ]  
    
    playoff_team_features = playoff_team_data.merge(
        playoff_top1[["team_abbreviation","team_id", "season"] + top1_selected],
        on=["team_id", "season"],
        how="left"
    )

    reg_team_features = reg_team_data.merge(
        reg_top1[["team_abbreviation","team_id", "season"] + top1_selected],
        on=["team_id", "season"],
        how="left"
    )

    playoff_labels = build_label(playoff_team_data)

    playoff_team_features = playoff_team_features.merge(
            playoff_labels[
                ["team_id", "season", "playoff_stage"]],
            on=["team_id", "season"],
            how="left"
        )

    playoff_team_features[
        ["team_abbreviation","team_id","season"] + 
        team_selected + 
        top1_selected + 
        ["playoff_stage"]
    ].to_csv(os.path.join(feature_path,"playoff_team_features.csv"),index=False)

    reg_team_features[
        ["team_abbreviation","team_id","season"] + 
        team_selected + 
        top1_selected
    ].to_csv(os.path.join(feature_path,"reg_team_features.csv"),index=False)

    # Create historical playoff outcome
    model_df = reg_team_features.merge(
        playoff_labels[
            ["team_id", "season", "playoff_stage", "made_conf_finals"]
        ],
        on=["team_id", "season"],
        how="left"
    )

    model_df["made_conf_finals"] = (
        model_df["made_conf_finals"]
        .fillna(0)
        .astype(int)
    )

    model_df.to_csv(
        os.path.join(
            feature_path,
            "modeling_features.csv"
        ),
        index=False
    )
    


if __name__ == "__main__":
    main()