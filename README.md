# NBA-playoff-impact-model

A season-aware machine learning model that predicts which NBA teams are likely to reach the Conference Finals using engineered efficiency metrics and star player impact from the 2015–16 through 2024–25 seasons.

# Project Objective

To estimate the probability that a team will make a deep playoff run (Conference Finals or beyond) based on regular-season efficiency metrics and top player performance.

A deep playoff run is defined as reaching:
- Conference Finals
- NBA Finals
- NBA Champion

# Modeling Approach
- Binary classification problem (making a deep playoff run or not)
- Random Forest Classifier
- Class weight balancing to address playoff class imbalance
- Leave-One-Season-Out cross-validation
- Most recent season held out for forecasting simulation
- Current regular season stats for all 30 teams appended to training data to enrich predictions
- Top player features derived by selecting each team's highest-PIE player from regular season data
- Evaluation metric: ROC-AUC
  
Mean Season-Based ROC-AUC: ~0.852

# Dashboard
Interactive dashboard visualizing model predictions, LOSO validation results, and feature importances.

- [View on Tableau Public](https://public.tableau.com/views/NBAPlayoffPredictionModel/Dashboard1?:language=en-US&publish=yes&:sid=&:redirect=auth&:display_count=n&:origin=viz_share_link)
- Download `dashboard/nba_playoff_dashboard.twbx` to open locally in Tableau

# Project Structure:
- collect_raw_data.py
- process_data.py
- feature_engineering.py
- modeling.py
- README.md

# Libraries

`nba_api`: An API Client Package to Access the APIs of NBA.com([Readmore](https://github.com/swar/nba_api)).

`pandas`: data structures and data analysis tools for the Python programming language.([Readmore](https://pandas.pydata.org/docs/index.html))

`scikit-learn`: Machine learning modeling and evaluation([Readmore](https://scikit-learn.org/stable/))

# Installing Libraries

```bash
pip install nba_api pandas scikit-learn
```


# collect_raw_data.py

This script collects NBA playoff data (2015–16 through 2024–25) directly from the NBA Stats API using the `nba_api` package.

It retrieves:
- Player per-game and advanced statistics (playoff and regular season)
- Team per-game and advanced statistics (playoff and regular season)

All datasets are saved to `data/raw/` for downstream processing.

# process_data.py

This script standardizes and prepares raw NBA playoff data for analysis.

It performs the following steps:
- Cleans and standardizes column names (lowercase, trimmed, underscores)
- Removes redundant ranking columns
- Merges per-game stats with advanced stats into unified team and player datasets
- Saves cleaned datasets to `data/processed/`

# feature_engineering.py

Creates the final modeling datasets from processed team and player statistics.

Features:
- Creates a binary playoff outcome label (`deep_playoff_run`) from playoff win totals
- Engineers interpretable team efficiency metrics (three-point rate, free throw rate)
- Selects each team's highest-PIE player per season and extracts their key stats, prefixed with `top1_`
- Saves playoff and regular season datasets separately to `data/features/`

# modeling.py

Trains and evaluates the predictive model using a season‑aware Random Forest classifier.

Key Features

- Merges historic playoff top player features into the playoff team dataset
- Appends the latest regular season team and player stats (tagged with a fake season label) to enrich training without leaking into the prediction target
- Leave-One-Season-Out (LOSO) validation to simulate real forecasting conditions
- Class-balanced Random Forest to address playoff outcome imbalance
- Predicts deep playoff run probability for each team in the most recent season

Output:
- Season-level ROC-AUC scores
- Latest-season predictions saved to data/results/latest_season_predictions.csv


# Future Improvements

- XGBoost or gradient boosting to extract more signal from the current feature set
- Late-season momentum metrics
- Hyperparameter tuning with cross-validation







