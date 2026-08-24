# NBA Playoff Forecasting Model

A season-aware machine learning model that predicts which NBA teams are likely to reach the Conference Finals using engineered efficiency metrics and star player impact from the 2015–16 through 2024–25 seasons.

# Project Objective

To estimate the probability that a team will make at least make the conference finals based on regular-season efficiency metrics and top player performance.


# Feature Selection

- Binary classification problem indicating whether a team reached the Conference Finals or beyond.
- Candidate features include team-level efficiency metrics and top player statistics.
- LASSO regression and classification trees are used to identify informative features.
- Classification tree feature selection is used for the final feature set due to stronger test performance than LASSO(0.59 vs 0.21).
- Selected features:
  - `net_rating`
  - `ts_pct`
  - `top1_oreb_pct`
  - `top1_pie`
  - `top1_usg_pct`
  - `pie`
  - `top1_net_rating`

# Modeling Approach
Two candidate models are trained and compared:
- **Logistic Regression**: standardized features with L2 regularization and balanced class weights.
- **Random Forest**: hyperparameters (`n_estimators`, `max_depth`, `min_samples_split`, `min_samples_leaf`) tuned via `RandomizedSearchCV` with 5-fold cross-validation, scored on ROC-AUC.

Both models are evaluated using walk-forward validation: starting from a minimum training window of 3 seasons, each model is trained only on prior seasons and tested on the following season, then the window expands by one season and the process repeats. This mimics how the model is actually used, forecasting a season it has not seen any outcomes from.

Final models are refit on all historical seasons (2015-16 through 2024-25) and used to forecast the current season. Teams are grouped by conference, and the two highest predicted probabilities per conference are selected as the model's projected Conference Finalists.


# Dashboard
Interactive dashboard visualizing model predictions, Walk-forward validation results, and feature importances.

- [View on Tableau Public](https://public.tableau.com/app/profile/abillash.thampiyah/viz/nba_playoff_dashboard/Dashboard1?publish=yes)
- Download `dashboard/nba_playoff_dashboard.twbx` to open locally in Tableau

# Project Structure:
- collect_raw_data.py
- process_data.py
- feature_analysis.py
- forecasting.py
- README.md

# Libraries

`nba_api`: An API Client Package to Access the APIs of NBA.com([Readmore](https://github.com/swar/nba_api)).

`pandas`: data structures and data analysis tools for the Python programming language([Readmore](https://pandas.pydata.org/docs/index.html)).

`scikit-learn`: Machine learning modelling and evaluation([Readmore](https://scikit-learn.org/stable/))

`scipy`: Statistical distributions and functions for hyperparameter search spaces ([Readmore](https://scipy.org))

# Installing Libraries

```bash
pip install nba_api pandas scikit-learn scipy
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

# feature_analysis.py

Performs feature selection on processed NBA team and player statistics.

Features:
- Creates a binary playoff outcome label (`made_conf_finals`) indicating whether a team reached the Conference Finals.
- Engineers interpretable team efficiency metrics, including three-point rate and free throw rate.
- Selects each team's highest-PIE player per season and extracts their key statistics, prefixed with `top1_`.
- Splits the data chronologically, using the first 80% of seasons for training and the final 20% for testing.
- Performs feature selection using LASSO regression and a classification tree.
- Saves the selected features for use in subsequent forecasting models.

# forecasting.py
Trains, validates, and evaluates both models, then generates Conference Finals forecasts for the current season.
- Loads engineered historical features from `modeling_features.csv` and current-season regular-season stats from `reg_team_features.csv`.
- Drops rows with missing values in the selected feature set.
- Runs walk-forward validation for both Logistic Regression and Random Forest, reporting mean ROC-AUC across test seasons.
- Evaluates both models on a chronological 80/20 historical train/test split for comparison.
- Refits both models on all historical seasons and forecasts probabilities for the current season.
- Maps each team to its conference and selects the top 2 teams per conference by predicted probability as the projected Conference Finalists.
- Saves predictions and walk-forward results to `data/results/`.


# Conclusions

The results indicate that regular-season team efficiency and top-player impact contain meaningful information about a team's likelihood of reaching the Conference Finals or beyond.

Random Forest achieved the highest mean walk-forward ROC-AUC of **0.845**, slightly outperforming Logistic Regression at **0.839**. However, Logistic Regression performed better on the chronological 80/20 historical holdout, achieving a ROC-AUC of **0.820** compared with **0.777** for Random Forest.

The difference between the two evaluation methods highlights the importance of season-aware validation when forecasting across NBA seasons. While the Random Forest captured slightly more signal across the expanding walk-forward windows, the stronger holdout performance of Logistic Regression suggests that a simpler model may generalize better to later seasons.

The final models provide probability-based rankings rather than deterministic playoff predictions. These probabilities can be used to identify teams with stronger estimated chances of making a deep playoff run while accounting for uncertainty in the prediction.

# Future Improvements

- Incorporate player availability and injury information to account for changes in team strength entering the playoffs.
- Add late-season and post-All-Star-Break performance metrics to capture current team form.
- Include multiple top-player metrics rather than only the highest-PIE player to better capture roster depth.
- Explore conference-specific models or conference-adjusted features to account for differences between the Eastern and Western Conference.
- Evaluate probability calibration to determine whether predicted probabilities accurately reflect observed outcomes.
- Investigate interactions between team-level efficiency and top-player impact.
- Incorporate potential playoff matchups and opponent strength into playoff probability estimates.








