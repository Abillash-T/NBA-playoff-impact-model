# NBA-playoff-impact-model

A season-aware machine learning project that predicts which NBA teams will reach the Conference Finals using engineered team efficiency metrics from 2015–16 through 2024–25.
The model is trained on historical seasons and evaluated using leave-one-season-out validation to simulate real-world forecasting conditions.

# Project Objective

To estimate the probability that a team will make a deep playoff run (Conference Finals or beyond) based on regular-season efficiency metrics.

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
- Evaluation metric: ROC-AUC
  
Mean Season-Based ROC-AUC: ~0.85

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
- Player per-game playoff statistics
- Player advanced playoff statistics
- Team playoff statistics
- Player game logs
- All datasets are saved in the data/raw/ directory for downstream processing.

# process_data.py

This script standardizes and prepares raw NBA playoff data for analysis.

It performs the following steps:
- Cleans and standardizes column names (lowercase, trimmed, underscores)
- Saves cleaned datasets to data/processed
- Merges player per-game stats with advanced stats into a single dataset
- Preserves original files for reproducibility

# feature_engineering.py

This script creates the final modeling dataset from processed team playoff statistics (2015–2025).

Features:
- Creates a multiclass playoff outcome variable (playoff_stage)
- Engineers interpretable team efficiency metrics
- Removes redundant ranking columns
- Exports a modeling-ready dataset and correlation matrix

# modeling.py

Trains and evaluates the predictive model using a random forest classifier.

Features:

- Leave-One-Season-Out validation
- Feature importance analysis
- Forecasting simulation for the most recent season
- Exports latest season predictions to data/results/



# Future Improvements

- Incorporate defensive rating splits
- Add late-season momentum metrics
- Include prior playoff experience
- Hyperparameter tuning with cross-validation







