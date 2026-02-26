# NBA-playoff-impact-model
A statistical modeling project estimating player contributions to NBA playoff wins using logistic regression, tree-based models, and feature importance analysis.

Project Structure:
- collect_raw_data.py
- process_data.py
- feature_engineering.py
- README.md

# Libraries

`nba_api`: An API Client Package to Access the APIs of NBA.com([Readmore](https://github.com/swar/nba_api)).

`pandas`: data structures and data analysis tools for the Python programming language.([Readmore](https://pandas.pydata.org/docs/index.html))

# Installing Libraries

```bash
pip install nba_api
```
```bash
pip install pandas
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





