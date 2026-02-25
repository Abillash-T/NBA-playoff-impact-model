# NBA-playoff-impact-model
A statistical modeling project estimating player contributions to NBA playoff wins using logistic regression, tree-based models, and feature importance analysis.

Project Structure:
- collect_raw_data.py
- process_data.py
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

This script makes use of the `nba_api` to collect data for playoff seasons 2014-15 through 2024-25 using official stats from nba.com. By running the script as is, it provides the user with player stats, advanced player stats, team stats, and game logs that will later be processed for use later. 

You can simply run the function,
```bash
python scripts/collect_raw_data.py
```
and you will see the output under data/raw.

# process_data.py

Assuming that `collect_raw_data.py` has been run, and as such the raw data files were created, this script creates the same CSV files but standardized to ensure consistency across datasets and prevents merge/key errors caused by inconsistent naming for future steps.

It also merges the `player_per_game_playoffs_2015_2025.csv`, and `player_advanced_playoffs_2015_2025.csv`, as `Player_stats_2015_2025.csv` to allow future analysis. The original files remain unmodified as well. 

# feature_engineering.py


