# NBA-playoff-impact-model
A statistical modeling project estimating player contributions to NBA playoff wins using logistic regression, tree-based models, and feature importance analysis.

Project Structure:
- collect_raw_data.py
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

This script makes use of the `nba_api` to collect data for playoff seasons 2014-15 through 2024-25 using nba.com. By running the script as is, it provides the user with player stats, advanced player stats, team stats, and game logs that will later be processed for use later. 

You can simply run the function,
```bash
python scripts/collect_raw_data.py
```
and you will see the output under data/raw.

