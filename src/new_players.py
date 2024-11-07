import pandas as pd

def new_players_per_year(players_teams_df: pd.DataFrame) -> pd.DataFrame:
    #known_players = set()

    for year in range(1, 11):
        new_players = 0
        known_players = set()
        filtered_df = players_teams_df[players_teams_df["year"] == year]
        for index, row in filtered_df.iterrows():
            if row["playerID"] not in known_players:
                known_players.add(row["playerID"])
                new_players = new_players+1
        print(f"{new_players} new players in year {year}")

new_players_per_year(pd.read_csv("../data/players_teams.csv"))