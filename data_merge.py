import pandas as pd
import data_clean

def count_awards(players_teams_df: pd.DataFrame, awards_df: pd.DataFrame) -> pd.DataFrame:
    awards_df = awards_df.rename(columns={"year": "awardYear"})
    merged = players_teams_df.merge(awards_df, how="left", on="playerID", validate="m:m")
    merged = merged[(merged["awardYear"].isna()) | (merged["awardYear"] <= merged["year"])]

    return merged.groupby(by=list(players_teams_df)).agg({"award": "count"}).reset_index()

def merge_players(players_teams_df: pd.DataFrame, players_df: pd.DataFrame) -> pd.DataFrame:
    players_df = players_df.rename(columns={"bioID": "playerID"})
    return players_teams_df.merge(players_df, how="inner", on=["playerID"], validate="m:1")

def count_colleges(teams_df: pd.DataFrame, players_teams_df: pd.DataFrame) -> pd.DataFrame:
    players_teams_df = players_teams_df.drop(["GP"], axis=1)
    print(players_teams_df["college"])
    merged = teams_df.merge(players_teams_df, how="inner", on=["tmID", "year"], validate="1:m")
    print(merged)
    print(list(teams_df))

    return merged.groupby(by=list(teams_df), dropna=False).agg({"college": pd.Series.mode}).reset_index()

#print(count_awards(data_clean.players_teams_df, data_clean.awards_df))
print(count_colleges(data_clean.teams_df, merge_players(data_clean.players_teams_df, data_clean.players_df)))

