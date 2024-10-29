import pandas as pd

def lag_row(df):
    df["playoff"] = df["playoff"].shift(-1)
    return df

def lag_playoffs(teams_df: pd.DataFrame):
    teams_df_grouped = teams_df.groupby('tmID')
    teams_df_grouped = teams_df_grouped.apply(lag_row)
    return teams_df_grouped

def merge_to_teams(teams_df: pd.DataFrame, other_df: pd.DataFrame) -> pd.DataFrame:
    return teams_df.merge(other_df, on=["year", "tmID"])

def add_coaches(row: pd.Series, coaches_df: pd.DataFrame) -> pd.Series:
    year = row["year"]
    tmID = row["tmID"]
    coaches = coaches_df[(coaches_df["year"] == year) & (coaches_df["tmID"] ==tmID)]
    coach_no = len(coaches)

    if coach_no == 3:
        print("THERE ARE THREE COACHES")

    if coach_no == 1:
        row["Coach1"] = coaches.iloc[0]["coachID"]
    if coach_no == 2:
        print(coaches.iloc[1]["coachID"])
        row["Coach2"] = coaches.iloc[1]["coachID"]

    return row

def merge_coaches(teams_df: pd.DataFrame, coaches_df: pd.DataFrame) -> pd.DataFrame:
    teams_df.insert(5, "Coach1", pd.NA)
    teams_df.insert(5, "Coach2", pd.NA)

    teams_df = teams_df.apply(lambda x: add_coaches(x, coaches_df), axis=1)

    return teams_df

def count_awards(row: pd.Series, awards_df: pd.DataFrame) -> pd.Series:
    grouped_df = awards_df.groupby("playerID").size()
    row["Award Count"] = grouped_df.loc[row["playerID"]] if row["playerID"] in grouped_df.index else 0

    return row

def merge_awards(teams_df: pd.DataFrame, players_teams_df: pd.DataFrame, awards_df: pd.DataFrame) -> pd.DataFrame:
    players_teams_df.insert(5, "Award Count", 0)

    players_teams_df = players_teams_df.apply(lambda x: count_awards(x, awards_df), axis=1)

    return players_teams_df

def merge_colleges(row: pd.Series, players_df: pd.DataFrame) -> pd.Series:
    filtered = players_df[players_df["bioID"] == row["playerID"]].reset_index()
    row["College"] = filtered.at[0, "college"]

    return row

def count_colleges(row: pd.Series, players_teams_df: pd.DataFrame) -> pd.Series:
    row["College Mode"] = players_teams_df[(players_teams_df["tmID"] == row["tmID"]) & (players_teams_df["year"] == row["year"])].mode().iloc[0]["College"]

    return row

def merge_college(teams_df: pd.DataFrame, players_teams_df: pd.DataFrame, players_df: pd.DataFrame) -> pd.DataFrame:
    players_teams_df.insert(5, "College", pd.NA)
    teams_df.insert(5, "College Mode", pd.NA)

    players_teams_df = players_teams_df.apply(lambda x: merge_colleges(x, players_df), axis=1)
    teams_df = teams_df.apply(lambda x: count_colleges(x, players_teams_df), axis=1)

    return teams_df


#merge_coaches(pd.read_csv("data/teams.csv"), pd.read_csv("data/coaches.csv"))

print(merge_college(pd.read_csv("data/teams.csv"), pd.read_csv("data/players_teams.csv"), pd.read_csv("data/players.csv")))

