import pandas as pd
from data_clean import *

### Data Merging Functions ###

def merge_to_teams(teams_df: pd.DataFrame, other_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the teams dataframe with another dataframe on the year and team ID (tmID).
    """
    return teams_df.merge(other_df, on=["year", "tmID"])

def add_coaches(row: pd.Series, coaches_df: pd.DataFrame) -> pd.Series:
    """
    Add the coaches to the teams row.
    """
    year = row["year"]
    tmID = row["tmID"]
    coaches = coaches_df[(coaches_df["year"] == year) & (coaches_df["tmID"] ==tmID)]
    coach_no = len(coaches)

    if coach_no == 3:
        print("THERE ARE THREE COACHES")

    if coach_no == 1:
        row["Coach1"] = coaches.iloc[0]["coachID"]
    if coach_no == 2:
        #print(coaches.iloc[1]["coachID"])
        row["Coach2"] = coaches.iloc[1]["coachID"]

    return row

def merge_coaches(teams_df: pd.DataFrame, coaches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the coaches dataframe with the teams dataframe. Adds two new columns, Coach1 and Coach2.
    """
    teams_df.insert(1, "Coach1", pd.NA)
    teams_df.insert(2, "Coach2", pd.NA)

    teams_df = teams_df.apply(lambda x: add_coaches(x, coaches_df), axis=1)

    return teams_df

def count_awards(row: pd.Series, awards_df: pd.DataFrame) -> pd.Series:
    """
    Count the number of awards a player has won and add it to the row.
    """
    grouped_df = awards_df.groupby("playerID").size()
    row["Award Count"] = grouped_df.loc[row["playerID"]] if row["playerID"] in grouped_df.index else 0

    return row

def merge_awards(players_teams_df: pd.DataFrame, awards_df: pd.DataFrame) -> pd.DataFrame:
    """
    Merge the awards dataframe with the players_teams dataframe. Adds a new column, Award Count.
    """
    players_teams_df.insert(5, "Award Count", 0)

    players_teams_df = players_teams_df.apply(lambda x: count_awards(x, awards_df), axis=1)

    return players_teams_df

def merge_colleges(row: pd.Series, players_df: pd.DataFrame) -> pd.Series:
    """
    // Adds the college of the player to the row.
    """
    filtered = players_df[players_df["bioID"] == row["playerID"]].reset_index()
    row["College"] = filtered.at[0, "college"]

    return row

def count_colleges(row: pd.Series, players_teams_df: pd.DataFrame) -> pd.Series:
    """
    Count the most common college of the players in the team and add it to the row.
    """
    row["College Mode"] = players_teams_df[(players_teams_df["tmID"] == row["tmID"]) & (players_teams_df["year"] == row["year"])].mode().iloc[0]["College"]

    return row

def merge_college(teams_df: pd.DataFrame, players_teams_df: pd.DataFrame, players_df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds to the teams dataframe the most common college of the players in the team.
    """
    players_teams_df.insert(1, "College", pd.NA)
    teams_df.insert(2, "College Mode", pd.NA)

    # Merge the players_teams_df with the players_df to get the college of each player.
    players_teams_df = players_teams_df.apply(lambda x: merge_colleges(x, players_df), axis=1)
    # Calculate the mode
    teams_df = teams_df.apply(lambda x: count_colleges(x, players_teams_df), axis=1)

    return teams_df

def calculate_player_prev_stats(players_teams_df: pd.DataFrame) -> pd.DataFrame:
    """
    Using exponential moving average, calculate the previous stats of the players.
    Replaces the stats columns with the previous stats.
    """

    stats = [
        "GP","GS","minutes","points","oRebounds","dRebounds",
        "rebounds","assists","steals","blocks","turnovers","PF",
        "fgAttempted","fgMade","ftAttempted","ftMade","threeAttempted",
        "threeMade","dq", "Award Count", # Non Post stats
        "PostGP","PostGS","PostMinutes","PostPoints","PostoRebounds",
        "PostdRebounds","PostRebounds","PostAssists","PostSteals",
        "PostBlocks","PostTurnovers","PostPF","PostfgAttempted",
        "PostfgMade","PostftAttempted","PostftMade",
        "PostthreeAttempted","PostthreeMade","PostDQ" # Post stats
    ]

    for stat in stats:
        #players_teams_df[stat] = players_teams_df.sort_values('year').groupby(by=['playerID'])[stat].expanding().mean().reset_index()[stat]
        players_teams_df[stat] = (
            players_teams_df
            .sort_values('year')
            .groupby(by=['playerID'])[stat]
            .apply(lambda x: x.ewm(alpha=0.7, adjust=False).mean()) # Alpha maior = mais peso para os valores mais recentes | Adjust faria os valores serem normalizados
            .reset_index(level=0, drop=True)
        )

    players_teams_df[stats] = players_teams_df.groupby('playerID')[stats].shift(periods=1)
    #players_teams_df["year"] = players_teams_df["year"].apply(lambda x: x+1)
    return players_teams_df

def calculate_team_players_average(teams_df: pd.DataFrame, players_teams_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the average of the previous years' player stats that belong to the team and add it to the team row.
    """
    stats = [
        "GP","GS","minutes","points","oRebounds","dRebounds",
        "rebounds","assists","steals","blocks","turnovers","PF",
        "fgAttempted","fgMade","ftAttempted","ftMade","threeAttempted",
        "threeMade","dq", "Award Count", # Non Post stats
        "PostGP","PostGS","PostMinutes","PostPoints","PostoRebounds",
        "PostdRebounds","PostRebounds","PostAssists","PostSteals",
        "PostBlocks","PostTurnovers","PostPF","PostfgAttempted",
        "PostfgMade","PostftAttempted","PostftMade",
        "PostthreeAttempted","PostthreeMade","PostDQ" # Post stats
    ]

    mean_series = players_teams_df.groupby(by=["tmID", "year"])[stats].mean()
    teams_df = teams_df.reset_index(drop=True)
    teams_df = teams_df.merge(mean_series, how="inner", on=["tmID", "year"], validate="1:1")
    teams_df = teams_df[teams_df["year"] > 1]
    teams_df.to_csv("wowe.csv")
    #teams_df[stat] = teams_df[['tmID', "year"]].map(mean_series)

    return teams_df