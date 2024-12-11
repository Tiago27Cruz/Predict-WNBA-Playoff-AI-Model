import pandas as pd
from data_clean import *
import numpy as np

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
    filtered_awards = awards_df[awards_df["year"] <= row["year"]]
    grouped_df = filtered_awards.groupby("playerID").size()
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
        "PostthreeAttempted","PostthreeMade","PostDQ", # Post stats
        "PostthreeRatio", "PostfgRatio", "PostftRatio", "ThreeRatio", "fgRatio", "ftRatio"
    ]

    stats = [stat for stat in stats if stat in players_teams_df]

    for stat in stats:
        
        if stat[:4] == "Post":
            if stat not in ["PostMinutes", "PostGS", "PostGP"]:
                players_teams_df[stat] = players_teams_df[stat] * players_teams_df["PostMinutes"] / players_teams_df["PostGP"]
        else:
            if stat not in ["minutes", "Award Count", "GP", "GS"]:
                players_teams_df[stat] = players_teams_df[stat] * players_teams_df["minutes"] / players_teams_df["GP"]
        
        #players_teams_df[stat] = players_teams_df.sort_values('year').groupby(by=['playerID'])[stat].expanding().mean().reset_index()[stat]
        players_teams_df[stat] = (
            players_teams_df
            .sort_values(['year', 'stint'])
            .groupby(by=['playerID'])[stat]
            .apply(lambda x: x.ewm(alpha=0.6, adjust=False).mean()) # Alpha maior = mais peso para os valores mais recentes | Adjust faria os valores serem normalizados
            .reset_index(level=0, drop=True)
        )

    players_teams_df[stats] = players_teams_df.groupby('playerID')[stats].shift(periods=1)
    #players_teams_df = players_teams_df.dropna()
    #players_teams_df["year"] = players_teams_df["year"].apply(lambda x: x+1)
    return players_teams_df

def bad_calculate_player_prev_stats(players_teams_df: pd.DataFrame) -> pd.DataFrame:
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
        "PostthreeAttempted","PostthreeMade","PostDQ", # Post stats
        "PostthreeRatio", "PostfgRatio", "PostftRatio", "ThreeRatio", "fgRatio", "ftRatio"
    ]

    stats = [stat for stat in stats if stat in players_teams_df]

    for stat in stats:
        
        if stat[:4] == "Post":
            if stat not in ["PostMinutes", "PostGS", "PostGP"]:
                players_teams_df[stat] = players_teams_df[stat] * players_teams_df["PostMinutes"] / players_teams_df["PostGP"]
        else:
            if stat not in ["minutes", "Award Count", "GP", "GS"]:
                players_teams_df[stat] = players_teams_df[stat] * players_teams_df["minutes"] / players_teams_df["GP"]
        
        players_teams_df[stat] = players_teams_df.sort_values('year').groupby(by=['playerID'])[stat].expanding().mean().reset_index()[stat]

    players_teams_df[stats] = players_teams_df.groupby('playerID')[stats].shift(periods=1)

    return players_teams_df

def remove_outliers_zscore(df, threshold=3):
    z_scores = np.abs((df - df.mean()) / df.std())
    filtered_df = df[(z_scores < threshold).all(axis=1)]
    num_removed = len(filtered_df)
    print(f"Number of values: {num_removed}")
    return filtered_df

def remove_outliers_iqr(df, factor=1.5):
    Q1 = df.quantile(0.25)
    Q3 = df.quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - (factor * IQR)
    upper_bound = Q3 + (factor * IQR)
    filtered_df = df[~((df < lower_bound) | (df > upper_bound)).any(axis=1)]
    #num_removed = len(df) - len(filtered_df)
    #print(f"Number of values removed: {num_removed}")
    return filtered_df

def calculate_mean_without_outliers(players_teams_df, stats, factor=1.5):
    # Group by tmID and year
    grouped = players_teams_df.groupby(by=["tmID", "year"])

    # Initialize an empty DataFrame to store the results
    mean_series_list = []

    # Iterate over each group
    for name, group in grouped:
        #print(group)
        # Remove outliers
        #print("Team: ", name[0], "Year: ", name[1])
        filtered_group = remove_outliers_iqr(group[stats], factor)
        
        # Calculate the mean of the filtered group
        mean_values = filtered_group.mean()
        
        # Create a Series with the mean values and add tmID and year
        mean_values["tmID"] = name[0]
        mean_values["year"] = name[1]
        
        # Append the mean values to the result list
        mean_series_list.append(mean_values)

    # Concatenate the list of Series into a DataFrame
    mean_series = pd.DataFrame(mean_series_list)

    return mean_series

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
        "PostthreeAttempted","PostthreeMade","PostDQ", # Post stats
        "PostthreeRatio", "PostfgRatio", "PostftRatio", "ThreeRatio", "fgRatio", "ftRatio"
    ]

    stats = [stat for stat in stats if stat in players_teams_df]

    #players_teams_df = players_teams_df[players_teams_df["minutes"] > 20]
    mean_series = calculate_mean_without_outliers(players_teams_df, stats, 5)
    #mean_series = players_teams_df.groupby(by=["tmID", "year"])[stats].mean()


    teams_df = teams_df.reset_index(drop=True)
    teams_df = teams_df.merge(mean_series, how="inner", on=["tmID", "year"], validate="1:1")
    teams_df = teams_df[teams_df["year"] > 1]


    return teams_df

def calculate_coach_prev_stats(coaches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Using exponential moving average, calculate the previous stats of the coaches.
    Replaces the stats columns with the previous stats.
    """

    stats = [
        "wr", "pwr"
    ]

    for stat in stats:
        #players_teams_df[stat] = players_teams_df.sort_values('year').groupby(by=['playerID'])[stat].expanding().mean().reset_index()[stat]
        coaches_df[stat] = (
            coaches_df
            .sort_values('year')
            .groupby(by=['coachID'])[stat]
            .apply(lambda x: x.ewm(alpha=0.7, adjust=False).mean()) # Alpha maior = mais peso para os valores mais recentes | Adjust faria os valores serem normalizados
            .reset_index(level=0, drop=True)
        )
    # TODO: change alpha
    coaches_df[stats] = coaches_df.groupby('coachID')[stats].shift(periods=1)
    #coaches_df = coaches_df.dropna()
    #players_teams_df["year"] = players_teams_df["year"].apply(lambda x: x+1)
    return coaches_df


def bad_calculate_coach_prev_stats(coaches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Using exponential moving average, calculate the previous stats of the coaches.
    Replaces the stats columns with the previous stats.
    """

    stats = [
        "won","lost","post_wins","post_losses"
    ]

    for stat in stats:
        coaches_df[stat] = coaches_df.sort_values('year').groupby(by=['coachID'])[stat].expanding().mean().reset_index()[stat]

    coaches_df[stats] = coaches_df.groupby('coachID')[stats].shift(periods=1)

    return coaches_df

def calculate_team_coaches_average(teams_df: pd.DataFrame, coaches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the average of the previous years' coach stats that belong to the team and add it to the team row.
    """
    stats = [
        "wr", "pwr"
    ]

    mean_series = coaches_df.groupby(by=["tmID", "year"])[stats].mean()
    mean_series = mean_series.rename(columns={stat: f'coach_{stat}' for stat in stats})
    
    teams_df = teams_df.reset_index(drop=True)

    

    teams_df = teams_df.merge(mean_series, how="inner", on=["tmID", "year"], validate="1:1")
    teams_df = teams_df[teams_df["year"] > 1]
    teams_df["coach_pwr"] = teams_df["coach_pwr"].fillna(0)
    #teams_df.to_csv("wowe.csv")
    #teams_df[stat] = teams_df[['tmID', "year"]].map(mean_series)

    return teams_df

def bad_calculate_team_coaches_average(teams_df: pd.DataFrame, coaches_df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate the average of the previous years' coach stats that belong to the team and add it to the team row.
    """
    stats = [
        "won","lost","post_wins","post_losses"
    ]

    mean_series = coaches_df.groupby(by=["tmID", "year"])[stats].mean()
    mean_series = mean_series.rename(columns={stat: f'coach_{stat}' for stat in stats})
    
    teams_df = teams_df.reset_index(drop=True)

    

    teams_df = teams_df.merge(mean_series, how="inner", on=["tmID", "year"], validate="1:1")
    teams_df = teams_df[teams_df["year"] > 1]
    return teams_df

# Y11

def ccps_y11(coaches_df: pd.DataFrame, coaches_y11, alpha) -> pd.DataFrame:
    """
    Using exponential moving average, calculate the previous stats of the coaches.
    Replaces the stats columns with the previous stats.
    """

    stats = [
        "wr", "pwr"
    ]

    for stat in stats:
        
        coaches_df[stat] = (
            coaches_df
            .sort_values(['year','stint'])
            .groupby(by=['coachID'])[stat]
            .apply(lambda x: x.ewm(alpha=alpha, adjust=False).mean())
            .reset_index(level=0, drop=True)
        )
    coaches_df = pd.concat([coaches_df, coaches_y11], ignore_index=True, sort=False)
    coaches_df[stats] = coaches_df.groupby('coachID')[stats].shift(periods=1)

    return coaches_df

def cpps_y11(players_teams_df: pd.DataFrame, players_teams_y11, alpha) -> pd.DataFrame:
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
        "PostthreeAttempted","PostthreeMade","PostDQ", # Post stats
        "PostthreeRatio", "PostfgRatio", "PostftRatio", "ThreeRatio", "fgRatio", "ftRatio"
    ]

    stats = [stat for stat in stats if stat in players_teams_df]

    for stat in stats:
        
        if stat[:4] == "Post":
            if stat not in ["PostMinutes", "PostGS", "PostGP"]:
                players_teams_df[stat] = players_teams_df[stat] * players_teams_df["PostMinutes"] / players_teams_df["PostGP"]
        else:
            if stat not in ["minutes", "Award Count", "GP", "GS"]:
                players_teams_df[stat] = players_teams_df[stat] * players_teams_df["minutes"] / players_teams_df["GP"]
        
        
        players_teams_df[stat] = (
            players_teams_df
            .sort_values(['year','stint'])
            .groupby(by=['playerID'])[stat]
            .apply(lambda x: x.ewm(alpha=alpha, adjust=False).mean())
            .reset_index(level=0, drop=True)
        )

    players_teams_df = pd.concat([players_teams_df, players_teams_y11], ignore_index=True, sort=False)
    players_teams_df[stats] = players_teams_df.groupby('playerID')[stats].shift(periods=1)

    return players_teams_df

def cewm_y11(teams_df, teams_y11, alpha1, alpha2):
    stats_list = [
        "o_fgm","o_fga","o_ftm","o_fta","o_3pm","o_3pa",
        "o_oreb","o_dreb","o_reb","o_asts","o_pf","o_stl","o_to",
        "o_blk","o_pts","d_fgm","d_fga","d_ftm","d_fta","d_3pm",
        "d_3pa","d_oreb","d_dreb","d_reb","d_asts","d_pf","d_stl",
        "d_to","d_blk","d_pts"
    ]

    team_stats = teams_df[stats_list].copy()
    team_stats[["tmID", "year"]] = teams_df[["tmID", "year"]]
    team_stats["wr"] = teams_df["won"] / (teams_df["won"] + teams_df["lost"])

    alpha = 0.085

    for stat in stats_list:
        team_stats[stat] = (
            team_stats
            .sort_values('year')
            .groupby(by=['tmID'])[stat]
            .apply(lambda x: x.ewm(alpha=alpha1, adjust=False).mean())
            .reset_index(level=0, drop=True)
        )
        
    team_stats["wr"] = (
        team_stats
        .sort_values('year')
        .groupby(by=['tmID'])["wr"]
        .apply(lambda x: x.ewm(alpha=alpha2, adjust=False).mean())
        .reset_index(level=0, drop=True)
    )

    teams_df = pd.concat([teams_df, teams_y11], ignore_index=True, sort=False)
    teams_df[stats_list] = team_stats.groupby('tmID')[stats_list].shift(periods=1)
    teams_df["wr"] = team_stats.groupby('tmID')["wr"].shift(periods=1)

    return teams_df
