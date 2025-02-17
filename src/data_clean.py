import pandas as pd

### Data Cleaning Functions ###

def clean_lgId(df: pd.DataFrame) -> pd.DataFrame:
    """
    Drop the lgID column from the dataframe.
    """
    return df.drop(["lgID"], axis=1)

def drop_forbidden_columns(teams_df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops the columns that are related to the playoffs information of a team from teh team df.
    """
    return teams_df.drop(columns=["firstRound", "semis", "finals", "rank"])

def drop_string_columns(teams_df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops the string columns from the team df: lgID, tmID, franchID, confID, divID, arena, name.
    """
    return teams_df.drop(columns=["lgID","tmID","franchID","confID","divID","arena", "name"])

def drop_team_info(teams_df: pd.DataFrame) -> pd.DataFrame:
    """
    Drops the columns that are related to the team information from the team df.
    """
    columns = ["tmORB","tmDRB","tmTRB","opptmORB",
                "opptmDRB","opptmTRB","won","lost","GP","homeW","homeL",
                "awayW","awayL","confW","confL","min","attend", "lgID",
                "franchID", "divID", "arena", "name", "rank",
                "firstRound", "semis", "finals", "seeded"]
    
    return teams_df.drop(columns=columns)

def transform_ch_stats_in_ratio(coaches_df: pd.DataFrame) -> pd.DataFrame:
    coaches_df["wr"] = coaches_df["won"] / (coaches_df["won"] + coaches_df["lost"])
    coaches_df["pwr"] = (coaches_df["post_wins"] / (coaches_df["post_wins"] + coaches_df["post_losses"])).fillna(0)
    coaches_df.drop(columns=["won", "lost", "post_wins", "post_losses"], inplace=True)

    return coaches_df

def transform_pl_stats_in_ratio(players_teams_df: pd.DataFrame) -> pd.DataFrame:
    players_teams_df["PostthreeRatio"] = (players_teams_df["PostthreeMade"] / players_teams_df["PostthreeAttempted"]).fillna(0)
    players_teams_df.drop(columns=["PostthreeMade", "PostthreeAttempted"], inplace=True)
    players_teams_df["PostfgRatio"] = (players_teams_df["PostfgMade"] / players_teams_df["PostfgAttempted"]).fillna(0)
    players_teams_df.drop(columns=["PostfgMade", "PostfgAttempted"], inplace=True)
    players_teams_df["PostftRatio"] = (players_teams_df["PostftMade"] / players_teams_df["PostftAttempted"]).fillna(0)
    players_teams_df.drop(columns=["PostftMade", "PostftAttempted"], inplace=True)

    # Regular
    players_teams_df["ThreeRatio"] = players_teams_df["threeMade"] / players_teams_df["threeAttempted"]
    players_teams_df.drop(columns=["threeMade", "threeAttempted"], inplace=True)
    players_teams_df["fgRatio"] = players_teams_df["fgMade"] / players_teams_df["fgAttempted"]
    players_teams_df.drop(columns=["fgMade", "fgAttempted"], inplace=True)
    players_teams_df["ftRatio"] = players_teams_df["ftMade"] / players_teams_df["ftAttempted"]
    players_teams_df.drop(columns=["ftMade", "ftAttempted"], inplace=True)

    return players_teams_df

def transform_pl_ch_stats_in_ratio(teams_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the stats of players and coaches in the teams_df to ratios based on attempted vs made and calculates wr and post-wr.
    """

    # Coaches
    #teams_df["coach_wr"] = teams_df["coach_won"] / (teams_df["coach_won"] + teams_df["coach_lost"])
    #teams_df["coach_pwr"] = teams_df["coach_post_wins"] / (teams_df["coach_post_wins"] + teams_df["coach_post_losses"]).fillna(0)
    #teams_df.drop(columns=["coach_won", "coach_lost", "coach_post_wins", "coach_post_losses"], inplace=True)

    # Post
    teams_df["PostthreeRatio"] = teams_df["PostthreeMade"] / teams_df["PostthreeAttempted"]
    teams_df.drop(columns=["PostthreeMade", "PostthreeAttempted"], inplace=True)
    teams_df["PostfgRatio"] = teams_df["PostfgMade"] / teams_df["PostfgAttempted"]
    teams_df.drop(columns=["PostfgMade", "PostfgAttempted"], inplace=True)
    teams_df["PostftRatio"] = teams_df["PostftMade"] / teams_df["PostftAttempted"]
    teams_df.drop(columns=["PostftMade", "PostftAttempted"], inplace=True)

    # Regular
    teams_df["ThreeRatio"] = teams_df["threeMade"] / teams_df["threeAttempted"]
    teams_df.drop(columns=["threeMade", "threeAttempted"], inplace=True)
    teams_df["fgRatio"] = teams_df["fgMade"] / teams_df["fgAttempted"]
    teams_df.drop(columns=["fgMade", "fgAttempted"], inplace=True)
    teams_df["ftRatio"] = teams_df["ftMade"] / teams_df["ftAttempted"]
    teams_df.drop(columns=["ftMade", "ftAttempted"], inplace=True)

    return teams_df

def transform_team_stats_in_ratio(teams_df: pd.DataFrame) -> pd.DataFrame:
    """
    Transforms the stats of the team in the teams_df to ratios based on attempted vs made.
    """

    # Offensive Statistics
    teams_df["o_fgm_ratio"] = teams_df["o_fgm"] / teams_df["o_fga"]
    teams_df["o_ftm_ratio"] = teams_df["o_ftm"] / teams_df["o_fta"]
    teams_df["o_3pm_ratio"] = teams_df["o_3pm"] / teams_df["o_3pa"]
    
    # Defensive Statistics
    teams_df["d_fgm_ratio"] = teams_df["d_fgm"] / teams_df["d_fga"]
    teams_df["d_ftm_ratio"] = teams_df["d_ftm"] / teams_df["d_fta"]
    teams_df["d_3pm_ratio"] = teams_df["d_3pm"] / teams_df["d_3pa"]

    # General
    teams_df["wr"] = teams_df["won"] / (teams_df["won"] + teams_df["lost"])
    teams_df["home_wr"] = teams_df["homeW"] / (teams_df["homeW"] + teams_df["homeL"])
    teams_df["away_wr"] = teams_df["awayW"] / (teams_df["awayW"] + teams_df["awayL"])
    teams_df["conf_wr"] = teams_df["confW"] / (teams_df["confW"] + teams_df["confL"])
    
    teams_df.drop(columns=["o_fgm","o_fga","o_ftm","o_fta","o_3pm","o_3pa","d_fgm","d_fga","d_ftm","d_fta","d_3pm","d_3pa","won","lost","homeW","homeL","awayW","awayL","confW","confL"], inplace=True)

    return teams_df