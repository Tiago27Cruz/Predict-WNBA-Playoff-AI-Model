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