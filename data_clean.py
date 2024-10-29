import pandas as pd

def clean_lgId(df: pd.DataFrame) -> pd.DataFrame:
    return df.drop(["lgID"], axis=1)

awards_df = clean_lgId(pd.read_csv("data/awards_players.csv"))
coaches_df = pd.read_csv("data/coaches.csv")
players_teams_df = clean_lgId(pd.read_csv("data/players_teams.csv"))
players_df = pd.read_csv("data/players.csv")
teams_df = clean_lgId(pd.read_csv("data/teams.csv"))