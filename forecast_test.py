import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing

to_transform = ["points"]

def predict_player(name: str, players_teams_df: pd.DataFrame):
    filtered_df = players_teams_df[players_teams_df["playerID"] == name]
    for attr in to_transform:
        filtered_df[attr] = filtered_df[attr]/filtered_df["minutes"]
    
    filtered_df = filtered_df.reset_index()
    print(filtered_df["points"])
    
    model = ExponentialSmoothing(filtered_df["points"].iloc[:-1], initialization_method="estimated").fit()

    return model.forecast(1)
    

#pagemu01w
print(predict_player("swoopsh01w", pd.read_csv("data/players_teams.csv")))
