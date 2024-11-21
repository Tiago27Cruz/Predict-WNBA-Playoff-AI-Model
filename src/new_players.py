import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

barwidth = 0.25
plt.subplots(figsize =(12, 8))

def new_players_per_year(players_teams_df: pd.DataFrame) -> pd.DataFrame:
    #known_players = set()
    new_players_l = []
    total_players_l = []

    for year in range(1, 11):
        new_players = 0
        total_players = 0
        known_players = set()
        filtered_df = players_teams_df[players_teams_df["year"] == year]
        for index, row in filtered_df.iterrows():
            total_players += 1
            if row["playerID"] not in known_players:
                known_players.add(row["playerID"])
                new_players = new_players+1
        new_players_l.append(new_players_l)
        total_players_l.append(total_players_l)
        print(f"{new_players} new players in year {year}")

    br1 = np.arange(len(new_players_l))  
    br2 = [x + barwidth for x in br1]

    print("helo")
    plt.bar(br1, total_players_l, color ='b', width = barwidth, 
        edgecolor ='grey', label ='Total Players') 
    print("helo2")
    plt.bar(br2, new_players_l, color ='o', width = barwidth, 
        edgecolor ='grey', label ='New Players')  
    plt.show()
    plt.savefig("newplayers.png")
    plt.close()

new_players_per_year(pd.read_csv("../data/players_teams.csv"))