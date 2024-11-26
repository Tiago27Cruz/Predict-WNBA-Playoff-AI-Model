import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

barwidth = 0.25
plt.subplots(figsize =(12, 8))

def new_players_per_year(players_teams_df: pd.DataFrame) -> pd.DataFrame:
    known_players = set()
    new_players_l = []
    total_players_l = []

    for year in range(1, 11):
        new_players = 0
        total_players = 0
        #known_players = set()
        filtered_df = players_teams_df[players_teams_df["year"] == year]
        for index, row in filtered_df.iterrows():
            total_players += 1
            if row["playerID"] not in known_players:
                known_players.add(row["playerID"])
                new_players = new_players+1
        new_players_l.append(new_players)
        total_players_l.append(total_players)
        print(f"{new_players} new players in year {year}")

    br1 = np.arange(len(new_players_l))  
    br2 = [x + barwidth/2 for x in br1]
    br3 = [x - barwidth/2 for x in br1]

    print("helo")
    plt.bar(br3, total_players_l, color ='b', width = barwidth, 
        label ='Total Players') 
    print("helo2")
    plt.bar(br2, new_players_l, color ='orange', width = barwidth, 
        label ='New Players')  
    plt.xlabel('Year', fontweight ='bold', fontsize = 15) 
    plt.ylabel('Player amount', fontweight ='bold', fontsize = 15) 
    plt.xticks(range(len(new_players_l)), 
        list(map(lambda x: x+1, br1)))
    plt.savefig("newplayers.png")
    plt.close()

def new_teams_per_year(teams_df: pd.DataFrame) -> pd.DataFrame:
    known_teams = set()
    new_teams_l = []
    total_teams_l = []

    for year in range(1, 11):
        new_teams = 0
        total_teams = 0
        #known_teams = set()
        filtered_df = teams_df[teams_df["year"] == year]
        for index, row in filtered_df.iterrows():
            total_teams += 1
            if row["tmID"] not in known_teams:
                known_teams.add(row["tmID"])
                new_teams = new_teams+1
        new_teams_l.append(new_teams)
        total_teams_l.append(total_teams)
        print(f"{new_teams} new teams in year {year}")

    br1 = np.arange(len(new_teams_l))  
    br2 = [x + barwidth/2 for x in br1]
    br3 = [x - barwidth/2 for x in br1]

    plt.bar(br3, total_teams_l, color ='b', width = barwidth, 
        label ='Total Teams') 
    plt.bar(br2, new_teams_l, color ='orange', width = barwidth, 
        label ='New Teams')  
    plt.xlabel('Year', fontweight ='bold', fontsize = 15) 
    plt.ylabel('Team amount', fontweight ='bold', fontsize = 15) 
    plt.xticks(range(len(new_teams_l)), 
        list(map(lambda x: x+1, br1)))
    plt.savefig("newteams.png")
    plt.close()

#new_players_per_year(pd.read_csv("../data/players_teams.csv"))

new_teams_per_year(pd.read_csv("../data/teams.csv"))