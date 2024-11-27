from data_merge import *

### Data Preparation Functions ###

def prepare_expanding_average(teams_df: pd.DataFrame) -> pd.DataFrame:
    stats = ["o_fgm","o_fga","o_ftm","o_fta","o_3pm","o_3pa","o_oreb","o_dreb","o_reb","o_asts","o_pf","o_stl","o_to","o_blk","o_pts","d_fgm","d_fga","d_ftm","d_fta","d_3pm","d_3pa","d_oreb","d_dreb","d_reb","d_asts","d_pf","d_stl","d_to","d_blk","d_pts","tmORB","tmDRB","tmTRB","opptmORB","opptmDRB","opptmTRB","won","lost","GP","homeW","homeL","awayW","awayL","confW","confL","min","attend"]
    teams_df[stats] = teams_df.sort_values('year').groupby(by=['tmID'])[stats]\
        .expanding().mean().reset_index()[stats]
    
    teams_df[stats] = teams_df.groupby('tmID')[stats].shift(periods=1)
    stats.extend(["tmID", "year"])
    teams_df = teams_df[stats]

    return teams_df

def prepare_model_data_teams() -> pd.DataFrame:
    teams_df = pd.read_csv("../data/teams.csv")

    model_data = drop_forbidden_columns(prepare_expanding_average(teams_df))
    model_data = model_data[model_data["playoff"].notnull()]
    model_data = drop_string_columns(model_data)

    return model_data

def calculate_ewm(teams_df):
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
            .apply(lambda x: x.ewm(alpha=alpha, adjust=False).mean())
            .reset_index(level=0, drop=True)
        )
        teams_df[stat] = team_stats.groupby('tmID')[stat].shift(periods=1)
    team_stats["wr"] = (
        team_stats
        .sort_values('year')
        .groupby(by=['tmID'])["wr"]
        .apply(lambda x: x.ewm(alpha=alpha, adjust=False).mean())
        .reset_index(level=0, drop=True)
    )

    teams_df["wr"] = team_stats.groupby('tmID')["wr"].shift(periods=1)

    return teams_df

def prepare_data(fillna = True) -> pd.DataFrame:
    teams_df = pd.read_csv("../data/teams.csv")
    awards_df = pd.read_csv("../data/awards_players.csv")
    players_teams_df = pd.read_csv("../data/players_teams.csv")
    coaches_df = pd.read_csv("../data/coaches.csv")

    # Handling Teams csv values
    teams_df = calculate_ewm(teams_df)
    teams_df = drop_team_info(teams_df)

    # Handling Players + Awards csv values
    players_teams_df = merge_awards(players_teams_df, awards_df)
    players_teams_df = transform_pl_stats_in_ratio(players_teams_df)
    players_teams_df = calculate_player_prev_stats(players_teams_df)

    # Handling Coaches csv values
    coaches_df = transform_ch_stats_in_ratio(coaches_df)
    coaches_df = calculate_coach_prev_stats(coaches_df)
    
    # Merging
    teams_df = calculate_team_players_average(teams_df, players_teams_df)
    teams_df = calculate_team_coaches_average(teams_df, coaches_df)
    #teams_df = transform_pl_ch_stats_in_ratio(teams_df)

    # Others
    ## Handle useless rows
    teams_df = teams_df[teams_df["year"] > 1]
    ## Handle N/A Values if wanted
    ## TODO: Some other method of leading with N/A might give us better results (???) or only in some fields
    if (fillna): teams_df = teams_df.fillna(0)

    teams_df.to_csv("wewo.csv", index=False)

    ## Needed for models
    teams_df = teams_df.drop(columns="tmID")
    teams_df['playoff'] = teams_df['playoff'].map({'N':0,'Y':1})

    return teams_df
