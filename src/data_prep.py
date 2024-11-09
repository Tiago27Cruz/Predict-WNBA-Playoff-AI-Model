from data_merge import *

### Data Preparation Functions ###

def lag_row(df):
    df["playoff"] = df["playoff"].shift(-1)
    return df

def lag_playoffs(teams_df: pd.DataFrame):
    teams_df_grouped = teams_df.groupby('tmID')
    teams_df_grouped = teams_df_grouped.apply(lag_row)
    return teams_df_grouped

def prepare_expanding_average(teams_df: pd.DataFrame) -> pd.DataFrame:
    teams_df[["o_fgm","o_fga","o_ftm","o_fta","o_3pm","o_3pa","o_oreb","o_dreb","o_reb","o_asts","o_pf","o_stl","o_to","o_blk","o_pts","d_fgm","d_fga","d_ftm","d_fta","d_3pm","d_3pa","d_oreb","d_dreb","d_reb","d_asts","d_pf","d_stl","d_to","d_blk","d_pts","tmORB","tmDRB","tmTRB","opptmORB","opptmDRB","opptmTRB","won","lost","GP","homeW","homeL","awayW","awayL","confW","confL","min","attend"]] = teams_df.sort_values('year').groupby(by=['tmID'])[["o_fgm","o_fga","o_ftm","o_fta","o_3pm","o_3pa","o_oreb","o_dreb","o_reb","o_asts","o_pf","o_stl","o_to","o_blk","o_pts","d_fgm","d_fga","d_ftm","d_fta","d_3pm","d_3pa","d_oreb","d_dreb","d_reb","d_asts","d_pf","d_stl","d_to","d_blk","d_pts","tmORB","tmDRB","tmTRB","opptmORB","opptmDRB","opptmTRB","won","lost","GP","homeW","homeL","awayW","awayL","confW","confL","min","attend"]]\
        .expanding().mean().reset_index()[["o_fgm","o_fga","o_ftm","o_fta","o_3pm","o_3pa","o_oreb","o_dreb","o_reb","o_asts","o_pf","o_stl","o_to","o_blk","o_pts","d_fgm","d_fga","d_ftm","d_fta","d_3pm","d_3pa","d_oreb","d_dreb","d_reb","d_asts","d_pf","d_stl","d_to","d_blk","d_pts","tmORB","tmDRB","tmTRB","opptmORB","opptmDRB","opptmTRB","won","lost","GP","homeW","homeL","awayW","awayL","confW","confL","min","attend"]]
    
    teams_df["playoff"] = teams_df.groupby('tmID')['playoff'].shift(periods=-1)

    return teams_df

def prepare_model_data_teams() -> pd.DataFrame:
    teams_df = pd.read_csv("../data/teams.csv")

    model_data = drop_forbidden_columns(prepare_expanding_average(teams_df))
    model_data = model_data[model_data["playoff"].notnull()]
    model_data = drop_string_columns(model_data)

    return model_data

def prepare_model_data_players_rf() -> pd.DataFrame:
    teams_df = pd.read_csv("../data/teams.csv")
    awards_df = pd.read_csv("../data/awards_players.csv")
    players_teams_df = pd.read_csv("../data/players_teams.csv")
    coaches_df = pd.read_csv("../data/coaches.csv")

    teams_df = lag_playoffs(teams_df)
    teams_df = drop_team_info(teams_df)

    players_teams_df = merge_awards(players_teams_df, awards_df)
    players_teams_df = calculate_player_prev_stats(players_teams_df)

    teams_df = calculate_team_players_average(teams_df, players_teams_df)
    teams_df = calculate_team_coaches_average(teams_df, coaches_df)
    teams_df = transform_pl_ch_stats_in_ratio(teams_df)
    #print(teams_df)
    #teams_df.to_csv("wewo.csv")

    teams_df = teams_df.drop(columns="tmID")

    teams_df = teams_df[teams_df["playoff"].notnull()]
    teams_df["playoff"] = teams_df['playoff'].map({'N':0,'Y':1})

    return teams_df

def prepare_global_model():
    teams_df = pd.read_csv("../data/teams.csv")
    awards_df = pd.read_csv("../data/awards_players.csv")
    players_teams_df = pd.read_csv("../data/players_teams.csv")
    coaches_df = pd.read_csv("../data/coaches.csv")

    teams_df = drop_forbidden_columns(prepare_expanding_average(teams_df))
    teams_df = teams_df[teams_df["playoff"].notnull()]

    players_teams_df = merge_awards(players_teams_df, awards_df)
    players_teams_df = calculate_player_prev_stats(players_teams_df)

    teams_df = calculate_team_players_average(teams_df, players_teams_df)
    teams_df = calculate_team_coaches_average(teams_df, coaches_df)
    
    teams_df = transform_pl_ch_stats_in_ratio(teams_df)
    teams_df = transform_team_stats_in_ratio(teams_df)
    
    teams_df = drop_string_columns(teams_df)

    #teams_df.to_csv("wewo.csv")

    return teams_df


