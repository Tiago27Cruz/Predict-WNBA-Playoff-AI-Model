import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

def lag_row(df):
    df["playoff"] = df["playoff"].shift(-1)
    return df

def lag_playoffs(teams_df: pd.DataFrame):
    teams_df_grouped = teams_df.groupby('tmID')
    teams_df_grouped = teams_df_grouped.apply(lag_row)
    return teams_df_grouped

def merge_to_teams(teams_df: pd.DataFrame, other_df: pd.DataFrame) -> pd.DataFrame:
    return teams_df.merge(other_df, on=["year", "tmID"])

def add_coaches(row: pd.Series, coaches_df: pd.DataFrame) -> pd.Series:
    year = row["year"]
    tmID = row["tmID"]
    coaches = coaches_df[(coaches_df["year"] == year) & (coaches_df["tmID"] ==tmID)]
    coach_no = len(coaches)

    if coach_no == 3:
        print("THERE ARE THREE COACHES")

    if coach_no == 1:
        row["Coach1"] = coaches.iloc[0]["coachID"]
    if coach_no == 2:
        print(coaches.iloc[1]["coachID"])
        row["Coach2"] = coaches.iloc[1]["coachID"]

    return row

def merge_coaches(teams_df: pd.DataFrame, coaches_df: pd.DataFrame) -> pd.DataFrame:
    teams_df.insert(5, "Coach1", pd.NA)
    teams_df.insert(5, "Coach2", pd.NA)

    teams_df = teams_df.apply(lambda x: add_coaches(x, coaches_df), axis=1)

    return teams_df

def count_awards(row: pd.Series, awards_df: pd.DataFrame) -> pd.Series:
    grouped_df = awards_df.groupby("playerID").size()
    row["Award Count"] = grouped_df.loc[row["playerID"]] if row["playerID"] in grouped_df.index else 0

    return row

def merge_awards(teams_df: pd.DataFrame, players_teams_df: pd.DataFrame, awards_df: pd.DataFrame) -> pd.DataFrame:
    players_teams_df.insert(5, "Award Count", 0)

    players_teams_df = players_teams_df.apply(lambda x: count_awards(x, awards_df), axis=1)

    return players_teams_df

def merge_colleges(row: pd.Series, players_df: pd.DataFrame) -> pd.Series:
    filtered = players_df[players_df["bioID"] == row["playerID"]].reset_index()
    row["College"] = filtered.at[0, "college"]

    return row

def count_colleges(row: pd.Series, players_teams_df: pd.DataFrame) -> pd.Series:
    row["College Mode"] = players_teams_df[(players_teams_df["tmID"] == row["tmID"]) & (players_teams_df["year"] == row["year"])].mode().iloc[0]["College"]

    return row

def merge_college(teams_df: pd.DataFrame, players_teams_df: pd.DataFrame, players_df: pd.DataFrame) -> pd.DataFrame:
    players_teams_df.insert(5, "College", pd.NA)
    teams_df.insert(5, "College Mode", pd.NA)

    players_teams_df = players_teams_df.apply(lambda x: merge_colleges(x, players_df), axis=1)
    teams_df = teams_df.apply(lambda x: count_colleges(x, players_teams_df), axis=1)

    return teams_df

def prepare_expanding_average(teams_df: pd.DataFrame) -> pd.DataFrame:
    teams_df[["o_fgm","o_fga","o_ftm","o_fta","o_3pm","o_3pa","o_oreb","o_dreb","o_reb","o_asts","o_pf","o_stl","o_to","o_blk","o_pts","d_fgm","d_fga","d_ftm","d_fta","d_3pm","d_3pa","d_oreb","d_dreb","d_reb","d_asts","d_pf","d_stl","d_to","d_blk","d_pts","tmORB","tmDRB","tmTRB","opptmORB","opptmDRB","opptmTRB","won","lost","GP","homeW","homeL","awayW","awayL","confW","confL","min","attend"]] = teams_df.sort_values('year').groupby(by=['tmID'])[["o_fgm","o_fga","o_ftm","o_fta","o_3pm","o_3pa","o_oreb","o_dreb","o_reb","o_asts","o_pf","o_stl","o_to","o_blk","o_pts","d_fgm","d_fga","d_ftm","d_fta","d_3pm","d_3pa","d_oreb","d_dreb","d_reb","d_asts","d_pf","d_stl","d_to","d_blk","d_pts","tmORB","tmDRB","tmTRB","opptmORB","opptmDRB","opptmTRB","won","lost","GP","homeW","homeL","awayW","awayL","confW","confL","min","attend"]]\
        .expanding().mean().reset_index()[["o_fgm","o_fga","o_ftm","o_fta","o_3pm","o_3pa","o_oreb","o_dreb","o_reb","o_asts","o_pf","o_stl","o_to","o_blk","o_pts","d_fgm","d_fga","d_ftm","d_fta","d_3pm","d_3pa","d_oreb","d_dreb","d_reb","d_asts","d_pf","d_stl","d_to","d_blk","d_pts","tmORB","tmDRB","tmTRB","opptmORB","opptmDRB","opptmTRB","won","lost","GP","homeW","homeL","awayW","awayL","confW","confL","min","attend"]]
    
    teams_df["playoff"] = teams_df.groupby('tmID')['playoff'].shift(periods=-1)

    return teams_df

def drop_forbidden_columns(teams_df: pd.DataFrame) -> pd.DataFrame:
    return teams_df.drop(columns=["firstRound", "semis", "finals", "rank"])

def drop_string_columns(teams_df: pd.DataFrame) -> pd.DataFrame:
    return teams_df.drop(columns=["lgID","tmID","franchID","confID","divID","arena", "name"])

def team_values_model_rf():
    model_data = drop_forbidden_columns(prepare_expanding_average(pd.read_csv("data/teams.csv")))
    model_data = model_data[model_data["playoff"].notnull()]
    model_data = drop_string_columns(model_data)
    X = model_data.drop('playoff', axis=1)
    y = model_data['playoff'].map({'N':0,'Y':1})

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    rf = RandomForestClassifier()
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

def team_values_model_gs():
    """
    Train a Random Forest model using GridSearchCV to predict playoffs qualification.

    This function performs the following steps:
    - Loads and preprocesses team data.
    - Splits the data into training and test sets.
    - Defines a parameter grid for hyperparameter tuning.
    - Performs grid search with cross-validation to find the best parameters.
    - Trains the Random Forest classifier with the best parameters.
    - Evaluates the model on the test set and prints the results.
    """

    #merge_coaches(pd.read_csv("data/teams.csv"), pd.read_csv("data/coaches.csv"))

    #print(merge_college(pd.read_csv("data/teams.csv"), pd.read_csv("data/players_teams.csv"), pd.read_csv("data/players.csv")))
    model_data = drop_forbidden_columns(prepare_expanding_average(pd.read_csv("data/teams.csv")))
    model_data = model_data[model_data["playoff"].notnull()]
    model_data = drop_string_columns(model_data)
    X = model_data.drop('playoff', axis=1)
    y = model_data['playoff'].map({'N':0,'Y':1})

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }

    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1)
    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Best parameters found:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)
    print("Test set accuracy:", accuracy)


if __name__ == "__main__":
    #team_values_model_gs()
    team_values_model_rf()
