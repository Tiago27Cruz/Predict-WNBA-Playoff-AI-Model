from data_prep import *

from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV

### Model Training Functions ###

def team_values_model_rf():
    model_data = prepare_model_data_teams()
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
    model_data = prepare_model_data_teams()
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

def player_values_model_rf():
    teams_df = pd.read_csv("data/teams.csv")
    coaches_df = pd.read_csv("data/coaches.csv")
    players_df = pd.read_csv("data/players.csv")
    awards_df = pd.read_csv("data/awards_players.csv")
    players_teams_df = pd.read_csv("data/players_teams.csv")
