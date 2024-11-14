from data_prep import *

from sklearn.metrics import accuracy_score, make_scorer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
import numpy as np
from sklearn.inspection import permutation_importance
from matplotlib import pyplot as plt

### Model Training Functions ###

def my_custom_loss(ground_truth, predictions):
    total = sum(ground_truth)
    total_pred = sum(predictions)
    predictions = list(map(lambda x: total*x/total_pred, predictions))
    return np.sum(np.abs(ground_truth - predictions))

loss = make_scorer(my_custom_loss, greater_is_better=False, response_method="predict_proba")

def team_values_model_rf():
    model_data = prepare_model_data_teams()
    X = model_data.drop('playoff', axis=1)
    y = model_data['playoff'].map({'N':0,'Y':1})

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(random_state=42)
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
    model_data = prepare_model_data_teams()
    X = model_data.drop('playoff', axis=1)
    y = model_data['playoff'].map({'N':0,'Y':1})

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }

    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=2)
    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Best parameters found:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)
    print("Test set accuracy:", accuracy)

def player_values_model_rf():
    df = prepare_model_data_players_rf().drop(columns=["year"])

    X = df.drop('playoff', axis=1)
    y = df['playoff']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    importance = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)

    forest_importances = pd.Series(importance, index=X.columns)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()

    result = permutation_importance(
        rf, X_test, y_test, n_repeats=20, random_state=42, n_jobs=2
    )
    forest_importances = pd.Series(result.importances_mean, index=X.columns)

    fig, ax = plt.subplots()
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.show()

    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)

def player_values_model_gs():
    df = prepare_model_data_players_rf()

    X = df.drop('playoff', axis=1)
    y = df['playoff'].map({'N':0,'Y':1})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }

    rf = RandomForestClassifier()
    grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, verbose=3, scoring="roc_auc")
    grid_search.fit(X_train, y_train)

    best_rf = grid_search.best_estimator_
    y_pred = best_rf.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    print("Best parameters found:", grid_search.best_params_)
    print("Best cross-validation score:", grid_search.best_score_)
    print("Test set accuracy:", accuracy)

def player_values_model_rf_custom_metric():
    df = prepare_model_data_players_rf()
    df['playoff'] = df['playoff'].map({'N':0,'Y':1})

    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }
    
    for year in range(3, 11):
        filtered_df = df[df["year"] < year].drop(columns=["year"])
        target_df = df[df["year"] == year].drop(columns=["year"])

        X_train = filtered_df.drop(columns=["playoff"])
        y_train = filtered_df["playoff"]

        X_test = target_df.drop(columns=["playoff"])
        y_test = target_df["playoff"]
        
        rf = RandomForestClassifier()
        #rf.fit(X_train, y_train)
        #pyplot.figure(dpi=1200)
        #tree.plot_tree(rf, feature_names=list(X_train))
        
        #pyplot.savefig(f"year{year}.png")
        grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5, n_jobs=-1, scoring="roc_auc")

        grid_search.fit(X_train, y_train)

        y_pred = grid_search.best_estimator_.predict_proba(X_test)[:,1]
        y_pred_sum = sum(y_pred)
        y_pred = list(map(lambda x: 8*x/y_pred_sum, y_pred))
        
        error = 0
        for i in range(len(y_pred)):
            error += abs(y_pred[i] - list(y_test)[i])

        print(f"predicting year {year}: error was {error}")

def global_model_rf():
    df = prepare_global_model()

    print(df)

    X = df.drop('playoff', axis=1)
    y = df['playoff'].map({'N':0,'Y':1})

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X_train, y_train)

    y_pred = rf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    print("Accuracy:", accuracy)
