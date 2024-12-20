import optuna
import sklearn
from xgboost import XGBClassifier
from data_prep import *
from analysis import *

from sklearn.metrics import accuracy_score, make_scorer
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, AdaBoostClassifier, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import tree 
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
from sklearn.svm import SVC
from sklearn import svm
from sklearn.neural_network import MLPClassifier
from itertools import product
import xgboost as xgb

### Utils for Models ###

def my_custom_loss(ground_truth, predictions):
    total = sum(ground_truth)
    total_pred = sum(predictions)
    predictions = list(map(lambda x: total*x/total_pred, predictions))
    return np.sum(np.abs(ground_truth - predictions))

loss = make_scorer(my_custom_loss, greater_is_better=False, response_method="predict_proba")

def custom_split(df, year, usepca):
    """
        Split the data into training and test sets based on the year.
    """
    if year == 11: filtered_df = df[df["year"] < year].drop(columns=["year", "confID", "tmID"])
    else: filtered_df = df[df["year"] < year].drop(columns=["year", "confID"])

    target_df = df[df["year"] == year].drop(columns=["year"])

    X_train = filtered_df.drop(columns=["playoff"])
    y_train = filtered_df["playoff"]
    cols = X_train.columns

    pca = PCA(n_components=11)
    scaler = StandardScaler()
    if usepca:
        X_train = pca.fit_transform(scaler.fit_transform(X_train))

        pcas = pd.DataFrame(pca.components_,columns=cols)
    
    # Year 11 doesn't have the playoff column
    if(year == 11):
        X_test = target_df.drop(columns=["confID", "tmID", "playoff"])
        y_test = target_df.filter(items=["confID", "tmID", "playoff"])
    else:
        X_test = target_df.drop(columns=["playoff", "confID"])
        y_test = target_df.filter(items=["playoff", "confID"])

    if usepca:
        X_test = pca.transform(scaler.transform(X_test))
        
    return X_train, y_train, X_test, y_test, filtered_df.drop(columns=["playoff"])
    

### Model Training Functions ###
def train(df: pd.DataFrame, estimator: any, param_grid: dict, name: str, importances = False, usepca = True):
    errors = []
    error = 0
    feature_names = list(df)
    feature_names.remove("year")
    feature_names.remove("playoff")

    for year in range(3, 11):
        X_train, y_train, X_test, y_test, X = custom_split(df, year, usepca)

        grid_search = GridSearchCV(estimator=estimator, refit=True, verbose=False, param_grid=param_grid, n_jobs=-1, scoring="accuracy")
        grid_search.fit(X_train, y_train)

        # Predictions on training set
        y_pred_full = grid_search.best_estimator_.predict_proba(X_test)
        y_pred = y_pred_full[:,1]

        error = predict_error(y_pred, y_test, year)
        errors.append(str(error))

        if name == "decisiontree":
            tree.plot_tree(grid_search.best_estimator_, feature_names=feature_names)
            plt.savefig(f"tree{year}", dpi=300)
            plt.close()

        calculate_curves(f"{name}/year{year}", y_test, y_pred)
        plot_confusion_matrix(f"{name}/year{year}", y_test, y_pred_full)
        #if (importances): calculate_importances(f"{name}/year{year}", grid_search.best_estimator_, X, X_test, y_test)

    with open(f"results_{name}.txt", "w") as f:
        for error in errors:
            f.write(f"{error}\n")
    return error

def objective(trial):
    alpha1 = trial.suggest_float("alpha1", 0.8, 0.97)
    alpha2 = trial.suggest_float("alpha2", 0.8, 0.97)
    alpha3 = trial.suggest_float("alpha3", 0.8, 0.97)
    alpha4 = trial.suggest_float("alpha4", 0.5, 0.75)

    df = prepare_data_y11(alpha1, alpha2, alpha3, alpha4)
    #train_x, train_y, valid_x, valid_y, unused = custom_split(df, 10, True)

    df = df[df["year"] <= 10].drop(columns="year")
    target = df["playoff"]
    data = df.drop(columns=["playoff"])

    train_x, valid_x, train_y, valid_y = train_test_split(data, target, test_size=0.2, stratify=target)

    classifier = XGBClassifier()

    param = {
        "verbosity": 0,
        # defines booster, gblinear for linear functions.
        # L2 regularization weight.
        "lambda": trial.suggest_float("lambda", 1e-5, 10.0, log=True),
        # L1 regularization weight.
        "alpha": trial.suggest_float("alpha", 1e-5, 10.0, log=True),
        # sampling ratio for training data.
        #"subsample": trial.suggest_float("subsample", 0.85, 1.0),
        # sampling according to each tree.
        #"colsample_bytree": trial.suggest_float("colsample_bytree", 0.85, 1.0),
    }

    # maximum depth of the tree, signifies complexity of the tree.
    param["max_depth"] = trial.suggest_int("max_depth", 3, 10, step=1)
    # minimum child weight, larger the term more conservative the tree.
    param["min_child_weight"] = trial.suggest_int("min_child_weight", 1, 4)
    param["eta"] = trial.suggest_float("eta", 0.2, 0.4)
    # defines how selective algorithm is.
    param["gamma"] = trial.suggest_float("gamma", 1e-8, 1.0, log=True)

    classifier.set_params(**param)
    classifier.fit(train_x, train_y)
    y_pred_full = classifier.predict_proba(valid_x)
    y_pred = y_pred_full[:,1]

    error1, error2 = predict_error_2metric(y_pred, valid_y)
    return error1, error2

def model_xgboost_year11():
    params = {'lambda': 4.789447948189175e-07, 'alpha': 5.917773858702537e-08, 'max_depth': 5, 'min_child_weight': 2, 'eta': 0.2353935522417914, 'gamma': 0.00027391060978191527}
    alpha1 = 0.8178930575309684
    alpha2= 0.8957139155243756
    alpha3= 0.823118808259113
    alpha4 = 0.5439571944325344
    estimator = XGBClassifier(random_state=42)
    
    df = prepare_data_y11(alpha1, alpha2, alpha3, alpha4, True, True)

    estimator = XGBClassifier()
    estimator.set_params(**params)

    X_train, y_train, X_test, y_test, X = custom_split(df, 11, False)

    grid_search = GridSearchCV(estimator=estimator, refit=True, verbose=False, param_grid={}, n_jobs=-1, scoring="accuracy")
    grid_search.fit(X_train, y_train)

    # Predictions on training set
    y_pred_full = grid_search.best_estimator_.predict_proba(X_test)
    y_pred = y_pred_full[:,1]

    predict_y11(y_pred, y_test)


def model_xgboost2():
    """
        XGBoost model using optuna (Competition Day 4)
    """

    study = optuna.create_study(directions=["minimize", "minimize"])
    study.optimize(objective, n_trials=3, timeout=800)

    print("Number of finished trials: ", len(study.trials))
    print("Best trial:")
    trials = study.best_trials
    trial = sorted(trials, key=lambda t: (t.values[0], t.values[1]))[0]

    print("  Value: {} | {}".format(trial.values[0], trial.values[1]))
    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

def model_xgboost():
    
    params = {
        'min_child_weight': [1, 5, 10],
        'gamma': [0.1, 0.3, 0.5],
        'max_depth': [5, 6, 7]
    }
    estimator = XGBClassifier(random_state=42)
    
    df = prepare_data_y11(0.9, 0.8, 0.8, 0.6)
    train(df, estimator, params, "xgboost", True, False)

    ''' Code used in Day 2 of the competition

    alpha_values = np.linspace(0.80, 0.95, num=4)  # Generate 11 values between 0 and 1
    alpha_combinations = list(product(alpha_values, repeat=4))  # Generate all combinations of 4 alphas
    
    min_error = float('inf')
    best_alphas = None

    for alphas in alpha_combinations:
        alpha1, alpha2, alpha3, alpha4 = alphas
        alpha1= alpha1.item() 
        alpha2= alpha2.item()
        alpha3= alpha3.item() 
        alpha4= alpha4.item() - 0.3
        alphas = alpha1, alpha2, alpha3, alpha4

        df = prepare_data_y11(alpha1, alpha2, alpha3, alpha4)
        print(f"Testing with alphas: {alphas}")
        error = train(df, estimator, params, f"xgboost_alpha_{alpha1}_{alpha2}_{alpha3}_{alpha4}", False)
        if error < min_error:
            min_error = error
            best_alphas = alphas
            print (f"New best error: {min_error} with alphas: {best_alphas}")

    print(f"Best alphas: {best_alphas} - Error: {min_error}")
    '''

def model_randomforest():
    df = prepare_data_y11(0.9, 0.8, 0.8, 0.6)

    param_grid = {
        'n_estimators': [100, 200, 500],
        'max_depth': [None, 5, 10],
        'min_samples_split': [2, 5],
        'min_samples_leaf': [1, 2],
        'bootstrap': [True, False]
    }
    estimator = RandomForestClassifier(random_state=42)

    train(df, estimator, param_grid, "randomforest", False)

def model_gradientboost():
    df = prepare_data_y11(0.9, 0.8, 0.8, 0.6)
    gradient_boosting_params = {
        'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
        'max_leaf_nodes': [20,30,40]
    }
    estimator = GradientBoostingClassifier(random_state=42)
    train(df, estimator, gradient_boosting_params, "gradientboost", False)

def model_gradientboost_nopca():
    df = prepare_data_y11(0.9, 0.8, 0.8, 0.6)
    gradient_boosting_params = {
        'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
        'max_leaf_nodes': [20,30,40]
    }
    estimator = GradientBoostingClassifier(random_state=42)
    train(df, estimator, gradient_boosting_params, "gradientboost_noPCA", False, False)

def model_badgb():
    df = prepare_bad_data()
    gradient_boosting_params = {
        'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
        'max_leaf_nodes': [20,30,40]
    }
    estimator = GradientBoostingClassifier(random_state=42)
    train(df, estimator, gradient_boosting_params, "bad_gradientboost", True)

def model_svc():
    df = prepare_data_y11(0.9, 0.8, 0.8, 0.6)
    alpha_values = np.linspace(0.25, 0.75, num=2)  # Generate 11 values between 0 and 1
    alpha_combinations = list(product(alpha_values, repeat=4))  # Generate all combinations of 4 alphas

    param_grid = {'C': [0.1, 1, 10],
                  'gamma': [1, 0.1, 0.01, 0.001], 
                  'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}
    svc = SVC(probability=True, random_state=42)
    min_error = float('inf')
    best_alphas = None
    train(df, svc, param_grid, "svc", False)
    '''for alphas in alpha_combinations:
        alpha1, alpha2, alpha3, alpha4 = alphas
        df = prepare_data_y11(alpha1.item(), alpha2.item(), alpha3.item(), alpha4.item())
        print(f"Testing with alphas: {alphas}")
        error = train(df, svc, param_grid, f"svc_alpha_{alpha1.item()}_{alpha2.item()}_{alpha3.item()}_{alpha4.item()}", False)
        if error < min_error:
            min_error = error
            best_alphas = alphas
            print (f"New best error: {min_error} with alphas: {best_alphas}")
    print(f"Best alphas: {best_alphas}")'''

def model_adaboost():
    df = prepare_data_y11(0.9, 0.8, 0.8, 0.6)
    param_grid = {
        'n_estimators': [100, 200, 500],
        'learning_rate': [0.5,1,2,5],
        'algorithm': ["SAMME"]
    }
    estimator = AdaBoostClassifier(random_state=42)
    train(df, estimator, param_grid, "adaboost")

def model_knn():
    df = prepare_data_y11(0.9, 0.8, 0.8, 0.6)
    param_grid = {
        'n_neighbors': list(range(1, 27)),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    estimator = KNeighborsClassifier()
    train(df, estimator, param_grid, "knn")

def model_decisiontree():
    df = prepare_data_y11(0.9, 0.8, 0.8, 0.6)
    param_grid = { 'criterion':['gini','entropy'],'max_depth': np.arange(3, 15)}
    estimator = DecisionTreeClassifier(random_state=42)
    train(df, estimator, param_grid, "decisiontree")

def model_mlp():
    df = prepare_data_y11(0.9, 0.8, 0.8, 0.6)
    param_grid = {
        'hidden_layer_sizes': [(25, 25, 25)],
        'max_iter': [400, 600],
        'activation': ['tanh', 'relu'],
        'solver': ['adam'],
        'alpha': [ 0.0001, 0.001],
        'learning_rate': ['constant', 'adaptive'],
        'learning_rate_init': [0.0001, 0.001],
    }
    estimator = MLPClassifier(random_state=42)
    train(df, estimator, param_grid, "mlp")
