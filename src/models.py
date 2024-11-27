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

from statsmodels.stats.contingency_tables import mcnemar

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
    filtered_df = df[df["year"] < year].drop(columns=["year"])
    target_df = df[df["year"] == year].drop(columns=["year"])

    X_train = filtered_df.drop(columns=["playoff"])
    y_train = filtered_df["playoff"]
    cols = X_train.columns

    pca = PCA(n_components=11)
    scaler = StandardScaler()
    if usepca:
        X_train = pca.fit_transform(scaler.fit_transform(X_train))

        pcas = pd.DataFrame(pca.components_,columns=cols)
        sorted_columns = pcas.apply(lambda row: [col for col, _ in sorted(row.items(), key=lambda x: x[1])][:4], axis=1)
        print(sorted_columns)

    X_test = target_df.drop(columns=["playoff"])
    y_test = target_df["playoff"]
    if usepca:
        X_test = pca.transform(scaler.transform(X_test))
        
    return X_train, y_train, X_test, y_test, filtered_df.drop(columns=["playoff"])
    

### Model Training Functions ###
def train(df: pd.DataFrame, estimator: any, param_grid: dict, name: str, importances = False, usepca = True):
    errors = []
    feature_names = list(df)
    feature_names.remove("year")
    feature_names.remove("playoff")

    for year in range(3, 11):
        X_train, y_train, X_test, y_test, X = custom_split(df, year, usepca)

        grid_search = GridSearchCV(estimator=estimator, refit=True, verbose=False, param_grid=param_grid, cv=5, n_jobs=-1, scoring="accuracy")
        grid_search.fit(X_train, y_train)
        print(grid_search.best_estimator_)

        # Predictions on training set
        y_train_pred_full = grid_search.best_estimator_.predict_proba(X_train)
        y_train_pred = y_train_pred_full[:, 1]

        # Predictions on test set
        y_test_pred_full = grid_search.best_estimator_.predict_proba(X_test)
        y_test_pred = y_test_pred_full[:, 1]

        # Calculate performance metrics
        train_accuracy = accuracy_score(y_train, (y_train_pred > 0.5).astype(int))
        test_accuracy = accuracy_score(y_test, (y_test_pred > 0.5).astype(int))

        print(f"Year: {year}, Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}")

        errors.append(str(predict_error(y_test_pred, y_test, year)))
        if name == "decisiontree":
            tree.plot_tree(grid_search.best_estimator_, feature_names=feature_names)
            plt.savefig(f"tree{year}", dpi=300)
            plt.close()

        calculate_curves(f"{name}/year{year}", y_test, y_test_pred)
        plot_confusion_matrix(f"{name}/year{year}", y_test, y_test_pred_full)
        if (importances): calculate_importances(f"{name}/year{year}", grid_search.best_estimator_, X, X_test, y_test)

    with open(f"results_{name}.txt", "w") as f:
        for error in errors:
            f.write(f"{error}\n")
 

def model_randomforest():
    df = prepare_data()

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
    df = prepare_data()
    gradient_boosting_params = {
        'learning_rate': [0.001, 0.005, 0.01, 0.05, 0.1, 0.5],
        'max_leaf_nodes': [20,30,40]
    }
    estimator = GradientBoostingClassifier(random_state=42)
    train(df, estimator, gradient_boosting_params, "gradientboost", True)

def model_gradientboost_nopca():
    df = prepare_data()
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
    df = prepare_data()
    param_grid = {'C': [0.1, 1, 10],
              'gamma': [1, 0.1, 0.01, 0.001], 
              'kernel': ['linear', 'rbf', 'poly', 'sigmoid']}
    svc = SVC(probability=True, random_state=42)
    train(df, svc, param_grid, "svc", False)

def model_adaboost():
    df = prepare_data()
    param_grid = {
        'n_estimators': [100, 200, 500],
        'learning_rate': [0.5,1,2,5],
        'algorithm': ["SAMME"]
    }
    estimator = AdaBoostClassifier(random_state=42)
    train(df, estimator, param_grid, "adaboost")

def model_knn():
    df = prepare_data()
    param_grid = {
        'n_neighbors': list(range(1, 27)),
        'weights': ['uniform', 'distance'],
        'metric': ['euclidean', 'manhattan', 'minkowski']
    }
    estimator = KNeighborsClassifier()
    train(df, estimator, param_grid, "knn")

def model_decisiontree():
    df = prepare_data()
    param_grid = { 'criterion':['gini','entropy'],'max_depth': np.arange(3, 15)}
    estimator = DecisionTreeClassifier(random_state=42)
    train(df, estimator, param_grid, "decisiontree")

def model_mlp():
    df = prepare_data()
    param_grid = {
        'hidden_layer_sizes': [(50,), (100,), (50, 50), (50, 50, 50), (50,50,50,50,50)],
        'max_iter': [400, 600, 800],
        'activation': ['tanh', 'relu'],
        'solver': ['adam'],
        'alpha': [ 0.0001, 0.001],
        'learning_rate': ['constant', 'adaptive'],
        'learning_rate_init': [0.0001, 0.001],
    }
    estimator = MLPClassifier(random_state=42)
    train(df, estimator, param_grid, "mlp")
