from sklearn import metrics
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc, f1_score
from sklearn.inspection import permutation_importance

import matplotlib
from matplotlib import pyplot as plt
matplotlib.use('Agg')

import numpy as np
import pandas as pd


def plot_confusion_matrix(name, y_test, y_scores):
    confusion_matrix = metrics.confusion_matrix(y_test, np.argmax(y_scores, axis=1))
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix = confusion_matrix, display_labels = [0, 1])
    cm_display.plot()
    plt.savefig("../results/" + name + "matrix.png")
    plt.close()



def calculate_curves(name, y_test, y_scores):
    """
        Calculate ROC and Precision-Recall curves. Saves figures using the given name inside /results.
        Also prints the ROC AUC and F1 scores.
    """
    # ROC AUC + F1 score
    roc_auc = roc_auc_score(y_test, y_scores)
    #f1 = f1_score(y_test, y_scores)
    print("ROC AUC Score:", roc_auc)
    #print("F1 Score:", f1)

    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_scores)
    plt.figure()
    plt.plot(fpr, tpr, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(name + " ROC Curve")
    plt.legend(loc='lower right')
    plt.savefig("../results/" + name + "_roc_curve.png")
    plt.close()

    # Precision-Recall curve
    precision, recall, _ = precision_recall_curve(y_test, y_scores)
    pr_auc = auc(recall, precision)
    plt.figure()
    plt.plot(recall, precision, label='Precision-Recall curve (area = %0.2f)' % pr_auc)
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title(name + " Precision-Recall Curve")
    plt.legend(loc='lower right')
    plt.savefig("../results/" + name + "_pr_curve.png")
    plt.close()

def calculate_importances(name, rf, X, X_test, y_test):
    importance = rf.feature_importances_
    std = np.std([tree.feature_importances_ for tree in rf.estimators_], axis=0)

    forest_importances = pd.Series(importance, index=X.columns)

    fig, ax = plt.subplots(figsize=(20, 16))
    forest_importances.plot.bar(yerr=std, ax=ax)
    ax.set_title("Feature importances using MDI")
    ax.set_ylabel("Mean decrease in impurity")
    fig.tight_layout()
    plt.savefig("../results/" + name + "_importances1.png")
    plt.close()

    result = permutation_importance(
        rf, X_test, y_test, n_repeats=20, random_state=42, n_jobs=2
    )
    forest_importances = pd.Series(result.importances_mean, index=X.columns)

    fig, ax = plt.subplots(figsize=(20, 16))
    forest_importances.plot.bar(yerr=result.importances_std, ax=ax)
    ax.set_title("Feature importances using permutation on full model")
    ax.set_ylabel("Mean accuracy decrease")
    fig.tight_layout()
    plt.savefig("../results/" + name + "_importances2.png")
    plt.close()

def predict_error(y_pred, y_test, year):
    y_pred_sum = sum(y_pred)
    y_pred = list(map(lambda x: 8*x/y_pred_sum, y_pred))
    
    error = 0
    for i in range(len(y_pred)):
        error += abs(y_pred[i] - list(y_test)[i])

    print(f"predicting year {year}: error was {error}")
    return error