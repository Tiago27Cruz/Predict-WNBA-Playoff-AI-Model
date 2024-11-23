from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve, auc, f1_score
from matplotlib import pyplot as plt

def calculate_curves(name, y_test, y_scores):
    """
        Calculate ROC and Precision-Recall curves. Saves figures using the given name inside /results.
        Also prints the ROC AUC and F1 scores.
    """
    # ROC AUC + F1 score
    roc_auc = roc_auc_score(y_test, y_scores)
    f1 = f1_score(y_test, y_scores)
    print("ROC AUC Score:", roc_auc)
    print("F1 Score:", f1)

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