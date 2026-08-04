import pandas as pd

from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
PRIMARY_METRIC = "Recall"  #failing to identify a customer who will churn is more expensive than incorrectly flagging a customer who won't churn. Therefore, maximizing Recall is often the priority.
def evaluate_model(
    y_test,
    y_pred,
    y_prob,
)->dict:
    """
    Compute evaluation metrics.

    Returns:
        Dictionary containing all evaluation metrics.
    """
    metrics={
        "Confusion Matrix": confusion_matrix(y_test,y_pred),
        "Accuracy": accuracy_score(y_test,y_pred),
        "Precision": precision_score(y_test,y_pred),
        "Recall": recall_score(y_test,y_pred),
        "F1 Score": f1_score(y_test,y_pred),
        "ROC AUC Score": roc_auc_score(y_test,y_prob),
    }
    return metrics

def create_comparison_table(
    
)