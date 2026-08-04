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
    baseline_results:dict,
    tuned_results:dict,
)->pd.DataFrame:
    
    """
    Create a comparison table for all models.

    Returns:
        Ranked pandas DataFrame.
    """
    rows=[]
    for model_name,metrics in baseline_results.items():
        rows.append(
            {
                "Model":f"{model_name} (Baseline)",
                "Accuracy":metrics["Accuracy"],
                "Precision":metrics["Precision"],
                "Recall":metrics["Recall"],
                "F1 Score":metrics["F1 Score"],
                "ROC AUC Score":metrics["ROC AUC Score"],             
            }
        )
        for model_name, metrics in tuned_results.items():

            rows.append(
            {
                "Model": f"{model_name} (Tuned)",
                "Accuracy": metrics["Accuracy"],
                "Precision": metrics["Precision"],
                "Recall": metrics["Recall"],
                "F1 Score": metrics["F1 Score"],
                "ROC-AUC": metrics["ROC-AUC"],
            }
        )
    results=pd.DataFrame(rows)
    results=results.sort_values(by=PRIMARY_METRIC,ascending=False).reset_index(drop=True)# Sorts wrt RECALL and renumber the indexes
    results.insert(
        0,  
        "Rank",
        range(1,len(results)+1),
    )
    
    return results

def save_results(
    comparison_table: pd.DataFrame,
    output_path: str = "results/model_comparison.csv",
):
    """
    Save comparison table as CSV.
    """

    comparison_table.to_csv(
        output_path,
        index=False,
    )

    print(f"\nResults saved to {output_path}")
    
def print_confusion_matrix(
    model_name: str,
    confusion_mat,
):
    """
    Print confusion matrix.
    """

    print(f"\n{model_name} Confusion Matrix")
    print("-" * 40)
    print(confusion_mat)


def print_comparison_table(
    comparison_table: pd.DataFrame,
):
    """
    Print final ranked comparison table.
    """

    print("\n")
    print("=" * 70)
    print("FINAL MODEL RANKING")
    print("=" * 70)

    print(comparison_table.to_string(index=False))

    print("\nPrimary Metric:", PRIMARY_METRIC)
        