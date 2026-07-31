from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
def build_baseline_model(
    class_weights:dict,
)->LogisticRegression:
    model=LogisticRegression(
        class_weight=class_weights,
        random_state=42,
        max_iter=1000,
    )
    return model