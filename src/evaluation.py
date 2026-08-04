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
