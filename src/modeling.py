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
        max_iter=1000,   #iterate upto 1000 iterations to find the best weight
    )
    return model

def train_model(
    model,
    X_train,
    y_train,
):
    """
    Train the model.
    """
    model.fit(
        X_train,
        y_train,
    )
    return model

def predict_model(
    model,
    X_test,
):
    """
    Predict class labels and probabilities.
    """
    y_pred=model.predict(X_test)
    y_prob=model.predict_proba(X_test)[:,1] #These are the probabilities that each customer will churn
    
    return y_pred,y_prob

def evaluate_model(
    y_test,
    y_pred,
    y_prob,
):
    """
    Compute evaluation metrics.
    """
    metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "Precision": precision_score(y_test, y_pred),
        "Recall": recall_score(y_test, y_pred),
        "F1 Score": f1_score(y_test, y_pred),
        "ROC-AUC": roc_auc_score(y_test, y_prob),
    }

    return metrics