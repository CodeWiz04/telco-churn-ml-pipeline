from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import numpy as np
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
        "Accuracy": accuracy_score(y_test, y_pred),   #Out of all predictions, how many were correct((TP+TN)/(TP+TN+FP+FN))
        "Precision": precision_score(y_test, y_pred), #Out of everyone the model predicted will churn, how many actually churned((TP)/(TP+FP))
        "Recall": recall_score(y_test, y_pred),       #(TPR/Sensitivity)Out of all customers who actually churned, how many did the model successfully find((TP)/(TP+FN))
        "F1 Score": f1_score(y_test, y_pred),         #A balanced measure that is high only when both Precision and Recall are high.((2*Precision*recall)/(Precision+Recall))
        "ROC-AUC": roc_auc_score(y_test, y_prob),     #plots a curve between FPR and TPR for every possible threshold (1.0, 0.99, 0.98, …, 0.0)
    }

    return metrics

def get_models(class_weights:dict)->dict:
    """
    Return all models to compare.
    """

    models = {
        "Logistic Regression": LogisticRegression(
            class_weight=class_weights,
            random_state=42,
            max_iter=1000,
        ),
        "Random Forest": RandomForestClassifier(
            class_weight=class_weights,
            random_state=42,
        ),
    }

    return models

    