from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import (
    cross_val_score,
    GridSearchCV
    )
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
import numpy as np


def build_baseline_model(
    class_weights: dict,
) -> LogisticRegression:
    model = LogisticRegression(
        class_weight=class_weights,
        random_state=42,
        max_iter=1000,   #iterate upto 1000 iterations to find the best weight
    )
    return model


def get_models(class_weights: dict) -> dict:
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

def get_parameters_grids(
    
)->dict:
    """
    Return hyperparameter grids for each model.
    """
    parameters_grid={
        "Logistic Regression":{
            "C":[0.01,0.1,1,10],  #inverse of regularization strengths
            "solver":[            #optimization algos to min log loss and finds the best weights
                "liblinear",      #works well fr small datasets,good for binary classification
                "lbfgs"           #works well for medium/large datasets
            ],
        },
        "Random Forest":{
            "n_estimators": [100, 200],  #no.of trees
            "max_depth": [None, 10, 20],
            "min_samples_split": [2, 5], 
            "min_samples_leaf": [1, 2],
        },
    }
    return parameters_grid

def train_model(
    model,
    X_train,
    y_train,
):
    """
    Train a machine learning model.
    """
    model.fit(X_train, y_train)
    return model


def cross_validate_model(
    model,
    X_train,
    y_train,
    cv: int = 5,
):
    """
    Perform k-fold cross validation.

    Returns:
        Mean CV score,
        Standard deviation,
        Individual fold scores.
    """
    scores = cross_val_score(
        estimator=model,
        X=X_train,
        y=y_train,
        cv=cv,
        scoring="f1",  #as dataset is imbalanced so F1 score is a better metric to evaluate the model performance(since it requires both precison and recall to be high)
    )

    return {
        "cv_scores": scores,
        "cv_mean": np.mean(scores),
        "cv_std": np.std(scores),
    }

def tune_model(
    model,
    param_grid:dict,
    X_train,
    y_train,
    cv:int=5,
):
    """
    Perform Grid Search hyperparameter tuning.

    Returns:
        Best trained model,
        Best parameters,
        Best cross-validation score.
    """
    grid_search=GridSearchCV( 
        estimator=model,        #which model to tune
        param_grid=param_grid,  #list of parameter values to try
        cv=cv,                  #every parameter combination is evaluated on 5-fold
        scoring="f1",           #chooses parameter combination which gives best f1 score
        n_jobs=-1,              #use all cpu cores so that model training could be done in parallel
    )
    grid_search.fit(X_train,y_train)  #for every parameter combination runs 5-fold, compute f1 and store 
    return (
        grid_search.best_estimator_, #returns trained model with best parameters
        grid_search.best_params_,    #returns best parameters values
        grid_search.best_score_,     #returns highest average cross validation score
    )
def predict_model(
    model,
    X_test,
):
    """
    Predict class labels and probabilities.
    """
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]   #Convert into Probablities that each customer will churn (1) or not churn (0)

    return y_pred, y_prob


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


def train_and_evaluate(
    model,
    X_train,
    X_test,
    y_train,
    y_test,
):
    """
    Train, cross validate and evaluate one model.
    """

    cv_results = cross_validate_model(
        model,
        X_train,
        y_train,
    )

    model = train_model(
        model,
        X_train,
        y_train,
    )

    y_pred, y_prob = predict_model(
        model,
        X_test,
    )

    metrics = evaluate_model(
        y_test,
        y_pred,
        y_prob,
    )

    metrics.update(cv_results)

    return model, metrics

def tune_and_evaluate(
    model,
    param_grid,
    X_train,
    X_test,
    y_train,
    y_test,
):
    """
    Tune, train and evaluate one model.
    """

    best_model, best_params, best_cv_score = tune_model(
        model,
        param_grid,
        X_train,
        y_train,
    )

    y_pred, y_prob = predict_model(
        best_model,
        X_test,
    )

    metrics = evaluate_model(
        y_test,
        y_pred,
        y_prob,
    )

    metrics["Best Parameters"] = best_params
    metrics["Best CV Score"] = best_cv_score

    return best_model, metrics

