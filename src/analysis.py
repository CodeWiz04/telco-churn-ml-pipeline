from scipy.stats import ttest_rel
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def compare_cv_scores(
    baseline_results: dict,
    tuned_results: dict,
):
    """
    Compare the top two models using a paired t-test
    on their cross-validation scores.
    """

    lr_scores = baseline_results["Logistic Regression"]["cv_scores"]
    rf_scores = tuned_results["Random Forest"]["cv_scores"]

    t_statistic, p_value = ttest_rel(   #t_statistics=>Average diff/variation in differences(How large is the difference compared to the amount of variation)
                                        #p_value=>2*p(>=|t|)(tells us how likely it is that the difference between two models happened just by chance.)
        lr_scores,
        rf_scores,
    )

    print("\n" + "=" * 70)
    print("QUESTION 2")
    print("=" * 70)

    print("\nLogistic Regression CV Scores:")
    print(lr_scores)

    print("\nRandom Forest CV Scores:")
    print(rf_scores)

    print(f"\nT-Statistic : {t_statistic:.4f}")
    print(f"P-Value     : {p_value:.4f}")

    if p_value < 0.05:
        print("\nConclusion:")
        print("The difference between the two models is statistically significant.")
    else:
        print("\nConclusion:")
        print("The difference between the two models is NOT statistically significant.")
        print("The observed difference is likely due to random variation in the CV folds.")
        
def get_feature_importance(
    model,
    preprocessor,
) -> pd.DataFrame:
    """
    Extract feature importance (Random Forest)
    or coefficients (Logistic Regression).

    Returns:
        DataFrame sorted by feature importance.
    """
    feature_names=preprocessor.get_feature_names_out()  #give the name of features after we applied one hot encoding
    if isinstance(model,RandomForestClassifier):
        importance=model.feature_importances_   #Returns importance of each feature in prediction
    elif isinstance(model,LogisticRegression):
        importance=abs(model.coef_[0])          #has shape (1,no.of features), so returns one and only row of coefficients
    else:
        raise ValueError("Unsupported model type.")

    feature_importance = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": importance,
        }
    )

    feature_importance = feature_importance.sort_values(
        by="Importance",
        ascending=False,
    ).reset_index(drop=True)

    return feature_importance
def display_feature_importance(
    feature_importance: pd.DataFrame,
    top_n: int = 10,
):
    """
    Display the most important features.
    """

    print("\n" + "=" * 70)
    print("QUESTION 3")
    print("=" * 70)

    print(f"\nTop {top_n} Most Important Features:\n")

    print(feature_importance.head(top_n))

def save_feature_importance(
    feature_importance: pd.DataFrame,
    output_path,
):
    """
    Save feature importance to CSV.
    """

    feature_importance.to_csv(
        output_path,
        index=False,
    )
    
def analyze_model_errors(
    X_test:pd.DataFrame,
    y_test,
    y_pred,
):
    """
    Find which customer segments have the highest prediction error.

    Returns:
        Dictionary containing error rates for different segments.
    """
    error_df=X_test.copy()
    error_df["Actual"]=y_test.values
    error_df["Predicted"]=y_pred
    error_df["Incorrect"] = (
        error_df["Actual"] != error_df["Predicted"]
    )
    segment_errors = {}

    categorical_columns =(
        X_test.select_dtypes(include="object").columns.tolist()
        )
    for column in categorical_columns:
       segment_errors[column] = (     
        error_df
        .groupby(column)["Incorrect"]
        .mean()
        .sort_values(ascending=False)
    )
    error_df["Tenure Group"]=pd.cut(
        error_df["tenure"],
        bins=[0, 12, 24, 48, 72],
        labels=[
           "0-12",
           "13-24",
           "25-48",
           "49-72",
        ],
    )
    segment_errors["Tenure Group"] = (
       error_df
       .groupby("Tenure Group")["Incorrect"]
       .mean()
       .sort_values(ascending=False)
    )
    return segment_errors

def display_error_analysis(
    segment_errors: dict,
):
    """
    Display customer segments with the highest error rate.
    """

    print("\n" + "=" * 70)
    print("QUESTION 4")
    print("=" * 70)

    for segment, errors in segment_errors.items():

        print(f"\n{segment}")
        print("-" * 40)

        print(errors)

        worst_segment = errors.idxmax()
        worst_error = errors.max()

        print(
            f"\nHighest Error: {worst_segment} "
            f"({worst_error:.2%})"
        )
    