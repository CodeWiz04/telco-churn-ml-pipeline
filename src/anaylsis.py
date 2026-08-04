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
        
        