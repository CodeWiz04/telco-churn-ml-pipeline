from src.preprocessing import (
    load_dataset,
    validate_dataset,
    dataset_summary,
    clean_dataset,
    build_features,
    identify_feature_types,
    get_train_test_split,
    encode_target,
    build_preprocessor,
    preprocess_data,
    compute_class_weights,
)

from src.modeling import (
    get_models,
    train_and_evaluate,
    get_parameter_grids,
    tune_and_evaluate,
)
from src.evaluation import (
    create_comparison_table,
    save_results,
    print_comparison_table,
    print_confusion_matrix,
)
from src.analysis import (
    compare_cv_scores,
    get_feature_importance,
    display_feature_importance,
    save_feature_importance,
    analyze_model_errors,
    display_error_analysis,
)


def main():
    """Run the complete Telco Churn ML pipeline."""

    print("=" * 60)
    print("TELCO CUSTOMER CHURN ML PIPELINE")
    print("=" * 60)

    # Load dataset
    print("\nLoading dataset...")
    df = load_dataset()

    # Validate dataset
    print("\nValidating dataset...")
    validate_dataset(df)

    # Dataset summary
    print("\nDataset Summary")
    dataset_summary(df)

    # Clean dataset
    print("\nCleaning dataset...")
    df = clean_dataset(df)

    # Build features and target
    print("\nBuilding features and target...")
    X, y = build_features(df)

    # Identify feature types
    print("\nIdentifying feature types...")
    numerical_features, categorical_features = identify_feature_types(X)

    # Train-Test Split
    print("\nSplitting dataset...")
    X_train, X_test, y_train, y_test = get_train_test_split(
        X,
        y,
    )

    # Encode target labels
    print("\nEncoding target labels...")
    y_train = encode_target(y_train)
    y_test = encode_target(y_test)

    # Build preprocessor
    print("\nBuilding preprocessing pipeline...")
    preprocessor = build_preprocessor(
        numerical_features,
        categorical_features,
    )

    # Preprocess data
    print("\nPreprocessing dataset...")
    X_train_processed, X_test_processed = preprocess_data(
        preprocessor,
        X_train,
        X_test,
    )

    # Compute class weights
    print("\nComputing class weights...")
    class_weights = compute_class_weights(y_train)

    # Create models
    print("\nCreating models...")
    models = get_models(class_weights)

    # Hyperparameter grids
    parameter_grids = get_parameter_grids()

    baseline_models = {}
    tuned_models = {}

    baseline_results = {}
    tuned_results = {}

    # ======================================================
    # Baseline Models
    # ======================================================

    print("\n")
    print("=" * 70)
    print("BASELINE MODEL COMPARISON")
    print("=" * 70)

    for model_name, model in models.items():

        print(f"\nTraining {model_name}...")

        trained_model, metrics = train_and_evaluate(
            model,
            X_train_processed,
            X_test_processed,
            y_train,
            y_test,
        )

        baseline_models[model_name] = trained_model
        baseline_results[model_name] = metrics

    # ======================================================
    # Hyperparameter Tuning
    # ======================================================

    print("\n")
    print("=" * 70)
    print("HYPERPARAMETER TUNING")
    print("=" * 70)

    for model_name, model in models.items():

        print(f"\nTuning {model_name}...")

        tuned_model, tuned_metrics = tune_and_evaluate(
            model,
            parameter_grids[model_name],
            X_train_processed,
            X_test_processed,
            y_train,
            y_test,
        )

        tuned_models[model_name] = tuned_model
        tuned_results[model_name] = tuned_metrics

    # ======================================================
    # Final Results
    # ======================================================

    print("\n")
    print("=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    for model_name in models.keys():

        baseline = baseline_results[model_name]
        tuned = tuned_results[model_name]

        print(f"\n{model_name}")
        print("-" * 60)

        print("\nBaseline Results")
        print(f"Accuracy : {baseline['Accuracy']:.4f}")
        print(f"Precision: {baseline['Precision']:.4f}")
        print(f"Recall   : {baseline['Recall']:.4f}")
        print(f"F1 Score : {baseline['F1 Score']:.4f}")
        print(f"ROC-AUC  : {baseline['ROC-AUC']:.4f}")
        print(f"CV Mean  : {baseline['cv_mean']:.4f}")
        print(f"CV Std   : {baseline['cv_std']:.4f}")

        print("\nBest Parameters")
        print(tuned["Best Parameters"])

        print(f"\nBest CV Score : {tuned['Best CV Score']:.4f}")

        print("\nTuned Results")
        print(f"Accuracy : {tuned['Accuracy']:.4f}")
        print(f"Precision: {tuned['Precision']:.4f}")
        print(f"Recall   : {tuned['Recall']:.4f}")
        print(f"F1 Score : {tuned['F1 Score']:.4f}")
        print(f"ROC-AUC  : {tuned['ROC-AUC']:.4f}")

    # ======================================================
    # Model Evaluation
    # ======================================================

    comparison_table = create_comparison_table(
        baseline_results,
        tuned_results,
    )

    print_comparison_table(
        comparison_table,
    )

    save_results(
        comparison_table,
    )

    # Print confusion matrices
    print("\n")
    print("=" * 70)
    print("CONFUSION MATRICES")
    print("=" * 70)

    for model_name in models.keys():

        print_confusion_matrix(
            f"{model_name} (Baseline)",
            baseline_results[model_name]["Confusion Matrix"],
        )

        print_confusion_matrix(
            f"{model_name} (Tuned)",
            tuned_results[model_name]["Confusion Matrix"],
        )

    # ======================================================
    # QUESTION 1
    # ======================================================

    print("\n" + "=" * 70)
    print("QUESTION 1")
    print("=" * 70)

    baseline_recall = baseline_results["Logistic Regression"]["Recall"]

    all_models = {
        "Logistic Regression (Baseline)": baseline_results["Logistic Regression"]["Recall"],
        "Logistic Regression (Tuned)": tuned_results["Logistic Regression"]["Recall"],
        "Random Forest (Baseline)": baseline_results["Random Forest"]["Recall"],
        "Random Forest (Tuned)": tuned_results["Random Forest"]["Recall"],
    }

    best_model = max(all_models, key=all_models.get)
    best_recall = all_models[best_model]

    improvement = best_recall - baseline_recall

    print(f"\nPrimary Metric : Recall")
    print(f"Best Model     : {best_model}")
    print(f"Best Recall    : {best_recall:.4f}")
    print(f"Baseline Recall: {baseline_recall:.4f}")
    print(f"Improvement    : {improvement:.4f} ({improvement*100:.2f} percentage points)")

    if improvement > 0:
        print("\nConclusion:")
        print(f"{best_model} outperformed the baseline on Recall.")
    elif improvement < 0:
        print("\nConclusion:")
        print("The baseline Logistic Regression achieved the highest Recall.")
    else:
        print("\nConclusion:")
        print("No improvement over the baseline. Both models achieved the same Recall.")

    # ======================================================
    # QUESTION 2
    # ======================================================

    compare_cv_scores(
        baseline_results,
        tuned_results,
    )

    # ======================================================
    # QUESTION 3
    # ======================================================

    best_model_object = tuned_models["Random Forest"]

    feature_importance = get_feature_importance(
        best_model_object,
        preprocessor,
    )

    display_feature_importance(
        feature_importance,
    )

    save_feature_importance(
        feature_importance,
        "results/feature_importance.csv",
    )

    # ======================================================
    # QUESTION 4
    # ======================================================

    best_predictions = tuned_results["Random Forest"]["Predictions"]

    segment_errors = analyze_model_errors(
        X_test,
        y_test,
        best_predictions,
    )

    display_error_analysis(
        segment_errors,
    )

    print("\n")
    print("=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()