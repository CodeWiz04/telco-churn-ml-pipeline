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

    print("\n")
    print("=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()