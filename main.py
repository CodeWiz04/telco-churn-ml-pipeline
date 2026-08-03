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

    # Build models
    print("\nCreating models...")
    models = get_models(class_weights)

    results = {}

    # Train and evaluate each model
    for model_name, model in models.items():

        print(f"\n{'=' * 60}")
        print(f"Training {model_name}")
        print(f"{'=' * 60}")

        trained_model, metrics = train_and_evaluate(
            model,
            X_train_processed,
            X_test_processed,
            y_train,
            y_test,
        )

        results[model_name] = metrics

    # ======================================================
    # Final Results
    # ======================================================

    print("\n")
    print("=" * 70)
    print("FINAL MODEL COMPARISON")
    print("=" * 70)

    for model_name, metrics in results.items():

        print(f"\n{model_name}")
        print("-" * 50)

        print(f"Accuracy : {metrics['Accuracy']:.4f}")
        print(f"Precision: {metrics['Precision']:.4f}")
        print(f"Recall   : {metrics['Recall']:.4f}")
        print(f"F1 Score : {metrics['F1 Score']:.4f}")
        print(f"ROC-AUC  : {metrics['ROC-AUC']:.4f}")

        print("\nCross Validation")

        print(f"Fold Scores : {metrics['cv_scores']}")
        print(f"Mean Score  : {metrics['cv_mean']:.4f}")
        print(f"Std Dev     : {metrics['cv_std']:.4f}")

    print("\n")
    print("=" * 70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)


if __name__ == "__main__":
    main()