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


def main():
    """Run the complete data preprocessing pipeline."""

    print("=" * 60)
    print("TELCO CUSTOMER CHURN PREPROCESSING PIPELINE")
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

    # Train-test split
    print("\nSplitting dataset...")
    X_train, X_test, y_train, y_test = get_train_test_split(X, y)

    # Encode target labels
    print("\nEncoding target labels...")
    y_train = encode_target(y_train)
    y_test = encode_target(y_test)

    # Build preprocessing pipeline
    print("\nBuilding preprocessing pipeline...")
    preprocessor = build_preprocessor(
        numerical_features,
        categorical_features,
    )

    # Apply preprocessing
    print("\nApplying preprocessing...")
    X_train_processed, X_test_processed = preprocess_data(
        preprocessor,
        X_train,
        X_test,
    )

    # Compute class weights
    print("\nComputing class weights...")
    class_weights = compute_class_weights(y_train)

    # -----------------------------
    # Results
    # -----------------------------
    print("\n" + "=" * 60)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nDataset Shapes")
    print(f"Original X : {X.shape}")
    print(f"Original y : {y.shape}")

    print("\nTrain/Test Shapes")
    print(f"X_train : {X_train.shape}")
    print(f"X_test  : {X_test.shape}")
    print(f"y_train : {y_train.shape}")
    print(f"y_test  : {y_test.shape}")

    print("\nProcessed Data Shapes")
    print(f"Processed X_train : {X_train_processed.shape}")
    print(f"Processed X_test  : {X_test_processed.shape}")

    print("\nTarget Distribution (Training)")
    print(y_train.value_counts(normalize=True))

    print("\nTarget Distribution (Testing)")
    print(y_test.value_counts(normalize=True))

    print("\nNumerical Features")
    print(numerical_features)

    print("\nCategorical Features")
    print(categorical_features)

    print("\nComputed Class Weights")
    print(class_weights)


if __name__ == "__main__":
    main()