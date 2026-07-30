from src.preprocessing import(
    load_dataset,
    validate_dataset,
    dataset_summary,
    clean_dataset,
    build_features
)

def main():
    """Run the preprocessing pipeline."""

    print("=" * 60)
    print("Loading dataset...")
    df = load_dataset()

    print("\nValidating dataset...")
    validate_dataset(df)

    print("\nDataset summary:")
    dataset_summary(df)

    print("\nCleaning dataset...")
    df = clean_dataset(df)

    print("\nBuilding features and target...")
    X, y = build_features(df)

    print("\nPipeline completed successfully!")
    print("=" * 60)

    print(f"\nFeature matrix shape (X): {X.shape}")
    print(f"Target shape (y): {y.shape}")

    print("\nFeature columns:")
    print(X.columns.tolist())

    print("\nTarget distribution:")
    print(y.value_counts())

    print("\nFirst five rows of X:")
    print(X.head())

    print("\nFirst five values of y:")
    print(y.head())


if __name__ == "__main__":
    main()
