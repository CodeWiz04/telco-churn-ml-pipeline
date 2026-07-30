from pathlib import Path  #Treats file path as objects e.g. instead of writing file_path = "data/raw/telco.csv" we can write file_path = Path("data") / "raw" / "telco.csv"
import pandas as pd
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data")
RAW_DATA_PATH = DATA_DIR / "raw" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
PROCESSED_DATA_PATH = DATA_DIR / "processed" / "processed_telco_churn.csv"

def load_dataset(path:Path=RAW_DATA_PATH)->pd.DataFrame:
    """
    Load te Telco Customer Churn dataset
    Args:
        path: Path to the raw CSV dataset.

    Returns:
        Loaded pandas DataFrame.
    """
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}"
        )
    return pd.read_csv(path)


def validate_dataset(df:pd.Dataframe)->None:
    """
    Validate that the dataset contains the expected columns
    and is not empty.
    """
    expected_columns={
        "customerID",
        "gender",
        "SeniorCitizen",
        "Partner",
        "Dependents",
        "tenure",
        "PhoneService",
        "MultipleLines",
        "InternetService",
        "OnlineSecurity",
        "OnlineBackup",
        "DeviceProtection",
        "TechSupport",
        "StreamingTV",
        "StreamingMovies",
        "Contract",
        "PaperlessBilling",
        "PaymentMethod",
        "MonthlyCharges",
        "TotalCharges",
        "Churn"
    }
    if df.empty:
        raise ValueError("Dataset is empty")
    missing=expected_columns-set(df.columns)
    if missing:
        raise ValueError(
            f"Missing columns: {missing}"
        )
        
def dataset_summary(df:pd.DataFrame)->None:
    """Print a quick overview of the dataset"""
    print("=" * 50)
    print("Dataset Shape")
    print(df.shape)
    
    print("\nData Types")
    print(df.dtypes)
    
    print("\nMissing Values")
    print(df.isnull().sum())
    
    print("\nTarget Distribution")
    print(df["Churn"].value_counts(normalize=True))
    

    