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


    