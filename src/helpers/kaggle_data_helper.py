import os
import glob
import pandas as pd
import sqlite3
import kagglehub

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../tests"))
DATA_DIR = os.path.join(BASE_DIR, "data")
os.makedirs(DATA_DIR, exist_ok=True)

def download_and_retrieve_data_from_kaggle(handle: str, relative_output_path: str) -> pd.DataFrame:
    """
    Downloads a Kaggle dataset (if not already stored locally), extracts the first 
    data file (CSV/Parquet/Excel/SQLite), stores it under `output_path`, and returns a DataFrame.
    If `output_path` already exists, the function simply loads and returns the DataFrame.
    """
    output_path = os.path.join(DATA_DIR, relative_output_path)
    # -----------------------------------------------------
    # 1. If output file exists: load and return
    # -----------------------------------------------------
    if os.path.exists(output_path):
        print(f"File already exists → loading from {output_path}")
        return pd.read_csv(output_path)

    # Ensure directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    # -----------------------------------------------------
    # 2. Download the dataset
    # -----------------------------------------------------
    try:
        local_path = kagglehub.dataset_download(handle=handle)
        print(f"Dataset downloaded to: {local_path}")
    except Exception as e:
        print("Dataset download failed — check dataset handle or permissions.")
        raise

    # -----------------------------------------------------
    # 3. Search for dataset files
    # -----------------------------------------------------
    patterns = ["**/*.csv", "**/*.parquet", "**/*.xlsx", "**/*.xls", "**/*.sqlite", "**/*.db"]
    all_files = []
    for pattern in patterns:
        all_files.extend(glob.glob(os.path.join(local_path, pattern), recursive=True))

    if not all_files:
        raise FileNotFoundError("No data files found (.csv, .parquet, .xlsx, .sqlite, .db).")

    # Pick the first file (or implement selection logic)
    target = all_files[0]
    ext = os.path.splitext(target)[1].lower()

    # -----------------------------------------------------
    # 4. Load into DataFrame depending on extension
    # -----------------------------------------------------
    if ext == ".csv":
        df = pd.read_csv(target)

    elif ext == ".parquet":
        df = pd.read_parquet(target)

    elif ext in [".xlsx", ".xls"]:
        df = pd.read_excel(target)

    elif ext in [".sqlite", ".db"]:
        conn = sqlite3.connect(target)
        tables = pd.read_sql("SELECT name FROM sqlite_master WHERE type='table';", conn)
        first_table = tables["name"].iloc[0]
        df = pd.read_sql(f"SELECT * FROM {first_table}", conn)
        conn.close()

    else:
        raise ValueError(f"Unsupported file type: {ext}")

    # -----------------------------------------------------
    # 5. Save CSV output and return DataFrame
    # -----------------------------------------------------
    df.to_csv(output_path, index=False)
    print(f"Saved data → {output_path}")

    return df
