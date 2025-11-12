import os
import glob
import pandas as pd
import kagglehub


def download_data_from_kaggle(handle: str) -> None:
    try:
        local_path = kagglehub.dataset_download(handle=handle)  # full dataset download
    except Exception as e:
        print("Dataset download failed — likely wrong handle or private/deleted dataset.")
        raise

    patterns = ["**/*.csv", "**/*.parquet", "**/*.xlsx", "**/*.xls", "**/*.sqlite", "**/*.db"]
    all_files = []
    for p in patterns:
        all_files.extend(glob.glob(os.path.join(local_path, p), recursive=True))

    if not all_files:
        raise FileNotFoundError("No CSV/Parquet/Excel/SQLite files found in dataset.")

    target = all_files[0]

    ext = os.path.splitext(target)[1].lower()

    if ext == ".csv":
        df = pd.read_csv(target)
        out_dir = os.path.join("data")
        os.makedirs(out_dir, exist_ok=True)

        out_path = os.path.join(out_dir, "cds_data.csv")
        df.to_csv(out_path)
        print("Downloaded to:", out_path)