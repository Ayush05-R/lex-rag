import pandas as pd
from pathlib import Path

RAW_DIR = Path("data/raw")
PROCESSED_DIR = Path("data/processed")
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

SPLITS = {
    "train": "cjpe_dataset_single_train.csv",
    "dev": "cjpe_dataset_single_dev.csv",
    "test": "cjpe_dataset_test.csv",
}


def load_split(filename: str) -> pd.DataFrame:
    return pd.read_csv(RAW_DIR / filename)


def drop_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    before = len(df)
    df = df.drop_duplicates(subset="text", keep="first")
    after = len(df)
    print(f"Dropped {before - after} duplicate rows.")
    return df


def clean_split(name: str, filename: str) -> None:
    print(f"--- Cleaning {name} ---")
    df = load_split(filename)

    if name == "train":
        df = drop_duplicates(df)

    out_path = PROCESSED_DIR / f"{name}_clean.csv"
    df.to_csv(out_path, index=False)
    print(f"--- {name} cleaned and saved to {out_path} ---")


if __name__ == "__main__":
    for name, filename in SPLITS.items():
        clean_split(name, filename)
