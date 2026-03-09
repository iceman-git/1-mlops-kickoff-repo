from pathlib import Path
import numpy as np
import pandas as pd
from src.utils import read_config, setup_logger

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = df.columns.astype(str).str.strip().str.replace(" ", "_", regex=False)
    return df

def clean_data(df: pd.DataFrame, config: dict) -> pd.DataFrame:
    logger = setup_logger(config["logging"]["log_file"], config["logging"]["level"])

    df = standardize_columns(df)

    required = set(config["schema"]["required_columns"])
    target = config["schema"]["target"]

    missing = required - set(df.columns)
    if missing:
        logger.error("Missing required columns: %s", sorted(missing))
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    if target not in df.columns:
        logger.error("Missing target column: %s", target)
        raise ValueError(f"Missing target column: {target}")

    start_rows = df.shape[0]

    df = df.drop_duplicates()

    if "Sex" in df.columns:
        df["Sex"] = df["Sex"].astype(str).str.lower().str.strip()
    
    if "Embarked" in df.columns:
        df["Embarked"] = df["Embarked"].astype("string").str.upper().str.strip()
        if df["Embarked"].isna().any():
            df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

    if "Age" in df.columns:
        global_median = df["Age"].median()
        medians = df.groupby(["Sex", "Pclass"])["Age"].median() if "Sex" in df.columns and "Pclass" in df.columns else None

        def fill_age(row):
            if pd.notna(row["Age"]):
                return row["Age"]
            if medians is not None:
                key = (row.get("Sex"), row.get("Pclass"))
                value = medians.get(key, np.nan)
            else:
                value = np.nan
            return global_median if pd.isna(value) else value

        df["Age"] = df.apply(fill_age, axis=1)

    if "Fare" in df.columns:
        df["Fare"] = pd.to_numeric(df["Fare"], errors="coerce")
        df["Fare"] = df["Fare"].fillna(df["Fare"].median())

    df = df.reset_index(drop=True)

    end_rows = df.shape[0]
    logger.info("Cleaned data: start_rows=%s end_rows=%s dropped=%s", start_rows, end_rows, start_rows - end_rows)

    return df

def save_clean(df: pd.DataFrame, config: dict) -> Path:
    out_path = Path(config["data"]["processed_path"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path

if __name__ == "__main__":
    config = read_config("config.yaml")
    logger = setup_logger(config["logging"]["log_file"], config["logging"]["level"])

    df = pd.read_csv(config["data"]["raw_path"])
    df_clean = clean_data(df, config)
    out = save_clean(df_clean, config)

    logger.info("Saved cleaned file: %s shape=%s", out, df_clean.shape)
    print(df_clean.head(3))