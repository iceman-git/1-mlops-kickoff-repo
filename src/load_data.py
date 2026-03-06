from pathlib import Path
import pandas as pd
from src.utils import read_config, setup_logger

def load_data(config: dict) -> pd.DataFrame:
    raw_path = Path(config["data"]["raw_path"])
    logger = setup_logger(config["logging"]["log_file"], config["logging"]["level"])

    if not raw_path.exists():
        logger.error("Raw file not found: %s", raw_path)
        raise FileNotFoundError(f"Raw file not found: {raw_path}")

    if not raw_path.is_file():
        logger.error("Raw path is not a file: %s", raw_path)
        raise ValueError(f"Raw path is not a file: {raw_path}")

    try:
        df = pd.read_csv(raw_path)
    except Exception as e:
        logger.error("Failed reading CSV: %s | %s", raw_path, str(e))
        raise ValueError(f"Failed reading CSV: {raw_path}") from e

    if df.shape[0] == 0:
        logger.error("Loaded dataset is empty: %s", raw_path)
        raise ValueError(f"Loaded dataset is empty: {raw_path}")

    logger.info("Loaded raw data: path=%s shape=%s", raw_path, df.shape)
    return df

if __name__ == "__main__":
    config = read_config("config.yaml")
    df = load_data(config)
    print(df.head(3))