import pandas as pd
from src.load_data import load_data

def test_load_data_success():
    config = {
        "data": {"raw_path": "data/raw/Titanic-Dataset.csv"},
        "logging": {"log_file": "data/reports/test.log", "level": "INFO"},
    }

    df = load_data(config)

    assert isinstance(df, pd.DataFrame)
    assert df.shape[0] > 0
    assert df.shape[1] > 0