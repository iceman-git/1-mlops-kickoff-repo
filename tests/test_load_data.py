import pandas as pd
from src.load_data import load_data
import pytest


def test_load_data_success_tmp(tmp_path):
    # Arrange: tiny CSV in temp directory
    raw_path = tmp_path / "example.csv"
    pd.DataFrame({"a": [1, 2], "b": [3, 4]}).to_csv(raw_path, index=False)

    config = {
        "data": {"raw_path": str(raw_path)},
        "logging": {"log_file": str(tmp_path / "test.log"), "level": "INFO"},
    }

    # Act
    df = load_data(config)

    # Assert
    assert isinstance(df, pd.DataFrame)
    assert df.shape == (2, 2)


def test_load_data_missing_file_raises(tmp_path):
    config = {
        "data": {"raw_path": str(tmp_path / "missing.csv")},
        "logging": {"log_file": str(tmp_path / "test.log"), "level": "INFO"},
    }

    with pytest.raises(FileNotFoundError):
        load_data(config)


def test_load_data_empty_file_raises(tmp_path):
    raw_path = tmp_path / "empty.csv"
    pd.DataFrame().to_csv(raw_path, index=False)

    config = {
        "data": {"raw_path": str(raw_path)},
        "logging": {"log_file": str(tmp_path / "test.log"), "level": "INFO"},
    }

    with pytest.raises(ValueError):
        load_data(config)