import pandas as pd
from src.clean_data import clean_data

def test_clean_data_success():
    config = {
        "data": {"processed_path": "data/processed/titanic_clean.csv"},
        "schema": {
            "target": "Survived",
            "required_columns": [
                "PassengerId","Survived","Pclass","Name","Sex","Age",
                "SibSp","Parch","Ticket","Fare","Cabin","Embarked"
            ],
        },
        "logging": {"log_file": "data/reports/test.log", "level": "INFO"},
    }

    df = pd.read_csv("data/raw/Titanic-Dataset.csv")
    df_clean = clean_data(df, config)

    assert df_clean.shape[0] > 0
    assert list(df_clean.index) == list(range(df_clean.shape[0]))
    assert df_clean.duplicated().sum() == 0
    assert "Survived" in df_clean.columns