import pandas as pd
from src.clean_data import clean_data

def test_clean_data_success_tmp(tmp_path):
    df = pd.DataFrame(
        {
            "PassengerId": [1, 2],
            "Survived": [0, 1],
            "Pclass": [3, 1],
            "Name": ["A", "B"],
            "Sex": ["male", "female"],
            "Age": [22.0, 38.0],
            "SibSp": [1, 0],
            "Parch": [0, 0],
            "Ticket": ["X", "Y"],
            "Fare": [7.25, 71.28],
            "Cabin": [None, None],
            "Embarked": ["S", "C"],
        }
    )

    config = {
        "schema": {
            "target": "Survived",
            "required_columns": [
                "PassengerId","Survived","Pclass","Name","Sex","Age",
                "SibSp","Parch","Ticket","Fare","Cabin","Embarked"
            ],
        },
        "logging": {"log_file": str(tmp_path / "test.log"), "level": "INFO"},
    }

    df_clean = clean_data(df, config)

    assert df_clean.shape[0] == 2
    assert df_clean.duplicated().sum() == 0
    assert "Survived" in df_clean.columns

    # Stronger contracts:
    assert list(df_clean.index) == list(range(len(df_clean)))
    assert all(col in df_clean.columns for col in config["schema"]["required_columns"])