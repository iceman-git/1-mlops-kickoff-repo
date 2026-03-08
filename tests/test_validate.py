import pytest
import pandas as pd
from src.validate import validate, DataValidationError


def make_valid_df():
    return pd.DataFrame({
        "PassengerId": [1, 2, 3],
        "Survived": [0, 1, 0],
        "Pclass": [3, 1, 2],
        "Name": ["Smith, Mr. John", "Doe, Mrs. Jane", "Brown, Miss. Emily"],
        "Sex": ["male", "female", "female"],
        "Age": [22.0, 38.0, 26.0],
        "SibSp": [1, 0, 1],
        "Parch": [0, 1, 0],
        "Ticket": ["A/5 21171", "PC 17599", "STON/O2. 3101282"],
        "Fare": [7.25, 71.2833, 7.925],
        "Cabin": [None, "C85", None],
        "Embarked": ["S", "C", "S"],
    })


# a valid DataFrame
def test_valid_passes():
    df = make_valid_df()
    assert validate(df) is True


# wrong dtypes for numeric columns
def test_wrong_dtypes():
    df = make_valid_df()
    df["Age"] = df["Age"].astype(str)
    df["Fare"] = df["Fare"].astype(str)

    with pytest.raises(DataValidationError) as exc:
        validate(df)
    msg = str(exc.value).lower()
    assert "age" in msg and "float" in msg
    assert "fare" in msg and "float" in msg



# duplicate PassengerId values
def test_duplicate_passengerid():
    df = make_valid_df()
    df.loc[2, "PassengerId"] = df.loc[0, "PassengerId"]

    with pytest.raises(DataValidationError) as exc:
        validate(df)
    assert "duplicate" in str(exc.value).lower()


# unexpected categorical values (Sex, Embarked)
def test_unexpected_categorical_values():
    df = make_valid_df()
    df.loc[0, "Sex"] = "unknown"
    df.loc[1, "Embarked"] = "X"

    with pytest.raises(DataValidationError) as exc:
        validate(df)
    msg = str(exc.value)
    assert "sex" in msg.lower() and "unknown" in msg
    assert "embarked" in msg.lower() and "X" in msg


# missing required (non-nullable) columns
def test_missing_required_columns():
    df = make_valid_df().drop(columns=["Sex", "SibSp", "Parch", "Name"])
    with pytest.raises(DataValidationError) as exc:
        validate(df)
    msg = str(exc.value).lower()
    assert "missing" in msg
    for c in ["sex", "sibsp", "parch", "name"]:
        assert c in msg


# extra/unexpected columns are flagged
def test_extra_columns_flagged():
    df = make_valid_df().copy()
    df["ExtraColumn"] = ["x", "y", "z"]

    with pytest.raises(DataValidationError) as exc:
        validate(df)
    assert "extra" in str(exc.value).lower() or "unexpected" in str(exc.value).lower()


# nulls in non-nullable columns
def test_nulls_in_non_nullable_columns():
    df = make_valid_df()
    df.loc[1, "Name"] = None
    df.loc[2, "Sex"] = None

    with pytest.raises(DataValidationError) as exc:
        validate(df)
    msg = str(exc.value).lower()
    assert "name" in msg and "null" in msg
    assert "sex" in msg and "null" in msg


# negative values in columns that must be non-negative
@pytest.mark.parametrize(
    "col, values, expected_substr",
    [
        ("Age", [22.0, -38.0, 26.0], "age"),
        ("SibSp", [1, 0, -1], "sibsp"),
        ("Fare", [7.25, -71.28, 7.925], "fare"),
    ],
)
def test_negative_values_in_non_negative_columns(col, values, expected_substr):
    df = make_valid_df()
    df[col] = values
    with pytest.raises(DataValidationError) as exc:
        validate(df)
    assert expected_substr in str(exc.value).lower()


# empty DataFrame should fail due to missing required columns
def test_empty_dataframe():
    df = pd.DataFrame()
    with pytest.raises(DataValidationError) as exc:
        validate(df)
    assert "missing" in str(exc.value).lower()