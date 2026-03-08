import pytest
import pandas as pd

from src.validate import validate_dataframe


@pytest.fixture
def required_columns():
    # minimal, generic required schema
    return ["id", "age", "fare", "category"]


def make_valid_df():
    return pd.DataFrame(
        {
            "id": [1, 2, 3],
            "age": [22.0, 38.0, 26.0],
            "fare": [7.25, 71.2833, 7.925],
            "category": ["A", "B", "A"],
        }
    )


def test_valid_dataframe_passes(required_columns):
    df = make_valid_df()
    assert validate_dataframe(df, required_columns) is True


def test_none_dataframe_fails(required_columns):
    with pytest.raises(ValueError) as exc:
        validate_dataframe(None, required_columns)
    assert "empty" in str(exc.value).lower() or "dataframe" in str(exc.value).lower()


def test_empty_dataframe_fails(required_columns):
    df = pd.DataFrame()
    with pytest.raises(ValueError) as exc:
        validate_dataframe(df, required_columns)
    assert "empty" in str(exc.value).lower() or "missing" in str(exc.value).lower()


def test_missing_required_columns_fails(required_columns):
    df = make_valid_df().drop(columns=["fare", "category"])
    with pytest.raises(ValueError) as exc:
        validate_dataframe(df, required_columns)
    msg = str(exc.value).lower()
    assert "missing required columns" in msg
    assert "fare" in msg
    assert "category" in msg


def test_nulls_in_required_columns_fails(required_columns):
    df = make_valid_df()
    df.loc[1, "age"] = None
    with pytest.raises(ValueError) as exc:
        validate_dataframe(df, required_columns)
    msg = str(exc.value).lower()
    assert "null" in msg
    assert "age" in msg


def test_duplicate_rows_fails(required_columns):
    df = make_valid_df()
    df = pd.concat([df, df.iloc[[0]]], ignore_index=True)  # add an exact duplicate row
    with pytest.raises(ValueError) as exc:
        validate_dataframe(df, required_columns)
    assert "duplicate" in str(exc.value).lower()


def test_zero_variance_numeric_column_fails(required_columns):
    df = make_valid_df()
    df["fare"] = 1.23  # constant numeric column -> zero variance
    with pytest.raises(ValueError) as exc:
        validate_dataframe(df, required_columns)
    msg = str(exc.value).lower()
    assert "zero variance" in msg
    assert "fare" in msg


def test_nulls_in_non_required_columns_are_allowed(required_columns):
    # your validator only checks nulls in required_columns, so nulls elsewhere should pass
    df = make_valid_df()
    df["optional_note"] = [None, None, None]
    assert validate_dataframe(df, required_columns) is True