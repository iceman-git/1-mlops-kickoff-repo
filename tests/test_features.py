import pytest
import pandas as pd
from sklearn.compose import ColumnTransformer

from src.features import get_feature_preprocessor


def test_returns_column_transformer():
    pre = get_feature_preprocessor(
        quantile_bin_cols=["age"],
        categorical_onehot_cols=["sex"],
        numeric_passthrough_cols=["fare"],
    )
    assert isinstance(pre, ColumnTransformer)


def test_invalid_n_bins_fails():
    with pytest.raises(ValueError):
        get_feature_preprocessor(quantile_bin_cols=["age"], n_bins=1)


def test_overlapping_columns_fails():
    with pytest.raises(ValueError) as exc:
        get_feature_preprocessor(
            quantile_bin_cols=["age"],
            categorical_onehot_cols=["age"],  # overlap
        )
    assert "multiple" in str(exc.value).lower() or "overlap" in str(exc.value).lower()


def test_no_columns_with_drop_fails():
    with pytest.raises(ValueError):
        get_feature_preprocessor(remainder="drop")


def test_can_fit_and_transform():
    df = pd.DataFrame(
        {
            "age": [22.0, 38.0, 26.0],
            "fare": [7.25, 71.2833, 7.925],
            "sex": ["male", "female", "female"],
        }
    )

    pre = get_feature_preprocessor(
        quantile_bin_cols=["age"],
        categorical_onehot_cols=["sex"],
        numeric_passthrough_cols=["fare"],
        n_bins=3,
        remainder="drop",
    )

    Xt = pre.fit_transform(df)

    # Should be 2D with same number of rows
    assert Xt.shape[0] == df.shape[0]
    assert Xt.ndim == 2


def test_remainder_passthrough_allows_empty_lists():
    df = pd.DataFrame({"x": [1, 2, 3]})
    pre = get_feature_preprocessor(remainder="passthrough")
    Xt = pre.fit_transform(df)
    assert Xt.shape[0] == 3