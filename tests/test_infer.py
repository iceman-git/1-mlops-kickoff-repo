import pandas as pd
import pytest
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.infer import run_inference


# =============================================================================
# SHARED FIXTURES
# =============================================================================

@pytest.fixture
def sample_X_infer():
    """
    A small, valid inference DataFrame with the 7 feature columns the
    Titanic pipeline expects after clean_dataframe() has run.

    Index starts at 10 (not 0) intentionally — this is the key detail
    that exposes index misalignment bugs. If a test used index [0,1,2]
    it might pass even if the index was reset incorrectly inside run_inference.
    """
    return pd.DataFrame(
        {
            "Pclass":     [3,       1,        2     ],
            "Sex":        ["male",  "female", "male"],
            "Age":        [25.0,    35.0,     45.0  ],
            "Fare":       [8.05,    71.2833,  21.0  ],
            "Embarked":   ["S",     "C",      "S"   ],
            "FamilySize": [2,       2,        1     ],
            "Title":      ["Mr",    "Mrs",    "Mr"  ],
        },
        index=[10, 11, 12],  # non-zero index — intentional
    )


@pytest.fixture
def fitted_model(sample_X_infer):
    """
    A minimal fitted sklearn Pipeline that mirrors the production pipeline
    structure (ColumnTransformer + LogisticRegression) without depending
    on the full training pipeline. Kept simple so tests remain fast and
    isolated from changes in train.py or features.py.
    """
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(),                          ["Age", "Fare", "FamilySize"]),
            ("cat", OneHotEncoder(handle_unknown="ignore"),    ["Pclass", "Sex", "Embarked", "Title"]),
        ]
    )
    pipeline = Pipeline([
        ("preprocess", preprocessor),
        ("model",      LogisticRegression(max_iter=500)),
    ])
    y_train = pd.Series([0, 1, 0], index=sample_X_infer.index)
    pipeline.fit(sample_X_infer, y_train)
    return pipeline


# =============================================================================
# TESTS
# =============================================================================

def test_output_is_dataframe(fitted_model, sample_X_infer):
    """
    model.predict() returns a numpy array. run_inference() must wrap it
    in a DataFrame. This guards against someone simplifying run_inference
    to just return model.predict() directly, which would silently break
    every downstream step that expects a DataFrame.
    """
    result = run_inference(fitted_model, sample_X_infer)
    assert isinstance(result, pd.DataFrame), (
        f"run_inference must return a pd.DataFrame, got {type(result)}"
    )


def test_output_has_single_prediction_column(fitted_model, sample_X_infer):
    """
    The contract requires exactly one column named 'prediction'.
    Any rename or accidental extra column (e.g. appending input features)
    breaks downstream joins and saved CSV schemas.
    """
    result = run_inference(fitted_model, sample_X_infer)
    assert list(result.columns) == ["prediction"], (
        f"Expected columns ['prediction'], got {list(result.columns)}"
    )


def test_output_index_matches_input_index(fitted_model, sample_X_infer):
    """
    Index alignment is the most critical correctness guarantee.
    If the index is reset inside run_inference, joining predictions
    back to passenger records by index produces silently wrong row matches.
    The non-zero fixture index [10, 11, 12] makes this failure visible.
    """
    result = run_inference(fitted_model, sample_X_infer)
    assert list(result.index) == list(sample_X_infer.index), (
        f"Output index {list(result.index)} does not match "
        f"input index {list(sample_X_infer.index)}"
    )


def test_output_length_matches_input(fitted_model, sample_X_infer):
    """
    One prediction row must be produced for every input row.
    Verifies no rows are silently dropped or duplicated during prediction.
    """
    result = run_inference(fitted_model, sample_X_infer)
    assert len(result) == len(sample_X_infer), (
        f"Expected {len(sample_X_infer)} prediction rows, got {len(result)}"
    )


def test_predictions_are_binary(fitted_model, sample_X_infer):
    """
    For a binary Titanic survival classifier, all prediction values must
    be either 0 (did not survive) or 1 (survived). Any value outside this
    set indicates a problem with the model, label encoding, or postprocessing.
    """
    result = run_inference(fitted_model, sample_X_infer)
    unexpected = set(result["prediction"].unique()) - {0, 1}
    assert not unexpected, (
        f"Unexpected prediction values found: {unexpected}. "
        f"All values must be 0 or 1."
    )