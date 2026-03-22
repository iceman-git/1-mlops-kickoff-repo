"""
Module: Feature Engineering
---------------------------
Role: Define the transformation recipe (binning, encoding, scaling) bundled with the model.
Input: Configuration (lists of column names).
Output: scikit-learn ColumnTransformer object.
"""

import logging
from typing import List, Optional

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import KBinsDiscretizer, OneHotEncoder

logger = logging.getLogger("mlops")


def _dedupe_preserve_order(cols: List[str]) -> List[str]:
    seen = set()
    out = []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def get_feature_preprocessor(
    quantile_bin_cols: Optional[List[str]] = None,
    categorical_onehot_cols: Optional[List[str]] = None,
    numeric_passthrough_cols: Optional[List[str]] = None,
    n_bins: int = 3,
    remainder: str = "drop",
) -> ColumnTransformer:
    """
    Inputs:
    - quantile_bin_cols: Numeric columns to discretize into quantile bins.
    - categorical_onehot_cols: Categorical columns to one-hot encode.
    - numeric_passthrough_cols: Numeric columns to pass through unchanged.
    - n_bins: Number of bins for KBinsDiscretizer.
    - remainder: What to do with unspecified columns ("drop" or "passthrough").

    Outputs:
    - preprocessor: A scikit-learn ColumnTransformer (NOT fitted).
    """
    logger.info("[features] Building ColumnTransformer feature recipe.")

    quantile_bin_cols = _dedupe_preserve_order(quantile_bin_cols or [])
    categorical_onehot_cols = _dedupe_preserve_order(categorical_onehot_cols or [])
    numeric_passthrough_cols = _dedupe_preserve_order(numeric_passthrough_cols or [])

    if n_bins < 2:
        raise ValueError("n_bins must be >= 2 for discretization to be meaningful.")

    all_cols = quantile_bin_cols + categorical_onehot_cols + numeric_passthrough_cols
    if len(all_cols) != len(set(all_cols)):
        overlaps = []
        for c in set(all_cols):
            count = (c in quantile_bin_cols) + (c in categorical_onehot_cols) + (c in numeric_passthrough_cols)
            if count > 1:
                overlaps.append(c)
        raise ValueError(
            f"Feature spec invalid: columns appear in multiple transformer lists: {sorted(overlaps)}"
        )

    if not (quantile_bin_cols or categorical_onehot_cols or numeric_passthrough_cols) and remainder == "drop":
        raise ValueError(
            "Feature spec invalid: no feature columns were provided and remainder='drop' "
            "would drop all features. Provide at least one column list or set remainder='passthrough'."
        )

    num_binner = KBinsDiscretizer(
        n_bins=n_bins,
        encode="onehot-dense",
        strategy="quantile",
        quantile_method="linear",
    )

    try:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        ohe = OneHotEncoder(handle_unknown="ignore", sparse=False)

    transformers = []

    if quantile_bin_cols:
        transformers.append(("quantile_bin", num_binner, quantile_bin_cols))
    if categorical_onehot_cols:
        transformers.append(("categorical_onehot", ohe, categorical_onehot_cols))
    if numeric_passthrough_cols:
        transformers.append(("numeric_passthrough", "passthrough", numeric_passthrough_cols))

    logger.info(
        "[features] quantile_bin=%s | categorical_onehot=%s | numeric_passthrough=%s",
        quantile_bin_cols,
        categorical_onehot_cols,
        numeric_passthrough_cols,
    )

    preprocessor = ColumnTransformer(
        transformers=transformers,
        remainder=remainder,
        verbose_feature_names_out=True,
    )

    return preprocessor