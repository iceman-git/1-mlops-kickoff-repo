"""
Module: Feature Engineering
---------------------------
Role: Define the transformation "recipe" (binning, encoding, scaling) to be bundled with the model.
Input: Configuration (lists of column names).
Output: scikit-learn ColumnTransformer object.
"""

from typing import Optional
import pandas as pd
import re
from sklearn.preprocessing import FunctionTransformer

def _extract_title_series(name_series: pd.Series) -> pd.Series:
    def _extract(name):
        if pd.isna(name):
            return "Unknown"
        m = re.search(r",\s*(.*?)\.", str(name))
        return m.group(1).strip() if m else "Unknown"
    return name_series.apply(_extract)

def _feature_adder(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds:
      - FamilySize = SibSp + Parch + 1
      - Title extracted from Name (rare titles grouped as 'Other')
    Safe behavior:
      - If SibSp/Parch missing -> treated as 0
      - If Name missing -> Title = 'Unknown'
      - Ensures categorical columns are strings for OneHotEncoder
    """
    df = df.copy()

    # Ensure sibling/parent columns exist
    if "SibSp" not in df.columns:
        df["SibSp"] = 0
    if "Parch" not in df.columns:
        df["Parch"] = 0

    df["FamilySize"] = df["SibSp"].fillna(0).astype(int) + df["Parch"].fillna(0).astype(int) + 1

    # Title extraction and normalization (preserve Unknown, map uncommon -> 'Other')
    common_titles = {
        "Mr", "Mrs", "Miss", "Master", "Dr", "Rev", "Ms", "Mlle", "Mme",
        "Major", "Col", "Capt", "Sir", "Lady", "Don", "Countess", "Jonkheer", "Dona"
    }

    if "Name" in df.columns:
        titles = _extract_title_series(df["Name"])
        # normalize synonyms
        titles = titles.replace({"Mlle": "Miss", "Ms": "Miss", "Mme": "Mrs"})
        # keep explicit Unknown
        is_unknown = titles == "Unknown"
        # map anything not in the whitelist to "Other"
        titles = titles.where(titles.isin(common_titles), other="Other")
        titles[is_unknown] = "Unknown"
        df["Title"] = titles
    else:
        df["Title"] = pd.Series(["Unknown"] * len(df), index=df.index)

    # Ensure categorical columns exist and are object dtype (not stringified 'nan')
    for c in ("Sex", "Embarked"):
        if c in df.columns:
            df[c] = df[c].astype(object)
        else:
            df[c] = pd.Series([None] * len(df), index=df.index, dtype=object)

    # Title already created above; make it object dtype but keep actual values (including 'Unknown'/'Other')
    df["Title"] = df["Title"].astype(object)

    return df

# sklearn-compatible transformer to insert as the first step of a Pipeline
FeatureAdder = FunctionTransformer(func=_feature_adder, validate=False)

def get_feature_adder():
    """Return a transformer that adds FamilySize and Title to a DataFrame."""
    return FeatureAdder