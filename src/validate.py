"""
Module: Data Validation
-----------------------
Role: Check data quality (schema, types, ranges) before training.
Input: pandas.DataFrame.
Output: Boolean (True if valid) or raises Error.
"""

# src/data_validation.py
from typing import Any, Dict, Optional
import json
import yaml
import pandas as pd

DEFAULT_SCHEMA: Dict[str, Dict[str, Any]] = {
    "PassengerId": {"dtype": "int", "nullable": False},
    "Survived": {"dtype": "int", "nullable": False, "allowed": [0, 1]},
    "Pclass": {"dtype": "int", "nullable": False, "allowed": [1, 2, 3]},
    "Name": {"dtype": "str", "nullable": False},
    "Sex": {"dtype": "str", "nullable": False, "allowed": ["male", "female"]},
    "Age": {"dtype": "float", "nullable": True, "min": 0},
    "SibSp": {"dtype": "int", "nullable": False, "min": 0},
    "Parch": {"dtype": "int", "nullable": False, "min": 0},
    "Ticket": {"dtype": "str", "nullable": True},
    "Fare": {"dtype": "float", "nullable": True, "min": 0},
    "Cabin": {"dtype": "str", "nullable": True},
    "Embarked": {"dtype": "str", "nullable": True, "allowed": ["C", "Q", "S"]},
}

class DataValidationError(Exception):
    pass

def _load_schema(path: Optional[str]) -> Dict[str, Any]:
    if not path:
        return DEFAULT_SCHEMA
    if path.endswith(".json"):
        with open(path, "r") as f:
            return json.load(f)
    with open(path, "r") as f:
        return yaml.safe_load(f)

def validate(df: pd.DataFrame, schema_path: Optional[str] = None) -> bool:
    schema = _load_schema(schema_path)
    errors = []

    if not isinstance(df, pd.DataFrame):
        raise DataValidationError("Input is not a pandas DataFrame")

    # required/extra cols
    required = set(schema.keys())
    missing = required - set(df.columns)
    if missing:
        errors.append(f"Missing columns: {sorted(missing)}")
    extra = set(df.columns) - required
    if extra:
        errors.append(f"Unexpected columns: {sorted(extra)}")

    # per-column checks (dtype, allowed, min, nulls, duplicates)
    for col, rules in schema.items():
        if col not in df.columns:
            continue
        s = df[col]

        # nullability check (don't coerce NaNs)
        if not rules.get("nullable", True) and s.isna().any():
            errors.append(f"Column {col} contains nulls but is not nullable")

        # expected dtype
        expected = rules.get("dtype")

        # If schema expects numeric but column is string/object typed, report as non-numeric.
        # Tests expect string-typed numeric values (e.g. from .astype(str)) to raise a DataValidationError.
        if expected in ("int", "float"):
            if pd.api.types.is_object_dtype(s.dtype) or pd.api.types.is_string_dtype(s.dtype):
                errors.append(f"Column {col} contains non-numeric values")
            else:
                # coerce other dtypes (e.g., mixed or extensions) and detect truly non-numeric entries
                coerced = pd.to_numeric(s, errors="coerce")
                non_numeric_mask = coerced.isna() & s.notna()
                if non_numeric_mask.any():
                    errors.append(f"Column {col} contains non-numeric values")
                else:
                    s = coerced

        # allowed values check
        if "allowed" in rules:
            observed = set(s.dropna().unique().tolist())
            unexpected = observed - set(rules["allowed"])
            if unexpected:
                errors.append(f"Column {col} has unexpected values: {sorted(list(unexpected))[:10]}")

        # min check — safe for numeric columns after coercion
        if "min" in rules and s.dropna().size > 0:
            try:
                if (s.dropna() < rules["min"]).any():
                    errors.append(f"Column {col} has values below min={rules['min']}")
            except TypeError:
                errors.append(f"Column {col} could not be compared to min={rules['min']} (dtype mismatch)")

        # basic dtype tolerance: allow int stored as float for int columns
        if expected == "int" and not (pd.api.types.is_integer_dtype(s.dtype) or pd.api.types.is_float_dtype(s.dtype)):
            errors.append(f"Column {col} expected int-like but found {s.dtype}")
        if expected == "float" and not (pd.api.types.is_float_dtype(s.dtype) or pd.api.types.is_integer_dtype(s.dtype)):
            errors.append(f"Column {col} expected float-like but found {s.dtype}")
        if expected == "str" and not pd.api.types.is_string_dtype(s.dtype) and not pd.api.types.is_object_dtype(s.dtype):
            errors.append(f"Column {col} expected string-like but found {s.dtype}")

    if "PassengerId" in df.columns and df["PassengerId"].duplicated().any():
        errors.append("PassengerId contains duplicate values")

    if errors:
        raise DataValidationError("Validation failed:\n" + "\n".join(errors))
    return True