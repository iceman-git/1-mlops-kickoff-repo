"""
Module: Data Validation
-----------------------
Role: Check data quality (schema, types, ranges) before training.
Input: pandas.DataFrame.
Output: Boolean (True if valid) or raises Error.
"""

"""
Educational Goal:
- Why this module exists in an MLOps system: Catch obvious data problems early to prevent silent model failures.
- Responsibility (separation of concerns): Validation is not cleaning; it is a fail-fast quality gate.
- Pipeline contract (inputs and outputs): Input = df + required columns list, Output = True if valid else raise.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

import pandas as pd


def validate_dataframe(df: pd.DataFrame, required_columns: list) -> bool:
    """
    Inputs:
    - df: DataFrame to validate.
    - required_columns: List of columns that must exist.

    Outputs:
    - True if valid.
    - Raises ValueError if validation fails.

    Why this contract matters:
    - Fail-fast validation prevents wasting compute and time training on broken datasets.
    - Protects downstream pipeline stages (features, training, evaluation).
    """

    print("[validate.validate_dataframe] Validating dataframe")

    
    # 1. Basic structural validation
    
    if df is None or len(df) == 0:
        raise ValueError(
            "Validation failed: DataFrame is empty. Check ingestion and cleaning steps."
        )

    missing = [c for c in required_columns if c not in df.columns]
    if missing:
        raise ValueError(
            f"Validation failed: Missing required columns: {missing}"
        )

    
    # 2. Null value check (critical baseline protection)
    
    null_counts = df[required_columns].isnull().sum()
    if null_counts.any():
        raise ValueError(
            f"Validation failed: Null values detected in required columns:\n{null_counts}"
        )

    
    # 3. Duplicate row check
    
    duplicate_count = df.duplicated().sum()
    if duplicate_count > 0:
        raise ValueError(
            f"Validation failed: {duplicate_count} duplicate rows detected."
        )

    
    # 4. Basic numeric sanity checks
    
    numeric_cols = df[required_columns].select_dtypes(include=["number"]).columns

    for col in numeric_cols:
        if df[col].min() == df[col].max():
            raise ValueError(
                f"Validation failed: Column '{col}' has zero variance."
            )

    print("[validate.validate_dataframe] Validation successful")

    return True