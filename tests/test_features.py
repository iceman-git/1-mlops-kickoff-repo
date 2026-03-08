# does feature adder exist
from src.features import get_feature_adder

def test_get_feature_adder():
    assert get_feature_adder() is not None

# functional correctness of Title extraction and FamilySize calculation
import pandas as pd     

def test_feature_adder_functionality():
    from src.features import _feature_adder

    # Create a test DataFrame
    df = pd.DataFrame({
        "Name": ["Smith, Mr. John", "Doe, Mrs. Jane", "Brown, Miss. Emily", None],
        "SibSp": [1, 0, 2, 1],
        "Parch": [0, 1, 0, 0]
    })

    # Apply the feature adder
    transformed_df = _feature_adder(df)

    # Check FamilySize calculation
    assert transformed_df.loc[0, "FamilySize"] == 2  # 1 sibling + 0 parents + 1
    assert transformed_df.loc[1, "FamilySize"] == 2  # 0 siblings + 1 parent + 1
    assert transformed_df.loc[2, "FamilySize"] == 3  # 2 siblings + 0 parents + 1
    assert transformed_df.loc[3, "FamilySize"] == 2  # treated as (0 siblings + 0 parents + 1)

    # Check Title extraction and grouping
    assert transformed_df.loc[0, "Title"] == "Mr"
    assert transformed_df.loc[1, "Title"] == "Mrs"
    assert transformed_df.loc[2, "Title"] == "Miss"
    assert transformed_df.loc[3, "Title"] == "Unknown"

#Edge case: missing SibSp/Parch columns
def test_feature_adder_missing_columns():
    from src.features import _feature_adder

    df = pd.DataFrame({
        "Name": ["Smith, Mr. John", "Doe, Mrs. Jane"]
    })

    transformed_df = _feature_adder(df)

    # Should create SibSp and Parch with default 0, so FamilySize = 1
    assert transformed_df.loc[0, "FamilySize"] == 1
    assert transformed_df.loc[1, "FamilySize"] == 1

#Types of categorical columns should be strings
def test_feature_adder_categorical_types():
    from src.features import _feature_adder

    df = pd.DataFrame({
        "Name": ["Smith, Mr. John", "Doe, Mrs. Jane"],
        "SibSp": [1, 0],
        "Parch": [0, 1]
    })

    transformed_df = _feature_adder(df)

    # Check that categorical columns are of type string
    assert transformed_df["Sex"].dtype == "object"
    assert transformed_df["Embarked"].dtype == "object"
    assert transformed_df["Title"].dtype == "object"

#handles missing values
def test_feature_adder_handles_missing_values():
    from src.features import _feature_adder

    df = pd.DataFrame({
        "Name": ["Smith, Mr. John", None],
        "SibSp": [1, None],
        "Parch": [0, None]
    })

    transformed_df = _feature_adder(df)

    # Missing Name should yield Title = 'Unknown'
    assert transformed_df.loc[1, "Title"] == "Unknown"

    # Missing SibSp/Parch should be treated as 0, so FamilySize = 1
    assert transformed_df.loc[1, "FamilySize"] == 1