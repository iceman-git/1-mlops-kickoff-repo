"""
Educational Goal:
- Why this module exists in an MLOps system: Provide automated proof that the pipeline runs end-to-end and produces required artifacts.
- Responsibility (separation of concerns): Validate orchestration behavior (paths, artifacts, and minimal schema contracts).
- Pipeline contract (inputs and outputs): Running src.main.main() must materialize clean.csv, model.joblib, and predictions.csv.

TODO: Replace print statements with standard library logging in a later session
TODO: Any temporary or hardcoded variable or parameter will be imported from config.yml in a later session
"""

from pathlib import Path

import pandas as pd


def test_pipeline_end_to_end_regression(tmp_path, monkeypatch):
    """
    Inputs:
    - tmp_path: pytest-provided temporary directory.
    - monkeypatch: pytest fixture used to change working directory safely.
    Outputs:
    - None (asserts pipeline artifacts and contracts).
    Why this contract matters for reliable ML delivery:
    - A green end-to-end test is the fastest way to detect breaking refactors before they hit production.
    """
    print("[tests] Running end-to-end regression pipeline test in isolated temp directory.")  # TODO: replace with logging later

    # Run the pipeline in an isolated working directory so it writes artifacts under tmp_path/
    monkeypatch.chdir(tmp_path)

    # Import inside the test so it uses the current working directory for relative paths.
    import src.main as main_mod

    # TODO_STUDENT:
    # - Add more assertions here as your pipeline grows (e.g., schema checks, metric thresholds).
    # - Keep this test fast; heavy tests belong in separate suites.
    main_mod.main()

    clean_path = Path("data/processed/clean.csv")
    model_path = Path("models/model.joblib")
    preds_path = Path("reports/predictions.csv")

    assert clean_path.exists(), "Expected artifact missing: data/processed/clean.csv"
    assert model_path.exists(), "Expected artifact missing: models/model.joblib"
    assert preds_path.exists(), "Expected artifact missing: reports/predictions.csv"

    df_preds = pd.read_csv(preds_path)
    assert list(df_preds.columns) == ["prediction"], "predictions.csv must have exactly one column named 'prediction'"
    assert len(df_preds) > 0, "predictions.csv should contain at least one prediction row"


def test_pipeline_end_to_end_classification(tmp_path, monkeypatch):
    """
    Inputs:
    - tmp_path: pytest-provided temporary directory.
    - monkeypatch: pytest fixture used to change working directory safely.
    Outputs:
    - None (asserts pipeline runs for classification too).
    Why this contract matters for reliable ML delivery:
    - Ensures the same scaffolding supports both regression and classification paths without refactors breaking one mode.
    """
    print("[tests] Running end-to-end classification pipeline test in isolated temp directory.")  # TODO: replace with logging later

    monkeypatch.chdir(tmp_path)

    # Prepare a tiny classification dataset at the expected raw path so the run is deterministic.
    raw_path = Path("data/raw/example.csv")
    raw_path.parent.mkdir(parents=True, exist_ok=True)

    df_cls = pd.DataFrame(
        {
            "num_feature": [0.0, 1.0, 2.0, 3.0, 4.0, 5.0],
            "cat_feature": ["A", "B", "A", "B", "C", "C"],
            "target": [0, 1, 0, 1, 0, 1],  # binary labels
        }
    )
    df_cls.to_csv(raw_path, index=False)

    import src.main as main_mod

    # Temporarily switch SETTINGS to classification, then restore afterwards.
    original_settings = {k: v for k, v in main_mod.SETTINGS.items()}
    try:
        main_mod.SETTINGS["problem_type"] = "classification"
        main_mod.SETTINGS["is_example_config"] = True  # keep example mode on for scaffolding

        # TODO_STUDENT:
        # - Consider adding multi-class labels here once your real use case requires it.
        main_mod.main()
    finally:
        # Restore SETTINGS to avoid cross-test contamination.
        main_mod.SETTINGS.clear()
        main_mod.SETTINGS.update(original_settings)

    preds_path = Path("reports/predictions.csv")
    assert preds_path.exists(), "Expected artifact missing: reports/predictions.csv"

    df_preds = pd.read_csv(preds_path)
    assert list(df_preds.columns) == ["prediction"], "predictions.csv must have exactly one column named 'prediction'"
    assert len(df_preds) > 0, "predictions.csv should contain at least one prediction row"