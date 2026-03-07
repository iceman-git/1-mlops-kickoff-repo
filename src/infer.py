"""
Module: Inference
-----------------
Role: Make predictions on new, unseen data with a trained model.
Input: Trained Model + Pre-cleaned features (X)
Output: Predictions (labels and/or probabilities as DataFrame or array)

NOTE: Assumes input data has been cleaned and preprocessed by clean_data.py.
      The trained model (Pipeline) includes all necessary transformations.
"""

import pandas as pd
import numpy as np
import joblib


def load_model(model_path):
    """
    Load a trained scikit-learn Pipeline from disk.
    
    Parameters
    ----------
    model_path : str
        Path to the saved model file (e.g., 'models/logistic_titanic_pipeline.joblib')
    
    Returns
    -------
    sklearn Pipeline
        Trained model ready for predictions
    
    Raises
    ------
    FileNotFoundError
        If model file does not exist
    """
    try:
        model = joblib.load(model_path)
        print(f"Model loaded successfully from {model_path}")
        return model
    except FileNotFoundError:
        raise FileNotFoundError(f"Model file not found at: {model_path}")


def predict_labels(model, X):
    """
    Generate class label predictions for new data.
    
    Parameters
    ----------
    model : sklearn Pipeline
        Trained model with predict method
    X : pd.DataFrame or np.ndarray
        Feature data (cleaned, no missing values)
    
    Returns
    -------
    np.ndarray
        Predicted class labels (0 or 1 for binary classification)
    """
    return model.predict(X)


def predict_probabilities(model, X):
    """
    Generate probability predictions for new data.
    
    Parameters
    ----------
    model : sklearn Pipeline
        Trained model with predict_proba method
    X : pd.DataFrame or np.ndarray
        Feature data (cleaned, no missing values)
    
    Returns
    -------
    np.ndarray
        Predicted probabilities for both classes [shape: (n_samples, 2)]
    """
    return model.predict_proba(X)


def predict_with_confidence(model, X, prob_threshold=0.5, return_proba=True):
    """
    Generate predictions with confidence scores and optionally filter by threshold.
    
    Parameters
    ----------
    model : sklearn Pipeline
        Trained model with predict and predict_proba methods
    X : pd.DataFrame or np.ndarray
        Feature data (cleaned, no missing values)
    prob_threshold : float, optional
        Confidence threshold for filtering predictions (default: 0.5)
        Only used if return_proba=True
    return_proba : bool, optional
        If True, return both predictions and probabilities (default: True)
    
    Returns
    -------
    dict
        Dictionary containing:
        - 'predictions': predicted class labels
        - 'probabilities': predicted probabilities for positive class (if return_proba=True)
        - 'confidence': max probability for each prediction (if return_proba=True)
        - 'high_confidence': boolean mask for predictions above threshold (if return_proba=True)
    """
    predictions = model.predict(X)
    
    result = {'predictions': predictions}
    
    if return_proba:
        proba = model.predict_proba(X)
        pos_proba = proba[:, 1]
        confidence = np.max(proba, axis=1)
        high_confidence = confidence >= prob_threshold
        
        result['probabilities'] = pos_proba
        result['confidence'] = confidence
        result['high_confidence'] = high_confidence
    
    return result


def batch_predict(model, X, batch_size=None):
    """
    Generate predictions for large datasets in batches to manage memory.
    
    Parameters
    ----------
    model : sklearn Pipeline
        Trained model with predict method
    X : pd.DataFrame or np.ndarray
        Feature data (cleaned, no missing values)
    batch_size : int, optional
        Number of samples per batch (default: None, process all at once)
    
    Returns
    -------
    np.ndarray
        Predicted class labels
    """
    if batch_size is None or len(X) <= batch_size:
        return model.predict(X)
    
    predictions = []
    n_samples = len(X)
    
    for start_idx in range(0, n_samples, batch_size):
        end_idx = min(start_idx + batch_size, n_samples)
        batch_preds = model.predict(X.iloc[start_idx:end_idx] if isinstance(X, pd.DataFrame) else X[start_idx:end_idx])
        predictions.extend(batch_preds)
    
    return np.array(predictions)


def predictions_to_dataframe(predictions_dict, X_original=None, include_index=True):
    """
    Convert predictions dictionary to a formatted DataFrame.
    
    Parameters
    ----------
    predictions_dict : dict
        Dictionary from predict_with_confidence() containing predictions, probabilities, etc.
    X_original : pd.DataFrame, optional
        Original feature DataFrame to include in output (default: None)
    include_index : bool, optional
        If True, include original index from X_original (default: True)
    
    Returns
    -------
    pd.DataFrame
        DataFrame with predictions and optional original features
    """
    result_df = pd.DataFrame({
        'predicted_label': predictions_dict['predictions']
    })
    
    if 'probabilities' in predictions_dict:
        result_df['survival_probability'] = predictions_dict['probabilities']
    
    if 'confidence' in predictions_dict:
        result_df['confidence'] = predictions_dict['confidence']
    
    if 'high_confidence' in predictions_dict:
        result_df['high_confidence'] = predictions_dict['high_confidence']
    
    # Add original features if provided
    if X_original is not None:
        if include_index and isinstance(X_original, pd.DataFrame):
            result_df.index = X_original.index
        result_df = pd.concat([X_original.reset_index(drop=True), result_df.reset_index(drop=True)], axis=1)
    
    return result_df


def run_inference_pipeline(model_path, X):
    """
    End-to-end inference: load model and generate predictions.
    
    Parameters
    ----------
    model_path : str
        Path to saved model file
    X : pd.DataFrame or np.ndarray
        Feature data (cleaned, no missing values)
    
    Returns
    -------
    dict
        Predictions dictionary with labels and probabilities
    """
    model = load_model(model_path)
    return predict_with_confidence(model, X)