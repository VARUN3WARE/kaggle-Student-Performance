"""Model loading and prediction helpers.

This module centralizes loading of model artifacts and provides small helper
functions for making predictions from raw inputs. The Streamlit app will use
these functions so model-loading logic is kept out of the UI file.
"""
from __future__ import annotations

import os
from typing import Tuple

import joblib
import numpy as np
import pandas as pd


def load_models(model_dir: str = "models") -> Tuple[object, object, object]:
    """Load pretrained model artifacts from disk.

    Expects the following files to exist in `model_dir`:
    - best_ridge_model.pkl
    - best_rf_model.pkl
    - transformer.pkl (ColumnTransformer / scaler)

    Returns (ridge_model, rf_model, scaler)
    """
    ridge_path = os.path.join(model_dir, "best_ridge_model.pkl")
    rf_path = os.path.join(model_dir, "best_rf_model.pkl")
    transformer_path = os.path.join(model_dir, "transformer.pkl")

    if not (os.path.exists(ridge_path) and os.path.exists(rf_path) and os.path.exists(transformer_path)):
        raise FileNotFoundError(
            f"Model files not found in '{model_dir}'. Expected: {ridge_path}, {rf_path}, {transformer_path}"
        )

    ridge = joblib.load(ridge_path)
    rf = joblib.load(rf_path)
    transformer = joblib.load(transformer_path)
    return ridge, rf, transformer


def _ensure_array(x) -> np.ndarray:
    arr = np.asarray(x)
    if arr.ndim == 1:
        arr = arr.reshape(1, -1)
    return arr


def predict_from_array(
    X, ridge_model, rf_model, scaler, model_choice: str = "Ridge Regression"
) -> float:
    """Scale X and return a scalar prediction using the chosen model.

    X: array-like with shape (n_features,) or (1, n_features)
    model_choice: either 'Ridge Regression' or 'Random Forest' (case-insensitive)

    Returns a Python float (single prediction).
    """
    # Prefer to keep DataFrame inputs as DataFrames (so ColumnTransformer
    # selectors which rely on column names work). Only convert to numpy array
    # for scalers that expect raw arrays.
    X_df = None
    if isinstance(X, pd.DataFrame):
        X_df = X
        X_arr = None
    else:
        X_arr = _ensure_array(X)

    # Choose the model up-front so we can infer which feature set it expects.
    chosen_name = "ridge" if str(model_choice).lower().startswith("ridge") else "rf"
    chosen = ridge_model if chosen_name == "ridge" else rf_model

    def _model_is_pipeline_with_preprocessor(model):
        try:
            if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
                return True
        except Exception:
            pass
        return False

    # If the chosen model has a preprocessor inside its pipeline, prefer to
    # construct a DataFrame matching that preprocessor's feature names. If not,
    # fall back to the standalone `scaler`'s feature names (if available).
    expected_names = None
    if _model_is_pipeline_with_preprocessor(chosen):
        try:
            expected_names = list(chosen.named_steps["preprocessor"].feature_names_in_)
        except Exception:
            expected_names = None
    if expected_names is None and hasattr(scaler, "feature_names_in_"):
        expected_names = list(getattr(scaler, "feature_names_in_"))

    if expected_names is not None:
        if X_df is None:
            X_df = pd.DataFrame(X_arr, columns=expected_names)
        else:
            # Ensure DataFrame has the expected columns. If some columns are
            # missing, create them and fill with NaN so imputers can handle.
            if list(X_df.columns) != list(expected_names):
                new_row = {}
                for c in expected_names:
                    if c in X_df.columns:
                        new_row[c] = X_df.iloc[0][c]
                    else:
                        # Try a few simple name variants mapping (common renames)
                        if c == "extracurricular_activities" and "extracurricular" in X_df.columns:
                            new_row[c] = X_df.iloc[0]["extracurricular"]
                        elif c == "sample_question_papers_practiced" and "sample_papers_practiced" in X_df.columns:
                            new_row[c] = X_df.iloc[0]["sample_papers_practiced"]
                        else:
                            new_row[c] = np.nan
                X_df = pd.DataFrame([new_row], columns=expected_names)
        # we'll pass the DataFrame to the model or scaler depending on model type
    else:
        if X_arr is None:
            X_arr = np.asarray(X_df)

    chosen = ridge_model if str(model_choice).lower().startswith("ridge") else rf_model

    # If the chosen model is a pipeline with its own preprocessor, pass raw
    # DataFrame/array and let the pipeline handle transformation.
    if _model_is_pipeline_with_preprocessor(chosen):
        if X_df is not None:
            pred = chosen.predict(X_df)
        else:
            pred = chosen.predict(X_arr)
    else:
        # Model expects transformed arrays
        if hasattr(scaler, "feature_names_in_"):
            X_scaled = scaler.transform(X_df)
        else:
            X_scaled = scaler.transform(X_arr)
        pred = chosen.predict(X_scaled)

    return float(pred[0])


def predict_from_inputs(
    hours_studied: float,
    previous_scores: float,
    extracurricular: str | int,
    sleep_hours: float,
    sample_papers_practiced: int,
    ridge_model, rf_model, scaler,
    model_choice: str = "Ridge Regression",
) -> float:
    """Convenience wrapper that builds the feature array from named inputs.

    `extracurricular` may be provided as 'Yes'/'No', True/False, or 1/0.
    """
    if isinstance(extracurricular, str):
        extracurricular_numeric = 1 if extracurricular.lower().startswith("y") else 0
    else:
        extracurricular_numeric = int(bool(extracurricular))

    X = np.array([
        hours_studied,
        previous_scores,
        extracurricular_numeric,
        sleep_hours,
        sample_papers_practiced,
    ])

    # If transformer was fitted with feature names, build a DataFrame matching
    # those names so ColumnTransformer selectors work. Fill unknown features with
    # zeros (imputer in transformer can handle missingness where appropriate).
    if hasattr(scaler, "feature_names_in_"):
        cols = list(getattr(scaler, "feature_names_in_"))
        row = {}
        for c in cols:
            if c == "hours_studied":
                row[c] = hours_studied
            elif c == "previous_scores":
                row[c] = previous_scores
            elif c == "extracurricular" or c == "extracurricular_numeric":
                # transformer may expect either the original categorical name or a numeric column
                row[c] = extracurricular_numeric
            elif c == "sleep_hours":
                row[c] = sleep_hours
            elif c == "sample_papers_practiced":
                row[c] = sample_papers_practiced
            else:
                # default filler; prefer np.nan so imputers can act, but fall back to 0
                row[c] = np.nan

        X_df = pd.DataFrame([row], columns=cols)
        return predict_from_array(X_df, ridge_model, rf_model, scaler, model_choice=model_choice)

    return predict_from_array(X, ridge_model, rf_model, scaler, model_choice=model_choice)
