"""Feature engineering and transformer utilities."""
from __future__ import annotations

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler
from typing import Tuple


def _numeric_columns(X: pd.DataFrame) -> list:
    return X.select_dtypes(include=["number"]).columns.tolist()


def build_transformer(X: pd.DataFrame):
    """Build a simple ColumnTransformer that standardizes numeric features.

    Returns a fitted transformer when `fit()` is called later.
    """
    numeric_cols = _numeric_columns(X)
    transformer = ColumnTransformer([("num", StandardScaler(), numeric_cols)], remainder="passthrough")
    return transformer


def fit_transformer(X: pd.DataFrame, save_path: str | None = None):
    """Fit transformer on DataFrame X and optionally persist it with joblib."""
    transformer = build_transformer(X)
    transformer.fit(X)
    if save_path:
        joblib.dump(transformer, save_path)
    return transformer


def transform_X(transformer, X: pd.DataFrame) -> np.ndarray:
    """Apply a fitted transformer to DataFrame X and return a numpy array."""
    return transformer.transform(X)
