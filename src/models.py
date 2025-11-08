"""Model training and evaluation helpers.

This module provides small wrappers around common scikit-learn models and
utilities for training and evaluating models. It's intentionally lightweight
so the training script can call into it for experiments.
"""
from __future__ import annotations

import joblib
import os
from typing import Any, Dict

from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def train_ridge(X, y, **kwargs) -> Ridge:
    model = Ridge(**kwargs)
    model.fit(X, y)
    return model


def train_random_forest(X, y, **kwargs) -> RandomForestRegressor:
    model = RandomForestRegressor(**kwargs)
    model.fit(X, y)
    return model


def evaluate_model(model, X, y) -> Dict[str, float]:
    preds = model.predict(X)
    return {
        "mae": float(mean_absolute_error(y, preds)),
        "mse": float(mean_squared_error(y, preds)),
        "r2": float(r2_score(y, preds)),
    }


def save_model(model: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)
