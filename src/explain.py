"""Explainability helpers using SHAP.

This module provides a thin wrapper around SHAP to compute explanations for
tree and linear models. SHAP is optional and an ImportError will be raised if
it is not installed.
"""
from __future__ import annotations

import numpy as np


def compute_shap_explainer(model, X_sample):
    """Return a SHAP explainer and shap values for X_sample.

    Note: SHAP must be installed in the environment for this to work.
    """
    try:
        import shap
    except Exception as exc:
        raise ImportError("SHAP is required for explainability. Install with `pip install shap`.") from exc

    # Use the model-appropriate explainer
    try:
        explainer = shap.Explainer(model, X_sample)
    except Exception:
        # fallback to KernelExplainer for models shap can't handle directly
        explainer = shap.KernelExplainer(model.predict, X_sample)

    shap_values = explainer(X_sample)
    return explainer, shap_values
