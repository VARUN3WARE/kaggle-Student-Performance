import os
import sys

import numpy as np

# Ensure the repository root is on sys.path so `src` can be imported during tests
repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src import predict


def test_load_models_and_predict():
    """Ensure models load and a prediction can be produced for a sample input.

    This test uses the pre-saved artifacts in `models/` located at the repo root.
    It verifies the predict function returns a numeric scalar within a reasonable range.
    """
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    # Normalize the path
    model_dir = os.path.normpath(model_dir)

    ridge, rf, scaler = predict.load_models(model_dir=model_dir)

    # use a typical input similar to the defaults in the app
    hours_studied = 8
    previous_scores = 50
    extracurricular = "Yes"
    sleep_hours = 8
    sample_papers_practiced = 10

    pred = predict.predict_from_inputs(
        hours_studied,
        previous_scores,
        extracurricular,
        sleep_hours,
        sample_papers_practiced,
        ridge,
        rf,
        scaler,
        model_choice="Ridge Regression",
    )

    assert isinstance(pred, float)
    # Prediction should be within 0-100 for this dataset
    assert 0.0 <= pred <= 100.0
