import os
import sys
import pathlib
import numpy as np

# Ensure repo root on path
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src import predict


def test_predict_from_inputs_with_saved_models():
    model_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    model_dir = os.path.normpath(model_dir)
    ridge, rf, transformer = predict.load_models(model_dir=model_dir)

    # typical inputs
    hours_studied = 6
    previous_scores = 70
    extracurricular = "No"
    sleep_hours = 7
    sample_papers_practiced = 5

    pred = predict.predict_from_inputs(
        hours_studied,
        previous_scores,
        extracurricular,
        sleep_hours,
        sample_papers_practiced,
        ridge,
        rf,
        transformer,
        model_choice="Ridge Regression",
    )

    assert isinstance(pred, float)
    assert not np.isnan(pred)
 