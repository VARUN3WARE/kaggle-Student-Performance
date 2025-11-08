import sys
import pathlib
import numpy as np
import pandas as pd

# Ensure repo root on path
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_training import get_preprocessor


def test_get_preprocessor_and_transform():
    # tiny synthetic dataset with numeric and categorical columns
    df = pd.DataFrame({
        "hours_studied": [1, 2, 3, 4],
        "previous_scores": [50, 60, 70, 80],
        "extracurricular": ["yes", "no", "yes", "no"],
        "sleep_hours": [6, 7, 8, 5],
    })

    pre = get_preprocessor(df)

    # fit and transform should run without error and produce a numeric array
    X_trans = pre.fit_transform(df)
    assert X_trans.shape[0] == df.shape[0]
    assert np.isfinite(X_trans).all()
 