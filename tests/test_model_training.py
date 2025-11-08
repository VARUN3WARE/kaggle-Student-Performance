import os
import sys
import pathlib
import pandas as pd
import numpy as np

# Ensure repository root is on sys.path when tests are run directly
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.model_training import train_models


def test_train_models_quick(tmp_path):
    # Create a tiny synthetic dataset
    n = 80
    rng = np.random.RandomState(0)
    df = pd.DataFrame({
        "hours_studied": rng.uniform(0, 10, size=n),
        "previous_scores": rng.uniform(30, 90, size=n),
        "extracurricular": rng.choice(["yes", "no"], size=n),
        "sleep_hours": rng.uniform(4, 9, size=n),
    })
    # simple synthetic target correlated with hours_studied and previous_scores
    y = 0.6 * df["hours_studied"] + 0.4 * (df["previous_scores"] / 10.0) + rng.normal(0, 1, size=n)

    out_dir = str(tmp_path / "models_test")
    df_res = train_models(df, pd.Series(y), out_dir=out_dir, cv=2, quick=True, log_mlflow=False)

    # Basic sanity checks
    assert not df_res.empty
    # training_summary.csv should exist
    assert os.path.exists(os.path.join(out_dir, "training_summary.csv"))