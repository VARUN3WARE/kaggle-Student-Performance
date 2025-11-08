"""Lightweight smoke training script used by CI.

This script trains on a small subset of the dataset (first N rows) to
verify the training pipeline works quickly in CI.
"""
from __future__ import annotations

import argparse
import os
import json

from src import data as data_mod
from src import features as feat_mod
from src import models as models_mod


def main(args):
    df = data_mod.load_raw_data(args.data)
    df = data_mod.clean_data(df)

    # small subset
    df = df.head(args.n_rows)

    X_train, X_test, y_train, y_test = data_mod.train_test_split(df, test_size=0.2)

    transformer = feat_mod.fit_transformer(X_train)
    X_train_t = feat_mod.transform_X(transformer, X_train)
    X_test_t = feat_mod.transform_X(transformer, X_test)

    ridge = models_mod.train_ridge(X_train_t, y_train)
    ridge_metrics = models_mod.evaluate_model(ridge, X_test_t, y_test)

    rf = models_mod.train_random_forest(X_train_t, y_train, n_estimators=10)
    rf_metrics = models_mod.evaluate_model(rf, X_test_t, y_test)

    os.makedirs(args.out_dir, exist_ok=True)
    models_mod.save_model(ridge, os.path.join(args.out_dir, "best_ridge_model.pkl"))
    models_mod.save_model(rf, os.path.join(args.out_dir, "best_rf_model.pkl"))
    # persist transformer for prediction steps
    import joblib

    joblib.dump(transformer, os.path.join(args.out_dir, "transformer.pkl"))

    metrics = {"ridge": ridge_metrics, "random_forest": rf_metrics}
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    print("Smoke training complete.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="datasets/Student_Performance.csv")
    p.add_argument("--out-dir", default="models-smoke")
    p.add_argument("--n-rows", type=int, default=200)
    args = p.parse_args()
    main(args)
