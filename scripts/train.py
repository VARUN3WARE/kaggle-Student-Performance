"""Simple training script for the Student Performance project.

Usage:
    python scripts/train.py --data datasets/Student_Performance.csv --out-dir models

This script is intentionally minimal: it demonstrates end-to-end flow using
the helpers in `src/` and saves a scaler/transformer and best model to disk.
"""
from __future__ import annotations

import argparse
import os
import sys
import json

import joblib

# Ensure the repository root is on sys.path so `src` package imports work when
# running this file as a script (python scripts/train.py).
repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src import data as data_mod
from src import features as feat_mod
from src import models as models_mod


def main(args):
    df = data_mod.load_raw_data(args.data)
    df = data_mod.clean_data(df)

    X_train, X_test, y_train, y_test = data_mod.train_test_split(df, test_size=args.test_size)

    # Fit transformer on training data and persist
    transformer = feat_mod.fit_transformer(X_train, save_path=os.path.join(args.out_dir, "transformer.pkl"))

    X_train_t = feat_mod.transform_X(transformer, X_train)
    X_test_t = feat_mod.transform_X(transformer, X_test)

    # Train models
    print("Training Ridge...")
    ridge = models_mod.train_ridge(X_train_t, y_train)
    ridge_metrics = models_mod.evaluate_model(ridge, X_test_t, y_test)

    print("Training RandomForest...")
    rf = models_mod.train_random_forest(X_train_t, y_train, n_estimators=100)
    rf_metrics = models_mod.evaluate_model(rf, X_test_t, y_test)

    # Choose best by R2
    best_model = ridge if ridge_metrics["r2"] >= rf_metrics["r2"] else rf
    best_name = "ridge" if best_model is ridge else "random_forest"

    os.makedirs(args.out_dir, exist_ok=True)
    models_mod.save_model(best_model, os.path.join(args.out_dir, f"best_{best_name}_model.pkl"))
    joblib.dump(transformer, os.path.join(args.out_dir, "transformer.pkl"))

    metrics = {"ridge": ridge_metrics, "random_forest": rf_metrics, "best": best_name}
    with open(os.path.join(args.out_dir, "metrics.json"), "w") as fh:
        json.dump(metrics, fh, indent=2)

    print("Training complete. Models and transformer saved to:", args.out_dir)


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="datasets/Student_Performance.csv")
    p.add_argument("--out-dir", default="models")
    p.add_argument("--test-size", type=float, default=0.2)
    args = p.parse_args()
    main(args)
