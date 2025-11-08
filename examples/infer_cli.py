"""Small CLI example that loads saved artifacts and runs a prediction.

Usage:
    python examples/infer_cli.py --hours 8 --prev 50 --extra Yes --sleep 8 --papers 10

This is a minimal example to show programmatic inference outside Streamlit.
"""
from __future__ import annotations

import argparse
import os
import sys

# ensure repo root on path when running as script
repo_root = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from src.predict import load_models, predict_from_inputs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--hours", type=float, default=8)
    p.add_argument("--prev", type=float, default=50)
    p.add_argument("--extra", type=str, default="Yes")
    p.add_argument("--sleep", type=float, default=8)
    p.add_argument("--papers", type=int, default=10)
    p.add_argument("--model-dir", type=str, default="models")
    p.add_argument("--model", type=str, default="Ridge Regression", help="Ridge Regression or Random Forest")
    args = p.parse_args()

    ridge, rf, transformer = load_models(args.model_dir)

    pred = predict_from_inputs(
        args.hours,
        args.prev,
        args.extra,
        args.sleep,
        args.papers,
        ridge,
        rf,
        transformer,
        model_choice=args.model,
    )

    print(f"Prediction (model={args.model}): {pred:.2f}")


if __name__ == "__main__":
    main()
