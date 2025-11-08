"""Generate SHAP explainability reports for trained models.

Usage:
    python scripts/generate_shap_reports.py --model-dir models --out-dir reports/feature_importance --model ridge

The script loads a saved model (Pipeline or raw estimator + transformer),
computes SHAP values (TreeExplainer for tree models, KernelExplainer for others),
and saves a SHAP summary plot to the out-dir.
"""
from __future__ import annotations

import argparse
import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

try:
    import shap
except Exception as e:
    raise ImportError("SHAP is required for explainability. Install with `pip install shap`") from e


def _load_model_and_transformer(model_dir: str, model_name: str):
    model_path = os.path.join(model_dir, f"best_{model_name}_model.pkl")
    transformer_path = os.path.join(model_dir, "transformer.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found: {model_path}")
    model = joblib.load(model_path)
    transformer = joblib.load(transformer_path) if os.path.exists(transformer_path) else None
    return model, transformer


def _prepare_X(df: pd.DataFrame, transformer):
    # If transformer is None, return DataFrame values
    if transformer is None:
        return df.values

    # If transformer is a ColumnTransformer fitted on DataFrame columns, it
    # will expect the same columns we pass. We assume `df` already has the
    # correct columns. Try to transform; otherwise fall back to values.
    try:
        Xt = transformer.transform(df)
        return Xt
    except Exception:
        return df.values


def generate_shap_summary(df: pd.DataFrame, model, transformer, out_path: str, sample_n: int = 200):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    # Sample data for SHAP calculations to keep runtime reasonable
    if df.shape[0] > sample_n:
        df_sample = df.sample(sample_n, random_state=42)
    else:
        df_sample = df

    # Determine expected feature names from model pipeline or transformer
    expected_cols = None
    # If model is a Pipeline and contains a preprocessor, try to get its feature names
    try:
        if hasattr(model, "named_steps") and "preprocessor" in model.named_steps:
            expected_cols = list(model.named_steps["preprocessor"].feature_names_in_)
    except Exception:
        expected_cols = None

    if expected_cols is None and transformer is not None and hasattr(transformer, "feature_names_in_"):
        expected_cols = list(transformer.feature_names_in_)

    # If we have expected columns, reindex the DataFrame to that set (fill missing with NaN)
    if expected_cols is not None:
        df_sample = df_sample.reindex(columns=expected_cols)

    # choose explainer type
    # If the model is a pipeline with a final tree estimator, use TreeExplainer on the final estimator
    final_estimator = None
    if hasattr(model, "named_steps") and "model" in model.named_steps:
        final_estimator = model.named_steps["model"]

    # Build a prediction wrapper that accepts numpy arrays and returns predictions
    def predict_fn(x_array):
        import numpy as _np
        # If input is 1D, make it 2D
        arr = _np.asarray(x_array)
        if arr.ndim == 1:
            arr = arr.reshape(1, -1)
        # If we know expected columns, convert to DataFrame so transformers/pipelines that
        # expect column names work correctly
        if expected_cols is not None:
            try:
                x_df = pd.DataFrame(arr, columns=expected_cols)
                return model.predict(x_df)
            except Exception:
                pass
        # Fallback: pass array directly
        return model.predict(arr)

    # Use TreeExplainer for tree-based final estimators
    if final_estimator is not None and final_estimator.__class__.__name__.lower().startswith(("xgb", "randomforest", "decisiontree", "lgbm")):
        # Determine X input for explainer: transformed array if needed
        X_for_shap = _prepare_X(df_sample, transformer)
        explainer = shap.TreeExplainer(final_estimator)
        try:
            shap_values = explainer.shap_values(X_for_shap)
        except Exception:
            # As a fallback, wrap final estimator with predict_fn
            explainer = shap.KernelExplainer(predict_fn, df_sample.iloc[:min(50, len(df_sample))])
            X_for_shap = _prepare_X(df_sample, transformer)
            shap_values = explainer.shap_values(X_for_shap)
    else:
        # For other models use KernelExplainer but ensure the prediction function
        background = df_sample.iloc[:min(50, len(df_sample))]
        explainer = shap.KernelExplainer(predict_fn, background)
        X_for_shap = _prepare_X(df_sample, transformer)
        shap_values = explainer.shap_values(X_for_shap)

    # Plot summary
    plt.figure(figsize=(10, 6))
    try:
        shap.summary_plot(shap_values, X_for_shap, show=False)
        plt.tight_layout()
        plt.savefig(out_path, dpi=150)
        plt.close()
    except Exception:
        # Try saving via shap's matplotlib fallback
        shap.summary_plot(shap_values, X_for_shap)
        plt.savefig(out_path, dpi=150)
        plt.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model-dir", default="models")
    p.add_argument("--out-dir", default="reports/feature_importance")
    p.add_argument("--model", default="ridge", help="Name prefix used in models (ridge, random_forest, xgboost)")
    p.add_argument("--data", default="data/processed/processed_student_data.csv")
    p.add_argument("--sample-n", type=int, default=200, help="Number of rows to sample for SHAP computations (smaller is faster)")
    args = p.parse_args()

    df = pd.read_csv(args.data)
    model, transformer = _load_model_and_transformer(args.model_dir, args.model)

    out_path = os.path.join(args.out_dir, f"shap_summary_{args.model}.png")
    print(f"Generating SHAP summary for model {args.model} -> {out_path}")
    generate_shap_summary(df, model, transformer, out_path, sample_n=args.sample_n)
    print("Done")


if __name__ == "__main__":
    main()
