"""Model training utilities: pipelines, CV, hyperparameter tuning and tracking.

This module provides a reusable `train_models` function that fits multiple
models using scikit-learn Pipelines, performs cross-validation and hyperparameter
tuning, logs experiments via MLflow (optional), saves trained models and a
results summary in `models/`.
"""
from __future__ import annotations

import json
import os
from typing import Any, Dict, List, Tuple

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

try:
    import mlflow
    import mlflow.sklearn

    MLFLOW_AVAILABLE = True
except Exception:
    MLFLOW_AVAILABLE = False

try:
    import wandb

    WANDB_AVAILABLE = True
except Exception:
    WANDB_AVAILABLE = False


def get_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    # avoid including the target if present
    # For categoricals, pick object or category dtype
    categorical_cols = X.select_dtypes(include=["object", "category"]).columns.tolist()

    numeric_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ])

    categorical_transformer = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore")),
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    return preprocessor


def evaluate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    rmse = float(np.sqrt(mse))
    r2 = r2_score(y_true, y_pred)
    return {"mae": float(mae), "mse": float(mse), "rmse": rmse, "r2": float(r2)}


def build_pipeline(preprocessor: ColumnTransformer, estimator) -> Pipeline:
    return Pipeline(steps=[("preprocessor", preprocessor), ("model", estimator)])


def _save_model(model: Any, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    joblib.dump(model, path)


def train_models(
    X: pd.DataFrame,
    y: pd.Series,
    out_dir: str = "models",
    cv: int = 5,
    quick: bool = False,
    log_mlflow: bool = False,
    log_wandb: bool = False,
) -> pd.DataFrame:
    """Train a set of candidate models, tune hyperparameters and save results.

    Returns a DataFrame summarizing model metrics and paths to saved artifacts.
    """
    from sklearn.linear_model import Ridge
    from sklearn.ensemble import RandomForestRegressor

    results: List[Dict[str, Any]] = []

    preprocessor = get_preprocessor(X)

    # Candidate estimators and parameter grids
    candidates: List[Tuple[str, object, Dict[str, List[Any]]]] = []

    # Ridge
    candidates.append(
        (
            "ridge",
            Ridge(),
            {"model__alpha": [0.1, 1.0, 10.0]} if not quick else {"model__alpha": [1.0]},
        )
    )

    # Random Forest
    candidates.append(
        (
            "random_forest",
            RandomForestRegressor(random_state=42),
            {"model__n_estimators": [50, 100], "model__max_depth": [None, 5, 10]} if not quick else {"model__n_estimators": [50]},
        )
    )

    # Try to include XGBoost if installed and if we're not in quick/test mode.
    # XGBoost's sklearn wrapper may not be compatible with very new sklearn
    # versions in some environments, and it slows CI; include only for full runs.
    if not quick:
        try:
            from xgboost import XGBRegressor

            candidates.append(
                (
                    "xgboost",
                    XGBRegressor(objective="reg:squarederror", random_state=42, verbosity=0),
                    {"model__n_estimators": [50, 100], "model__max_depth": [3, 6]},
                )
            )
        except Exception:
            # If xgboost is present but incompatible, skip it to keep training robust.
            pass

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    if log_mlflow and MLFLOW_AVAILABLE:
        mlflow.set_experiment("student_performance_v2")

    if log_wandb and WANDB_AVAILABLE:
        # Initialize a W&B run for the whole training session. Individual
        # GridSearchCV runs will log params/metrics as separate runs within
        # the same project if desired — here we create a parent run.
        wandb.init(project="student_performance_v2", reinit=True)

    for name, estimator, param_grid in candidates:
        pipe = build_pipeline(preprocessor, estimator)
        gs = GridSearchCV(pipe, param_grid=param_grid, cv=cv, n_jobs=-1, scoring="neg_mean_absolute_error")
        gs.fit(X_train, y_train)

        best = gs.best_estimator_
        preds = best.predict(X_test)
        metrics = evaluate(y_test, preds)

        model_path = os.path.join(out_dir, f"best_{name}_model.pkl")
        _save_model(best, model_path)

        # Save a small JSON of metrics
        os.makedirs(out_dir, exist_ok=True)
        with open(os.path.join(out_dir, f"metrics_{name}.json"), "w") as fh:
            json.dump(metrics, fh, indent=2)


        if log_mlflow and MLFLOW_AVAILABLE:
            with mlflow.start_run(run_name=name):
                mlflow.log_params(gs.best_params_)
                mlflow.log_metrics(metrics)
                mlflow.sklearn.log_model(best, artifact_path=name)

        if log_wandb and WANDB_AVAILABLE:
            # Log params and metrics to W&B and upload the model file as an
            # artifact so it can be downloaded later.
            wandb.log({**{"model": name}, **{f"metrics/{k}": v for k, v in metrics.items()}})
            wandb.config.update(gs.best_params_)
            # Save local artifact and upload
            artifact = wandb.Artifact(name=f"best-{name}-model", type="model")
            artifact.add_file(model_path)
            wandb.log_artifact(artifact)

        results.append({"model": name, "metrics": metrics, "path": model_path, "best_params": gs.best_params_})

    # Summarize results into DataFrame
    rows = []
    for r in results:
        row = {"model": r["model"]}
        row.update(r["metrics"])
        row.update({"path": r["path"]})
        row.update({f"param_{k}": v for k, v in r["best_params"].items()})
        rows.append(row)

    df_res = pd.DataFrame(rows).sort_values(by="r2", ascending=False)
    # Save summary
    os.makedirs(out_dir, exist_ok=True)
    df_res.to_csv(os.path.join(out_dir, "training_summary.csv"), index=False)

    # Plot comparison (bar plot of MAE, RMSE, R2)
    try:
        import plotly.express as px

        # Expand metrics for plotting
        metrics_df = df_res[["model", "mae", "rmse", "r2"]]
        fig = px.bar(metrics_df.melt(id_vars=["model"], value_vars=["mae", "rmse", "r2"]), x="model", y="value", color="variable", barmode="group", title="Model comparison")
        fig.write_html(os.path.join(out_dir, "model_comparison.html"))
    except Exception:
        pass

    return df_res


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/processed/processed_student_data.csv")
    p.add_argument("--out-dir", default="models")
    p.add_argument("--cv", type=int, default=3)
    p.add_argument("--quick", action="store_true", help="Run quick, lightweight tuning")
    p.add_argument("--mlflow", action="store_true", help="Log runs to MLflow if available")
    args = p.parse_args()

    df = pd.read_csv(args.data)
    # Assume target column is performance_index; otherwise try last column
    if "performance_index" in df.columns:
        y = df["performance_index"]
        X = df.drop(columns=["performance_index"])
    else:
        y = df[df.columns[-1]]
        X = df.drop(columns=[df.columns[-1]])

    summary = train_models(X, y, out_dir=args.out_dir, cv=args.cv, quick=args.quick, log_mlflow=args.mlflow)
    print(summary)
