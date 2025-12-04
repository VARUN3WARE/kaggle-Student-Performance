# Kaggle Student Performance Prediction — v2

This project is based on the **Kaggle Student Performance** dataset, which is used to predict students' final grades based on various features like study time, past grades, and school-related factors. The project includes several machine learning models to predict student performance and compares them after hyperparameter tuning. The app is deployed using Streamlit for interactive visualization.
![....](image.png)
[Live Demo](https://kaggle-student-performance-varunrao.streamlit.app/)

[Link to kaggle Notebook](https://www.kaggle.com/code/varunraosfanlkan/notebook3ac0f15a42)

A compact, reproducible ML project that predicts student final scores and explains predictions with SHAP.

This repository includes:

- A modular `src/` package (prediction, training, explainability helpers)
- Training utilities and scripts (`src/model_training.py`, `scripts/train.py`)
- Explainability scripts (`scripts/generate_shap_reports.py`) and precomputed SHAP visuals
- An interactive Streamlit demo: `app/app_v2.py`

## Key artifacts / visuals

- SHAP summary plots: `reports/feature_importance/shap_summary_*.png`
- Saved models & metrics: `models/` (contains `best_*_model.pkl`, `transformer.pkl`, `training_summary.csv`, `model_comparison.html`)
- Processed dataset: `data/processed/processed_student_data.csv`
- Notebooks: `notebooks/Student_Performance.ipynb`, `notebooks/explainability.ipynb`

## Quickstart (local)

1. Create and activate a virtual environment, then install dependencies:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

2. Run the Streamlit demo:

```bash
streamlit run app/app_v2.py
```

Open http://localhost:8501 in your browser.

3. Regenerate SHAP summary images (fast sample):

```bash
python scripts/generate_shap_reports.py --model-dir models --out-dir reports/feature_importance --sample-n 200
```

## Troubleshooting & notes

- Module import: if `app/app_v2.py` fails with `ModuleNotFoundError: No module named 'src'`, you can install the package in editable mode (recommended for development):

```bash
# create a minimal pyproject.toml or setup.cfg, then:
pip install -e .
```

Or keep the local `sys.path` workaround (already present in `app/app_v2.py`) for quick local runs.

- Pickle/sklearn warnings: if you see `InconsistentVersionWarning` when loading model pickles, re-train and re-save models in this environment or pin `scikit-learn` to match the version used to save artifacts.

## What changed in v2 (short)

- Modularized code under `src/` and added training utilities
- Added SHAP explainability scripts and precomputed images in `reports/feature_importance/`
- Streamlit v2 app with cached explainers and per-sample waterfall plots
- pytest: quick training smoke test

## Local MLflow-only (no external services)

If you prefer to keep everything local and avoid external services, there's a lightweight compose stack that runs MLflow using a local SQLite backend and a filesystem artifact store — no MinIO, no Postgres, no paid services.

Files to use:

- `docker-compose.mlflow-local.yml` — starts a single MLflow service backed by sqlite and local disk artifacts.

Quick start (local-only):

```bash
# build and start the local MLflow server
docker compose -f docker-compose.mlflow-local.yml up -d --build

# open MLflow UI
open http://localhost:5000
```

Point your training runs to the server:

```bash
export MLFLOW_TRACKING_URI=http://localhost:5000
python src/model_training.py --data data/processed/processed_student_data.csv --out-dir models --mlflow --tracking-uri $MLFLOW_TRACKING_URI
```

Notes:

- This setup stores artifacts and the MLflow sqlite DB under a Docker volume named `mlflow_data`. You can back this up or mount a host directory if you want persistent files outside Docker.
- This stack is for local development and experimentation only. For production consider using managed storage and a proper database.

### Registering best model in MLflow Model Registry

If you want training to automatically register the best model in the MLflow Model Registry (local server), use the `--register` flag together with `--mlflow` when running the training script:

```bash
# locally (requires MLFLOW_TRACKING_URI pointing at your local server):
python src/model_training.py --data data/processed/processed_student_data.csv --out-dir models --mlflow --register

# or using the trainer image (Linux) with host networking:
docker run --rm -it --network host \
	-v "$PWD":/workspace:cached \
	-v "$PWD/mlflow":/mlflow \
	-e MLFLOW_TRACKING_URI=http://localhost:5000 \
	-w /workspace \
	kaggle-student-performance-trainer \
	python src/model_training.py --data data/processed/processed_student_data.csv --out-dir models --mlflow --register
```

The script will register the top-performing model (by R^2) under the registry name `student_performance_v2`. You can view registered models in the MLflow UI under the "Models" tab.

### Convenience: Makefile targets

I've added a small `Makefile` with helpful targets for local development:

```bash
# build and start mlflow + trainer image
make mlflow-up

# build trainer image only (injects your UID/GID so files aren't root-owned)
make build-trainer

# run training using the trainer image on the host network (Linux)
make train

# stop mlflow stack
make mlflow-down
```

These targets just wrap the same docker-compose / docker run commands demonstrated earlier and are intended for developer convenience.
