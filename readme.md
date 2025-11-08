# Kaggle Student Performance Prediction — v2

This project is based on the **Kaggle Student Performance** dataset, which is used to predict students' final grades based on various features like study time, past grades, and school-related factors. The project includes several machine learning models to predict student performance and compares them after hyperparameter tuning. The app is deployed using Streamlit for interactive visualization.

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
