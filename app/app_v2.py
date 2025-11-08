"""Streamlit App v2: interactive prediction dashboard and explainability.

Features:
- Sidebar model selection and input sliders
- Real-time prediction
- SHAP explainability: cached explainer, summary plot, per-sample waterfall
"""
from __future__ import annotations

import os
import sys
import pathlib
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Ensure the repository root is on sys.path so `from src import ...` works when
# running via `streamlit run` (Streamlit may change how the script is executed).
repo_root = pathlib.Path(__file__).resolve().parents[1]
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src import predict

import shap


def load_models(model_dir: str = "models"):
    models = {}
    try:
        ridge, rf, transformer = predict.load_models(model_dir=model_dir)
        models["Ridge"] = ridge
        models["Random Forest"] = rf
        models["transformer"] = transformer
    except Exception:
        # graceful fallback
        models = {}
    return models


@st.cache_resource
def load_models_cached(model_dir: str = "models"):
    return load_models(model_dir)


def _get_expected_columns(models: dict) -> list | None:
    transformer = models.get("transformer")
    expected = None
    try:
        m = models.get("Ridge")
        if hasattr(m, "named_steps") and "preprocessor" in m.named_steps:
            expected = list(m.named_steps["preprocessor"].feature_names_in_)
    except Exception:
        expected = None
    if expected is None and transformer is not None and hasattr(transformer, "feature_names_in_"):
        expected = list(transformer.feature_names_in_)
    return expected


@st.cache_data
def load_background_data(path: str = "data/processed/processed_student_data.csv", expected_cols: list | None = None, n: int = 200):
    df = pd.read_csv(path)
    if expected_cols is not None:
        df = df.reindex(columns=expected_cols)
    if df.shape[0] > n:
        return df.sample(n, random_state=42)
    return df


@st.cache_resource
def build_explainer(_model, background_df: pd.DataFrame):
    """Build and return a SHAP explainer for `_model` using `background_df`.

    Note: the leading underscore in `_model` prevents Streamlit from trying to
    hash the (unhashable) sklearn Pipeline object when creating the cache key.
    """
    try:
        expl = shap.Explainer(_model, background_df)
        return expl
    except Exception:
        # Fallback: KernelExplainer with a prediction wrapper
        def predict_fn(x):
            import numpy as _np
            arr = _np.asarray(x)
            if arr.ndim == 1:
                arr = arr.reshape(1, -1)
            try:
                df = pd.DataFrame(arr, columns=background_df.columns)
                return _model.predict(df)
            except Exception:
                return _model.predict(arr)

        return shap.KernelExplainer(predict_fn, background_df.iloc[:min(50, len(background_df))])


def sidebar_inputs():
    st.sidebar.header("Model & input settings")
    model_choice = st.sidebar.selectbox("Model", ["Ridge", "Random Forest"]) 
    hours_studied = st.sidebar.slider("Hours studied per week", 0.0, 40.0, 8.0)
    previous_scores = st.sidebar.slider("Average previous score", 0, 100, 70)
    extracurricular = st.sidebar.selectbox("Extracurricular activities", ["Yes", "No"]) 
    sleep_hours = st.sidebar.slider("Average sleep hours", 0.0, 12.0, 7.0)
    sample_papers = st.sidebar.number_input("Practice papers taken", min_value=0, max_value=100, value=5)
    return model_choice, hours_studied, previous_scores, extracurricular, sleep_hours, sample_papers


def main():
    st.set_page_config(page_title="Student Performance — V2", layout="wide")
    st.title("Student Performance — v2")
    st.markdown("A compact interactive dashboard with model selection, real-time prediction, and explainability.")

    models = load_models_cached()

    model_choice, hours_studied, previous_scores, extracurricular, sleep_hours, sample_papers = sidebar_inputs()

    col1, col2 = st.columns([2, 3])

    with col1:
        st.subheader("Prediction")
        if "transformer" in models and ("Ridge" in models or "Random Forest" in models):
            # Use predict helper which handles pipelines/transformers
            pred = predict.predict_from_inputs(
                hours_studied,
                previous_scores,
                extracurricular,
                sleep_hours,
                sample_papers,
                models.get("Ridge"),
                models.get("Random Forest"),
                models.get("transformer"),
                model_choice=("Ridge" if model_choice == "Ridge" else "Random Forest"),
            )
            st.metric("Predicted performance", f"{pred:.2f}")
        else:
            st.info("No models loaded — train models with the training script to enable predictions.")

    with col2:
        st.subheader("Explainability")
        st.write("SHAP summary (sample) and per-sample explanation. Explainers are cached to speed up interactions.")

        expected_cols = _get_expected_columns(models)
        data_sample = None
        if expected_cols is not None:
            try:
                data_sample = load_background_data(expected_cols=expected_cols, n=200)
            except Exception:
                data_sample = None

        if data_sample is not None and ("Ridge" in models or "Random Forest" in models):
            selected_key = "Ridge" if model_choice == "Ridge" else "Random Forest"
            model_obj = models.get(selected_key)
            if model_obj is not None:
                with st.spinner("Building SHAP explainer (cached)..."):
                    explainer = build_explainer(model_obj, data_sample)

                # Summary plot
                try:
                    shap_vals = explainer(data_sample)
                    fig = plt.figure(figsize=(8, 4))
                    shap.summary_plot(shap_vals, data_sample, show=False)
                    st.pyplot(fig)
                except Exception as e:
                    st.write("Could not compute SHAP summary:", e)

                st.markdown("---")
                st.markdown("**Explain this input**")
                if st.button("Explain current input"):
                    # Build single-row DataFrame aligned to expected columns
                    row = {c: np.nan for c in (expected_cols or [])}
                    mapping = {
                        "hours_studied": hours_studied,
                        "previous_scores": previous_scores,
                        "extracurricular": 1 if str(extracurricular).lower().startswith("y") else 0,
                        "sleep_hours": sleep_hours,
                        "sample_question_papers_practiced": sample_papers,
                    }
                    for k, v in mapping.items():
                        if expected_cols and k in expected_cols:
                            row[k] = v

                    x_df = pd.DataFrame([row], columns=expected_cols)
                    try:
                        shap_single = explainer(x_df)
                        fig2 = plt.figure(figsize=(6, 4))
                        shap.plots.waterfall(shap_single[0], show=False)
                        st.pyplot(fig2)
                    except Exception as e:
                        st.write("Failed to compute per-sample SHAP:", e)
            else:
                st.info("Selected model not available for explainability.")
        else:
            # fallback to precomputed image
            img_path = os.path.join("reports", "feature_importance", f"shap_summary_{'ridge' if model_choice=='Ridge' else 'random_forest'}.png")
            if os.path.exists(img_path):
                st.image(img_path, caption="SHAP summary (precomputed)")
            else:
                st.info("No background data or models available to compute SHAP. Run the explain script to generate reports.")

    st.sidebar.markdown("---")
    st.sidebar.subheader("About / Version")
    st.sidebar.write("COOL but still not better than v1, OG matters.")


if __name__ == "__main__":
    main()
