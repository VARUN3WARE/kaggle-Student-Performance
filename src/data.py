"""Data loading and cleaning utilities for Student Performance project."""
from __future__ import annotations

import os
from typing import Tuple

import pandas as pd


def load_raw_data(path: str) -> pd.DataFrame:
    """Load CSV data from `path` into a pandas DataFrame."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"Data file not found: {path}")
    return pd.read_csv(path)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Perform light cleaning and canonicalize column names.

    - Normalize column names to snake_case
    - Map 'Yes'/'No' extracurricular to 1/0 under `extracurricular`
    - Convert numeric columns to numeric types
    - Drop rows with missing values
    """
    df = df.copy()
    # normalize column names
    df.columns = [c.strip().replace(" ", "_").replace("__", "_").lower() for c in df.columns]

    # map extracurricular activities
    if "extracurricular_activities" in df.columns:
        df["extracurricular"] = df["extracurricular_activities"].map({"Yes": 1, "No": 0})
        df.drop(columns=["extracurricular_activities"], inplace=True)

    # coerce numeric columns to numeric types
    numeric_cols = [
        "hours_studied",
        "previous_scores",
        "sleep_hours",
        "sample_question_papers_practiced",
        "performance_index",
    ]
    for c in numeric_cols:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")

    df = df.dropna().reset_index(drop=True)
    return df


def train_test_split(df: pd.DataFrame, target: str = "performance_index", test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """Return X_train, X_test, y_train, y_test using sklearn's train_test_split."""
    from sklearn.model_selection import train_test_split

    X = df.drop(columns=[target])
    y = df[target]
    return train_test_split(X, y, test_size=test_size, random_state=random_state)
