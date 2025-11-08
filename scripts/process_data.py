"""Data processing and feature engineering script for v2.

This script is robust to different student dataset schemas. It will:
- Load the CSV (default: datasets/Student_Performance.csv)
- Clean missing values and coerce types
- Handle outliers via IQR capping
- Encode common categorical columns (Yes/No -> 1/0) and one-hot other categoricals
- Engineer features when source columns exist:
  - total_study_hours (studytime * absences) if both exist
  - parental_involvement (mean of Medu, Fedu, famrel if present)
  - G_avg (mean of G1 and G2) or use Previous Scores as proxy
  - attendance_ratio (1 - absences / (max_absences + 1))
- Save processed CSV to data/processed/processed_student_data.csv

Usage:
    python scripts/process_data.py --input datasets/Student_Performance.csv --out-dir data/processed
"""
from __future__ import annotations

import argparse
import os
from typing import List

import numpy as np
import pandas as pd


def cap_iqr(series: pd.Series) -> pd.Series:
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - 1.5 * iqr
    upper = q3 + 1.5 * iqr
    return series.clip(lower, upper)


def encode_yes_no(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    for c in cols:
        if c in df.columns:
            df[c] = df[c].map({"Yes": 1, "No": 0}).fillna(df[c])
    return df


def process(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # Normalize column names
    df.columns = [c.strip().replace(" ", "_").replace("__", "_").lower() for c in df.columns]

    # Coerce numeric columns where possible
    for col in df.columns:
        if df[col].dtype == object:
            # try to coerce
            df[col] = pd.to_numeric(df[col], errors="ignore")

    # Common yes/no columns
    df = encode_yes_no(df, ["extracurricular_activities", "extracurricular", "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic"])

    # Missing values: numeric -> median, categorical -> mode
    for col in df.columns:
        if df[col].dtype.kind in "biufc":
            df[col] = df[col].fillna(df[col].median())
        else:
            df[col] = df[col].fillna(df[col].mode().iloc[0] if not df[col].mode().empty else "")

    # Outlier handling: cap numeric columns using IQR
    for col in df.select_dtypes(include=[np.number]).columns:
        df[col] = cap_iqr(df[col])

    cols = df.columns.tolist()

    # Feature: total_study_hours = studytime * absences (if present)
    if "studytime" in cols and "absences" in cols:
        df["total_study_hours"] = df["studytime"] * df["absences"]

    # If Hours Studied exists (different schema), map to studytime_total
    if "hours_studied" in cols and "sample_question_papers_practiced" in cols:
        # create a proxy total study hours by combining hours_studied and sample papers
        df["total_study_hours_proxy"] = df["hours_studied"] * (1 + df["sample_question_papers_practiced"]/10.0)

    # Parental involvement: combine Medu, Fedu, famrel when available
    pinv_cols = [c for c in ["medu", "fedu", "famrel"] if c in cols]
    if pinv_cols:
        df["parental_involvement"] = df[pinv_cols].mean(axis=1)

    # G_avg: average of G1 and G2 if present, else use previous_scores if present
    if "g1" in cols and "g2" in cols:
        df["g_avg"] = df[["g1", "g2"]].mean(axis=1)
    elif "previous_scores" in cols:
        df["g_avg"] = df["previous_scores"]

    # Attendance ratio
    if "absences" in cols:
        max_abs = df["absences"].max()
        df["attendance_ratio"] = 1 - (df["absences"] / (max_abs + 1))

    # One-hot encode small-cardinality categoricals (<= 10 unique values)
    cat_cols = [c for c in df.select_dtypes(include=[object, "category"]).columns]
    for c in cat_cols:
        if df[c].nunique() <= 10:
            dummies = pd.get_dummies(df[c], prefix=c, drop_first=True)
            df = pd.concat([df.drop(columns=[c]), dummies], axis=1)

    return df


def main(args):
    df = pd.read_csv(args.input)
    processed = process(df)

    out_dir = args.out_dir
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "processed_student_data.csv")
    processed.to_csv(out_path, index=False)
    print(f"Saved processed data to: {out_path}")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--input", default="datasets/Student_Performance.csv")
    p.add_argument("--out-dir", default="data/processed")
    args = p.parse_args()
    main(args)
