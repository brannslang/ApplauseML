"""
Train the ApplauseML risk classifier and precompute risk tables.

Run once (and monthly) to refresh model artifacts:
    python model/train.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder

from config import (
    ARTIFACTS_DIR, CATEGORICAL_FEATURES, DATA_DIR,
    MIN_BUGS_FOR_TABLE, NUMERIC_FEATURES, TARGET,
)

os.makedirs(ARTIFACTS_DIR, exist_ok=True)


def load_data() -> pd.DataFrame:
    bugs = pd.read_excel(os.path.join(DATA_DIR, "bugdetails.xlsx"), engine="openpyxl")
    cycles = pd.read_excel(os.path.join(DATA_DIR, "testcycles.xlsx"), engine="openpyxl")

    cycle_cols = [
        "Test Cycle Id",
        "Test Cycle Testing Type",
        "Test Cycle Duration Activation to Lock/Close/Today",
        "Testing Approach",
    ]
    df = bugs.merge(cycles[cycle_cols], on="Test Cycle Id", how="left")

    for col in NUMERIC_FEATURES:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    df[TARGET] = df["Bug Severity"].isin(["Critical", "High"]).astype(int)
    return df


def compute_risk_tables(df: pd.DataFrame) -> dict:
    baseline = df[TARGET].mean()
    tables = {
        "baseline": baseline,
        "total_bugs": len(df),
    }

    risk_dims = [
        "App Component",
        "Parent App Component",
        "Platform Product Name",
        "Development Stage",
        "Testing Approach",
        "Bug Source Type",
        "Customer",
    ]
    for col in risk_dims:
        if col not in df.columns:
            continue
        tbl = (
            df.groupby(col)[TARGET]
            .agg(["mean", "count", "sum"])
            .rename(columns={"mean": "hc_rate", "count": "n_bugs", "sum": "n_hc"})
            .reset_index()
        )
        tbl["vs_baseline"] = tbl["hc_rate"] - baseline
        tbl = tbl[tbl["n_bugs"] >= MIN_BUGS_FOR_TABLE].sort_values(
            "hc_rate", ascending=False
        )
        tables[col] = tbl

    date_col = next(
        (c for c in df.columns if "create" in c.lower() and "date" in c.lower()), None
    )
    if date_col:
        df = df.copy()
        df["_month"] = pd.to_datetime(df[date_col], errors="coerce").dt.to_period("M")
        monthly = (
            df.groupby("_month")[TARGET]
            .agg(["mean", "count"])
            .rename(columns={"mean": "hc_rate", "count": "n_bugs"})
            .reset_index()
        )
        monthly["_month"] = monthly["_month"].astype(str)
        monthly = monthly.rename(columns={"_month": "month"})
        tables["monthly_trend"] = monthly

    return tables


def train_classifier(df: pd.DataFrame):
    available_cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    available_num = [c for c in NUMERIC_FEATURES if c in df.columns]
    features = available_cat + available_num

    X = df[features].copy()
    y = df[TARGET]

    cat_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="constant", fill_value="Unknown")),
        ("encoder", OrdinalEncoder(
            handle_unknown="use_encoded_value", unknown_value=-1
        )),
    ])
    num_pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
    ])

    preprocessor = ColumnTransformer([
        ("cat", cat_pipe, available_cat),
        ("num", num_pipe, available_num),
    ])

    clf = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=300,
            max_depth=12,
            min_samples_leaf=5,
            class_weight="balanced",
            random_state=42,
            n_jobs=-1,
        )),
    ])

    scores = cross_val_score(clf, X, y, cv=5, scoring="roc_auc", n_jobs=-1)
    print(f"  CV ROC-AUC: {scores.mean():.3f} ± {scores.std():.3f}")

    clf.fit(X, y)

    feature_info = {
        "features": features,
        "cat_cols": available_cat,
        "num_cols": available_num,
        "categories": {
            col: sorted(df[col].dropna().astype(str).unique().tolist())
            for col in available_cat
        },
    }
    return clf, feature_info


def main():
    print("Loading data...")
    df = load_data()
    print(f"  {len(df):,} bugs  |  baseline H/C rate: {df[TARGET].mean():.1%}")

    print("Computing risk tables...")
    risk_tables = compute_risk_tables(df)
    print(f"  {len(risk_tables) - 2} dimension tables built")

    print("Training classifier...")
    clf, feature_info = train_classifier(df)

    joblib.dump(clf, os.path.join(ARTIFACTS_DIR, "classifier.joblib"))
    joblib.dump(risk_tables, os.path.join(ARTIFACTS_DIR, "risk_tables.joblib"))
    joblib.dump(feature_info, os.path.join(ARTIFACTS_DIR, "feature_info.joblib"))

    print(f"\nArtifacts saved to {ARTIFACTS_DIR}/")
    print("  Done. Run 'streamlit run app/Home.py' to launch the dashboard.")


if __name__ == "__main__":
    main()
