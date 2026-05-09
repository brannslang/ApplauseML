import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import joblib
import pandas as pd

from config import ARTIFACTS_DIR

_model = None
_risk_tables = None
_feature_info = None


def _load():
    global _model, _risk_tables, _feature_info
    if _model is not None:
        return
    classifier_path = os.path.join(ARTIFACTS_DIR, "classifier.joblib")
    if not os.path.exists(classifier_path):
        raise FileNotFoundError(
            "Model artifacts not found. Run 'python model/train.py' first."
        )
    _model = joblib.load(classifier_path)
    _risk_tables = joblib.load(os.path.join(ARTIFACTS_DIR, "risk_tables.joblib"))
    _feature_info = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_info.joblib"))


def artifacts_exist() -> bool:
    return os.path.exists(os.path.join(ARTIFACTS_DIR, "classifier.joblib"))


def get_risk_tables() -> dict:
    _load()
    return _risk_tables


def get_feature_info() -> dict:
    _load()
    return _feature_info


def predict_release_risk(inputs: dict) -> dict:
    """
    inputs: dict mapping feature names to values (matching CATEGORICAL/NUMERIC_FEATURES)
    Returns dict with risk_score, baseline, risk_delta, risk_label, component_breakdown
    """
    _load()
    features = _feature_info["features"]
    row = {f: inputs.get(f) for f in features}
    X = pd.DataFrame([row])

    prob = float(_model.predict_proba(X)[0][1])
    baseline = _risk_tables["baseline"]
    delta = prob - baseline

    if prob >= 0.60:
        label, color = "High Risk", "#d62728"
    elif prob >= 0.40:
        label, color = "Elevated Risk", "#ff7f0e"
    else:
        label, color = "Normal Risk", "#2ca02c"

    component_tbl = _risk_tables.get("App Component", pd.DataFrame())

    selected_component = inputs.get("App Component")
    selected_parent = inputs.get("Parent App Component")
    selected_platform = inputs.get("Platform Product Name")

    return {
        "risk_score": prob,
        "baseline": baseline,
        "risk_delta": delta,
        "risk_label": label,
        "risk_color": color,
        "component_breakdown": component_tbl,
        "selected_component": selected_component,
        "selected_parent": selected_parent,
        "selected_platform": selected_platform,
    }
