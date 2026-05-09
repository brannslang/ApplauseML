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


def get_customer_list() -> list[str]:
    _load()
    return sorted(_risk_tables.get("customer_tables", {}).keys())


def get_customer_risk_tables(customer: str | None) -> dict:
    """Return customer-filtered risk tables, or all-data tables if customer is None."""
    _load()
    if customer:
        cust_tables = _risk_tables.get("customer_tables", {})
        if customer in cust_tables:
            return cust_tables[customer]
    return _risk_tables


def predict_release_risk(inputs: dict, customer: str | None = None) -> dict:
    """
    inputs: dict mapping feature names to values (matching CATEGORICAL/NUMERIC_FEATURES)
    customer: if provided, the component breakdown uses that customer's historical data only
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

    view_tables = get_customer_risk_tables(customer)
    component_tbl = view_tables.get("App Component", pd.DataFrame())
    view_baseline = view_tables["baseline"]

    return {
        "risk_score": prob,
        "baseline": baseline,
        "view_baseline": view_baseline,
        "risk_delta": delta,
        "risk_label": label,
        "risk_color": color,
        "component_breakdown": component_tbl,
        "selected_component": inputs.get("App Component"),
        "selected_platform": inputs.get("Platform Product Name"),
    }
