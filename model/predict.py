import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import joblib
import numpy as np
import pandas as pd

from config import ARTIFACTS_DIR, N_SVD_COMPONENTS, N_EMB_COMPONENTS, N_NMF_FACTORS

_model = None
_risk_tables = None
_feature_info = None
_text_pipeline = None
_nmf_model = None
_graph_artifacts = None
_sentence_model = None


def _safe_load(path):
    return joblib.load(path) if os.path.exists(path) else None


def _load():
    global _model, _risk_tables, _feature_info
    global _text_pipeline, _nmf_model, _graph_artifacts, _sentence_model

    if _model is not None:
        return

    classifier_path = os.path.join(ARTIFACTS_DIR, "classifier.joblib")
    if not os.path.exists(classifier_path):
        raise FileNotFoundError(
            "Model artifacts not found. Run 'python model/train.py' first."
        )

    _model        = joblib.load(classifier_path)
    _risk_tables  = joblib.load(os.path.join(ARTIFACTS_DIR, "risk_tables.joblib"))
    _feature_info = joblib.load(os.path.join(ARTIFACTS_DIR, "feature_info.joblib"))

    _text_pipeline  = _safe_load(os.path.join(ARTIFACTS_DIR, "text_pipeline.joblib"))
    _nmf_model      = _safe_load(os.path.join(ARTIFACTS_DIR, "nmf_model.joblib"))
    _graph_artifacts = _safe_load(os.path.join(ARTIFACTS_DIR, "graph_artifacts.joblib"))

    if _text_pipeline is not None:
        from sentence_transformers import SentenceTransformer
        _sentence_model = SentenceTransformer(_text_pipeline["sentence_model_name"])


def artifacts_exist() -> bool:
    return os.path.exists(os.path.join(ARTIFACTS_DIR, "classifier.joblib"))


def get_risk_tables() -> dict:
    _load()
    return _risk_tables


def get_feature_info() -> dict:
    _load()
    return _feature_info


def _apply_text_features(row: dict, bug_subject: str, bug_result: str) -> None:
    """Compute and insert text feature values into row in-place."""
    import re
    text = (str(bug_subject or "") + " " + str(bug_result or "")).strip()

    for col, pattern in _text_pipeline["keyword_groups"].items():
        row[col] = int(bool(re.search(pattern, text, re.IGNORECASE)))

    tfidf_vec = _text_pipeline["tfidf"].transform([text])
    svd_vec   = _text_pipeline["svd"].transform(tfidf_vec)[0]
    n_svd     = _text_pipeline["n_svd"]
    for i in range(N_SVD_COMPONENTS):
        row[f"text_svd_{i}"] = float(svd_vec[i]) if i < n_svd else 0.0

    if text and _sentence_model is not None:
        embedding = _sentence_model.encode([text])
    else:
        embedding = np.zeros((1, 384))
    pca_vec = _text_pipeline["pca"].transform(embedding)[0]
    n_emb   = _text_pipeline["n_emb"]
    for i in range(N_EMB_COMPONENTS):
        row[f"text_emb_{i}"] = float(pca_vec[i]) if i < n_emb else 0.0


def _apply_nmf_features(row: dict, inputs: dict) -> None:
    """Compute and insert NMF factor values into row in-place."""
    feature_names = _nmf_model["feature_names"]
    entity_vec = pd.DataFrame(0.0, index=[0], columns=feature_names)
    for col in _nmf_model["entity_cols"]:
        val = inputs.get(col)
        if val:
            col_name = f"{col}_{val}"
            if col_name in entity_vec.columns:
                entity_vec[col_name] = 1.0
    nmf_vec = _nmf_model["nmf"].transform(entity_vec)[0]
    for i in range(N_NMF_FACTORS):
        row[f"nmf_factor_{i}"] = float(nmf_vec[i]) if i < len(nmf_vec) else 0.0


def _apply_graph_features(row: dict, inputs: dict) -> None:
    """Look up and insert graph metric values into row in-place."""
    node_metrics = _graph_artifacts["node_metrics"]

    def lookup(col, val, metric):
        if not val:
            return 0.0
        return node_metrics.get(f"{col}:{val}", {}).get(metric, 0.0)

    comp     = inputs.get("App Component")
    platform = inputs.get("Platform Product Name")
    customer = inputs.get("Customer")

    row["graph_comp_pagerank"]     = lookup("App Component", comp, "pagerank")
    row["graph_comp_degree"]       = lookup("App Component", comp, "degree_centrality")
    row["graph_comp_clustering"]   = lookup("App Component", comp, "clustering")
    row["graph_platform_pagerank"] = lookup("Platform Product Name", platform, "pagerank")
    row["graph_customer_pagerank"] = lookup("Customer", customer, "pagerank")


def predict_release_risk(inputs: dict) -> dict:
    """
    inputs: dict mapping feature names to values.
            Optional keys: 'bug_subject', 'bug_result' for text features.
    Returns dict with risk_score, baseline, risk_delta, risk_label, component_breakdown.
    """
    _load()

    is_multimodal = _feature_info.get("has_multimodal", False)
    features      = _feature_info["features"]

    # Base categorical + original numeric features from user inputs
    row = {f: inputs.get(f) for f in features}

    # Engineered features (only when model was trained with them)
    if is_multimodal:
        if _text_pipeline is not None:
            _apply_text_features(
                row,
                inputs.get("bug_subject", ""),
                inputs.get("bug_result", ""),
            )
        if _nmf_model is not None:
            _apply_nmf_features(row, inputs)
        if _graph_artifacts is not None:
            _apply_graph_features(row, inputs)

    X = pd.DataFrame([{f: row.get(f) for f in features}])

    prob     = float(_model.predict_proba(X)[0][1])
    baseline = _risk_tables["baseline"]
    delta    = prob - baseline

    if prob >= 0.60:
        label, color = "High Risk", "#d62728"
    elif prob >= 0.40:
        label, color = "Elevated Risk", "#ff7f0e"
    else:
        label, color = "Normal Risk", "#2ca02c"

    component_tbl      = _risk_tables.get("App Component", pd.DataFrame())
    selected_component = inputs.get("App Component")
    selected_parent    = inputs.get("Parent App Component")
    selected_platform  = inputs.get("Platform Product Name")

    return {
        "risk_score":          prob,
        "baseline":            baseline,
        "risk_delta":          delta,
        "risk_label":          label,
        "risk_color":          color,
        "component_breakdown": component_tbl,
        "selected_component":  selected_component,
        "selected_parent":     selected_parent,
        "selected_platform":   selected_platform,
    }
