import os
import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import joblib
import numpy as np
import pandas as pd

from config import ARTIFACTS_DIR, N_SVD_COMPONENTS, N_EMB_COMPONENTS, N_NMF_FACTORS

_model          = None
_risk_tables    = None
_feature_info   = None
_text_pipeline  = None
_text_profiles  = None
_nmf_model      = None
_graph_artifacts = None
_sentence_model = None


def _safe_load(path):
    return joblib.load(path) if os.path.exists(path) else None


def _load():
    global _model, _risk_tables, _feature_info
    global _text_pipeline, _text_profiles, _nmf_model, _graph_artifacts, _sentence_model

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
    _text_profiles  = _safe_load(os.path.join(ARTIFACTS_DIR, "text_profiles.joblib"))
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


def _apply_text_features_from_profile(row: dict, inputs: dict) -> None:
    """
    Impute text features from the historical per-entity text profile rather than
    requiring raw bug text. Resolution order: component → platform → global mean.
    """
    comp     = inputs.get("App Component")
    platform = inputs.get("Platform Product Name")

    all_text_cols = _text_profiles["all_text_cols"]

    comp_profile = _text_profiles["by_component"].get(comp, {}).get("all", {}) if comp else {}
    plat_profile = _text_profiles["by_platform"].get(platform, {}).get("all", {}) if platform else {}
    global_profile = _text_profiles["global"]["all"]

    for col in all_text_cols:
        row[col] = (
            comp_profile.get(col)
            if comp_profile.get(col) is not None
            else plat_profile.get(col)
            if plat_profile.get(col) is not None
            else global_profile.get(col, 0.0)
        )


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


def get_feature_importances() -> pd.DataFrame:
    """
    Return a DataFrame of feature name + importance from the trained RandomForest,
    with a 'group' column for aggregated display.
    """
    _load()
    features    = _feature_info["features"]
    importances = _model.named_steps["classifier"].feature_importances_

    df = pd.DataFrame({"feature": features, "importance": importances})

    def _group(f):
        if f.startswith("text_svd_"):   return "Text: TF-IDF Topics"
        if f.startswith("text_emb_"):   return "Text: Semantic Embeddings"
        if f.startswith("text_flag_"):  return "Text: Keyword Flags"
        if f.startswith("nmf_factor_"): return "NMF: Latent Risk Archetypes"
        if f.startswith("graph_"):      return "Graph: Network Metrics"
        return "Core Features"

    df["group"] = df["feature"].apply(_group)
    return df.sort_values("importance", ascending=False).reset_index(drop=True)


def get_graph_network_data() -> pd.DataFrame:
    """
    Return graph node metrics as a DataFrame for scatter/bubble visualization.
    Columns: entity_type, entity_name, pagerank, degree_centrality, clustering.
    """
    _load()
    if _graph_artifacts is None:
        return pd.DataFrame()

    rows = []
    for node_key, metrics in _graph_artifacts["node_metrics"].items():
        entity_type, entity_name = node_key.split(":", 1)
        rows.append({
            "entity_type":       entity_type,
            "entity_name":       entity_name,
            "pagerank":          metrics["pagerank"],
            "degree_centrality": metrics["degree_centrality"],
            "clustering":        metrics["clustering"],
        })
    return (
        pd.DataFrame(rows)
        .sort_values("pagerank", ascending=False)
        .reset_index(drop=True)
    )


def get_text_risk_signals(component: str = None, platform: str = None) -> list:
    """
    Return keyword flag elevations for the given component vs. the global H/C baseline.

    Each item: {col, label, elevation, component_hc_rate, global_hc_rate}
    Sorted descending by elevation. Only returned when the component has enough
    H/C bugs for a stable estimate (MIN_BUGS_FOR_TABLE enforced at training time).
    Returns [] when no profile is available or artifacts are old-format.
    """
    _load()
    if _text_profiles is None:
        return []

    flag_cols  = _text_profiles["flag_cols"]
    global_hc  = _text_profiles["global"]["hc"]
    comp_data  = _text_profiles["by_component"].get(component, {}) if component else {}
    comp_hc    = comp_data.get("hc", {})

    if not comp_hc:
        return []

    signals = []
    for col in flag_cols:
        comp_rate   = comp_hc.get(col, 0.0)
        global_rate = global_hc.get(col, 0.0)
        signals.append({
            "col":               col,
            "elevation":         comp_rate - global_rate,
            "component_hc_rate": comp_rate,
            "global_hc_rate":    global_rate,
        })

    return sorted(signals, key=lambda x: x["elevation"], reverse=True)


def get_customers() -> list:
    """Return sorted list of customers present in training data."""
    _load()
    return _risk_tables.get("customers", [])


def get_customer_risk_tables(customer: str) -> dict:
    """Return risk tables scoped to a single customer."""
    _load()
    return _risk_tables.get("by_customer", {}).get(customer, {})


def get_bubble_data() -> pd.DataFrame:
    """Return precomputed bubble chart DataFrame, or empty if not yet generated."""
    path = os.path.join(ARTIFACTS_DIR, "bubble_data.joblib")
    if not os.path.exists(path):
        return pd.DataFrame()
    return joblib.load(path)


def predict_release_risk(inputs: dict) -> dict:
    """
    inputs: dict mapping feature names to values.
    Returns dict with risk_score, baseline, risk_delta, risk_label, component_breakdown.
    """
    _load()

    is_multimodal = _feature_info.get("has_multimodal", False)
    features      = _feature_info["features"]

    row = {f: inputs.get(f) for f in features}

    if is_multimodal:
        if _text_profiles is not None:
            _apply_text_features_from_profile(row, inputs)
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
