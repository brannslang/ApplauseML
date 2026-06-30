"""
Train the ApplauseML risk classifier and precompute risk tables.

Run once (and monthly) to refresh model artifacts:
    python model/train.py
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import joblib
import networkx as nx
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import NMF, PCA, TruncatedSVD
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import cross_val_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder, StandardScaler

from config import (
    ARTIFACTS_DIR, CATEGORICAL_FEATURES, DATA_DIR,
    GRAPH_FEATURES, KEYWORD_GROUPS, MIN_BUGS_FOR_TABLE,
    N_EMB_COMPONENTS, N_NMF_FACTORS, N_SVD_COMPONENTS,
    NMF_ENTITY_COLS, NMF_FEATURES, NUMERIC_FEATURES,
    TARGET, TEXT_EMB_FEATURES, TEXT_FLAG_FEATURES,
    TEXT_SVD_FEATURES,
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


def build_text_features(df: pd.DataFrame) -> tuple:
    """Stream A: keyword flags, TF-IDF + SVD, sentence embeddings + PCA."""
    subject = df["Bug Subject"].fillna("") if "Bug Subject" in df.columns else pd.Series("", index=df.index)
    result  = df["Bug Result"].fillna("")  if "Bug Result"  in df.columns else pd.Series("", index=df.index)
    text = (subject + " " + result).str.strip()

    # Layer 1: keyword flags
    flag_df = pd.DataFrame(index=df.index)
    for col, pattern in KEYWORD_GROUPS.items():
        flag_df[col] = text.str.contains(pattern, case=False, regex=True).astype(int)

    # Layer 2: TF-IDF + Truncated SVD
    tfidf = TfidfVectorizer(
        min_df=5, max_features=2000, stop_words="english", ngram_range=(1, 2)
    )
    tfidf_matrix = tfidf.fit_transform(text)
    n_svd = min(N_SVD_COMPONENTS, tfidf_matrix.shape[1] - 1)
    svd = TruncatedSVD(n_components=n_svd, random_state=42)
    svd_arr = svd.fit_transform(tfidf_matrix)
    svd_df = pd.DataFrame(index=df.index)
    for i in range(N_SVD_COMPONENTS):
        svd_df[f"text_svd_{i}"] = svd_arr[:, i] if i < n_svd else 0.0

    # Layer 3: sentence embeddings + PCA
    print("    Encoding bug text with sentence transformer (this may take a minute)...")
    st_model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = st_model.encode(text.tolist(), show_progress_bar=True, batch_size=128)
    n_emb = min(N_EMB_COMPONENTS, embeddings.shape[0] - 1, embeddings.shape[1])
    pca = PCA(n_components=n_emb, random_state=42)
    pca_arr = pca.fit_transform(embeddings)
    emb_df = pd.DataFrame(index=df.index)
    for i in range(N_EMB_COMPONENTS):
        emb_df[f"text_emb_{i}"] = pca_arr[:, i] if i < n_emb else 0.0

    feature_df = pd.concat([flag_df, svd_df, emb_df], axis=1)
    artifacts = {
        "tfidf": tfidf,
        "svd": svd,
        "pca": pca,
        "n_svd": n_svd,
        "n_emb": n_emb,
        "sentence_model_name": "all-MiniLM-L6-v2",
        "keyword_groups": KEYWORD_GROUPS,
    }
    return feature_df, artifacts


def build_nmf_features(df: pd.DataFrame) -> tuple:
    """Stream B: NMF latent factors from entity co-occurrence matrix."""
    available_cols = [c for c in NMF_ENTITY_COLS if c in df.columns]

    encoded_parts = []
    for col in available_cols:
        dummies = pd.get_dummies(df[col].fillna("Unknown"), prefix=col)
        encoded_parts.append(dummies)

    entity_matrix = pd.concat(encoded_parts, axis=1).astype(float)
    feature_names = list(entity_matrix.columns)

    nmf = NMF(n_components=N_NMF_FACTORS, random_state=42, max_iter=500)
    nmf_arr = nmf.fit_transform(entity_matrix)
    feature_df = pd.DataFrame(nmf_arr, columns=NMF_FEATURES, index=df.index)

    artifacts = {
        "nmf": nmf,
        "entity_cols": available_cols,
        "feature_names": feature_names,
    }
    return feature_df, artifacts


def build_graph_features(df: pd.DataFrame) -> tuple:
    """Stream C: property graph — entity co-occurrence network weighted by above-baseline H/C rate."""
    baseline = df[TARGET].mean()
    G = nx.Graph()

    entity_pairs = [
        ("App Component", "Platform Product Name"),
        ("App Component", "Customer"),
        ("App Component", "Development Stage"),
        ("App Component", "Testing Approach"),
        ("Platform Product Name", "Customer"),
    ]

    for col_a, col_b in entity_pairs:
        if col_a not in df.columns or col_b not in df.columns:
            continue
        agg = (
            df.groupby([col_a, col_b])[TARGET]
            .agg(["mean", "count"])
            .reset_index()
        )
        agg = agg[agg["count"] >= MIN_BUGS_FOR_TABLE]
        agg["weight"] = (agg["mean"] - baseline).clip(lower=0)
        for _, row in agg.iterrows():
            if row["weight"] > 0:
                node_a = f"{col_a}:{row[col_a]}"
                node_b = f"{col_b}:{row[col_b]}"
                if G.has_edge(node_a, node_b):
                    G[node_a][node_b]["weight"] = max(
                        G[node_a][node_b]["weight"], row["weight"]
                    )
                else:
                    G.add_edge(node_a, node_b, weight=row["weight"])

    node_metrics = {}
    if len(G.nodes()) > 0:
        pr = nx.pagerank(G, weight="weight")
        dc = nx.degree_centrality(G)
        try:
            cl = nx.clustering(G, weight="weight")
        except Exception:
            cl = {n: 0.0 for n in G.nodes()}
        for node in G.nodes():
            node_metrics[node] = {
                "pagerank": pr.get(node, 0.0),
                "degree_centrality": dc.get(node, 0.0),
                "clustering": cl.get(node, 0.0),
            }

    def make_lookup(prefix, metric):
        return {
            k.split(":", 1)[1]: v[metric]
            for k, v in node_metrics.items()
            if k.startswith(f"{prefix}:")
        }

    comp_pr = make_lookup("App Component", "pagerank")
    comp_dc = make_lookup("App Component", "degree_centrality")
    comp_cl = make_lookup("App Component", "clustering")
    plat_pr = make_lookup("Platform Product Name", "pagerank")
    cust_pr = make_lookup("Customer", "pagerank")

    feature_df = pd.DataFrame(index=df.index)
    feature_df["graph_comp_pagerank"]     = df["App Component"].map(comp_pr).fillna(0.0)          if "App Component"         in df.columns else 0.0
    feature_df["graph_comp_degree"]       = df["App Component"].map(comp_dc).fillna(0.0)          if "App Component"         in df.columns else 0.0
    feature_df["graph_comp_clustering"]   = df["App Component"].map(comp_cl).fillna(0.0)          if "App Component"         in df.columns else 0.0
    feature_df["graph_platform_pagerank"] = df["Platform Product Name"].map(plat_pr).fillna(0.0)  if "Platform Product Name" in df.columns else 0.0
    feature_df["graph_customer_pagerank"] = df["Customer"].map(cust_pr).fillna(0.0)               if "Customer"              in df.columns else 0.0

    return feature_df, {"node_metrics": node_metrics}


def compute_text_profiles(df: pd.DataFrame) -> dict:
    """
    Compute per-component and per-platform mean text feature profiles.

    Two sub-profiles per entity:
      'all' — mean across all bugs (used to impute text features at prediction time)
      'hc'  — mean across H/C bugs only (used to surface risk signals in the UI)
    """
    all_text_cols = [c for c in TEXT_FLAG_FEATURES + TEXT_SVD_FEATURES + TEXT_EMB_FEATURES if c in df.columns]
    flag_cols     = [c for c in TEXT_FLAG_FEATURES if c in df.columns]

    def means(subset: pd.DataFrame) -> dict:
        return subset[all_text_cols].mean().to_dict() if len(subset) > 0 else {}

    global_profile = {
        "all": means(df),
        "hc":  means(df[df[TARGET] == 1]),
    }

    by_component = {}
    if "App Component" in df.columns:
        for comp, grp in df.groupby("App Component"):
            hc_grp = grp[grp[TARGET] == 1]
            by_component[comp] = {
                "all":    means(grp),
                "hc":     means(hc_grp) if len(hc_grp) >= MIN_BUGS_FOR_TABLE else {},
                "n_bugs": len(grp),
                "n_hc":   int(hc_grp[TARGET].sum()),
            }

    by_platform = {}
    if "Platform Product Name" in df.columns:
        for plat, grp in df.groupby("Platform Product Name"):
            by_platform[plat] = {"all": means(grp)}

    # Monthly keyword flag trends
    date_col = next(
        (c for c in df.columns if "create" in c.lower() and "date" in c.lower()), None
    )
    monthly_flag_trends = None
    if date_col and flag_cols:
        _df = df.copy()
        _df["_month"] = pd.to_datetime(_df[date_col], errors="coerce").dt.to_period("M")
        agg = {f: "mean" for f in flag_cols}
        agg[TARGET] = "count"
        monthly_flag_trends = (
            _df.groupby("_month")
            .agg(agg)
            .reset_index()
            .rename(columns={"_month": "month", TARGET: "n_bugs"})
        )
        monthly_flag_trends["month"] = monthly_flag_trends["month"].astype(str)

    return {
        "by_component":       by_component,
        "by_platform":        by_platform,
        "global":             global_profile,
        "flag_cols":          flag_cols,
        "all_text_cols":      all_text_cols,
        "monthly_flag_trends": monthly_flag_trends,
    }


def compute_risk_tables(df: pd.DataFrame) -> dict:
    baseline = df[TARGET].mean()
    tables = {
        "baseline":   baseline,
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
    all_numeric = (
        NUMERIC_FEATURES
        + TEXT_FLAG_FEATURES
        + TEXT_SVD_FEATURES
        + TEXT_EMB_FEATURES
        + NMF_FEATURES
        + GRAPH_FEATURES
    )
    available_cat = [c for c in CATEGORICAL_FEATURES if c in df.columns]
    available_num = [c for c in all_numeric if c in df.columns]
    features = available_cat + available_num

    X = df[features].copy()
    for col in available_num:
        X[col] = pd.to_numeric(X[col], errors="coerce")
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
        "features":     features,
        "cat_cols":     available_cat,
        "num_cols":     available_num,
        "base_num_cols": [c for c in NUMERIC_FEATURES if c in df.columns],
        "categories": {
            col: sorted(df[col].dropna().astype(str).unique().tolist())
            for col in available_cat
        },
        "has_multimodal": True,
    }
    return clf, feature_info


BUBBLE_GROUP_COLS = [
    "Platform Product Name",
    "Mobile OS Major Version",
    "App Component Name",
]

CLUSTER_NAMES = {0: "Stable", 1: "Nuisance Zone", 2: "Critical Hazard"}
CLUSTER_COLORS = {
    "Critical Hazard": "#d62728",
    "Nuisance Zone":   "#ff7f0e",
    "Stable":          "#2ca02c",
}


def compute_bubble_data(df: pd.DataFrame, clf, feature_info: dict) -> pd.DataFrame:
    """
    Build one row per (Platform, OS Major Version, App Component) group with:
      - ml_prob:       mean classifier P(High/Critical) across bugs in the group
      - failure_rate:  test case failure rate from devicetestruns
      - n_bugs, severity counts: for hover tooltips
      - cluster:       KMeans k=3 auto-labeled by centroid position
    """
    # --- Step 1: per-bug ML probabilities on the main training set ---
    features = feature_info["features"]
    available = [f for f in features if f in df.columns]
    X_all = df[available].copy()
    for col in feature_info.get("num_cols", []):
        if col in X_all.columns:
            X_all[col] = pd.to_numeric(X_all[col], errors="coerce")
    # Pad any missing features with NaN so shape matches
    for f in features:
        if f not in X_all.columns:
            X_all[f] = np.nan
    X_all = X_all[features]

    df = df.copy()
    df["_ml_prob"] = clf.predict_proba(X_all)[:, 1]

    # --- Step 2: load device-level tables ---
    device_bugs = pd.read_excel(os.path.join(DATA_DIR, "devicebugs.xlsx"), engine="openpyxl")
    device_runs = pd.read_excel(os.path.join(DATA_DIR, "devicetestruns.xlsx"), engine="openpyxl")

    # Normalise the severity column name (may have trailing spaces)
    sev_col = next((c for c in device_bugs.columns if c.strip() == "Bug Severity  Old".strip() or c.strip() == "Bug Severity Old"), None)

    # --- Step 3: join devicebugs → bugdetails to pull ml_prob ---
    # bugdetails key column is "Bug"; devicebugs key is "Bug Id"
    bug_probs = df[["Bug", "_ml_prob"]].rename(columns={"Bug": "Bug Id"})
    dbugs = device_bugs.merge(bug_probs, on="Bug Id", how="left")

    # --- Step 4: aggregate per group ---
    agg_dict = {
        "ml_prob": ("_ml_prob", "mean"),
        "n_bugs":  ("Bug Id",   "count"),
    }
    if sev_col:
        agg_dict["n_critical"] = (sev_col, lambda x: (x == "Critical").sum())
        agg_dict["n_high"]     = (sev_col, lambda x: (x == "High").sum())
        agg_dict["n_medium"]   = (sev_col, lambda x: (x == "Medium").sum())
        agg_dict["n_low"]      = (sev_col, lambda x: (x == "Low").sum())

    bug_agg = (
        dbugs.groupby(BUBBLE_GROUP_COLS)
        .agg(**agg_dict)
        .reset_index()
    )

    # Fill groups where no bugdetails join matched with global baseline
    global_baseline = df["_ml_prob"].mean()
    bug_agg["ml_prob"] = bug_agg["ml_prob"].fillna(global_baseline)

    # --- Step 5: failure rate from device test runs ---
    run_totals  = device_runs.groupby(BUBBLE_GROUP_COLS).size().rename("n_runs")
    run_fails   = device_runs[device_runs["Result Status"] == "Failed"].groupby(BUBBLE_GROUP_COLS).size().rename("n_failed")
    run_agg = pd.concat([run_totals, run_fails], axis=1).fillna(0).reset_index()
    run_agg["failure_rate"] = run_agg["n_failed"] / run_agg["n_runs"].replace(0, np.nan)
    run_agg["failure_rate"] = run_agg["failure_rate"].fillna(0.0)
    run_agg = run_agg[BUBBLE_GROUP_COLS + ["failure_rate", "n_runs", "n_failed"]]

    # --- Step 6: merge ---
    bubble = bug_agg.merge(run_agg, on=BUBBLE_GROUP_COLS, how="outer")
    bubble["ml_prob"]     = bubble["ml_prob"].fillna(global_baseline)
    bubble["failure_rate"]= bubble["failure_rate"].fillna(0.0)
    bubble["n_bugs"]      = bubble["n_bugs"].fillna(0).astype(int)
    for col in ["n_critical", "n_high", "n_medium", "n_low"]:
        if col in bubble.columns:
            bubble[col] = bubble[col].fillna(0).astype(int)

    # --- Step 7: KMeans clustering on (failure_rate, ml_prob) ---
    n_clusters = min(3, len(bubble))
    if n_clusters >= 2:
        scaler = StandardScaler()
        coords = scaler.fit_transform(bubble[["failure_rate", "ml_prob"]])
        km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
        labels = km.fit_predict(coords)

        centroid_scores = km.cluster_centers_[:, 0] + km.cluster_centers_[:, 1]
        rank = centroid_scores.argsort().argsort()  # 0=lowest … 2=highest

        bubble["cluster"] = [CLUSTER_NAMES.get(rank[l], "Unknown") for l in labels]
    else:
        bubble["cluster"] = "Stable"

    bubble["cluster_color"] = bubble["cluster"].map(CLUSTER_COLORS).fillna("#aec7e8")
    return bubble


def main():
    print("Loading data...")
    df = load_data()
    print(f"  {len(df):,} bugs  |  baseline H/C rate: {df[TARGET].mean():.1%}")

    print("Building text features (keyword flags, TF-IDF/SVD, embeddings)...")
    text_features, text_artifacts = build_text_features(df)
    df = pd.concat([df, text_features], axis=1)
    print(f"  {len(text_features.columns)} text feature columns added")

    print("Computing per-entity text profiles...")
    text_profiles = compute_text_profiles(df)
    print(f"  {len(text_profiles['by_component'])} component profiles  |  {len(text_profiles['by_platform'])} platform profiles")

    print("Building NMF association features...")
    nmf_features, nmf_artifacts = build_nmf_features(df)
    df = pd.concat([df, nmf_features], axis=1)
    print(f"  {N_NMF_FACTORS} NMF factors added")

    print("Building property graph features...")
    graph_features, graph_artifacts = build_graph_features(df)
    df = pd.concat([df, graph_features], axis=1)
    print(f"  {len(graph_artifacts['node_metrics'])} graph nodes  |  {len(graph_features.columns)} graph feature columns added")

    print("Computing risk tables...")
    risk_tables = compute_risk_tables(df)
    print(f"  {len(risk_tables) - 2} dimension tables built")

    print("Training classifier...")
    clf, feature_info = train_classifier(df)

    print("Computing bubble map data...")
    bubble_data = compute_bubble_data(df, clf, feature_info)
    print(f"  {len(bubble_data):,} bubbles  |  clusters: {bubble_data['cluster'].value_counts().to_dict()}")

    joblib.dump(clf,            os.path.join(ARTIFACTS_DIR, "classifier.joblib"))
    joblib.dump(risk_tables,    os.path.join(ARTIFACTS_DIR, "risk_tables.joblib"))
    joblib.dump(feature_info,   os.path.join(ARTIFACTS_DIR, "feature_info.joblib"))
    joblib.dump(text_artifacts, os.path.join(ARTIFACTS_DIR, "text_pipeline.joblib"))
    joblib.dump(text_profiles,  os.path.join(ARTIFACTS_DIR, "text_profiles.joblib"))
    joblib.dump(nmf_artifacts,  os.path.join(ARTIFACTS_DIR, "nmf_model.joblib"))
    joblib.dump(graph_artifacts, os.path.join(ARTIFACTS_DIR, "graph_artifacts.joblib"))
    joblib.dump(bubble_data,    os.path.join(ARTIFACTS_DIR, "bubble_data.joblib"))

    print(f"\nArtifacts saved to {ARTIFACTS_DIR}/")
    print("  Done. Run 'streamlit run app/Home.py' to launch the dashboard.")


if __name__ == "__main__":
    main()
