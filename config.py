import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
ARTIFACTS_DIR = os.path.join(ROOT, "model", "artifacts")

CATEGORICAL_FEATURES = [
    "App Component",
    "Parent App Component",
    "Platform Product Name",
    "Development Stage",
    "Bug Request Source",
    "Bug Source Type",
    "Testing Approach",
]

NUMERIC_FEATURES = [
    "Bug Rate Amount",
    "Test Cycle Duration Activation to Lock/Close/Today",
]

TARGET = "is_high_crit"
MIN_BUGS_FOR_TABLE = 10

# --- Multi-modal enrichment constants ---

N_SVD_COMPONENTS = 25
N_EMB_COMPONENTS = 20
N_NMF_FACTORS = 15

KEYWORD_GROUPS = {
    "text_flag_crash": r"\b(crash|freeze|hang|unresponsive|force.?close)\b",
    "text_flag_data_integrity": r"\b(data.?loss|incorrect|missing|wrong|corrupt(ed)?)\b",
    "text_flag_error": r"\b(error|exception|null|undefined|failed.?to.?load)\b",
    "text_flag_security": r"\b(security|unauthorized|unauthorised|exposed|bypass)\b",
    "text_flag_visibility": r"\b(blank|white.?screen|not.?loading|broken)\b",
    "text_flag_performance": r"\b(slow|timeout|time.?out|performance|lag|latency)\b",
    "text_flag_access": r"\b(login|auth(entication)?|permission|access.?denied|session)\b",
}

TEXT_FLAG_FEATURES = list(KEYWORD_GROUPS.keys())
TEXT_SVD_FEATURES = [f"text_svd_{i}" for i in range(N_SVD_COMPONENTS)]
TEXT_EMB_FEATURES = [f"text_emb_{i}" for i in range(N_EMB_COMPONENTS)]
NMF_FEATURES = [f"nmf_factor_{i}" for i in range(N_NMF_FACTORS)]

GRAPH_FEATURES = [
    "graph_comp_pagerank",
    "graph_comp_degree",
    "graph_comp_clustering",
    "graph_platform_pagerank",
    "graph_customer_pagerank",
]

NMF_ENTITY_COLS = [
    "App Component",
    "Platform Product Name",
    "Development Stage",
    "Testing Approach",
    "Bug Source Type",
    "Customer",
]
