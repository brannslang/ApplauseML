import os

ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(ROOT, "data")
ARTIFACTS_DIR = os.path.join(ROOT, "model", "artifacts")

CATEGORICAL_FEATURES = [
    "Customer",
    "Test Cycle Testing Type",
    "Platform Product Name",
    "Development Stage",
    "Bug Source Type",
    "App Component",
    "Bug Type",
]

NUMERIC_FEATURES = [
    "Bug Rate Amount",
    "Test Cycle Duration Activation to Lock/Close/Today",
]

TARGET = "is_high_crit"
MIN_BUGS_FOR_TABLE = 10
