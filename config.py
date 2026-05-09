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
