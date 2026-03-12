from pathlib import Path

# Base folders
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
OUTPUTS_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUTS_DIR / "plots"
TABLES_DIR = OUTPUTS_DIR / "tables"

# Input file
DATA_PATH = DATA_DIR / "walmart.csv"

# Create output folders if they do not exist
PLOTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

# Reproducibility
RANDOM_STATE = 42

# Clustering settings
K_RANGE_ELBOW = range(5, 15)
K_RANGE_METRICS = range(2, 15)
FINAL_K = 12
SILHOUETTE_SAMPLE_FRACTION = 0.05

# Market basket settings
TARGET_CLUSTER_FOR_RULES = 10
MIN_SUPPORT = 0.10
MIN_CONFIDENCE = 0.05
TOP_N_RULES = 5
