from pathlib import Path
import os
from dotenv import load_dotenv

# Project root directory
BASE_DIR = Path(__file__).resolve().parent.parent

# Load .env (only for optional settings)
load_dotenv(BASE_DIR / ".env")

# ===============================
# Dataset & Artifacts
# ===============================
DATA_PATH = BASE_DIR / "data" / "raw" / "car.csv"
MODEL_PATH = BASE_DIR / "models" / "model.pkl"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.pkl"

# ===============================
# Reports
# ===============================
METRICS_PATH = BASE_DIR / "reports" / "metrics.csv"
BEST_PARAMS_PATH = BASE_DIR / "reports" / "best_params.json"

# ===============================
# Logging
# ===============================
LOG_FILE = BASE_DIR / "logs" / "project.log"

# ===============================
# Train-Test Split
# ===============================
TEST_SIZE = float(os.getenv("TEST_SIZE", 0.2))
RANDOM_STATE = int(os.getenv("RANDOM_STATE", 42))

# ===============================
# Cross Validation
# ===============================
CV = int(os.getenv("CV", 5))

# ===============================
# Hyperparameter Tuning
# ===============================
N_ITER = int(os.getenv("N_ITER", 20))
N_JOBS = int(os.getenv("N_JOBS", -1))
VERBOSE = int(os.getenv("VERBOSE", 1))