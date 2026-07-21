import os
import sys
import pickle
import logging
from pathlib import Path

from dotenv import load_dotenv

# ==========================
# Load Environment Variables
# ==========================

load_dotenv()

# ==========================
# Project Paths
# ==========================

BASE_DIR = Path(__file__).resolve().parent.parent

DATA_PATH = os.getenv("DATA_PATH")
MODEL_PATH = os.getenv("MODEL_PATH")
PREPROCESSOR_PATH = os.getenv("PREPROCESSOR_PATH")
METRICS_PATH = os.getenv("METRICS_PATH")
BEST_PARAMS_PATH = os.getenv("BEST_PARAMS_PATH")

TEST_SIZE = float(os.getenv("TEST_SIZE"))
RANDOM_STATE = int(os.getenv("RANDOM_STATE"))
CV = int(os.getenv("CV"))
N_ITER = int(os.getenv("N_ITER"))
N_JOBS = int(os.getenv("N_JOBS"))
VERBOSE = int(os.getenv("VERBOSE"))

# ==========================
# Logging
# ==========================

LOG_DIR = BASE_DIR / "reports"
LOG_DIR.mkdir(parents=True, exist_ok=True)

LOG_FILE = LOG_DIR / "project.log"

logging.basicConfig(
    filename=str(LOG_FILE),
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logger = logging.getLogger(__name__)

# ==========================
# Custom Exception
# ==========================

class CustomException(Exception):
    def __init__(self, error_message, error_detail: sys):
        super().__init__(error_message)
        self.error_message = str(error_message)

    def __str__(self):
        return self.error_message

# ==========================
# Save Object
# ==========================

def save_object(file_path, obj):
    """
    Save a Python object using pickle.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file:
            pickle.dump(obj, file)

        logger.info(f"Object saved successfully at {file_path}")

    except Exception as e:
        logger.exception("Error while saving object.")
        raise CustomException(e, sys)

# ==========================
# Load Object
# ==========================

def load_object(file_path):
    """
    Load a pickled Python object.
    """
    try:
        with open(file_path, "rb") as file:
            obj = pickle.load(file)

        logger.info(f"Object loaded successfully from {file_path}")

        return obj

    except Exception as e:
        logger.exception("Error while loading object.")
        raise CustomException(e, sys)