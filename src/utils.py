import os
import pickle
import sys

from src.exception import CustomException
from src.logger import logging


def save_object(file_path, obj):
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)

        with open(file_path, "wb") as file:
            pickle.dump(obj, file)

        logging.info(f"Object saved at {file_path}")

    except Exception as e:
        raise CustomException(e, sys)


def load_object(file_path):
    try:
        with open(file_path, "rb") as file:
            obj = pickle.load(file)

        logging.info(f"Object loaded from {file_path}")
        return obj

    except Exception as e:
        raise CustomException(e, sys)