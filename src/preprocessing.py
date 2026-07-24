import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    DATA_PATH,
    PREPROCESSOR_PATH,
    RANDOM_STATE,
    TEST_SIZE,
)
from src.exception import CustomException
from src.logger import logging
from src.utils import save_object


class DataPreprocessor:

    def __init__(self):
        self.data_path = DATA_PATH
        self.preprocessor_path = PREPROCESSOR_PATH

    def clean_column_names(self, df):
        df = df.copy()
        df.columns = (
            df.columns.str.strip()
            .str.lower()
            .str.replace(".", "", regex=False)
            .str.replace(" ", "_")
        )
        return df

    def preprocess_data(self):
        try:
            logging.info("Loading dataset...")

            df = pd.read_csv(self.data_path)

            df = self.clean_column_names(df)
            df.drop_duplicates(inplace=True)

            # Data Cleaning
            df["levy"] = df["levy"].replace("-", np.nan).astype(float)

            df["mileage"] = (
                df["mileage"]
                .str.replace(" km", "", regex=False)
                .astype(int)
            )

            df["is_turbo"] = (
                df["engine_volume"]
                .str.contains("Turbo", case=False, na=False)
                .astype(int)
            )

            df["engine_volume"] = (
                df["engine_volume"]
                .str.replace(" Turbo", "", regex=False)
                .astype(float)
            )

            df["doors"] = df["doors"].replace({
                "04-May": "4",
                "02-Mar": "2",
                ">5": "5"
            })

            df["car_age"] = datetime.now().year - df["prod_year"]

            df.drop(columns=["model", "id"], inplace=True)

            # Remove Outliers
            lower = df["price"].quantile(0.01)
            upper = df["price"].quantile(0.99)

            df = df[
                (df["price"] >= lower) &
                (df["price"] <= upper)
            ]

            X = df.drop("price", axis=1)
            y = df["price"]

            X_train, X_test, y_train, y_test = train_test_split(
                X,
                y,
                test_size=TEST_SIZE,
                random_state=RANDOM_STATE
            )

            numerical_features = [
                "levy",
                "prod_year",
                "engine_volume",
                "mileage",
                "cylinders",
                "airbags",
                "car_age",
                "is_turbo"
            ]

            categorical_features = [
                "manufacturer",
                "category",
                "leather_interior",
                "fuel_type",
                "gear_box_type",
                "drive_wheels",
                "doors",
                "wheel",
                "color"
            ]

            numerical_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler())
            ])

            categorical_pipeline = Pipeline([
                ("imputer", SimpleImputer(strategy="most_frequent")),
                ("encoder", OneHotEncoder(handle_unknown="ignore"))
            ])

            logging.info("Creating preprocessing pipeline...")

            preprocessor = ColumnTransformer([
                ("num", numerical_pipeline, numerical_features),
                ("cat", categorical_pipeline, categorical_features)
            ])

            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)

            save_object(self.preprocessor_path, preprocessor)

            logging.info("Preprocessor saved successfully.")

            processed_dir = os.path.join(
                os.path.dirname(self.data_path),
                "processed"
            )

            os.makedirs(processed_dir, exist_ok=True)

            files = {
                "X_train.csv": X_train,
                "X_test.csv": X_test,
                "y_train.csv": y_train,
                "y_test.csv": y_test,
            }

            for file_name, data in files.items():
                data.to_csv(
                    os.path.join(processed_dir, file_name),
                    index=False
                )

            return (
                X_train_processed,
                X_test_processed,
                y_train,
                y_test,
                preprocessor,
            )

        except Exception as e:
            logging.error(str(e))
            raise CustomException(e, sys)


if __name__ == "__main__":
    preprocessor = DataPreprocessor()
    preprocessor.preprocess_data()
    print("Preprocessing completed successfully.")