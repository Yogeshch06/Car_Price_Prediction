import os
import sys
from datetime import datetime

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.utils import (
    logger,
    CustomException,
    save_object,
    DATA_PATH,
    PREPROCESSOR_PATH,
    TEST_SIZE,
    RANDOM_STATE
)


class DataPreprocessor:

    def __init__(self):
        self.data_path = DATA_PATH
        self.preprocessor_path = PREPROCESSOR_PATH

    def clean_column_names(self, df):

        df = df.copy()

        df.columns = (
            df.columns
            .str.strip()
            .str.lower()
            .str.replace(".", "", regex=False)
            .str.replace(" ", "_")
        )

        return df

    def preprocess_data(self):

        try:

            logger.info("Loading dataset...")

            df = pd.read_csv(self.data_path)

            logger.info(f"Dataset loaded successfully. Shape: {df.shape}")

            # Clean column names
            df = self.clean_column_names(df)
            logger.info("Column names cleaned successfully.")

            # Remove duplicate rows
            df.drop_duplicates(inplace=True)
            logger.info(f"Duplicates removed. Shape: {df.shape}")

            # Handle Levy column
            df["levy"] = df["levy"].replace("-", np.nan)
            df["levy"] = df["levy"].astype(float)

            # Handle Mileage column
            df["mileage"] = (
                df["mileage"]
                .str.replace(" km", "", regex=False)
                .astype(int)
            )

            # Handle Engine Volume and Turbo
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

            # Clean Doors column
            df["doors"] = df["doors"].replace({
                "04-May": "4",
                "02-Mar": "2",
                ">5": "5"
            })

            # Create Car Age Feature
            current_year = datetime.now().year
            df["car_age"] = current_year - df["prod_year"]

            logger.info("Feature engineering completed.")

             # Drop unnecessary columns
            df.drop(columns=["model", "id"], inplace=True)

            logger.info("Dropped model and id columns.")

            # Remove outliers from Price
            lower_limit = df["price"].quantile(0.01)
            upper_limit = df["price"].quantile(0.99)

            df = df[
                (df["price"] >= lower_limit) &
                (df["price"] <= upper_limit)
            ]

            logger.info(f"Outliers removed. Shape: {df.shape}")

            X = df.drop("price", axis=1)
            y = df["price"]

            logger.info("Features and target separated.")

            X_train, X_test, y_train, y_test = train_test_split( X,
                                                                y,
                                                                test_size=TEST_SIZE,
                                                                random_state=RANDOM_STATE
          )

            logger.info("Train-test split completed.")

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

            # Numerical Pipeline
            numerical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            # Categorical Pipeline
            categorical_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("encoder", OneHotEncoder(handle_unknown="ignore"))
                ]
            )

            # Column Transformer
            preprocessor = ColumnTransformer(
                transformers=[
                    ("num", numerical_pipeline, numerical_features),
                    ("cat", categorical_pipeline, categorical_features)
                ]
            )

            logger.info("Preprocessing pipeline created successfully.")

            X_train_processed = preprocessor.fit_transform(X_train)
            X_test_processed = preprocessor.transform(X_test)

            logger.info("Preprocessor fitted successfully.")

            save_object(
                self.preprocessor_path,
                preprocessor
            )

            logger.info("Preprocessor saved successfully.")

            processed_dir = os.path.join(
                os.path.dirname(self.data_path),
                "processed"
            )

            os.makedirs(processed_dir, exist_ok=True)

            pd.DataFrame(X_train).to_csv(
                os.path.join(processed_dir, "X_train.csv"),
                index=False
            )

            pd.DataFrame(X_test).to_csv(
                os.path.join(processed_dir, "X_test.csv"),
                index=False
            )

            pd.DataFrame(y_train).to_csv(
                os.path.join(processed_dir, "y_train.csv"),
                index=False
            )

            pd.DataFrame(y_test).to_csv(
                os.path.join(processed_dir, "y_test.csv"),
                index=False
            )

            logger.info("Processed datasets saved successfully.")

            return (
                X_train_processed,
                X_test_processed,
                y_train,
                y_test,
                preprocessor
            )

        except Exception as e:
            logger.error(f"Error during preprocessing: {str(e)}")
            raise CustomException(e, sys)
        
if __name__ == "__main__":

        preprocessor = DataPreprocessor()

        preprocessor.preprocess_data()

        print("Preprocessing completed successfully.")