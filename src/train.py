import os
import sys
import json

import numpy as np
import pandas as pd

from sklearn.linear_model import (
    LinearRegression,
    Ridge,
    Lasso
)

from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import RandomizedSearchCV

from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)

from src.utils import (
    logger,
    CustomException,
    save_object,
    load_object,
    MODEL_PATH,
    PREPROCESSOR_PATH,
    METRICS_PATH,
    BEST_PARAMS_PATH,
    TEST_SIZE,
    RANDOM_STATE,
    CV,
    N_ITER,
    N_JOBS,
    VERBOSE
)

class ModelTrainer:

    def __init__(self):

        self.model_path = MODEL_PATH
        self.preprocessor_path = PREPROCESSOR_PATH
        self.metrics_path = METRICS_PATH
        self.best_params_path = BEST_PARAMS_PATH

    def random_search(
        self,
        model,
        params,
        X_train,
        y_train
    ):

        search = RandomizedSearchCV(
            estimator=model,
            param_distributions=params,
            n_iter=N_ITER,
            cv=CV,
            scoring="r2",
            random_state=RANDOM_STATE,
            n_jobs=N_JOBS,
            verbose=VERBOSE
        )

        search.fit(X_train, y_train)

        return search.best_estimator_, search.best_params_
        
    def train(self):

        try:

            logger.info("Loading processed datasets...")

            X_train = pd.read_csv("data/processed/X_train.csv")
            X_test = pd.read_csv("data/processed/X_test.csv")

            y_train = (
                pd.read_csv("data/processed/y_train.csv")
                .values
                .ravel()
            )

            y_test = (
                pd.read_csv("data/processed/y_test.csv")
                .values
                .ravel()
            )

            logger.info("Processed datasets loaded successfully.")
            logger.info("Loading preprocessor...")

            preprocessor = load_object(
                self.preprocessor_path
            )

            logger.info("Preprocessor loaded successfully.")

            logger.info("Transforming datasets...")

            X_train = preprocessor.transform(X_train)
            X_test = preprocessor.transform(X_test)

            logger.info("Dataset transformation completed.")
            
            logger.info("Initializing models...")

            models = {

                "Linear Regression": LinearRegression(),

                "Ridge": Ridge(),

                "Lasso": Lasso(),

                "Decision Tree": DecisionTreeRegressor(
                    random_state=RANDOM_STATE
                ),

                "Random Forest": RandomForestRegressor(
                    random_state=RANDOM_STATE
                )

            }

            logger.info("Models initialized successfully.")

            param_grids = {

                "Ridge": {

                    "alpha": np.logspace(-3, 3, 50)

                },

                "Lasso": {

                    "alpha": np.logspace(-3, 3, 50)

                },

                "Decision Tree": {

                    "max_depth": [5, 10, 15, 20, None],

                    "min_samples_split": [2, 5, 10],

                    "min_samples_leaf": [1, 2, 4]

                },

                "Random Forest": {

                    "n_estimators": [100, 200, 300],

                    "max_depth": [10, 20, None],

                    "min_samples_split": [2, 5, 10],

                    "min_samples_leaf": [1, 2, 4]

                }

            }
            results = []

            best_params = {}

            best_model = None

            best_model_name = None

            best_r2 = float("-inf")

            for name, model in models.items():

                logger.info(f"Training {name}...")

                if name == "Linear Regression":

                    model.fit(X_train, y_train)

                    trained_model = model

                    params = {}
                else:

                    trained_model, params = self.random_search(

                        model,

                        param_grids[name],

                        X_train,

                        y_train

                    )

                best_params[name] = params

                predictions = trained_model.predict(X_test) 

                r2 = r2_score(y_test, predictions)

                mae = mean_absolute_error(
                    y_test,
                    predictions
                )

                mse = mean_squared_error(
                    y_test,
                    predictions
                )

                rmse = np.sqrt(mse)

                results.append({

                    "Model": name,

                    "R2": r2,

                    "MAE": mae,

                    "MSE": mse,

                    "RMSE": rmse

                })

                logger.info(
                    f"{name} | R2={r2:.4f} | MAE={mae:.2f}"
                )

                if r2 > best_r2:

                    best_r2 = r2

                    best_model = trained_model

                    best_model_name = name

            logger.info("Model evaluation completed.")

            logger.info(
                f"Best Model : {best_model_name}"
            )

            logger.info(
                f"Best R2 Score : {best_r2:.4f}"
            )
            metrics_df = pd.DataFrame(results)

            metrics_df.to_csv(
                self.metrics_path,
                index=False
            )

            logger.info(
                "Metrics report saved successfully."
            )
            with open(
                self.best_params_path,
                "w"
            ) as file:

                json.dump(
                    best_params,
                    file,
                    indent=4
                )

            logger.info(
                "Best parameters saved successfully."
            )
            save_object(

                self.model_path,

                best_model

            )

            logger.info(
                "Best model saved successfully."
            )
            return best_model
        
        except Exception as e:

            logger.error(
                f"Error during model training : {str(e)}"
            )

            raise CustomException(
                e,
                sys
            )
if __name__ == "__main__":

    trainer = ModelTrainer()

    trainer.train()

    print("Training completed successfully.")