import json
import sys

from src.utils import save_object, load_object
import numpy as np
import pandas as pd

from scipy.stats import randint, uniform

from sklearn.compose import TransformedTargetRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.metrics import (
    r2_score,
    mean_absolute_error,
    mean_squared_error
)
from sklearn.model_selection import RandomizedSearchCV
from sklearn.tree import DecisionTreeRegressor

from src.config import (
    MODEL_PATH,
    PREPROCESSOR_PATH,
    RANDOM_STATE
)

from src.logger import logging
from src.exception import CustomException

class ModelTrainer:

    def __init__(self):
        self.model_path = MODEL_PATH
        self.preprocessor_path = PREPROCESSOR_PATH

    @staticmethod
    def make_model(estimator):
        """
        Train on log(price) but predict actual prices.
        """
        return TransformedTargetRegressor(
            regressor=estimator,
            func=np.log1p,
            inverse_func=np.expm1
        )

    @staticmethod
    def tune_model(
        model,
        param_distributions,
        X_train,
        y_train,
        cv=5,
        scoring="r2",
        n_iter=40
    ):

        random_search = RandomizedSearchCV(
            estimator=model,
            param_distributions=param_distributions,
            n_iter=n_iter,
            cv=cv,
            scoring=scoring,
            n_jobs=-1,
            random_state=RANDOM_STATE,
            verbose=1
        )

        random_search.fit(X_train, y_train)

        logging.info(f"Best Parameters : {random_search.best_params_}")
        logging.info(f"Best CV Score : {random_search.best_score_:.4f}")

        return (
            random_search.best_estimator_,
            random_search.best_params_,
            random_search.best_score_
        )

    def train(self):

        try:

            logging.info("Loading processed dataset...")

            X_train = pd.read_csv("data/processed/X_train.csv")
            X_test = pd.read_csv("data/processed/X_test.csv")

            y_train = (
                pd.read_csv("data/processed/y_train.csv")
                .values.ravel()
            )

            y_test = (
                pd.read_csv("data/processed/y_test.csv")
                .values.ravel()
            )

            logging.info("Loading preprocessor...")

            preprocessor = load_object(
            self.preprocessor_path
            )

            X_train_t = preprocessor.transform(X_train)
            X_test_t = preprocessor.transform(X_test)

            logging.info("Creating baseline models...")

            baseline_models = {

                "LinearRegression": self.make_model(
                    LinearRegression()
                ),

                "Ridge": self.make_model(
                    Ridge(random_state=RANDOM_STATE)
                ),

                "Lasso": self.make_model(
                    Lasso(
                        random_state=RANDOM_STATE,
                        max_iter=5000
                    )
                ),

                "DecisionTree": self.make_model(
                    DecisionTreeRegressor(
                        random_state=RANDOM_STATE
                    )
                ),

                "RandomForest": self.make_model(
                    RandomForestRegressor(
                        random_state=RANDOM_STATE,
                        n_jobs=-1
                    )
                )
            }

            baseline_results = []

            logging.info("Training baseline models...")

            for name, model in baseline_models.items():

                logging.info(f"Training {name}")

                model.fit(
                    X_train_t,
                    y_train
                )

                preds = model.predict(X_test_t)

                baseline_results.append({

                    "Model": name,

                    "R2": r2_score(
                        y_test,
                        preds
                    ),

                    "MAE": mean_absolute_error(
                        y_test,
                        preds
                    ),

                    "RMSE": np.sqrt(
                        mean_squared_error(
                            y_test,
                            preds
                        )
                    )

                })

            baseline_df = (
                pd.DataFrame(baseline_results)
                .sort_values(
                    "R2",
                    ascending=False
                )
                .reset_index(drop=True)
            )

            best_model_name = baseline_df.iloc[0]["Model"]

            logging.info(
                f"Best Baseline Model : {best_model_name}"
            )

            param_distributions = {

                "Ridge": {

                    "regressor__alpha":
                    uniform(0.01, 300)

                },

                "Lasso": {

                    "regressor__alpha":
                    uniform(0.001, 30)

                },

                "DecisionTree": {

                    "regressor__max_depth":
                    [5, 7, 10, 12, 15, 18, 20],

                    "regressor__min_samples_split":
                    randint(2, 25),

                    "regressor__min_samples_leaf":
                    randint(1, 12),

                    "regressor__max_features":
                    ["sqrt", "log2", None]

                },

                "RandomForest": {

                    "regressor__n_estimators":
                    randint(100, 350),

                    "regressor__max_depth":
                    [8, 10, 12, 15, 18, 20],

                    "regressor__min_samples_split":
                    randint(2, 20),

                    "regressor__min_samples_leaf":
                    randint(2, 10),

                    "regressor__max_features":
                    ["sqrt", "log2", 0.5, 0.7],

                    "regressor__bootstrap":
                    [True, False]

                }

            }

            estimator_lookup = {

                "Ridge":
                Ridge(random_state=RANDOM_STATE),

                "Lasso":
                Lasso(
                    random_state=RANDOM_STATE,
                    max_iter=5000
                ),

                "DecisionTree":
                DecisionTreeRegressor(
                    random_state=RANDOM_STATE
                ),

                "RandomForest":
                RandomForestRegressor(
                    random_state=RANDOM_STATE,
                    n_jobs=-1
                )

            }
            if best_model_name == "LinearRegression":

                final_model = baseline_models["LinearRegression"]
                best_params = "N/A"
                best_cv_score = None

                logging.info(
                    "LinearRegression selected. No hyperparameter tuning required."
                )

            else:

                logging.info(
                    f"Tuning {best_model_name}..."
                )

                wrapped_estimator = self.make_model(
                    estimator_lookup[best_model_name]
                )

                final_model, best_params, best_cv_score = self.tune_model(
                    wrapped_estimator,
                    param_distributions[best_model_name],
                    X_train_t,
                    y_train
                )

            logging.info("Evaluating final model...")

            preds = final_model.predict(X_test_t)

            r2 = r2_score(y_test, preds)
            mae = mean_absolute_error(y_test, preds)
            mse = mean_squared_error(y_test, preds)
            rmse = np.sqrt(mse)

            final_results_df = pd.DataFrame([{

                "Model": f"{best_model_name} (Tuned)",
                "R2": r2,
                "MAE": mae,
                "MSE": mse,
                "RMSE": rmse

            }])

            final_results_df.to_csv(
                "reports/metrics.csv",
                index=False
            )

            logging.info(
                "Metrics saved successfully."
            )

            with open(
                "reports/best_params.json",
                "w"
            ) as file:

                json.dump({

                    "best_model": best_model_name,
                    "best_params": str(best_params),
                    "best_cv_r2": best_cv_score

                },
                file,
                indent=4)

            logging.info(
                "Best parameters saved successfully."
            )

            save_object(
                self.model_path,
                final_model
            )

            logging.info(
                "Final model saved successfully."
            )

            logging.info(
                f"""
            ==========================
            Training Completed
            ==========================
            Best Model : {best_model_name}
            R2 Score   : {r2:.4f}
            MAE        : {mae:.2f}
            RMSE       : {rmse:.2f}
            ==========================
            """
                        )

            return final_model

        except Exception as e:

            logging.error(
                f"Training failed : {str(e)}"
            )

            raise CustomException(
                e,
                sys
            )


if __name__ == "__main__":

    trainer = ModelTrainer()

    trainer.train()

    print("Model Training Completed Successfully.")