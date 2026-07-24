import sys
import pandas as pd

from src.utils import load_object
from src.logger import logging
from src.exception import CustomException
from src.config import MODEL_PATH, PREPROCESSOR_PATH

class PredictPipeline:

    def __init__(self):
        self.model_path = MODEL_PATH
        self.preprocessor_path = PREPROCESSOR_PATH

    def predict(self, features):

        try:

            logging.info("Loading preprocessor...")

            preprocessor = load_object(self.preprocessor_path)

            logging.info("Preprocessor loaded successfully.")

            logging.info("Loading trained model...")

            model = load_object(self.model_path)

            logging.info("Model loaded successfully.")

            logging.info("Transforming input data...")

            data_scaled = preprocessor.transform(features)

            logging.info("Prediction started...")

            prediction = model.predict(data_scaled)

            logging.info("Prediction completed successfully.")

            return prediction

        except Exception as e:

            logging.error(f"Prediction Error: {str(e)}")

            raise CustomException(e, sys)


class CustomData:

    def __init__(
        self,
        levy,
        prod_year,
        manufacturer,
        category,
        leather_interior,
        fuel_type,
        engine_volume,
        mileage,
        cylinders,
        gear_box_type,
        drive_wheels,
        doors,
        wheel,
        color,
        airbags,
        is_turbo,
        car_age
    ):

        self.levy = levy
        self.prod_year = prod_year
        self.manufacturer = manufacturer
        self.category = category
        self.leather_interior = leather_interior
        self.fuel_type = fuel_type
        self.engine_volume = engine_volume
        self.mileage = mileage
        self.cylinders = cylinders
        self.gear_box_type = gear_box_type
        self.drive_wheels = drive_wheels
        self.doors = doors
        self.wheel = wheel
        self.color = color
        self.airbags = airbags
        self.is_turbo = is_turbo
        self.car_age = car_age

    def get_data_as_dataframe(self):

        try:

            custom_data = {

                "levy": [self.levy],
                "prod_year": [self.prod_year],
                "manufacturer": [self.manufacturer],
                "category": [self.category],
                "leather_interior": [self.leather_interior],
                "fuel_type": [self.fuel_type],
                "engine_volume": [self.engine_volume],
                "mileage": [self.mileage],
                "cylinders": [self.cylinders],
                "gear_box_type": [self.gear_box_type],
                "drive_wheels": [self.drive_wheels],
                "doors": [self.doors],
                "wheel": [self.wheel],
                "color": [self.color],
                "airbags": [self.airbags],
                "is_turbo": [self.is_turbo],
                "car_age": [self.car_age]

            }

            df = pd.DataFrame(custom_data)

            logging.info("Input dataframe created successfully.")

            return df

        except Exception as e:

            logging.error(f"DataFrame Creation Error: {str(e)}")

            raise CustomException(e, sys)


if __name__ == "__main__":

    sample = CustomData(

        levy=919.0,
        prod_year=2012,
        manufacturer="MERCEDES-BENZ",
        category="Jeep",
        leather_interior="Yes",
        fuel_type="Diesel",
        engine_volume=3.0,
        mileage=140200,
        cylinders=6.0,
        gear_box_type="Automatic",
        drive_wheels="4x4",
        doors="2-3",
        wheel="Left wheel",
        color="Grey",
        airbags=10,
        is_turbo=1,
        car_age=14

    )

    data = sample.get_data_as_dataframe()

    predictor = PredictPipeline()

    prediction = predictor.predict(data)

    print(f"Predicted Car Price: {prediction[0]:.2f}")