import sys
import pandas as pd

from src.utils import (
    logger,
    CustomException,
    load_object,
    MODEL_PATH,
    PREPROCESSOR_PATH
)


class PredictPipeline:

    def __init__(self):
        self.model_path = MODEL_PATH
        self.preprocessor_path = PREPROCESSOR_PATH

    def predict(self, features):

        try:

            logger.info("Loading preprocessor...")

            preprocessor = load_object(self.preprocessor_path)

            logger.info("Preprocessor loaded successfully.")

            logger.info("Loading trained model...")

            model = load_object(self.model_path)

            logger.info("Model loaded successfully.")

            logger.info("Transforming input data...")

            data_scaled = preprocessor.transform(features)

            logger.info("Prediction started...")

            prediction = model.predict(data_scaled)

            logger.info("Prediction completed successfully.")

            return prediction

        except Exception as e:

            logger.error(f"Prediction Error: {str(e)}")

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

            logger.info("Input dataframe created successfully.")

            return df

        except Exception as e:

            logger.error(f"DataFrame Creation Error: {str(e)}")

            raise CustomException(e, sys)


if __name__ == "__main__":

    sample = CustomData(

        levy=1399,
        prod_year=2018,
        manufacturer="TOYOTA",
        category="Sedan",
        leather_interior="Yes",
        fuel_type="Petrol",
        engine_volume=2.0,
        mileage=45000,
        cylinders=4,
        gear_box_type="Automatic",
        drive_wheels="Front",
        doors="4",
        wheel="Left wheel",
        color="Black",
        airbags=8,
        is_turbo=0,
        car_age=7

    )

    data = sample.get_data_as_dataframe()

    predictor = PredictPipeline()

    prediction = predictor.predict(data)

    print(f"Predicted Car Price: {prediction[0]:.2f}")