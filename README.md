# Car Price Prediction

This project aims to predict the selling price of used cars using Machine Learning regression algorithms. It follows a modular project structure with separate notebooks for data analysis, preprocessing, model training, and evaluation, along with reusable Python modules and a Streamlit web application for making predictions.

## Project Structure

car_price_prediction/
│
├── README.md
├── requirements.txt
├── .gitignore
├── app.py
│
├── src/
│   ├── utils.py
│   ├── preprocessing.py
│   ├── train.py
│   └── predict.py
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│   ├── model.pkl
│   └── preprocessor.pkl
│
├── reports/
│   ├── figures/
│   └── metrics.csv
│
└── notebooks/
├── 01_EDA.ipynb
├── 02_Preprocessing.ipynb
├── 03_Model_Training.ipynb

## Project Workflow

- Data Collection
- Exploratory Data Analysis (EDA)
- Data Preprocessing
- Feature Engineering
- Model Training
- Hyperparameter Tuning (RandomizedSearchCV)
- Model Evaluation
- Model Saving
- Streamlit Deployment

## Machine Learning Models

- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor

Each model (except Linear Regression) is tuned using GridSearchCV with cross-validation to find optimal hyperparameters.

## Evaluation Metrics

- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)
- Cross Validation Score

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Joblib

## Dataset

data\raw\car.csv

## Author

Yogesh
