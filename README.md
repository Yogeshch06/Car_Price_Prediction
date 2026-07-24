# Car Price Prediction

A Machine Learning project that predicts the selling price of used cars using various regression algorithms. The project follows a modular and production-ready structure with separate components for preprocessing, model training, prediction, logging, exception handling, and a Streamlit web application.

---

## Overview

This project includes:

- End-to-end Machine Learning pipeline
- Exploratory Data Analysis (EDA)
- Data preprocessing and feature engineering
- Multiple regression models
- Hyperparameter tuning using RandomizedSearchCV
- Modular project structure
- Logging and custom exception handling
- Streamlit web application
- Model performance dashboard
- CSV prediction download
- Better alternative car recommendations

---

## Project Structure

```text
car_price_prediction/
│
├── app.py
├── README.md
├── requirements.txt
├── .env
├── .gitignore
│
├── data/
│   ├── raw/
│   │   └── car.csv
│   └── processed/
│       ├── X_train.csv
│       ├── X_test.csv
│       ├── y_train.csv
│       ├── y_test.csv
│       └── car_cleaned.csv
│
├── models/
│   ├── model.pkl
│   └── preprocessor.pkl
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_Preprocessing.ipynb
│   └── 03_Model_Training.ipynb
│
├── reports/
│   ├── metrics.csv
│   ├── best_params.json
│   └── figures/
│
├── logs/
│   └── project.log
│
└── src/
    ├── __init__.py
    ├── config.py
    ├── logger.py
    ├── exception.py
    ├── utils.py
    ├── preprocessing.py
    ├── train.py
    └── predict.py
```

---

## Workflow

1. Data Collection
2. Exploratory Data Analysis
3. Data Cleaning
4. Feature Engineering
5. Data Preprocessing
6. Train-Test Split
7. Model Training
8. Hyperparameter Tuning
9. Model Evaluation
10. Model Saving
11. Streamlit Deployment

---

## Machine Learning Models

- Linear Regression
- Ridge Regression
- Lasso Regression
- Decision Tree Regressor
- Random Forest Regressor

The best-performing model is selected after hyperparameter tuning using **RandomizedSearchCV**.

---

## Evaluation Metrics

- R² Score
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Root Mean Squared Error (RMSE)

---

## Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Scikit-learn
- Streamlit
- Joblib
- Python-dotenv

---

## Dataset

Used Car Price Prediction Dataset

```
data/raw/car.csv
```

---

## Installation

Clone the repository:

```bash
git clone https://github.com/YOUR_USERNAME/Car_Price_Prediction.git
cd Car_Price_Prediction
```

Create a virtual environment:

```bash
python -m venv .my_ml_env
```

Activate the environment (Windows):

```bash
.my_ml_env\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Train the model:

```bash
python -m src.train
```

Run the Streamlit application:

```bash
streamlit run app.py
```

---

## Streamlit Dashboard

The application provides:

- Used car price prediction
- Model performance metrics
- Actual vs Predicted visualization
- Prediction table
- CSV download
- Better alternative car recommendations

---

## Author

**Ch Yogesh**

B.Tech in Computer Science and Engineering

RVS College of Engineering & Technology, Jamshedpur

AI & ML Internship – PaulTech Software Services
