"""
app.py

Streamlit dashboard for the Car Price Prediction capstone project.

- Uses the EXISTING trained pipeline only (src/predict.py: PredictPipeline,
  CustomData). No retraining, no changes to preprocessing/model/pipeline.
- Section 1: Prediction form -> predicted price in a highlighted box.
- Section 2: Model Performance dashboard (metrics + actual vs predicted
  chart + scrollable prediction table with CSV download).

Run with:
    streamlit run app.py
"""

from datetime import datetime

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

from src.utils import logger, BASE_DIR
from src.predict import PredictPipeline, CustomData


# =============================================================================
# PAGE CONFIG
# =============================================================================
st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="Car",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Light custom styling on top of Streamlit's own theme (works in light & dark
# mode since it doesn't hardcode background colors, only spacing/accents).
st.markdown("""
    <style>
    .main .block-container { padding-top: 2rem; max-width: 1100px; }
    div[data-testid="stMetric"] {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 10px;
        padding: 14px 16px;
    }
    div[data-testid="stForm"] {
        border: 1px solid rgba(128,128,128,0.25);
        border-radius: 12px;
        padding: 1.5rem;
    }
    .stTabs [data-baseweb="tab"] { font-weight: 500; padding: 8px 18px; }
    </style>
""", unsafe_allow_html=True)


# =============================================================================
# SIDEBAR
# =============================================================================
with st.sidebar:
    st.markdown("## Car Price Predictor")
    st.divider()

    st.markdown("**Project**")
    st.write("Car Price Prediction")

    st.markdown("**Model**")
    st.write("Tuned Regression Model (RandomizedSearchCV)")

    st.markdown("**Dataset**")
    st.write("Used Car Price Dataset (Kaggle)")

    st.divider()

    st.markdown("**Developer**")
    st.write("Ch Yogesh")
    st.caption("B.Tech CSE, RVS College of Engineering & Technology")
    st.caption("AI & ML Internship - PaulTech Software Services")

    st.divider()
    st.caption("Built with scikit-learn, pandas & Streamlit")


# =============================================================================
# HEADER
# =============================================================================
st.title("Car Price Prediction Dashboard")
st.caption("Estimate a used car's resale price and review model performance on unseen test data.")
st.divider()

PROCESSED_DIR = BASE_DIR / "data" / "processed"


@st.cache_data
def load_alternatives_pool():
    """
    Loads a pool of real cars (with actual prices) to compare a prediction
    against. Prefers the full cleaned dataset (car_cleaned.csv) saved by
    02_Preprocessing.ipynb; falls back to X_test + y_test if that file
    isn't present.
    """
    cleaned_path = PROCESSED_DIR / "car_cleaned.csv"
    if cleaned_path.exists():
        pool = pd.read_csv(cleaned_path)
    else:
        X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
        y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv")
        pool = X_test.copy()
        pool["price"] = y_test.values.ravel()
    return pool


def find_better_alternatives(pool: pd.DataFrame, predicted_price: float,
                              tolerance: float = 0.15, top_n: int = 5) -> pd.DataFrame:
    """
    Finds cars priced within +/- `tolerance` of the predicted price, then
    ranks them by a simple "value" score: newer cars (lower car_age) and
    lower mileage rank better. Widens the price band once if too few
    matches are found.
    """
    def _filter(band):
        lo, hi = predicted_price * (1 - band), predicted_price * (1 + band)
        return pool[(pool["price"] >= lo) & (pool["price"] <= hi)].copy()

    candidates = _filter(tolerance)
    if len(candidates) < 3:
        candidates = _filter(tolerance * 2)

    if candidates.empty:
        return candidates

    # normalize mileage and car_age to 0-1 so both contribute equally
    for col in ["mileage", "car_age"]:
        if col in candidates.columns:
            col_range = candidates[col].max() - candidates[col].min()
            candidates[f"_{col}_norm"] = (
                0.0 if col_range == 0
                else (candidates[col] - candidates[col].min()) / col_range
            )
        else:
            candidates[f"_{col}_norm"] = 0.0

    candidates["value_score"] = candidates["_mileage_norm"] + candidates["_car_age_norm"]
    candidates = candidates.sort_values("value_score").head(top_n)

    display_cols = [c for c in [
        "manufacturer", "category", "prod_year", "mileage",
        "fuel_type", "gear_box_type", "price"
    ] if c in candidates.columns]

    return candidates[display_cols].round(0)


# =============================================================================
# SECTION 1 - PREDICTION
# =============================================================================
st.header("Predict Car Price")

with st.form("prediction_form"):
    col1, col2, col3 = st.columns(3)

    with col1:
        manufacturer = st.text_input("Manufacturer", placeholder="e.g. TOYOTA")
        category = st.selectbox(
            "Category",
            ["Sedan", "Jeep", "Hatchback", "Microbus", "Goods wagon",
             "Universal", "Coupe", "Minivan", "Cabriolet", "Limousine"],
        )
        prod_year = st.number_input("Production Year", min_value=1950, max_value=2026, value=2015)
        levy = st.number_input("Levy", min_value=0.0, value=800.0, step=10.0)
        mileage = st.number_input("Mileage (km)", min_value=0, value=100000, step=1000)

    with col2:
        engine_volume = st.number_input("Engine Volume (L)", min_value=0.1, max_value=10.0, value=2.0, step=0.1)
        is_turbo = st.selectbox("Turbo", ["No", "Yes"])
        cylinders = st.number_input("Cylinders", min_value=1, max_value=16, value=4)
        gear_box_type = st.selectbox("Gear Box Type", ["Automatic", "Tiptronic", "Manual", "Variator"])
        drive_wheels = st.selectbox("Drive Wheels", ["Front", "Rear", "4x4"])
        fuel_type = st.selectbox(
            "Fuel Type",
            ["Petrol", "Diesel", "CNG", "Hybrid", "Plug-in Hybrid", "LPG", "Hydrogen"],
        )

    with col3:
        # NOTE: matches the bucketed categories the preprocessor was
        # actually fit on ('2-3', '4-5', '>5') -- do not change without
        # re-checking the fitted OneHotEncoder categories.
        doors = st.selectbox("Doors", ["2-3", "4-5", ">5"])
        wheel = st.selectbox("Wheel", ["Left wheel", "Right-hand drive"])
        color = st.selectbox(
            "Color",
            ["Silver", "Black", "White", "Grey", "Blue", "Red",
             "Green", "Yellow", "Brown", "Orange", "Golden", "Purple", "Sky blue"],
        )
        leather_interior = st.selectbox("Leather Interior", ["Yes", "No"])
        airbags = st.number_input("Airbags", min_value=0, max_value=16, value=6)

    submitted = st.form_submit_button("Predict Price", use_container_width=True)

if submitted:
    car_age = datetime.now().year - prod_year
    is_turbo_flag = 1 if is_turbo == "Yes" else 0

    try:
        custom_data = CustomData(
            levy=levy,
            prod_year=prod_year,
            manufacturer=manufacturer,
            category=category,
            leather_interior=leather_interior,
            fuel_type=fuel_type,
            engine_volume=engine_volume,
            mileage=mileage,
            cylinders=cylinders,
            gear_box_type=gear_box_type,
            drive_wheels=drive_wheels,
            doors=doors,
            wheel=wheel,
            color=color,
            airbags=airbags,
            is_turbo=is_turbo_flag,
            car_age=car_age,
        )

        input_df = custom_data.get_data_as_dataframe()

        pipeline = PredictPipeline()
        prediction = pipeline.predict(input_df)[0]

        st.success(f"### Estimated Price: $ {prediction:,.2f}")
        st.caption(f"{manufacturer or 'Vehicle'} - {prod_year} - {mileage:,} km - {category}")

        # ---- Better alternatives in the same price range ----
        st.subheader("Better Alternatives in This Price Range")
        try:
            pool = load_alternatives_pool()
            alternatives = find_better_alternatives(pool, prediction)

            if alternatives.empty:
                st.info("No comparable listings found in this price range in the available data.")
            else:
                st.caption(
                    "Cars priced close to your estimate, ranked by lower mileage and newer "
                    "production year (better value for a similar price)."
                )
                st.dataframe(alternatives, use_container_width=True, hide_index=True)
        except FileNotFoundError as e:
            st.info("Alternative suggestions unavailable: reference data not found.")
            logger.warning(str(e))

    except Exception as e:
        st.error(f"Prediction failed: {e}")
        logger.error(f"Streamlit prediction error: {e}")

st.divider()


# =============================================================================
# SECTION 2 - MODEL PERFORMANCE DASHBOARD
# =============================================================================
st.header("Model Performance")


@st.cache_data
def load_test_data():
    X_test = pd.read_csv(PROCESSED_DIR / "X_test.csv")
    y_test = pd.read_csv(PROCESSED_DIR / "y_test.csv").values.ravel()
    return X_test, y_test


@st.cache_data
def get_predictions_on_test(X_test, y_test):
    pipeline = PredictPipeline()
    y_pred = pipeline.predict(X_test)
    return y_pred


try:
    X_test, y_test = load_test_data()
    y_pred = get_predictions_on_test(X_test, y_test)

    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)

    # ---- Metric cards ----
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("R2 Score", f"{r2:.3f}")
    m2.metric("MAE", f"$ {mae:,.0f}")
    m3.metric("MSE", f"{mse:,.2e}")
    m4.metric("RMSE", f"$ {rmse:,.0f}")

    st.divider()

    # ---- Actual vs Predicted chart ----
    chart_col, table_col = st.columns([1, 1])

    with chart_col:
        st.subheader("Actual vs Predicted")

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.scatter(y_test, y_pred, alpha=0.35, s=18, color="#2563EB", edgecolors="none")

        lims = [min(y_test.min(), y_pred.min()), max(y_test.max(), y_pred.max())]
        ax.plot(lims, lims, "--", color="red", linewidth=1.5, label="Perfect Prediction")

        ax.set_xlabel("Actual Price")
        ax.set_ylabel("Predicted Price")
        ax.set_title("Actual vs Predicted Car Prices", fontweight="bold")
        ax.spines[["top", "right"]].set_visible(False)
        ax.legend(frameon=False)
        fig.tight_layout()

        st.pyplot(fig)

    # ---- Prediction table + CSV download ----
    with table_col:
        st.subheader("Prediction Table")

        results_df = pd.DataFrame({
            "Actual Price": y_test,
            "Predicted Price": y_pred,
        }).round(2)

        st.dataframe(results_df.head(100), height=420, use_container_width=True)

        csv_bytes = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download Predictions as CSV",
            data=csv_bytes,
            file_name="predictions.csv",
            mime="text/csv",
            use_container_width=True,
        )

except FileNotFoundError as e:
    st.warning(
        f"Test data not found at {PROCESSED_DIR}. "
        "Make sure X_test.csv and y_test.csv exist in data/processed/."
    )
    logger.warning(str(e))

st.divider()
st.caption("Car Price Prediction - AI & ML Internship Capstone - PaulTech Software Services")