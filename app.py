
import os
import joblib
import pandas as pd
import streamlit as st

# Page settings
st.set_page_config(
    page_title="Starbucks Calories Predictor",
    page_icon="☕",
    layout="centered"
)

st.title("☕ Starbucks Calories Predictor")
st.write("Enter beverage information to estimate its calories.")

# Model file path
MODEL_PATH = "/kaggle/working/starbucks_calorie_model.pkl"

# Check model file
if not os.path.isfile(MODEL_PATH):
    st.error(
        "Model file not found. Please save "
        "starbucks_calorie_model.pkl in /kaggle/working."
    )
    st.stop()

# Load model package
@st.cache_resource
def load_model_package(path):
    return joblib.load(path)

try:
    package = load_model_package(MODEL_PATH)
    model = package["model"]
    features = package["features"]
    training_data = package.get("training_data")

    if training_data is None:
        st.error(
            "Training data is missing from the model package. "
            "Please retrain and save the model with training_data."
        )
        st.stop()

except Exception as error:
    st.error(f"Unable to load model: {error}")
    st.stop()

# Validate feature columns
missing_features = [
    feature for feature in features
    if feature not in training_data.columns
]

if missing_features:
    st.error(f"Missing feature columns: {missing_features}")
    st.stop()

# User input form
st.subheader("Enter Beverage Information")
user_values = {}

with st.form("calorie_prediction_form"):

    for feature in features:
        column = training_data[feature]

        if pd.api.types.is_numeric_dtype(column):
            numeric_column = pd.to_numeric(
                column, errors="coerce"
            )
            default_value = numeric_column.median()

            if pd.isna(default_value):
                default_value = 0.0

            user_values[feature] = st.number_input(
                label=str(feature),
                value=float(default_value),
                step=1.0,
                key=f"field_{feature}"
            )

        else:
            options = (
                column.dropna()
                .astype(str)
                .unique()
                .tolist()
            )

            if options:
                user_values[feature] = st.selectbox(
                    label=str(feature),
                    options=options,
                    key=f"field_{feature}"
                )
            else:
                user_values[feature] = ""

    predict_button = st.form_submit_button("Predict Calories")

# Make prediction
if predict_button:
    try:
        input_df = pd.DataFrame(
            [user_values],
            columns=features
        )

        for feature in features:
            if pd.api.types.is_numeric_dtype(
                training_data[feature]
            ):
                input_df[feature] = pd.to_numeric(
                    input_df[feature],
                    errors="coerce"
                )

        prediction = model.predict(input_df)[0]

        st.success(
            f"Estimated Calories: {prediction:.1f} kcal"
        )

        st.caption(
            "This is a machine-learning estimate, "
            "not an official Starbucks nutrition value."
        )

    except Exception as error:
        st.error(f"Prediction failed: {error}")

st.divider()
st.caption("Starbucks Calories Prediction | Machine Learning Project")
