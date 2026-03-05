
import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# ---------------------------------------
# CONFIG
# ---------------------------------------
MODEL_REPO_ID = "SunnyShaurya1981/engine-predictive-maintenance-model"

MODEL_FILE = "random_forest_model.joblib"

# ---------------------------------------
# LOAD MODEL
# ---------------------------------------
@st.cache_resource
def load_model():
    try:
        model_path = hf_hub_download(
            repo_id=MODEL_REPO_ID,
            filename=MODEL_FILE
        )
        model = joblib.load(model_path)
        return model
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None


model = load_model()

# ---------------------------------------
# STREAMLIT PAGE CONFIG
# ---------------------------------------
st.set_page_config(
    page_title="Engine Predictive Maintenance",
    layout="centered"
)

st.title("⚙️ Engine Predictive Maintenance")
st.write(
    "Predict whether an engine is **Healthy or Unhealthy** based on sensor readings."
)

# ---------------------------------------
# INPUT FORM
# ---------------------------------------
with st.form("prediction_form"):

    col1, col2 = st.columns(2)

    with col1:
        engine_rpm = st.number_input("Engine RPM", min_value=0, max_value=3000, value=750)

        lub_oil_pressure = st.number_input(
            "Lub Oil Pressure (MPa)",
            min_value=0.0,
            max_value=10.0,
            value=3.0
        )

        fuel_pressure = st.number_input(
            "Fuel Pressure (MPa)",
            min_value=0.0,
            max_value=30.0,
            value=6.0
        )

    with col2:
        coolant_pressure = st.number_input(
            "Coolant Pressure (MPa)",
            min_value=0.0,
            max_value=10.0,
            value=2.5
        )

        lub_oil_temp = st.number_input(
            "Lub Oil Temperature (°C)",
            min_value=50.0,
            max_value=150.0,
            value=75.0
        )

        coolant_temp = st.number_input(
            "Coolant Temperature (°C)",
            min_value=50.0,
            max_value=150.0,
            value=80.0
        )

    submitted = st.form_submit_button("Predict Engine Condition")

# ---------------------------------------
# PREDICTION
# ---------------------------------------
if submitted:

    if model is None:
        st.error("Model failed to load. Please check Hugging Face logs.")
    else:

        input_data = pd.DataFrame([{
            "Engine rpm": engine_rpm,
            "Lub oil pressure": lub_oil_pressure,
            "Fuel pressure": fuel_pressure,
            "Coolant pressure": coolant_pressure,
            "lub oil temp": lub_oil_temp,
            "Coolant temp": coolant_temp
        }])

        try:
            prediction = model.predict(input_data)[0]
            prediction_proba = model.predict_proba(input_data)[0]

            st.subheader("Prediction Result")

            if prediction == 0:
                st.success("✅ Engine Condition: **Healthy**")
                st.metric("Confidence (Healthy)", f"{prediction_proba[0]*100:.2f}%")
                st.metric("Confidence (Unhealthy)", f"{prediction_proba[1]*100:.2f}%")

            else:
                st.warning("⚠️ Engine Condition: **Unhealthy**")
                st.metric("Confidence (Unhealthy)", f"{prediction_proba[1]*100:.2f}%")
                st.metric("Confidence (Healthy)", f"{prediction_proba[0]*100:.2f}%")

        except Exception as e:
            st.error(f"Prediction error: {e}")

st.caption("⚠️ Machine Learning prediction for decision support only.")
