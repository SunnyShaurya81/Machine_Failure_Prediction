import streamlit as st
import pandas as pd
import joblib
from huggingface_hub import hf_hub_download

# ---------------------------------------
# CONFIG
# ---------------------------------------
MODEL_REPO_ID = "SunnyShaurya1981/engine-predictive-maintenance-model"

FILES = {
    "model": "random_forest_model.joblib",
    # No scaler, cat_cols, num_cols, processed_cols for this model as it's a simple RF without complex preprocessing artifacts
    # If the chosen model requires them, these paths should be uncommented and defined.
}

# ---------------------------------------
# LOAD ARTIFACTS
# ---------------------------------------
@st.cache_resource
def load_artifacts():
    model = joblib.load(hf_hub_download(MODEL_REPO_ID, FILES["model"]))
    # Uncomment and load other artifacts if they were part of your model pipeline
    # scaler = joblib.load(hf_hub_download(MODEL_REPO_ID, FILES["scaler"]))
    # categorical_cols = joblib.load(hf_hub_download(MODEL_REPO_ID, FILES["cat_cols"]))
    # numerical_cols = joblib.load(hf_hub_download(MODEL_REPO_ID, FILES["num_cols"]))
    # processed_columns = joblib.load(hf_hub_download(MODEL_REPO_ID, FILES["processed_cols"]))
    return model # , scaler, categorical_cols, numerical_cols, processed_columns


model = load_artifacts() # Unpack other artifacts if returned

# ---------------------------------------
# STREAMLIT UI
# ---------------------------------------
st.set_page_config(page_title="Engine Predictive Maintenance", layout="centered")
st.title("⚙️ Engine Predictive Maintenance")
st.write("Predict if an engine is in a healthy or unhealthy condition based on sensor readings.")

with st.form("prediction_form"):
    col1, col2 = st.columns(2)

    with col1:
        engine_rpm = st.number_input("Engine RPM", min_value=0, max_value=3000, value=750)
        lub_oil_pressure = st.number_input("Lub Oil Pressure (MPa)", min_value=0.0, max_value=10.0, value=3.0, format="%.6f")
        fuel_pressure = st.number_input("Fuel Pressure (MPa)", min_value=0.0, max_value=30.0, value=6.0, format="%.6f")

    with col2:
        coolant_pressure = st.number_input("Coolant Pressure (MPa)", min_value=0.0, max_value=10.0, value=2.5, format="%.6f")
        lub_oil_temp = st.number_input("Lub Oil Temperature (C)", min_value=50.0, max_value=150.0, value=75.0, format="%.6f")
        coolant_temp = st.number_input("Coolant Temperature (C)", min_value=50.0, max_value=150.0, value=80.0, format="%.6f")

    submitted = st.form_submit_button("Predict Engine Condition")

# ---------------------------------------
# PREDICTION
# ---------------------------------------
if submitted:
    # Create a DataFrame from the input values
    input_data = pd.DataFrame([{
        'Engine rpm': engine_rpm,
        'Lub oil pressure': lub_oil_pressure,
        'Fuel pressure': fuel_pressure,
        'Coolant pressure': coolant_pressure,
        'lub oil temp': lub_oil_temp,
        'Coolant temp': coolant_temp
    }])

    # Note: For this specific Random Forest model, if no scaling or one-hot encoding was applied in training,
    # then these steps are not needed here. Otherwise, include them based on your model_build_eval.py.
    # Example of how you would apply preprocessing if needed:
    # if scaler is not None:
    #     input_data[numerical_cols] = scaler.transform(input_data[numerical_cols])
    # if categorical_cols is not None:
    #     input_data_encoded = pd.get_dummies(input_data, columns=categorical_cols, drop_first=True)
    #     input_data_processed = input_data_encoded.reindex(columns=processed_columns, fill_value=0)
    # else:
    #     input_data_processed = input_data

    # In this case, assuming the RF model takes raw numerical input directly (after data preparation steps)
    # If your model_build_eval.py involved scaling, you would need to load and apply the scaler here.
    # For this simplified Random Forest, we assume direct input without further preprocessing artifacts.

    try:
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)[0]

        st.subheader("Prediction Result")
        if prediction == 0:
            st.success("✅ The engine is predicted to be in **Healthy** condition.")
            st.metric("Confidence (Healthy)", f"{prediction_proba[0]*100:.2f}%")
            st.metric("Confidence (Unhealthy)", f"{prediction_proba[1]*100:.2f}%")
        else:
            st.warning("⚠️ The engine is predicted to be in **Unhealthy** condition.")
            st.metric("Confidence (Unhealthy)", f"{prediction_proba[1]*100:.2f}%")
            st.metric("Confidence (Healthy)", f"{prediction_proba[0]*100:.2f}%")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")
        st.write("Please ensure all model artifacts are correctly loaded and input features match the model's expectations.")

    st.caption("⚠️ ML-based prediction for decision support only.")
