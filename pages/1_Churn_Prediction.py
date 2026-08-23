"""
1_Churn_Prediction.py

A page in the InsightHub multi-page Streamlit app.
Streamlit automatically turns any .py file inside a "pages/" folder
into a new page in the sidebar navigation — no extra config needed.

Lets a user fill in a customer's details and get a live churn prediction
from the model trained in Phase 2.
"""

from pathlib import Path
import sys

import joblib
import pandas as pd
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "data" / "processed" / "churn_best_model.joblib"

st.set_page_config(page_title="Churn Prediction", page_icon="📉")
st.title("📉 Customer Churn Predictor")
st.caption("Fill in a customer's details to predict their likelihood of churning.")


@st.cache_resource
def load_churn_model():
    return joblib.load(MODEL_PATH)


model = load_churn_model()

with st.form("churn_form"):
    col1, col2 = st.columns(2)

    with col1:
        tenure = st.number_input("Tenure (months)", min_value=0, max_value=100, value=12)
        monthlycharges = st.number_input("Monthly Charges ($)", min_value=0.0, value=70.0)
        totalcharges = st.number_input("Total Charges ($)", min_value=0.0, value=840.0)
        contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
        internetservice = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        paymentmethod = st.selectbox(
            "Payment Method",
            ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"],
        )

    with col2:
        gender = st.selectbox("Gender", ["Male", "Female"])
        seniorcitizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Has Partner", ["Yes", "No"])
        dependents = st.selectbox("Has Dependents", ["Yes", "No"])
        paperlessbilling = st.selectbox("Paperless Billing", ["Yes", "No"])
        phoneservice = st.selectbox("Phone Service", ["Yes", "No"])

    submitted = st.form_submit_button("Predict Churn Risk")

if submitted:
    # Build a single-row DataFrame matching the columns the model was trained on.
    # Any columns the model expects that aren't in this form get a reasonable default.
    input_data = pd.DataFrame([{
        "gender": gender,
        "seniorcitizen": seniorcitizen,
        "partner": partner,
        "dependents": dependents,
        "tenure": tenure,
        "phoneservice": phoneservice,
        "multiplelines": "No",
        "internetservice": internetservice,
        "onlinesecurity": "No",
        "onlinebackup": "No",
        "deviceprotection": "No",
        "techsupport": "No",
        "streamingtv": "No",
        "streamingmovies": "No",
        "contract": contract,
        "paperlessbilling": paperlessbilling,
        "paymentmethod": paymentmethod,
        "monthlycharges": monthlycharges,
        "totalcharges": totalcharges,
    }])

    prediction = model.predict(input_data)[0]
    probability = float(model.predict_proba(input_data)[0][1])

    st.divider()
    if prediction == 1:
        st.error(f"⚠️ High churn risk — {probability:.1%} probability")
    else:
        st.success(f"✅ Low churn risk — {probability:.1%} probability")

    st.progress(probability)
    st.caption(
        "This is a live prediction from the XGBoost model trained in Phase 2, "
        "served directly in this UI (no separate API call needed for the demo)."
    )