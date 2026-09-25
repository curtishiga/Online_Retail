import streamlit as st
import pandas as pd
import numpy as np
import joblib

MODEL_PATH = "Models/XGBoost/xgb_best_estim.joblib"
xgb_model = joblib.load(MODEL_PATH)

st.title("XGBoost Churn Prediction Dashboard")

st.header("Input customer order details")

order = {
    "RollingTotalPrice": st.number_input("Rolling Total Price ($)",
                                         min_value=0.0,
                                         step = 0.01),
    "DaysSinceLastPurchase": st.number_input("Days Since Last Purchase",
                                             min_value = 0,
                                             step = 1),
    "RollingNumInvoices": st.number_input("Rolling Number of Invoices",
                                          min_value = 0,
                                          step = 1),
    "InvoiceDate": st.date_input("Invoice Date",
                                 min_value=pd.to_datetime("2010-12-01")),
    "FirstInvoiceDate": st.date_input("First Invoice Date",
                                      min_value=pd.to_datetime("2010-12-01")),
    "Country": st.selectbox("Country",
                            ["United Kingdom", "Germany", "France", "EIRE","Other"])
}

if st.button("Predict Churn"):
    row = pd.DataFrame([order])
    prediction = xgb_model.predict(row)
    predict_prob = xgb_model.predict_proba(row)

    st.write(f"Churn Prediction: {'Yes' if prediction[0] == 1 else 'No'}")
    st.write(f"Churn Probability: {predict_prob[0][1]:.2%}")