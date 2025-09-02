
import streamlit as st
import pickle
import numpy as np
import pandas as pd
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, KFold, cross_val_score
from sklearn import metrics

# Set page config
st.set_page_config(page_title="Medical Insurance Charge Prediction", layout="centered")
st.title("🏥 Medical Insurance Charge Prediction")

Pkl_Filename = "xgb_tuned.pkl"

try:
    with open(Pkl_Filename, 'rb') as file:
        model = pickle.load(file)
except FileNotFoundError:
    st.error("Model file 'xgb_tuned.pkl' not found. Please ensure it's in the same directory.")
    st.stop()

st.write("Enter your details below to get an estimated insurance charge:")

col1, col2 = st.columns(2)

with col1:
    age = st.number_input("Age", min_value=18, max_value=100, value=30)
    bmi = st.number_input("BMI", min_value=10.0, max_value=50.0, value=25.0, format="%.2f")
    children = st.number_input("Number of Children", min_value=0, max_value=10, value=0)
    
with col2:
    sex = st.selectbox("Sex", options=[0, 1], format_func=lambda x: "Female" if x == 0 else "Male")
    smoker = st.selectbox("Smoker Status", options=[0, 1], format_func=lambda x: "Non-Smoker" if x == 0 else "Smoker")
    region = st.selectbox("Region", options=[0, 1, 2, 3], format_func=lambda x: ["Northeast", "Northwest", "Southeast", "Southwest"][x])
    diabetic = st.selectbox("Diabetic", options=[0, 1], format_func=lambda x: "No" if x == 0 else "Yes")

if st.button("🔮 Predict Insurance Charge", type="primary"):
    has_children = 1 if children > 0 else 0
    is_north = 1 if region in [0, 1] else 0
    bmi_age_interaction = bmi * age
    smoker_obese_interaction = 1 if (smoker == 1 and bmi >= 30) else 0
    bmi_children_interaction = bmi * children

    # BMI category
    if bmi < 18.5:
        bmi_category = 0
    elif bmi < 25:
        bmi_category = 1
    elif bmi < 30:
        bmi_category = 2
    else:
        bmi_category = 3

    # Age group
    if age < 30:
        age_group = 0
    elif age < 60:
        age_group = 1
    else:
        age_group = 2

    estimated_charge = (3000 + age * 100 + bmi * 50 + children * 1000)
    if smoker == 1:
        estimated_charge *= 4
    if diabetic == 1:
        estimated_charge *= 1.3
    if bmi >= 30:
        estimated_charge *= 1.2
    if age >= 60:
        estimated_charge *= 1.5
    estimated_charge = max(estimated_charge, 1000)

    log_charges = np.log(estimated_charge)
    high_charges = 1 if estimated_charge > 10000 else 0

    # Build feature array (adjust order as needed)
    features = np.array([age, bmi, children, has_children, is_north, bmi_age_interaction,
                         log_charges, smoker_obese_interaction, high_charges,
                         bmi_children_interaction, diabetic, sex, smoker, region,
                         bmi_category, age_group]).reshape(1, -1)

    try:
        prediction = model.predict(features)
        if prediction < 0:
            st.error("❌ Error: Negative prediction. Please check your inputs.")
        else:
            st.success(f"💰 **Estimated Insurance Charge: ${float(prediction):,.2f}**")
            st.info("📊 **Risk Factors:**")
            risk_factors = []
            if smoker == 1:
                risk_factors.append("🚬 Smoking status")
            if bmi >= 30:
                risk_factors.append("⚖️ High BMI (Obesity)")
            if diabetic == 1:
                risk_factors.append("🩺 Diabetic condition")
            if age >= 60:
                risk_factors.append("👴 Senior age group")
            if risk_factors:
                st.write("Factors increasing your premium:")
                for factor in risk_factors:
                    st.write(f"• {factor}")
            else:
                st.write("✅ Low risk profile detected!")

            

    except Exception as e:
        st.error(f"❌ Prediction Error: {str(e)}")
        st.write("Debug info:", str(features.shape), "features provided")
