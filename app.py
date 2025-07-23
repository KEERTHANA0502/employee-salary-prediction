import streamlit as st
import pandas as pd
import joblib
import os

# Get current directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Load the model and all encoders
model = joblib.load(os.path.join(current_dir, 'best_model (1).pkl'))
scaler = joblib.load(os.path.join(current_dir, 'scaler (1).pkl'))
target_encoder = joblib.load(os.path.join(current_dir, 'target_encoder.pkl'))

# Load encoders for categorical columns
education_encoder = joblib.load(os.path.join(current_dir, 'education_encoder.pkl'))
marital_status_encoder = joblib.load(os.path.join(current_dir, 'marital-status_encoder.pkl'))
occupation_encoder = joblib.load(os.path.join(current_dir, 'occupation_encoder.pkl'))
relationship_encoder = joblib.load(os.path.join(current_dir, 'relationship_encoder.pkl'))
race_encoder = joblib.load(os.path.join(current_dir, 'race_encoder.pkl'))
workclass_encoder = joblib.load(os.path.join(current_dir, 'workclass_encoder.pkl'))
native_country_encoder = joblib.load(os.path.join(current_dir, 'native-country_encoder.pkl'))

model_columns = joblib.load(os.path.join(current_dir, 'model_columns.pkl'))

st.title("Employee Salary Prediction App 💼")

# User inputs
age = st.number_input("Age", min_value=18, max_value=90, value=30)
workclass = st.selectbox("Workclass", workclass_encoder.classes_)
fnlwgt = st.number_input("FNLWGT", min_value=0, value=100000)
education = st.selectbox("Education", education_encoder.classes_)
educational_num = st.slider("Educational Number", 1, 16, 10)
marital_status = st.selectbox("Marital Status", marital_status_encoder.classes_)
occupation = st.selectbox("Occupation", occupation_encoder.classes_)
relationship = st.selectbox("Relationship", relationship_encoder.classes_)
race = st.selectbox("Race", race_encoder.classes_)
gender = st.selectbox("Gender", ["Male", "Female"])
capital_gain = st.number_input("Capital Gain", value=0)
capital_loss = st.number_input("Capital Loss", value=0)
hours_per_week = st.slider("Hours per Week", 1, 100, 40)
native_country = st.selectbox("Native Country", native_country_encoder.classes_)

# Encode inputs
input_data = {
    "age": age,
    "workclass": workclass_encoder.transform([workclass])[0],
    "fnlwgt": fnlwgt,
    "education": education_encoder.transform([education])[0],
    "educational-num": educational_num,
    "marital-status": marital_status_encoder.transform([marital_status])[0],
    "occupation": occupation_encoder.transform([occupation])[0],
    "relationship": relationship_encoder.transform([relationship])[0],
    "race": race_encoder.transform([race])[0],
    "gender": 1 if gender == "Male" else 0,
    "capital-gain": capital_gain,
    "capital-loss": capital_loss,
    "hours-per-week": hours_per_week,
    "native-country": native_country_encoder.transform([native_country])[0],
}

# Create DataFrame
input_df = pd.DataFrame([input_data])

# Reorder columns to match training data
input_df = input_df[model_columns]

# Scale input
input_scaled = scaler.transform(input_df)

# Predict
if st.button("Predict Salary"):
    prediction = model.predict(input_scaled)
    prediction_label = target_encoder.inverse_transform(prediction)[0]
    st.success(f"Predicted Salary Category: **{prediction_label}**")
