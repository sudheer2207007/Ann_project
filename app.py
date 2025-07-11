import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

# --- PAGE CONFIG ---
st.set_page_config(
    page_title="📊 Churn Predictor - Data Analyst Style",
    layout="centered",
    initial_sidebar_state="auto"
)

# --- BACKGROUND STYLE ---
def set_background():
    page_bg_style = '''
    <style>
    body {
        background: linear-gradient(to right, #74ebd5, #ACB6E5);
        color: #0f2027;
        font-family: 'Segoe UI', sans-serif;
    }
    .stApp {
        background-color: rgba(255,255,255,0.9);
        padding: 2rem;
        border-radius: 12px;
        box-shadow: 0 0 30px rgba(0,0,0,0.2);
    }
    .stSelectbox > div, .stNumberInput > div, .stSlider > div {
        background-color: #ffffff30 !important;
        border-radius: 10px;
    }
    </style>
    '''
    st.markdown(page_bg_style, unsafe_allow_html=True)

set_background()

# --- HEADER ---
st.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=80)
st.title("🔍 Customer Churn Prediction")
st.subheader("📈 Smart Dashboard for Data Analysts")
st.markdown("Fill in the customer's information below to check if they are likely to churn.")

# --- Load Model and Preprocessors ---
model = tf.keras.models.load_model('model.h5', compile=False)

with open('label_encoder_gender.pkl', 'rb') as file:
    label_encoder_gender = pickle.load(file)

with open('onehot_encoder_geo.pkl', 'rb') as file:
    onehot_encoder_geo = pickle.load(file)

with open('scaler.pk1', 'rb') as file:
    scaler = pickle.load(file)

# --- User Input ---
geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
gender = st.selectbox('🧑 Gender', label_encoder_gender.classes_)
age = st.slider('🎂 Age', 18, 92, 30)
balance = st.number_input('💰 Balance', value=0.0)
credit_score = st.number_input('💳 Credit Score', value=650)
estimated_salary = st.number_input('📊 Estimated Salary', value=50000.0)
tenure = st.slider('📅 Tenure (Years)', 0, 10, 3)
num_of_products = st.slider('🛍️ Number of Products', 1, 4, 1)
has_cr_card = st.selectbox('💳 Has Credit Card?', [0, 1])
is_active_member = st.selectbox('🔥 Is Active Member?', [0, 1])

# --- Prepare Input ---
input_data = pd.DataFrame({
    'CreditScore': [credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age': [age],
    'Tenure': [tenure],
    'Balance': [balance],
    'NumOfProducts': [num_of_products],
    'HasCrCard': [has_cr_card],
    'IsActiveMember': [is_active_member],
    'EstimatedSalary': [estimated_salary]
})

geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

# --- Scale and Predict ---
input_data_scaled = scaler.transform(input_data)
prediction = model.predict(input_data_scaled)
prediction_proba = prediction[0][0]

# --- Show Result ---
st.markdown("### 🎯 Prediction Result")
if prediction_proba > 0.5:
    st.error(f'⚠️ The customer is likely to churn.\n\n**Churn Probability:** `{prediction_proba:.2f}`')
else:
    st.success(f'✅ The customer is not likely to churn.\n\n**Churn Probability:** `{prediction_proba:.2f}`')
