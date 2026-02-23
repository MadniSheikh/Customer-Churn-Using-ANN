import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
import pandas as pd
import pickle

st.set_page_config(
    page_title="Customer Churn Predictor",
    page_icon="🏦",
    layout="centered",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    /* Mobile  */
    .main-title {
        font-size: 32px !important;
        font-weight: bold !important;
        color: #1E3A8A !important;
        text-align: center !important;
        margin-bottom: 0px !important;
        line-height: 1.2 !important;
    }
    .sub-title {
        font-size: 16px !important;
        color: #6B7280 !important;
        text-align: center !important;
        margin-bottom: 30px !important;
        line-height: 1.4 !important;
    }

    @media (min-width: 768px) {
        .main-title {
            font-size: 52px !important;
        }
        .sub-title {
            font-size: 22px !important;
        }
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    model = tf.keras.models.load_model('./Model/model.h5')
    with open('./encoders/label_encoder_gender.pkl', 'rb') as file:
        label_encoder_gender = pickle.load(file)
    with open('./encoders/OneHot_encoder_geo.pkl', 'rb') as file:
        onehot_encoder_geo = pickle.load(file)
    with open('./encoders/scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)
    return model, label_encoder_gender, onehot_encoder_geo, scaler

model, label_encoder_gender, onehot_encoder_geo, scaler = load_assets()

st.markdown('<p class="main-title">🏦 Customer Churn Predictor</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Analyze bank customer data to predict the likelihood of churning using Deep Learning.</p>', unsafe_allow_html=True)
st.divider()

st.write("### 👤 Customer Profile")
col1, col2, col3 = st.columns(3)

with col1:
    geography = st.selectbox('🌍 Geography', onehot_encoder_geo.categories_[0])
    age = st.slider('🎂 Age', 18, 92, 30)
with col2:
    gender = st.selectbox('🚻 Gender', label_encoder_gender.classes_)
    tenure = st.slider('⏳ Tenure (Years)', 0, 10, 5)
with col3:
    num_of_products = st.slider('📦 Number of Products', 1, 4, 1)

st.write("### 💳 Financial Metrics")
col4, col5 = st.columns(2)

with col4:
    balance = st.number_input('💰 Balance ($)', step=100.0, min_value=0.0, format="%.2f")
    estimated_salary = st.number_input('💵 Estimated Salary ($)', step=100.0, min_value=0.0, format="%.2f")
with col5:
    credit_score = st.number_input('📈 Credit Score', step=10, min_value=300, max_value=900, value=600)
    
    sub_col1, sub_col2 = st.columns(2)
    with sub_col1:
        has_cr_card = st.radio('💳 Credit Card?', ['Yes', 'No'])
    with sub_col2:
        is_active_member = st.radio('🟢 Active Member?', ['Yes', 'No'])

has_cr_card_val = 1 if has_cr_card == 'Yes' else 0
is_active_member_val = 1 if is_active_member == 'Yes' else 0

st.divider()

if st.button('🚀 Predict Churn Risk', use_container_width=True):
    
    with st.spinner('Analyzing data through the Neural Network...'):
        input_data = pd.DataFrame({
            'CreditScore': [credit_score],
            'Gender': [label_encoder_gender.transform([gender])[0]],
            'Age': [age],
            'Tenure': [tenure],
            'Balance': [balance],
            'NumOfProducts': [num_of_products],
            'HasCrCard': [has_cr_card_val],
            'IsActiveMember': [is_active_member_val],
            'EstimatedSalary': [estimated_salary]
        })

        geo_encoded = onehot_encoder_geo.transform([[geography]]).toarray()
        geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))

        input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
        input_data_scaled = scaler.transform(input_data)

        prediction = model.predict(input_data_scaled)
        prediction_proba = prediction[0][0] * 100 

        st.write("### 📊 Prediction Results")
        st.metric(label="Calculated Churn Probability", value=f"{prediction_proba:.2f}%")
        st.progress(int(prediction_proba))

        if prediction_proba > 50:
            st.error(f'⚠️ **High Risk!** This customer is likely to churn. Consider retention strategies.')
        elif prediction_proba > 30:
            st.warning(f'👀 **Moderate Risk.** Keep an eye on this account.')
        else:
            st.success(f'✅ **Safe.** This customer is likely to stay.')