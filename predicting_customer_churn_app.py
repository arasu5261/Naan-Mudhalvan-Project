import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import MinMaxScaler

# Load the trained model
model = joblib.load("churn_model_xgb.pkl")

# Define the preprocessing function
def preprocess_input(input_data, scaler, feature_columns):
    # Convert input dictionary to DataFrame
    df = pd.DataFrame([input_data])
    
    # Map binary columns
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        df[col] = df[col].map({'Yes': 1, 'No': 0})
    
    # One-hot encode categorical columns
    categorical_cols = ['gender', 'MultipleLines', 'InternetService', 'OnlineSecurity', 
                        'OnlineBackup', 'DeviceProtection', 'TechSupport', 'StreamingTV', 
                        'StreamingMovies', 'Contract', 'PaymentMethod']
    df = pd.get_dummies(df, columns=categorical_cols, drop_first=True)
    
    # Ensure all expected columns are present
    for col in feature_columns:
        if col not in df.columns:
            df[col] = 0
    
    # Reorder columns to match training data
    df = df[feature_columns]
    
    # Scale numerical columns
    df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(
        df[['tenure', 'MonthlyCharges', 'TotalCharges']]
    )
    
    return df

# Streamlit app
st.title("Customer Churn Prediction")

st.write("Enter customer details to predict churn likelihood.")

# Collect user inputs
with st.form("churn_form"):
    col1, col2 = st.columns(2)
    
    with col1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
        Partner = st.selectbox("Partner", ["Yes", "No"])
        Dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure (months)", 0, 72, 12)
        PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
        MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
        InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
    
    with col2:
        OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
        OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
        DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
        TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
        StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
        StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
        Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
        PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
        PaymentMethod = st.selectbox("Payment Method", [
            "Electronic check", "Mailed check", "Bank transfer (automatic)", 
            "Credit card (automatic)"
        ])
    
    MonthlyCharges = st.number_input("Monthly Charges ($)", min_value=0.0, value=50.0)
    TotalCharges = st.number_input("Total Charges ($)", min_value=0.0, value=500.0)
    
    submitted = st.form_submit_button("Predict")

# Define feature columns (based on training data after preprocessing)
feature_columns = [
    'SeniorCitizen', 'Partner', 'Dependents', 'tenure', 'PhoneService', 'PaperlessBilling',
    'MonthlyCharges', 'TotalCharges', 'gender_Male', 'MultipleLines_No phone service',
    'MultipleLines_Yes', 'InternetService_Fiber optic', 'InternetService_No',
    'OnlineSecurity_No internet service', 'OnlineSecurity_Yes', 
    'OnlineBackup_No internet service', 'OnlineBackup_Yes', 
    'DeviceProtection_No internet service', 'DeviceProtection_Yes', 
    'TechSupport_No internet service', 'TechSupport_Yes', 
    'StreamingTV_No internet service', 'StreamingTV_Yes', 
    'StreamingMovies_No internet service', 'StreamingMovies_Yes', 
    'Contract_One year', 'Contract_Two year', 'PaymentMethod_Credit card (automatic)',
    'PaymentMethod_Electronic check', 'PaymentMethod_Mailed check'
]

# Initialize scaler (fitted on training data ranges)
scaler = MinMaxScaler()
# Assuming training data ranges (you can adjust these if you have the exact ranges)
scaler.fit([
    [0, 0, 0],  # Min: tenure, MonthlyCharges, TotalCharges
    [72, 200, 10000]  # Max: tenure, MonthlyCharges, TotalCharges
])

# Process inputs and predict
if submitted:
    # Collect input data
    input_data = {
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'MultipleLines': MultipleLines,
        'InternetService': InternetService,
        'OnlineSecurity': OnlineSecurity,
        'OnlineBackup': OnlineBackup,
        'DeviceProtection': DeviceProtection,
        'TechSupport': TechSupport,
        'StreamingTV': StreamingTV,
        'StreamingMovies': StreamingMovies,
        'Contract': Contract,
        'PaperlessBilling': PaperlessBilling,
        'PaymentMethod': PaymentMethod,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges
    }
    
    # Preprocess input
    processed_input = preprocess_input(input_data, scaler, feature_columns)
    
    # Make prediction
    prediction = model.predict(processed_input)[0]
    probability = model.predict_proba(processed_input)[0][1]
    
    # Display results
    st.subheader("Prediction Result")
    if prediction == 1:
        st.error(f"The customer is likely to churn (Probability: {probability:.2%})")
    else:
        st.success(f"The customer is unlikely to churn (Probability: {probability:.2%})")
