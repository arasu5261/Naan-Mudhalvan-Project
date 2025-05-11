import os
import json
import streamlit as st
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# Set the title for the Streamlit app
st.title("Predicting Customer Churn Using Machine Learning")

# Set up Kaggle API credentials using Streamlit secrets
if "kaggle" in st.secrets:
    kaggle_api_token = {
        "username": st.secrets["kaggle"]["username"],
        "key": st.secrets["kaggle"]["key"]
    }
else:
    st.error("Kaggle API credentials not found in Streamlit secrets. Please add them in the app settings.")
    st.stop()

# Create the .kaggle directory and save the API token
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
with open(os.path.expanduser("~/.kaggle/kaggle.json"), "w") as f:
    json.dump(kaggle_api_token, f)
os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)

# Download the dataset using Kaggle API if it doesn't already exist
dataset_path = "data/WA_Fn-UseC_-Telco-Customer-Churn.csv"
if not os.path.exists(dataset_path):
    try:
        os.makedirs("data", exist_ok=True)
        st.write("Downloading dataset from Kaggle...")
        os.system("kaggle datasets download -d blastchar/telco-customer-churn --unzip -p ./data")
        st.success("Dataset downloaded successfully!")
    except Exception as e:
        st.error(f"Failed to download dataset: {e}")
        st.stop()
else:
    st.write("Dataset already exists, skipping download.")

# Load the dataset
try:
    df = pd.read_csv(dataset_path)
    st.success("Dataset loaded successfully!")
except FileNotFoundError:
    st.error("Dataset file not found. Please ensure the download was successful.")
    st.stop()

# Data Preprocessing
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
df.dropna(subset=['TotalCharges'], inplace=True)
df.reset_index(drop=True, inplace=True)

df.drop('customerID', axis=1, inplace=True)

df['Churn'] = df['Churn'].map({'Yes': 1, 'No': 0})

binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in binary_cols:
    df[col] = df[col].map({'Yes': 1, 'No': 0})

df = pd.get_dummies(df, drop_first=True)

# Scale numerical features
scaler = MinMaxScaler()
df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.fit_transform(df[['tenure', 'MonthlyCharges', 'TotalCharges']])

# Split data into features and target
X = df.drop('Churn', axis=1)
y = df['Churn']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

# Train and evaluate models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000),
    'Decision Tree': DecisionTreeClassifier(),
    'Random Forest': RandomForestClassifier(),
    'XGBoost': XGBClassifier(use_label_encoder=False, eval_metric='logloss')
}

results = []
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    results.append({
        'Model': name,
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_pred)
    })

# Display model evaluation results
results_df = pd.DataFrame(results).sort_values(by='ROC AUC', ascending=False)
st.write("\nModel Evaluation Summary:\n")
st.dataframe(results_df)

# Plot feature importances for the best model (XGBoost)
best_model = models['XGBoost']
importances = best_model.feature_importances_
features = X.columns

fig, ax = plt.subplots(figsize=(10, 6))
sns.barplot(x=importances, y=features, ax=ax)
plt.title("Feature Importances - XGBoost")
plt.tight_layout()
st.pyplot(fig)

# Save the best model
joblib.dump(best_model, "churn_model_xgb.pkl")
st.success("Model trained and saved as 'churn_model_xgb.pkl'!")

# Streamlit UI for predictions
st.header("Predict Customer Churn")

# Collect user inputs
gender = st.selectbox("Gender", ["Male", "Female"])
SeniorCitizen = st.selectbox("Senior Citizen", [0, 1])
Partner = st.selectbox("Partner", ["Yes", "No"])
Dependents = st.selectbox("Dependents", ["Yes", "No"])
tenure = st.slider("Tenure (months)", 0, 72, 12)
PhoneService = st.selectbox("Phone Service", ["Yes", "No"])
MultipleLines = st.selectbox("Multiple Lines", ["Yes", "No", "No phone service"])
InternetService = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
OnlineSecurity = st.selectbox("Online Security", ["Yes", "No", "No internet service"])
OnlineBackup = st.selectbox("Online Backup", ["Yes", "No", "No internet service"])
DeviceProtection = st.selectbox("Device Protection", ["Yes", "No", "No internet service"])
TechSupport = st.selectbox("Tech Support", ["Yes", "No", "No internet service"])
StreamingTV = st.selectbox("Streaming TV", ["Yes", "No", "No internet service"])
StreamingMovies = st.selectbox("Streaming Movies", ["Yes", "No", "No internet service"])
Contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
PaperlessBilling = st.selectbox("Paperless Billing", ["Yes", "No"])
PaymentMethod = st.selectbox("Payment Method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
MonthlyCharges = st.number_input("Monthly Charges", min_value=0.0, value=50.0)
TotalCharges = st.number_input("Total Charges", min_value=0.0, value=500.0)

# Prepare input data for prediction
input_data = {
    "gender": gender,
    "SeniorCitizen": SeniorCitizen,
    "Partner": Partner,
    "Dependents": Dependents,
    "tenure": tenure,
    "PhoneService": PhoneService,
    "MultipleLines": MultipleLines,
    "InternetService": InternetService,
    "OnlineSecurity": OnlineSecurity,
    "OnlineBackup": OnlineBackup,
    "DeviceProtection": DeviceProtection,
    "TechSupport": TechSupport,
    "StreamingTV": StreamingTV,
    "StreamingMovies": StreamingMovies,
    "Contract": Contract,
    "PaperlessBilling": PaperlessBilling,
    "PaymentMethod": PaymentMethod,
    "MonthlyCharges": MonthlyCharges,
    "TotalCharges": TotalCharges
}

# Convert to DataFrame
input_df = pd.DataFrame([input_data])

# Preprocess the input data
binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
for col in binary_cols:
    input_df[col] = input_df[col].map({'Yes': 1, 'No': 0})

input_df = pd.get_dummies(input_df, drop_first=True)

# Ensure all columns match the training data
expected_cols = X.columns
for col in expected_cols:
    if col not in input_df.columns:
        input_df[col] = 0
input_df = input_df[expected_cols]

# Scale numerical features
input_df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(input_df[['tenure', 'MonthlyCharges', 'TotalCharges']])

# Predict
if st.button("Predict"):
    try:
        model = joblib.load("churn_model_xgb.pkl")
        prediction = model.predict(input_df)[0]
        probability = model.predict_proba(input_df)[0][1]
        st.write(f"Prediction: {'Churn' if prediction == 1 else 'No Churn'}")
        st.write(f"Churn Probability: {probability:.2%}")
    except FileNotFoundError:
        st.error("Model file 'churn_model_xgb.pkl' not found. Please ensure the model has been trained and saved.")
