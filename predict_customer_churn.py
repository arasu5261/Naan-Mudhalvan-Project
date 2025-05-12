# Install dependencies
!pip install pandas scikit-learn xgboost matplotlib seaborn kaggle joblib streamlit pyngrok -q

# Import libraries
import os
import json
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
import joblib
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Set up Kaggle API credentials
# Replace these with your actual Kaggle username and API key
kaggle_api_token = {
    "username": "kashnitsky",  # Replace with your Kaggle username
    "key": "your_actual_kaggle_api_key"  # Replace with your Kaggle API key
}

# Create the .kaggle directory and save the API token
os.makedirs(os.path.expanduser("~/.kaggle"), exist_ok=True)
with open(os.path.expanduser("~/.kaggle/kaggle.json"), "w") as f:
    json.dump(kaggle_api_token, f)
os.chmod(os.path.expanduser("~/.kaggle/kaggle.json"), 0o600)

# Step 2: Download the dataset
print("Downloading dataset...")
os.system("kaggle datasets download -d blastchar/telco-customer-churn --unzip -p ./data")

# Step 3: Load and preprocess the data
df = pd.read_csv('./data/WA_Fn-UseC_-Telco-Customer-Churn.csv')

# Data preprocessing
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

# Step 4: Train the model
X = df.drop('Churn', axis=1)
y = df['Churn']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)

model_path = "churn_model_xgb.pkl"
if not os.path.exists(model_path):
    print("Training model...")
    model = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    model.fit(X_train, y_train)
    joblib.dump(model, model_path)
    # Evaluate model
    y_pred = model.predict(X_test)
    print("Model Evaluation:")
    print({
        'Accuracy': accuracy_score(y_test, y_pred),
        'Precision': precision_score(y_test, y_pred),
        'Recall': recall_score(y_test, y_pred),
        'F1 Score': f1_score(y_test, y_pred),
        'ROC AUC': roc_auc_score(y_test, y_pred)
    })

# Step 5: Feature Importance Plot
print("Generating feature importance plot...")
importances = model.feature_importances_
features = X.columns
plt.figure(figsize=(10, 6))
sns.barplot(x=importances, y=features)
plt.title("Feature Importances - XGBoost")
plt.tight_layout()
plt.show()

# Step 6: Load model for predictions
model = joblib.load(model_path)

# Step 7: Create a prediction interface using Colab forms
from google.colab import output
from IPython.display import display, HTML

# Define the input form
def create_prediction_form():
    display(HTML("<h3>Customer Churn Prediction</h3>"))
    gender = "Male"  # @param ["Male", "Female"]
    senior_citizen = 0  # @param [0, 1]
    partner = "Yes"  # @param ["Yes", "No"]
    dependents = "Yes"  # @param ["Yes", "No"]
    tenure = 12  # @param {type:"slider", min:0, max:72, step:1}
    phone_service = "Yes"  # @param ["Yes", "No"]
    multiple_lines = "No"  # @param ["No", "Yes", "No phone service"]
    internet_service = "DSL"  # @param ["DSL", "Fiber optic", "No"]
    online_security = "No"  # @param ["No", "Yes", "No internet service"]
    online_backup = "No"  # @param ["No", "Yes", "No internet service"]
    device_protection = "No"  # @param ["No", "Yes", "No internet service"]
    tech_support = "No"  # @param ["No", "Yes", "No internet service"]
    streaming_tv = "No"  # @param ["No", "Yes", "No internet service"]
    streaming_movies = "No"  # @param ["No", "Yes", "No internet service"]
    contract = "Month-to-month"  # @param ["Month-to-month", "One year", "Two year"]
    paperless_billing = "Yes"  # @param ["Yes", "No"]
    payment_method = "Electronic check"  # @param ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
    monthly_charges = 50.0  # @param {type:"number"}
    total_charges = 600.0  # @param {type:"number"}

    # Prepare input data
    input_data = {
        'gender': gender,
        'SeniorCitizen': senior_citizen,
        'Partner': partner,
        'Dependents': dependents,
        'tenure': tenure,
        'PhoneService': phone_service,
        'MultipleLines': multiple_lines,
        'InternetService': internet_service,
        'OnlineSecurity': online_security,
        'OnlineBackup': online_backup,
        'DeviceProtection': device_protection,
        'TechSupport': tech_support,
        'StreamingTV': streaming_tv,
        'StreamingMovies': streaming_movies,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges,
        'TotalCharges': total_charges
    }

    # Map categorical inputs to numerical values
    input_df = pd.DataFrame([input_data])
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling']
    for col in binary_cols:
        input_df[col] = input_df[col].map({'Yes': 1, 'No': 0})

    # Encode categorical variables
    input_df = pd.get_dummies(input_df, drop_first=True)

    # Align columns with training data
    input_df = input_df.reindex(columns=X.columns, fill_value=0)

    # Scale numerical features
    input_df[['tenure', 'MonthlyCharges', 'TotalCharges']] = scaler.transform(input_df[['tenure', 'MonthlyCharges', 'TotalCharges']])

    # Make prediction
    prediction = model.predict(input_df)[0]
    probability = model.predict_proba(input_df)[0][1]

    # Display results
    print(f"Churn Prediction: {'Yes' if prediction == 1 else 'No'}")
    print(f"Probability of Churn: {probability:.2%}")

# Run the prediction form
create_prediction_form()
