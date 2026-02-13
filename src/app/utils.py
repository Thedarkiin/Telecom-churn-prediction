import pandas as pd
import numpy as np

def preprocess_input(data, feature_names, scaler=None, imputer=None):
    """
    Transform raw input dictionary into a model-ready dataframe.
    Uses saved artifacts for consistent scaling/imputation.
    """
    # 1. Create DataFrame from dict
    df = pd.DataFrame([data])
    
    # 2. Binary Encoding (Manual Mappings from src/preprocessing.py)
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        if col in df.columns:
            df[col] = df[col].map({'Yes': 1, 'No': 0}).fillna(0)
            
    gender_map = {'Male': 1, 'Female': 0}
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map(gender_map).fillna(0)
        
    # Service Binary Encoding
    service_cols = [
        'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    for col in service_cols:
        if col in df.columns:
            # Map 'Yes'->1, everything else ('No', 'No internet service') -> 0
            df[col] = df[col].apply(lambda x: 1 if x == 'Yes' else 0)

    # 3. Numeric Conversions
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in numeric_cols:
        if col in df.columns:
            # Let imputer handle NaNs/Zeros if needed, but safe to coerce
            df[col] = pd.to_numeric(df[col], errors='coerce') 

    # 4. Feature Engineering
    # Service Count
    if set(service_cols).intersection(df.columns):
        df['total_services'] = df[service_cols].sum(axis=1)

    # 5. One-Hot Encoding (Manual for inference to ensure alignment)
    # We need to manually create the columns expected by the model
    # Contract
    df['Contract_One year'] = 1 if data.get('Contract') == 'One year' else 0
    df['Contract_Two year'] = 1 if data.get('Contract') == 'Two year' else 0
    # Month-to-month is the reference category (dropped)

    # InternetService
    df['InternetService_Fiber optic'] = 1 if data.get('InternetService') == 'Fiber optic' else 0
    df['InternetService_No'] = 1 if data.get('InternetService') == 'No' else 0
    # Reference category: DSL (implied 0,0)

    # PaymentMethod
    df['PaymentMethod_Credit card (automatic)'] = 1 if data.get('PaymentMethod') == 'Credit card (automatic)' else 0
    df['PaymentMethod_Electronic check'] = 1 if data.get('PaymentMethod') == 'Electronic check' else 0
    df['PaymentMethod_Mailed check'] = 1 if data.get('PaymentMethod') == 'Mailed check' else 0
    # Bank transfer is reference

    # 6. Align with Feature Names
    # Create a final dataframe with all expected columns initialized to 0
    final_df = pd.DataFrame(0, index=[0], columns=feature_names)
    
    # Fill in values from our processed df
    for col in feature_names:
        if col in df.columns:
            final_df[col] = df[col]
            
    # Handle Log1p if needed (TotalCharges)
    if 'TotalCharges' in final_df.columns:
         final_df['TotalCharges'] = np.log1p(final_df['TotalCharges'])

    # 7. Apply Imputer and Scaler
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
    present_numeric = [c for c in numeric_features if c in final_df.columns]
    
    if imputer and present_numeric:
        try:
             final_df[present_numeric] = imputer.transform(final_df[present_numeric])
        except Exception as e:
            print(f"Imputation warning: {e}")

    if scaler and present_numeric:
        try:
            final_df[present_numeric] = scaler.transform(final_df[present_numeric])
        except Exception as e:
            print(f"Scaling warning: {e}")
         
    return final_df

def preprocess_batch(df, feature_names, scaler=None, imputer=None):
    """
    Preprocess a batch of data (DataFrame) for inference.
    """
    # 1. Binary Encoding
    binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'Churn']
    for col in binary_cols:
        if col in df.columns:
            # Handle yes/no case insensitive
            df[col] = df[col].astype(str).str.lower().map({'yes': 1, 'no': 0}).fillna(0)
            
    gender_map = {'Male': 1, 'Female': 0, 'male': 1, 'female': 0}
    if 'gender' in df.columns:
        df['gender'] = df['gender'].map(gender_map).fillna(0)
        
    # Service Binary Encoding
    service_cols = [
        'MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies'
    ]
    for col in service_cols:
        if col in df.columns:
            # Map 'Yes'->1, everything else -> 0
            df[col] = df[col].astype(str).apply(lambda x: 1 if 'yes' in x.lower() else 0)

    # 2. Numeric Conversions
    numeric_cols = ['tenure', 'MonthlyCharges', 'TotalCharges']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')

    # 3. Feature Engineering
    if set(service_cols).intersection(df.columns):
        df['total_services'] = df[service_cols].sum(axis=1)

    # 4. One-Hot Encoding (Manual alignment)
    # Contract
    df['Contract_One year'] = (df['Contract'] == 'One year').astype(int)
    df['Contract_Two year'] = (df['Contract'] == 'Two year').astype(int)
    
    # InternetService
    df['InternetService_Fiber optic'] = (df['InternetService'] == 'Fiber optic').astype(int)
    df['InternetService_No'] = (df['InternetService'] == 'No').astype(int)

    # PaymentMethod
    df['PaymentMethod_Credit card (automatic)'] = (df['PaymentMethod'] == 'Credit card (automatic)').astype(int)
    df['PaymentMethod_Electronic check'] = (df['PaymentMethod'] == 'Electronic check').astype(int)
    df['PaymentMethod_Mailed check'] = (df['PaymentMethod'] == 'Mailed check').astype(int)

    # 5. Align with Feature Names
    final_df = pd.DataFrame(0, index=df.index, columns=feature_names)
    for col in feature_names:
        if col in df.columns:
            final_df[col] = df[col]
            
    # Log1p
    if 'TotalCharges' in final_df.columns:
         final_df['TotalCharges'] = np.log1p(final_df['TotalCharges'])

    # Apply artifacts
    numeric_features = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
    present_numeric = [c for c in numeric_features if c in final_df.columns]
    
    if imputer and present_numeric:
        try:
             final_df[present_numeric] = imputer.transform(final_df[present_numeric])
        except Exception as e:
            print(f"Batch Imputation warning: {e}")

    if scaler and present_numeric:
        try:
            final_df[present_numeric] = scaler.transform(final_df[present_numeric])
        except Exception as e:
            print(f"Batch Scaling warning: {e}")
         
    return final_df
