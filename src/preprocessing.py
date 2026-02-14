
import os
import logging
import joblib
import pandas as pd
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer, OneHotEncoder
from sklearn.impute import SimpleImputer, KNNImputer

logger = logging.getLogger(__name__)

class ChurnPreprocessor(BaseEstimator, TransformerMixin):
    """
    Unified Preprocessor for Churn Prediction.
    
    Handles:
    - Cleaning & Imputation
    - Feature Engineering (Domain specific)
    - Encoding (Binary & One-Hot)
    - Scaling
    
    Ensures EXACT consistency between Training and Inference.
    """
    def __init__(self, config=None):
        self.config = config
        self.numeric_imputer = None
        self.scaler = None
        self.discretizer = None
        self.train_columns = None # For alignment
        self.categorical_columns = None
        self.numeric_columns = None
        
    def fit(self, X, y=None):
        """
        Fit transformers on Training Data.
        """
        logger.info("Fitting ChurnPreprocessor...")
        
        # Create a copy to rely on current structure
        df = X.copy()
        
        # Identify Columns (Dynamic based on config or defaults)
        self.numeric_columns = ['tenure', 'MonthlyCharges', 'TotalCharges', 'SeniorCitizen']
        self.categorical_columns = ['Contract', 'PaymentMethod', 'InternetService']
        
        # 1. Fit Imputer (Numeric)
        # Handle TotalCharges being object/string usually
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
            
        self.numeric_imputer = SimpleImputer(strategy='median')
        self.numeric_imputer.fit(df[self.numeric_columns])
        
        # 2. Fit Discretizer (Binning)
        # We bin 'tenure' to capture effective contract stages
        if 'tenure' in df.columns:
            self.discretizer = KBinsDiscretizer(n_bins=4, encode='ordinal', strategy='quantile')
            self.discretizer.fit(df[['tenure']])
            
        # 3. Fit Scaler
        # Note: We fit scaler *after* initial cleaning in transform, but we need to know stats now.
        # Ideally, we chain this. For simplicity in this custom class, we'll fit on the simplified version.
        # But to be sklearn compliant, we usually do this:
        # We can't easily fit scaler here on *transformed* data without transforming it first.
        # So we will defer scaler fitting to specific method or do a partial transform here.
        
        # Let's do the standard approach: Transform local copy, then fit scaler.
        df_trans = self._transform_steps(df, training=True)
        
        # Store final columns for alignment
        self.train_columns = df_trans.columns.tolist()
        
        # Fit Scaler on the transformed numeric columns
        # (Log1p transformed 'TotalCharges' is already done in _transform_steps)
        scale_cols = [c for c in self.train_columns if c in ['tenure', 'MonthlyCharges', 'TotalCharges', 'total_services']]
        if scale_cols:
            self.scaler = StandardScaler()
            self.scaler.fit(df_trans[scale_cols])
            
        logger.info("ChurnPreprocessor fitted successfully.")
        return self

    def transform(self, X):
        """
        Apply transformations to Data.
        """
        # Apply core transformations
        df_trans = self._transform_steps(X.copy(), training=False)
        
        # Alignment (Critical for Inference)
        if self.train_columns:
            # Add missing columns with 0
            for col in self.train_columns:
                if col not in df_trans.columns:
                    df_trans[col] = 0
            
            # Drop extra columns (unseen categories)
            df_trans = df_trans[self.train_columns]
            
        # Apply Scaling
        if self.scaler:
             scale_cols = [c for c in df_trans.columns if c in ['tenure', 'MonthlyCharges', 'TotalCharges', 'total_services']]
             if scale_cols:
                 df_trans[scale_cols] = self.scaler.transform(df_trans[scale_cols])
                 
        return df_trans

    def _transform_steps(self, df, training=False):
        """
        Internal method containing the logic steps.
        """
        # 1. Clean & Coerce types
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce')
        
        # 2. Imputation (Numeric)
        # Ensure all numeric columns exist (critical for inference with partial inputs)
        for col in self.numeric_columns:
            if col not in df.columns:
                df[col] = np.nan

        if self.numeric_imputer:
            # We map back to DF so we don't lose context
            df[self.numeric_columns] = self.numeric_imputer.transform(df[self.numeric_columns])
        else:
            # Fallback for initial fit pass if needed, or simple fill
            df.fillna(0, inplace=True)
            
        # Deterministic Imputation Rule
        if 'TotalCharges' in df.columns and 'tenure' in df.columns:
             mask = (df['tenure'] == 0)
             df.loc[mask, 'TotalCharges'] = 0

        # 3. Feature Engineering
        # Service Count
        service_cols = ['MultipleLines', 'OnlineSecurity', 'OnlineBackup', 
                        'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies']
        present_services = [c for c in service_cols if c in df.columns]
        
        # Normalize service columns to 0/1 for counting
        for col in present_services:
            df[col] = df[col].astype(str).apply(lambda x: 1 if x.lower() in ['yes', 'true', '1'] else 0)
            
        if present_services:
            df['total_services'] = df[present_services].sum(axis=1)
            
        # Binning
        if self.discretizer and 'tenure' in df.columns:
            df['tenure_bin'] = self.discretizer.transform(df[['tenure']]).astype(int)

        # 4. Encoding
        # Binary
        # Handle SeniorCitizen specifically if it's already numeric (0/1)
        # We only map 'Yes'/'No' etc. if they exist.
        
        binary_map = {'Yes': 1, 'No': 0, 'True': 1, 'False': 0, 'Male': 1, 'Female': 0}
        binary_cols = ['Partner', 'Dependents', 'PhoneService', 'PaperlessBilling', 'gender']
        
        for col in binary_cols:
            if col in df.columns:
                df[col] = df[col].map(binary_map).fillna(0).astype(int)
                
        # SeniorCitizen is usually 0/1. If it's mixed, force it.
        if 'SeniorCitizen' in df.columns:
             # If it's string, try map. If it's numeric 0/1, keep it.
             # Coerce directly to numeric, then fillna(0)
             df['SeniorCitizen'] = pd.to_numeric(df['SeniorCitizen'], errors='coerce').fillna(0).astype(int)

        # One-Hot Encoding (Pandas get_dummies is easier for readability than OneHotEncoder here)
        # We rely on 'training=True' to generate cols, and 'alignment' in transform to fix them.
        if self.categorical_columns:
            df = pd.get_dummies(df, columns=[c for c in self.categorical_columns if c in df.columns], drop_first=True, dtype=int)
            
        # 5. Transformations
        if 'TotalCharges' in df.columns:
            df['TotalCharges'] = np.log1p(df['TotalCharges'])
            
        # Drop unused columns (like original categorical names if not dropped, or IDs)
        drop_cols = ['customerID']
        df.drop(columns=[c for c in drop_cols if c in df.columns], inplace=True)
        
        return df


def preprocess_data(config):
    """
    Pipeline integration wrapper.
    Splits data, Fits Preprocessor on Train, Transforms both.
    Arguments:
        config: Config object
    Returns:
        X_train, X_test, y_train, y_test, feature_names
    """
    logger.info("Starting (Simplified) Preprocessing...")
    
    df = pd.read_csv(config.DATA_PATH)
    target_col = config.TARGET_COLUMN
    
    # Split
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=[target_col])
    y = df[target_col].map({'Yes': 1, 'No': 0}).astype(int)
    
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, 
        test_size=config.PIPELINE_CONFIG["splitting"]["test_size"], 
        stratify=y, 
        random_state=config.RANDOM_STATE
    )
    
    # Init & Fit Preprocessor
    preprocessor = ChurnPreprocessor(config)
    preprocessor.fit(X_train_raw, y_train)
    
    # Transform
    X_train = preprocessor.transform(X_train_raw)
    X_test = preprocessor.transform(X_test_raw)
    
    feature_names = X_train.columns.tolist()
    


    # Save Preprocessor (THE TRUTH SOURCE)
    os.makedirs("models", exist_ok=True)
    joblib.dump(preprocessor, "models/preprocessor.pkl")
    logger.info("Saved unified 'models/preprocessor.pkl'")
    
    return X_train.values, X_test.values, y_train.values, y_test.values, feature_names
