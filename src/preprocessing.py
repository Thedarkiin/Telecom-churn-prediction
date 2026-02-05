import os
import logging
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.impute import KNNImputer

logger = logging.getLogger(__name__)


def encode_binary(df, col):
    """Encode binary Yes/No columns as 0/1."""
    mapping = {'Yes': 1, 'No': 0, 'Male': 1, 'Female': 0}
    if col in df.columns:
        df[col] = df[col].map(mapping).fillna(df[col])
    return df


def encode_service_binary(df, col):
    """Encode service columns: 'No internet/phone service' -> 0, 'No' -> 0, 'Yes' -> 1."""
    mapping = {
        'Yes': 1, 
        'No': 0, 
        'No internet service': 0, 
        'No phone service': 0
    }
    if col in df.columns:
        df[col] = df[col].map(mapping).fillna(0).astype(int)
    return df


def encode_categorical(df, cols):
    """One-hot encode categorical columns."""
    return pd.get_dummies(df, columns=cols, drop_first=True, dtype=int)


def deterministic_impute(df, config):
    """Apply deterministic imputation based on conditions."""
    imp_config = config.PIPELINE_CONFIG["imputation"]
    col = imp_config["deterministic"]["col"]
    value = imp_config["deterministic"]["value"]
    condition = imp_config["deterministic"]["condition"]
    
    # Convert TotalCharges to numeric (it's stored as object)
    if col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Apply deterministic rule: TotalCharges = 0 when tenure == 0
    mask = df.eval(condition)
    df.loc[mask, col] = value
    logger.info(f"Applied deterministic imputation: {col}={value} where {condition}")
    return df


def knn_impute(df, numeric_cols, n_neighbors=5):
    """Apply KNN imputation for remaining missing values."""
    if df[numeric_cols].isnull().any().any():
        imputer = KNNImputer(n_neighbors=n_neighbors)
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
        logger.info(f"Applied KNN imputation to numeric columns")
    return df


def apply_log1p(df, cols):
    """Apply log1p transformation to specified columns."""
    for col in cols:
        if col in df.columns:
            df[col] = np.log1p(df[col])
            logger.info(f"Applied log1p to {col}")
    return df


def fill_missing_values(df):
    """Fill any remaining missing values with median/mode."""
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logger.info(f"Filled missing in '{col}' with median: {median_val}")
            else:
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                logger.info(f"Filled missing in '{col}' with mode: {mode_val}")
    return df


def preprocess_data(config):
    """Main preprocessing pipeline with proper encoding and transformations."""
    logger.info("Starting preprocessing pipeline")
    
    df = pd.read_csv(config.DATA_PATH)
    logger.info(f"Loaded data with shape {df.shape}")
    
    pipeline_config = config.PIPELINE_CONFIG
    col_config = pipeline_config["columns"]
    
    # 1. Drop unwanted columns
    for col in col_config["drop"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            logger.info(f"Dropped column: {col}")
    
    # 2. Deterministic imputation (TotalCharges = 0 when tenure = 0)
    df = deterministic_impute(df, config)
    
    # 3. Encode binary columns (Yes/No, Male/Female)
    for col in col_config["binary"]:
        df = encode_binary(df, col)
        logger.info(f"Binary encoded: {col}")
    
    # 4. Encode service columns (handle "No internet/phone service")
    for col in col_config["service_binary"]:
        df = encode_service_binary(df, col)
        logger.info(f"Service binary encoded: {col}")
    
    # 5. One-hot encode categorical columns
    cat_cols_present = [c for c in col_config["categorical"] if c in df.columns]
    df = encode_categorical(df, cat_cols_present)
    logger.info(f"One-hot encoded: {cat_cols_present}")
    
    # 6. Fill any remaining missing values
    df = fill_missing_values(df)
    
    # 7. KNN imputation for numeric columns if needed
    numeric_cols = [c for c in col_config["numeric"] if c in df.columns]
    df = knn_impute(df, numeric_cols, pipeline_config["imputation"]["n_neighbors"])
    
    # 8. Apply log1p transformations
    log_cols = pipeline_config["transformations"]["log1p"]
    df = apply_log1p(df, log_cols)
    
    # 9. Split features and target
    X = df.drop(columns=[config.TARGET_COLUMN])
    y = df[config.TARGET_COLUMN].astype(int)
    
    # 10. Store feature names for later use
    feature_names = X.columns.tolist()
    
    # 11. Normalize numeric features
    normalize_cols = [c for c in pipeline_config["transformations"]["normalize"] if c in X.columns]
    if normalize_cols:
        scaler = StandardScaler()
        X[normalize_cols] = scaler.fit_transform(X[normalize_cols])
        logger.info(f"Normalized columns: {normalize_cols}")
    
    # 12. Save cleaned data
    cleaned_df = X.copy()
    cleaned_df[config.TARGET_COLUMN] = y
    os.makedirs("data", exist_ok=True)
    cleaned_df.to_csv("data/cleaned_data.csv", index=False)
    logger.info("Saved cleaned data to data/cleaned_data.csv")
    
    # 13. Stratified train-test split
    split_config = pipeline_config["splitting"]
    sss = StratifiedShuffleSplit(
        n_splits=1, 
        test_size=split_config["test_size"], 
        random_state=split_config["random_state"]
    )
    
    for train_idx, test_idx in sss.split(X, y):
        X_train = X.iloc[train_idx].values
        X_test = X.iloc[test_idx].values
        y_train = y.iloc[train_idx].values
        y_test = y.iloc[test_idx].values
    
    logger.info(f"Train set: {X_train.shape[0]}, Test set: {X_test.shape[0]}")
    logger.info(f"Train churn rate: {y_train.mean():.2%}, Test churn rate: {y_test.mean():.2%}")
    logger.info("Completed preprocessing pipeline")
    
    return X_train, X_test, y_train, y_test, feature_names


def get_preprocessed_dataframe(config):
    """Return preprocessed DataFrame for explainability and causal analysis."""
    df = pd.read_csv(config.DATA_PATH)
    pipeline_config = config.PIPELINE_CONFIG
    col_config = pipeline_config["columns"]
    
    # Drop unwanted columns
    for col in col_config["drop"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
    
    # Deterministic imputation
    df = deterministic_impute(df, config)
    
    # Encode binary columns
    for col in col_config["binary"]:
        df = encode_binary(df, col)
    
    # Encode service columns
    for col in col_config["service_binary"]:
        df = encode_service_binary(df, col)
    
    # One-hot encode categorical
    cat_cols_present = [c for c in col_config["categorical"] if c in df.columns]
    df = encode_categorical(df, cat_cols_present)
    
    # Fill missing
    df = fill_missing_values(df)
    
    # KNN impute
    numeric_cols = [c for c in col_config["numeric"] if c in df.columns]
    df = knn_impute(df, numeric_cols, pipeline_config["imputation"]["n_neighbors"])
    
    # Log1p
    log_cols = pipeline_config["transformations"]["log1p"]
    df = apply_log1p(df, log_cols)
    
    return df
