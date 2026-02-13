import os
import logging
import pandas as pd
import numpy as np

from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler, KBinsDiscretizer
from sklearn.impute import KNNImputer
from statsmodels.stats.outliers_influence import variance_inflation_factor
import joblib

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


# ===================== FEATURE ENGINEERING FUNCTIONS =====================
# Each function serves ONE purpose (Single Responsibility Principle)

def compute_vif(df, threshold=5.0):
    """
    Compute Variance Inflation Factor and remove multicollinear features.
    
    VIF > 5 indicates problematic multicollinearity.
    We preserve features with highest correlation to target.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if len(numeric_cols) < 2:
        return df, []
    
    removed_features = []
    df_vif = df[numeric_cols].copy()
    
    # Iteratively remove high VIF features
    while True:
        vif_data = []
        for i, col in enumerate(df_vif.columns):
            try:
                vif = variance_inflation_factor(df_vif.values, i)
                vif_data.append({'feature': col, 'vif': vif})
            except:
                vif_data.append({'feature': col, 'vif': 0})
        
        vif_df = pd.DataFrame(vif_data)
        max_vif = vif_df['vif'].max()
        
        if max_vif > threshold and len(df_vif.columns) > 1:
            # Remove feature with highest VIF
            worst_feature = vif_df.loc[vif_df['vif'].idxmax(), 'feature']
            df_vif = df_vif.drop(columns=[worst_feature])
            removed_features.append(worst_feature)
            logger.info(f"VIF: Removed '{worst_feature}' (VIF={max_vif:.2f})")
        else:
            break
    
    # Keep only low-VIF features in original df
    cols_to_keep = [c for c in df.columns if c not in removed_features]
    logger.info(f"VIF check complete: kept {len(cols_to_keep)} features, removed {len(removed_features)}")
    return df[cols_to_keep], removed_features


def select_by_correlation(df, target_col, min_threshold=0.02, max_inter_threshold=0.85):
    """
    Feature selection based on correlation analysis.
    
    - Remove features with |corr| < min_threshold with target (irrelevant)
    - Remove highly correlated feature pairs (keep one with higher target corr)
    """
    if target_col not in df.columns:
        return df, []
    
    removed_features = []
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if target_col in numeric_cols:
        numeric_cols.remove(target_col)
    
    # Step 1: Remove features with negligible target correlation
    target_corr = df[numeric_cols].corrwith(df[target_col]).abs()
    low_corr_features = target_corr[target_corr < min_threshold].index.tolist()
    for feat in low_corr_features:
        removed_features.append(feat)
        logger.info(f"Correlation: Removed '{feat}' (target_corr={target_corr[feat]:.4f})")
    
    remaining_cols = [c for c in numeric_cols if c not in low_corr_features]
    
    # Step 2: Remove highly inter-correlated features (keep the one closer to target)
    if len(remaining_cols) > 1:
        corr_matrix = df[remaining_cols].corr().abs()
        for i, col1 in enumerate(remaining_cols):
            for col2 in remaining_cols[i+1:]:
                if col1 in removed_features or col2 in removed_features:
                    continue
                if corr_matrix.loc[col1, col2] > max_inter_threshold:
                    # Keep feature with higher target correlation
                    if target_corr.get(col1, 0) >= target_corr.get(col2, 0):
                        removed_features.append(col2)
                        logger.info(f"Correlation: Removed '{col2}' (inter_corr={corr_matrix.loc[col1, col2]:.2f} with '{col1}')")
                    else:
                        removed_features.append(col1)
                        logger.info(f"Correlation: Removed '{col1}' (inter_corr={corr_matrix.loc[col1, col2]:.2f} with '{col2}')")
    
    cols_to_keep = [c for c in df.columns if c not in removed_features]
    logger.info(f"Correlation selection: kept {len(cols_to_keep)} features")
    return df[cols_to_keep], removed_features


def create_interaction_terms(df, pairs):
    """
    Create interaction features between specified column pairs.
    
    Interaction terms capture non-additive effects (e.g., tenure × charges
    captures how value perception changes with customer lifetime).
    """
    for col1, col2 in pairs:
        if col1 in df.columns and col2 in df.columns:
            interaction_name = f"{col1}_x_{col2}"
            df[interaction_name] = df[col1] * df[col2]
            logger.info(f"Created interaction: {interaction_name}")
    return df


def discretize_continuous(df, cols, n_bins=4, strategy='quantile'):
    """
    Bin continuous variables into discrete categories.
    
    Helps capture non-linear relationships and interactions.
    Strategy: 'quantile' (equal frequency) or 'uniform' (equal width).
    """
    for col in cols:
        if col in df.columns:
            new_col_name = f"{col}_bin"
            discretizer = KBinsDiscretizer(
                n_bins=n_bins, 
                encode='ordinal', 
                strategy=strategy,
                subsample=None  # Use all data for bin edges
            )
            df[new_col_name] = discretizer.fit_transform(df[[col]]).astype(int)
            logger.info(f"Discretized '{col}' into {n_bins} bins ({strategy})")
    return df


def create_polynomial_features(df, cols, degree=2):
    """
    Create polynomial features for selected columns.
    
    This captures non-linear relationships that LR cannot model otherwise.
    For example, churn risk may decrease with tenure but at a diminishing rate.
    
    Parameters
    ----------
    df : DataFrame
        Input dataframe.
    cols : list
        Columns to create polynomial features for.
    degree : int
        Maximum polynomial degree (default 2 for squared terms).
        
    Returns
    -------
    DataFrame with polynomial features added.
    """
    for col in cols:
        if col in df.columns:
            for d in range(2, degree + 1):
                new_col = f"{col}_pow{d}"
                df[new_col] = df[col] ** d
                logger.info(f"Created polynomial feature: {new_col}")
    return df




def create_service_count(df, service_cols):
    """
    Count total services subscribed per customer.
    
    Aggregate feature indicating engagement level.
    Higher count = more entrenched = lower churn risk.
    """
    existing_cols = [c for c in service_cols if c in df.columns]
    if existing_cols:
        df['total_services'] = df[existing_cols].sum(axis=1)
        logger.info(f"Created 'total_services' from {len(existing_cols)} service columns")
    return df


def preprocess_data(config):
    """
    Main preprocessing pipeline with STRICT train/test separation to prevent leakage.
    
    Flow:
    1. Load & Clean (Basic)
    2. Split Train/Test
    3. Fit transformations on Train, Apply to Test
    """
    logger.info("Starting preprocessing pipeline (Strict Separation)")
    
    df = pd.read_csv(config.DATA_PATH)
    logger.info(f"Loaded data with shape {df.shape}")
    
    pipeline_config = config.PIPELINE_CONFIG
    col_config = pipeline_config["columns"]
    target_col = config.TARGET_COLUMN
    
    # --- 1. INITIAL CLEANING (Safe to do before split) ---
    # Drop unwanted columns
    for col in col_config["drop"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            logger.info(f"Dropped column: {col}")
            
    # Deterministic Imputation (Row-wise, Safe)
    df = deterministic_impute(df, config)
    
    # Binary Encoding (Mapping is fixed, Safe)
    for col in col_config["binary"]:
        df = encode_binary(df, col)
        
    for col in col_config["service_binary"]:
        df = encode_service_binary(df, col)
        
    # --- 2. TRAIN-TEST SPLIT ---
    # Split NOW before any learnable parameters (mean, quantiles, etc.) are computed
    split_config = pipeline_config["splitting"]
    strat_col = df[target_col] if target_col in df.columns else None
    
    # Using simple train_test_split for clarity, respecting config
    from sklearn.model_selection import train_test_split
    X = df.drop(columns=[target_col])
    y = df[target_col].astype(int)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=split_config["test_size"], 
        stratify=y, # Important for churn
        random_state=split_config["random_state"]
    )
    
    logger.info(f"Data Split: Train={X_train.shape[0]}, Test={X_test.shape[0]}")
    
    # --- 3. LEARN & TRANSFORM (Stateful Steps) ---
    
    # Helper to apply one-hot (needs alignment)
    cat_cols = [c for c in col_config["categorical"] if c in X_train.columns]
    if cat_cols:
        # We use pd.get_dummies but must align columns
        X_train = pd.get_dummies(X_train, columns=cat_cols, drop_first=True, dtype=int)
        X_test = pd.get_dummies(X_test, columns=cat_cols, drop_first=True, dtype=int)
        
        # Align: Add missing columns to Test with 0, drop extra
        train_cols = X_train.columns
        X_test = X_test.reindex(columns=train_cols, fill_value=0)
        logger.info(f"One-hot encoded categorical columns (aligned)")

    # Helper to fit-transform specific columns
    def fit_impute_numeric(train_df, test_df, cols, n_neighbors):
        if not cols: return train_df, test_df
        
        if pipeline_config["imputation"].get("fallback") == "KNNImputer":
            imputer = KNNImputer(n_neighbors=n_neighbors)
        else:
            from sklearn.impute import SimpleImputer
            imputer = SimpleImputer(strategy='median')
            
        train_df[cols] = imputer.fit_transform(train_df[cols])
        test_df[cols] = imputer.transform(test_df[cols])
        
        # Save imputer
        os.makedirs("models", exist_ok=True)
        joblib.dump(imputer, "models/numeric_imputer.pkl")
        logger.info(f"Imputed numeric columns: {cols} and saved imputer to models/numeric_imputer.pkl")
        
        return train_df, test_df

    # Impute Numeric
    numeric_cols = [c for c in col_config["numeric"] if c in X_train.columns]
    X_train, X_test = fit_impute_numeric(
        X_train, X_test, numeric_cols, 
        pipeline_config["imputation"]["n_neighbors"]
    )
    
    # Fill remaining missing (Categorical/Binary via mode)
    # Simple mode fill based on TRAIN
    for col in X_train.columns:
        if X_train[col].isnull().any() or X_test[col].isnull().any():
            mode_val = X_train[col].mode()[0]
            X_train[col].fillna(mode_val, inplace=True)
            X_test[col].fillna(mode_val, inplace=True)

    # Log1p (Safe row-wise)
    log_cols = pipeline_config["transformations"]["log1p"]
    X_train = apply_log1p(X_train, log_cols)
    X_test = apply_log1p(X_test, log_cols)

    # --- FEATURE ENGINEERING ---
    fe_config = pipeline_config.get("feature_engineering", {})
    
    # Service Count
    service_cols = col_config.get("service_binary", [])
    X_train = create_service_count(X_train, service_cols)
    X_test = create_service_count(X_test, service_cols)
    
    # Interactions
    pairs = fe_config.get("interaction_pairs", [])
    if pairs:
        X_train = create_interaction_terms(X_train, pairs)
        X_test = create_interaction_terms(X_test, pairs)
        
    # Polynomial
    poly_config = fe_config.get("polynomial", {})
    if poly_config.get("enabled", False):
        cols = poly_config.get("columns", [])
        deg = poly_config.get("degree", 2)
        X_train = create_polynomial_features(X_train, cols, deg)
        X_test = create_polynomial_features(X_test, cols, deg)

    # Binning (Quantile - Learns from Train)
    bin_config = fe_config.get("binning", {})
    bin_cols = bin_config.get("columns", [])
    if bin_cols:
        est = KBinsDiscretizer(
            n_bins=bin_config.get("n_bins", 4),
            encode='ordinal',
            strategy=bin_config.get("strategy", "quantile"),
            subsample=None
        )
        # Handle if columns missing or error
        try:
            # Must process columns independently or together? Together is fine if cols exist
            present_bin_cols = [c for c in bin_cols if c in X_train.columns]
            if present_bin_cols:
                # Fit on Train
                est.fit(X_train[present_bin_cols])
                
                # Transform Train
                train_bins = est.transform(X_train[present_bin_cols]).astype(int)
                test_bins = est.transform(X_test[present_bin_cols]).astype(int)
                
                for i, col in enumerate(present_bin_cols):
                    X_train[f"{col}_bin"] = train_bins[:, i]
                    X_test[f"{col}_bin"] = test_bins[:, i]
                logger.info(f"Discretized {present_bin_cols} using Train quantiles")
        except Exception as e:
            logger.warning(f"Binning failed: {e}")

    # --- SELECTION (Correlation with Train Target) ---
    # We select features based on X_train + y_train relationship
    # Then drop same features from X_test
    corr_config = fe_config.get("correlation", {})
    if corr_config.get("enabled", True):
        # Combine just for correlation calculation
        temp_train = X_train.copy()
        temp_train['target'] = y_train
        
        _, removed_features = select_by_correlation(
            temp_train, 'target',
            min_threshold=corr_config.get("min_threshold", 0.02),
            max_inter_threshold=corr_config.get("max_threshold", 0.85)
        )
        
        if removed_features:
            X_train.drop(columns=removed_features, inplace=True, errors='ignore')
            X_test.drop(columns=removed_features, inplace=True, errors='ignore')
            logger.info(f"Dropped {len(removed_features)} features based on Train correlation")

    # VIF (Train only)
    vif_config = fe_config.get("vif", {})
    if vif_config.get("enabled", True):
        # reuse compute_vif but it returns df, removed.
        # We need to apply removed to X_test
        X_train, vif_removed = compute_vif(X_train, threshold=vif_config.get("threshold", 5.0))
        if vif_removed:
            X_test.drop(columns=vif_removed, inplace=True, errors='ignore')
            logger.info(f"Dropped {len(vif_removed)} features based on Train VIF")

    # --- NORMALIZATION (Fit Train, Transform Test) ---
    normalize_cols = [c for c in pipeline_config["transformations"]["normalize"] if c in X_train.columns]
    if normalize_cols:
        scaler = StandardScaler()
        X_train[normalize_cols] = scaler.fit_transform(X_train[normalize_cols])
        X_test[normalize_cols] = scaler.transform(X_test[normalize_cols])
        
        # Save scaler
        os.makedirs("models", exist_ok=True)
        joblib.dump(scaler, "models/scaler.pkl")
        logger.info(f"Normalized {len(normalize_cols)} columns (Fit on Train) and saved scaler to models/scaler.pkl")

    feature_names = X_train.columns.tolist()
    
    # Save processed (optional) - saving split
    # os.makedirs("data/processed", exist_ok=True)
    # pd.concat([X_train, y_train], axis=1).to_csv("data/processed/train.csv", index=False)
    # pd.concat([X_test, y_test], axis=1).to_csv("data/processed/test.csv", index=False)
    
    logger.info("Preprocessing Complete. Data Leakage Prevented.")
    return X_train.values, X_test.values, y_train.values, y_test.values, feature_names


def get_preprocessed_dataframe(config):
    """
    Return preprocessed DataFrame for explainability/causal analysis.
    
    NOTE: This applies transformations to the FULL dataset. 
    For strict training validation, use preprocess_data() which splits first.
    This function is for post-hoc analysis of the entire dataset distribution.
    """
    df = pd.read_csv(config.DATA_PATH)
    pipeline_config = config.PIPELINE_CONFIG
    col_config = pipeline_config["columns"]
    
    # Drop
    for col in col_config["drop"]:
        if col in df.columns:
            df.drop(columns=[col], inplace=True)
            
    # Deterministic Impute
    df = deterministic_impute(df, config)
    
    # Binary
    for col in col_config["binary"]:
        df = encode_binary(df, col)
        
    # Service Binary
    for col in col_config["service_binary"]:
        df = encode_service_binary(df, col)
        
    # One-Hot
    cat_cols_present = [c for c in col_config["categorical"] if c in df.columns]
    df = encode_categorical(df, cat_cols_present)
    
    # Fill Missing (Simple)
    df = fill_missing_values(df)
    
    # KNN (or Impute)
    numeric_cols = [c for c in col_config["numeric"] if c in df.columns]
    # Use simple imputation for speed in analysis
    from sklearn.impute import SimpleImputer
    if numeric_cols:
        imputer = SimpleImputer(strategy='median')
        df[numeric_cols] = imputer.fit_transform(df[numeric_cols])
    
    # Log1p
    log_cols = pipeline_config["transformations"]["log1p"]
    df = apply_log1p(df, log_cols)
    
    # FE
    df = create_service_count(df, col_config.get("service_binary", []))
    
    fe_config = pipeline_config.get("feature_engineering", {})
    pairs = fe_config.get("interaction_pairs", [])
    if pairs:
        df = create_interaction_terms(df, pairs)
        
    bin_cols = fe_config.get("binning", {}).get("columns", [])
    if bin_cols:
         df = discretize_continuous(df, bin_cols)

    poly_config = fe_config.get("polynomial", {})
    if poly_config.get("enabled", False):
        df = create_polynomial_features(df, poly_config.get("columns", []), degree=poly_config.get("degree", 2))
        
    # Final Normalize
    normalize_cols = [c for c in pipeline_config["transformations"]["normalize"] if c in df.columns]
    if normalize_cols:
        scaler = StandardScaler()
        df[normalize_cols] = scaler.fit_transform(df[normalize_cols])
        
    return df
