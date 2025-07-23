import os
import logging
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, chi2, mutual_info_classif

logger = logging.getLogger(__name__)

def fill_missing_values(df):
    for col in df.columns:
        if df[col].isnull().any():
            if pd.api.types.is_numeric_dtype(df[col]):
                median_val = df[col].median()
                df[col].fillna(median_val, inplace=True)
                logger.info(f"Filled missing numeric column '{col}' with median: {median_val}")
            else:
                mode_val = df[col].mode()[0]
                df[col].fillna(mode_val, inplace=True)
                logger.info(f"Filled missing categorical column '{col}' with mode: {mode_val}")
    return df

def handle_outliers(df, numerical_cols):
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        before_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
        df[col] = np.where(df[col] < lower_bound, lower_bound,
                           np.where(df[col] > upper_bound, upper_bound, df[col]))
        after_outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)].shape[0]
        logger.info(f"Capped outliers in '{col}': {before_outliers} values adjusted")
    return df

def preprocess_data(config):
    logger.info("Starting preprocessing pipeline")

    df = pd.read_csv(config.DATA_PATH)
    logger.info(f"Loaded data with shape {df.shape}")

    if "customerID" in df.columns:
        df.drop(columns=["customerID"], inplace=True)
        logger.info("Dropped 'customerID' column")

    if hasattr(config, "DROP_COLUMNS") and config.DROP_COLUMNS:
        df.drop(columns=config.DROP_COLUMNS, inplace=True)
        logger.info(f"Dropped configured columns: {config.DROP_COLUMNS}")

    df = fill_missing_values(df)

    num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    df = handle_outliers(df, num_cols)

    X = df.drop(columns=[config.TARGET_COLUMN])
    y = df[config.TARGET_COLUMN]

    # Label encode all categorical features
    cat_cols = X.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    for col in cat_cols:
        le = LabelEncoder()
        X[col] = le.fit_transform(X[col])
        logger.info(f"Label encoded categorical column: {col}")

    if y.nunique() == 2:
        y = LabelEncoder().fit_transform(y)
        logger.info("Label encoded target column")

    num_cols, _ = [], []
    if not X.empty:
        num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()

    if num_cols:
        scaler = StandardScaler()
        X_num = scaler.fit_transform(X[num_cols])

        vt = VarianceThreshold(threshold=0.01)
        X_num = vt.fit_transform(X_num)
        kept_num_cols = [col for i, col in enumerate(num_cols) if vt.get_support()[i]]
    else:
        X_num = np.empty((len(X), 0))
        kept_num_cols = []

    X_processed = X_num
    feature_names = kept_num_cols

    if len(feature_names) > 1:
        corr_df = pd.DataFrame(X_processed, columns=feature_names).corr().abs()
        upper = corr_df.where(np.triu(np.ones(corr_df.shape), k=1).astype(bool))
        to_drop = [col for col in upper.columns if any(upper[col] > 0.9)]
        if to_drop:
            mask = ~pd.Series(feature_names).isin(to_drop)
            X_processed = X_processed[:, mask.values]
            feature_names = list(pd.Series(feature_names)[mask.values])
            logger.info(f"Dropped {len(to_drop)} highly-correlated features: {to_drop}")

    if X_processed.size > 0:
        scaler_minmax = MinMaxScaler()
        X_scaled = scaler_minmax.fit_transform(X_processed)
        chi_scores, _ = chi2(X_scaled, y)
        mi_scores = mutual_info_classif(X_scaled, y, random_state=42)
    else:
        chi_scores, mi_scores = [], []

    scores_df = pd.DataFrame({
        "Feature": feature_names,
        "Chi2 Score": chi_scores,
        "Mutual Info": mi_scores
    }).sort_values("Mutual Info", ascending=False)

    scores_df.to_csv("results/metrics/feature_scores.csv", index=False)
    logger.info("Saved feature scores to results/metrics/feature_scores.csv")

    top_features = scores_df["Feature"].head(20).tolist()
    logger.info(f"Top 20 selected features: {top_features}")

    top_indices = [feature_names.index(f) for f in top_features if f in feature_names]
    X_selected = X_processed[:, top_indices] if top_indices else np.empty((X_processed.shape[0], 0))

    cleaned_df = pd.DataFrame(X_selected, columns=top_features)
    cleaned_df[config.TARGET_COLUMN] = y
    os.makedirs("data", exist_ok=True)
    cleaned_df.to_csv("data/cleaned_data.csv", index=False)
    logger.info("Saved cleaned data to data/cleaned_data.csv")

    X_train, X_test, y_train, y_test = train_test_split(
        X_selected, y,
        test_size=config.TEST_SIZE,
        random_state=config.RANDOM_STATE,
        stratify=y
    )

    logger.info("Completed preprocessing pipeline")
    return X_train, X_test, y_train, y_test
