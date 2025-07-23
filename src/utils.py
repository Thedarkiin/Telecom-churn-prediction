import os
import logging
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder

def setup_logger(log_file="logs/pipeline.log"):
    """
    Set up logging configuration. Ensures log directory exists.
    Returns a logger instance.
    """
    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger()
    if not logger.hasHandlers():
        logging.basicConfig(
            filename=log_file,
            level=logging.INFO,
            format="%(asctime)s:%(levelname)s:%(message)s"
        )
        logger.info("Logging initialized.")
    return logger

def save_plot(fig, filename, folder):
    """
    Save matplotlib figure to folder with filename and close figure.
    """
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Plot saved to {path}")

def split_numerical_categorical(df):
    """
    Returns two lists: numerical columns and categorical columns.
    """
    numerical = df.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical = df.select_dtypes(include=["object", "category", "bool"]).columns.tolist()
    return numerical, categorical

def check_missing_values(df):
    """
    Returns Series of missing counts sorted descending, logs the result.
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)
    if not missing.empty:
        logging.info("Missing values:\n" + str(missing))
    else:
        logging.info("No missing values found.")
    return missing

def label_encode(df, categorical_columns):
    """
    Label encode only binary categorical columns in the dataframe in place.
    """
    le = LabelEncoder()
    for col in categorical_columns:
        if df[col].nunique() == 2:
            df[col] = le.fit_transform(df[col])
            logging.info(f"Label encoded column: {col}")
    return df

def handle_outliers(df):
    """
    Handle outliers by capping them using the IQR method.
    """
    numerical_cols = df.select_dtypes(include=["int64", "float64"]).columns
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        df[col] = df[col].clip(lower_bound, upper_bound)
        logging.info(f"Outliers capped for column: {col}")
    return df
