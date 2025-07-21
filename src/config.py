import os

# Base path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Data paths
DATA_PATH = os.path.join(BASE_DIR, "../data/telecom_churn.csv")

# EDA
EDA_OUTPUT = os.path.join(BASE_DIR, "../results/eda/")
PLOTS_PATH = os.path.join(BASE_DIR, "../results/plots/")
METRICS_PATH = os.path.join(BASE_DIR, "../results/metrics/")
PREDICTIONS_PATH = os.path.join(BASE_DIR, "../results/predictions/")

# Preprocessing
TARGET_COLUMN = "Churn"  # Make sure this is correct based on actual CSV
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model Configs (we will use these in model training)
MODELS = ["xgboost", "logistic_regression", "decision_tree"]
