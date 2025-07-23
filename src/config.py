import os

class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    DATA_PATH = os.path.join(BASE_DIR, "../data/telecom_churn.csv")
    METRICS_PATH = os.path.join(BASE_DIR, "../results/metrics/")
    PREDICTIONS_PATH = os.path.join(BASE_DIR, "../results/predictions/")

    TARGET_COLUMN = "Churn"  # Verify this matches your dataset column name exactly
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    MODELS = ["xgboost", "logistic_regression", "decision_tree"]
