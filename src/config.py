import os

class Config:
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))

    DATA_PATH = os.path.join(BASE_DIR, "../data/telecom_churn.csv")
    METRICS_PATH = os.path.join(BASE_DIR, "../results/metrics/")
    PREDICTIONS_PATH = os.path.join(BASE_DIR, "../results/predictions/")
    EXPLAINABILITY_PATH = os.path.join(BASE_DIR, "../results/explainability/")
    CAUSAL_PATH = os.path.join(BASE_DIR, "../results/causal/")

    TARGET_COLUMN = "Churn"
    TEST_SIZE = 0.2
    RANDOM_STATE = 42

    MODELS = ["logistic_regression", "xgboost", "decision_tree"]

    # ================== PIPELINE CONFIGURATION ==================
    PIPELINE_CONFIG = {
        # 1. COLUMNS DEFINITION
        "columns": {
            # Binary Yes/No columns -> encode as 0/1
            "binary": [
                "gender", "Partner", "Dependents", "PhoneService", 
                "PaperlessBilling", "Churn"
            ],
            # Service columns with "No internet/phone service" -> treat as No, then binary
            "service_binary": [
                "MultipleLines", "OnlineSecurity", "OnlineBackup",
                "DeviceProtection", "TechSupport", "StreamingTV", "StreamingMovies"
            ],
            # Multi-category -> One-Hot encode
            "categorical": ["InternetService", "Contract", "PaymentMethod"],
            # Numeric columns
            "numeric": ["tenure", "MonthlyCharges", "TotalCharges", "SeniorCitizen"],
            # Columns to drop
            "drop": ["customerID"]
        },

        # 2. IMPUTATION
        "imputation": {
            "deterministic": {"col": "TotalCharges", "value": 0, "condition": "tenure == 0"},
            "fallback": "KNNImputer",
            "n_neighbors": 5
        },

        # 3. TRANSFORMATIONS
        "transformations": {
            "log1p": ["TotalCharges", "MonthlyCharges"],
            "normalize": ["tenure", "TotalCharges", "MonthlyCharges", "SeniorCitizen"]
        },

        # 4. SPLITTING
        "splitting": {
            "method": "StratifiedShuffleSplit",
            "test_size": 0.2,
            "random_state": 42
        },

        # 5. CROSS-VALIDATION
        "cv": {
            "method": "StratifiedKFold",
            "n_splits": 5,
            "shuffle": True,
            "random_state": 42
        },

        # 6. HYPERPARAMETER TUNING (Optuna)
        "tuning": {
            "n_trials": 50,
            "metric": "recall",
            "direction": "maximize",
            "param_space": {
                "C": {"type": "loguniform", "low": 0.001, "high": 100},
                "l1_ratio": {"type": "uniform", "low": 0.0, "high": 1.0}
            }
        },

        # 7. LOGISTIC REGRESSION CONFIG
        "logistic_regression": {
            "penalty": "elasticnet",
            "solver": "saga",
            "class_weight": "balanced",
            "max_iter": 2000,
            "random_state": 42
        },

        # 8. THRESHOLD OPTIMIZATION
        "threshold": {
            "optimize": True,
            "metric": "recall",  # Will also compute F1-optimized thresholds
            "search_range": [0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]
        },

        # 9. MONTE CARLO SIMULATION
        "monte_carlo": {
            "n_simulations": 100,
            "confidence_level": 0.95,
            "n_jobs": -1
        },

        # 10. DOUBLE ML (Causal Inference)
        "double_ml": {
            "treatment_variable": "Contract",  # Month-to-month vs longer
            "treatment_value": "Month-to-month",
            "n_folds": 5
        },

        # 11. EXPLAINABILITY
        "explainability": {
            "shap": True,
            "lime": True,
            "n_samples_lime": 100
        }
    }
