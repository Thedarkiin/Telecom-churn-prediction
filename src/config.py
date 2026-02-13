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
            "drop": ["customerID", "TotalCharges"]
        },

        # 2. IMPUTATION
        "imputation": {
            "deterministic": {"col": "TotalCharges", "value": 0, "condition": "tenure == 0"},
            "fallback": "KNNImputer",
            "n_neighbors": 5
        },

        # 3. TRANSFORMATIONS
        "transformations": {
            "log1p": [],  # Disabled - log transform may hurt LR performance
            "normalize": ["tenure", "TotalCharges", "MonthlyCharges", "SeniorCitizen"]
        },

        # 4. FEATURE ENGINEERING
        "feature_engineering": {
            # Interaction terms (non-additive effects)
            "interaction_pairs": [
                ["tenure", "MonthlyCharges"]  # Capture Customer Lifetime Value
            ],
            # Discretization for non-linear patterns
            "binning": {
                "columns": ["tenure", "MonthlyCharges"],
                "n_bins": 4,
                "strategy": "quantile"  # Equal frequency bins
            },
            # Polynomial features (DISABLED - user preference)
            "polynomial": {
                "enabled": False,  # Disabled per user request
                "columns": ["tenure", "MonthlyCharges", "TotalCharges"],
                "degree": 2
            },
            # Correlation-based feature selection (DISABLED - let regularization handle it)
            "correlation": {
                "enabled": False,  # ElasticNet handles multicollinearity via L1/L2
                "min_threshold": 0.01,
                "max_threshold": 0.95
            },
            # VIF-based multicollinearity removal (DISABLED - too aggressive)
            "vif": {
                "enabled": False,  # Regularization is sufficient
                "threshold": 10.0
            }
        },


        # 5. SPLITTING
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

        # 6. HYPERPARAMETER TUNING
        "tuning": {
            "cv_splits": 5,
            # XGBoost Search Space (Optuna)
            "xgboost": {
                "n_trials": 20,
                "param_space": {
                    "max_depth": {"type": "int", "low": 3, "high": 9},
                    "learning_rate": {"type": "loguniform", "low": 0.01, "high": 0.3},
                    "subsample": {"type": "float", "low": 0.6, "high": 1.0},
                    "colsample_bytree": {"type": "float", "low": 0.6, "high": 1.0},
                    "reg_alpha": {"type": "loguniform", "low": 1e-3, "high": 10.0},
                    "reg_lambda": {"type": "loguniform", "low": 1e-3, "high": 10.0}
                }
            },
            # Decision Tree Search Space (Grid/Random)
            "decision_tree": {
                "n_iter": 50,
                "param_space": {
                    "max_depth": [4, 6, 8, 10, 12, None],
                    "min_samples_split": [10, 20, 40, 60],
                    "min_samples_leaf": [5, 10, 20],
                    "max_features": ["sqrt", None]
                }
            }
        },

        # 7. LOGISTIC REGRESSION CONFIG
        "logistic_regression": {
            "penalty": "l2",
            "solver": "lbfgs",
            "class_weight": "balanced",
            "max_iter": 2000,
            "random_state": 42
        },

        # 8. THRESHOLD OPTIMIZATION
        "threshold": {
            "optimize": False,  # Use fixed threshold
            "metric": "recall",
            "default_threshold": 0.52,  # 52% probability for churn classification
            "search_range": [0.52]  # Fixed at 52%
        },


        # 9. MONTE CARLO SIMULATION (DISABLED)
        "monte_carlo": {
            "enabled": False,  # Disabled - using standard LR only
            "n_simulations": 30,
            "confidence_level": 0.95,
            "n_jobs": 1
        },

        # 10. DOUBLE ML (Causal Inference)
        "double_ml": {
            "treatment_variable": "Contract_Two year",  # Protective factor vs Month-to-month (reference)
            "treatment_value": "Contract_Two year",  # Exact column name after one-hot encoding
            "n_folds": 5
        },


        # 11. EXPLAINABILITY
        "explainability": {
            "shap": True,
            "lime": False,  # Disabled - SHAP plots are sufficient
            "n_samples_lime": 0
        }
    }
