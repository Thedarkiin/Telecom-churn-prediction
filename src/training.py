from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import train_test_split
import numpy as np

from src.utils import setup_logger

logger = setup_logger(__name__)

def train_models(X_train, y_train):
    logger.info("Training models...")

    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    models = {
        "xgboost": XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            use_label_encoder=False,
            random_state=42
        ),
        "logistic_regression": LogisticRegression(max_iter=1000),
        "decision_tree": DecisionTreeClassifier()
    }

    trained_models = {}
    for name, model in models.items():
        logger.info(f"Training {name}...")
        model.fit(X_train, y_train)
        trained_models[name] = model
        logger.info(f"{name} trained.")

    return trained_models
