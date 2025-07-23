import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)

from src.utils import save_plot, setup_logger
from src.config import Config

logger = setup_logger(__name__)

def evaluate_models(models, X_test, y_test):
    os.makedirs(Config.METRICS_PATH, exist_ok=True)
    os.makedirs(Config.PREDICTIONS_PATH, exist_ok=True)

    all_metrics = []

    for name, model in models.items():
        logger.info(f"Evaluating {name}")
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        auc = roc_auc_score(y_test, y_pred)

        all_metrics.append({
            "model": name,
            "accuracy": acc,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "roc_auc": auc
        })

        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots(figsize=(6, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
        save_plot(fig, f"{name}_confusion_matrix.png", Config.METRICS_PATH)

        dfp = pd.DataFrame({"Actual": y_test, "Predicted": y_pred})
        dfp.to_csv(os.path.join(Config.PREDICTIONS_PATH, f"{name}_predictions.csv"), index=False)

        if name == "xgboost":
            importances = model.feature_importances_
            feature_df = pd.DataFrame({
                "Feature": [f"f{i}" for i in range(len(importances))],
                "Importance": importances
            }).sort_values("Importance", ascending=False)
            feature_df.to_csv(f"{Config.METRICS_PATH}/{name}_feature_importance.csv", index=False)

    dfm = pd.DataFrame(all_metrics)
    dfm.to_csv(os.path.join(Config.METRICS_PATH, "all_metrics.csv"), index=False)
