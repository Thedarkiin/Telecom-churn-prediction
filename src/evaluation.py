"""
Evaluation module with comprehensive metrics, uncertainty quantification,
and threshold-aware predictions.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import logging

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    precision_recall_curve
)

from src.utils import save_plot, setup_logger
from src.config import Config

logger = setup_logger(__name__)


def evaluate_with_threshold(model, X_test, y_test, threshold=0.5):
    """
    Evaluate model using a specific threshold.
    
    Parameters
    ----------
    model : fitted model
        Model with predict_proba.
    X_test : array-like
        Test features.
    y_test : array-like
        Test labels.
    threshold : float
        Classification threshold.
        
    Returns
    -------
    metrics : dict
        Dictionary of metrics.
    y_pred : array
        Predictions at given threshold.
    """
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_test)[:, 1]
        y_pred = (probs >= threshold).astype(int)
    else:
        y_pred = model.predict(X_test)
        probs = y_pred
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probs),
        "threshold": threshold
    }
    
    return metrics, y_pred


def evaluate_monte_carlo(mc_model, X_test, y_test, threshold=0.5, alpha=0.05):
    """
    Evaluate Monte Carlo model with uncertainty metrics.
    
    Parameters
    ----------
    mc_model : MonteCarloLogisticRegression
        Fitted MC model.
    X_test : array-like
        Test features.
    y_test : array-like
        Test labels.
    threshold : float
        Classification threshold.
    alpha : float
        Significance level for CI.
        
    Returns
    -------
    metrics : dict
        Metrics including uncertainty measures.
    uncertainty_df : DataFrame
        Per-sample uncertainty data.
    """
    # Get predictions with intervals
    mean_prob, lower, upper, uncertainty = mc_model.predict_proba_interval(X_test, alpha=alpha)
    
    y_pred = (mean_prob >= threshold).astype(int)
    
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y_test, mean_prob),
        "threshold": threshold,
        "mean_uncertainty": uncertainty.mean(),
        "median_uncertainty": np.median(uncertainty),
        "high_uncertainty_pct": (uncertainty > 0.3).mean() * 100
    }
    
    uncertainty_df = pd.DataFrame({
        "actual": y_test,
        "predicted": y_pred,
        "prob_mean": mean_prob,
        "prob_lower": lower,
        "prob_upper": upper,
        "uncertainty": uncertainty
    })
    
    return metrics, uncertainty_df


def plot_roc_curve(model, X_test, y_test, model_name, output_dir):
    """Plot and save ROC curve."""
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        probs = model.predict(X_test)
    
    fpr, tpr, thresholds = roc_curve(y_test, probs)
    auc = roc_auc_score(y_test, probs)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(fpr, tpr, label=f'{model_name} (AUC = {auc:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', label='Random')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC Curve - {model_name}')
    ax.legend()
    
    save_plot(fig, f"{model_name}_roc_curve.png", output_dir)
    plt.close()


def plot_precision_recall_curve(model, X_test, y_test, model_name, output_dir):
    """Plot and save Precision-Recall curve."""
    if hasattr(model, 'predict_proba'):
        probs = model.predict_proba(X_test)[:, 1]
    else:
        return
    
    precision, recall, thresholds = precision_recall_curve(y_test, probs)
    
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(recall, precision)
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.set_title(f'Precision-Recall Curve - {model_name}')
    
    save_plot(fig, f"{model_name}_pr_curve.png", output_dir)
    plt.close()


def plot_uncertainty_distribution(uncertainty_df, model_name, output_dir):
    """Plot uncertainty distribution for Monte Carlo predictions."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Uncertainty histogram
    axes[0].hist(uncertainty_df['uncertainty'], bins=30, edgecolor='black', alpha=0.7)
    axes[0].set_xlabel('Prediction Uncertainty (CI Width)')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Distribution of Prediction Uncertainty')
    axes[0].axvline(uncertainty_df['uncertainty'].median(), color='r', linestyle='--', 
                    label=f'Median: {uncertainty_df["uncertainty"].median():.3f}')
    axes[0].legend()
    
    # Uncertainty by prediction correctness
    correct = uncertainty_df[uncertainty_df['actual'] == uncertainty_df['predicted']]['uncertainty']
    incorrect = uncertainty_df[uncertainty_df['actual'] != uncertainty_df['predicted']]['uncertainty']
    
    axes[1].boxplot([correct, incorrect], labels=['Correct', 'Incorrect'])
    axes[1].set_ylabel('Uncertainty')
    axes[1].set_title('Uncertainty by Prediction Correctness')
    
    plt.tight_layout()
    save_plot(fig, f"{model_name}_uncertainty.png", output_dir)
    plt.close()


def evaluate_models(models, X_test, y_test, optimal_thresholds=None, feature_names=None):
    """
    Evaluate all models with comprehensive metrics.
    
    Parameters
    ----------
    models : dict
        Dictionary of trained models.
    X_test : array-like
        Test features.
    y_test : array-like
        Test labels.
    optimal_thresholds : dict, optional
        Optimal thresholds per model.
    feature_names : list, optional
        Feature names for coefficient analysis.
        
    Returns
    -------
    all_metrics : DataFrame
        Metrics for all models.
    """
    os.makedirs(Config.METRICS_PATH, exist_ok=True)
    os.makedirs(Config.PREDICTIONS_PATH, exist_ok=True)
    
    if optimal_thresholds is None:
        optimal_thresholds = {name: 0.5 for name in models.keys()}
    
    all_metrics = []
    
    for name, model in models.items():
        logger.info(f"Evaluating {name}")
        
        threshold = optimal_thresholds.get(name, 0.5)
        
        # Special handling for Monte Carlo model
        if name == "monte_carlo_lr":
            metrics, uncertainty_df = evaluate_monte_carlo(
                model, X_test, y_test, threshold=threshold
            )
            # Save uncertainty data
            uncertainty_df.to_csv(
                os.path.join(Config.PREDICTIONS_PATH, f"{name}_uncertainty.csv"), 
                index=False
            )
            # Plot uncertainty
            plot_uncertainty_distribution(uncertainty_df, name, Config.METRICS_PATH)
            
            # Get coefficient intervals
            if feature_names:
                coef_intervals = model.get_coefficient_intervals(feature_names)
                if coef_intervals is not None:
                    coef_intervals.to_csv(
                        os.path.join(Config.METRICS_PATH, f"{name}_coef_intervals.csv"),
                        index=False
                    )
        else:
            metrics, y_pred = evaluate_with_threshold(model, X_test, y_test, threshold)
            
            # Confusion matrix
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots(figsize=(6, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                       xticklabels=['No Churn', 'Churn'],
                       yticklabels=['No Churn', 'Churn'])
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            ax.set_title(f'{name} Confusion Matrix (threshold={threshold:.2f})')
            save_plot(fig, f"{name}_confusion_matrix.png", Config.METRICS_PATH)
            plt.close()
            
            # ROC curve
            plot_roc_curve(model, X_test, y_test, name, Config.METRICS_PATH)
            
            # PR curve
            plot_precision_recall_curve(model, X_test, y_test, name, Config.METRICS_PATH)
            
            # Save predictions
            if hasattr(model, 'predict_proba'):
                probs = model.predict_proba(X_test)[:, 1]
            else:
                probs = y_pred
                
            dfp = pd.DataFrame({
                "Actual": y_test, 
                "Predicted": y_pred,
                "Probability": probs
            })
            dfp.to_csv(os.path.join(Config.PREDICTIONS_PATH, f"{name}_predictions.csv"), index=False)
        
        metrics["model"] = name
        all_metrics.append(metrics)
        
        # Log metrics
        logger.info(f"  {name} Results (threshold={threshold:.2f}):")
        logger.info(f"    Accuracy:  {metrics['accuracy']:.4f}")
        logger.info(f"    Precision: {metrics['precision']:.4f}")
        logger.info(f"    Recall:    {metrics['recall']:.4f}")
        logger.info(f"    F1:        {metrics['f1']:.4f}")
        logger.info(f"    ROC-AUC:   {metrics['roc_auc']:.4f}")
        
        # Feature importance for XGBoost
        if name == "xgboost" and feature_names:
            importances = model.feature_importances_
            feature_df = pd.DataFrame({
                "Feature": feature_names[:len(importances)],
                "Importance": importances
            }).sort_values("Importance", ascending=False)
            feature_df.to_csv(f"{Config.METRICS_PATH}/{name}_feature_importance.csv", index=False)
    
    # Save all metrics
    dfm = pd.DataFrame(all_metrics)
    
    # Add Git Hash
    try:
        import subprocess
        git_hash = subprocess.check_output(['git', 'rev-parse', '--short', 'HEAD']).strip().decode('utf-8')
        dfm['git_hash'] = git_hash
    except Exception as e:
        logger.warning(f"Could not retrieve git hash: {e}")
        dfm['git_hash'] = "unknown"
        
    dfm.to_csv(os.path.join(Config.METRICS_PATH, "all_metrics.csv"), index=False)
    logger.info("Saved all metrics to all_metrics.csv")
    
    return dfm
