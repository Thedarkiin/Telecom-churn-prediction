"""
Training module with Optuna hyperparameter optimization, 
stratified K-fold CV, and optimal threshold selection.
"""

import os
import logging
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.metrics import recall_score, f1_score, precision_score, make_scorer
from xgboost import XGBClassifier

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from src.utils import setup_logger
from src.monte_carlo_lr import MonteCarloLogisticRegression

logger = setup_logger(__name__)


def find_optimal_threshold(model, X_val, y_val, metric='recall', search_range=None):
    """
    Find optimal classification threshold by maximizing a metric.
    
    Parameters
    ----------
    model : fitted model
        Model with predict_proba.
    X_val : array-like
        Validation features.
    y_val : array-like
        Validation labels.
    metric : str
        Metric to optimize ('recall', 'f1', 'youden_j').
    search_range : list
        Thresholds to search.
        
    Returns
    -------
    optimal_threshold : float
        Best threshold found.
    best_score : float
        Score at optimal threshold.
    """
    if search_range is None:
        search_range = np.arange(0.1, 0.9, 0.05)
    
    probs = model.predict_proba(X_val)[:, 1]
    
    best_threshold = 0.5
    best_score = 0
    
    for threshold in search_range:
        y_pred = (probs >= threshold).astype(int)
        
        if metric == 'recall':
            score = recall_score(y_val, y_pred, zero_division=0)
        elif metric == 'f1':
            score = f1_score(y_val, y_pred, zero_division=0)
        elif metric == 'youden_j':
            # Youden's J = sensitivity + specificity - 1
            tp = np.sum((y_pred == 1) & (y_val == 1))
            tn = np.sum((y_pred == 0) & (y_val == 0))
            fp = np.sum((y_pred == 1) & (y_val == 0))
            fn = np.sum((y_pred == 0) & (y_val == 1))
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            score = sensitivity + specificity - 1
        else:
            score = recall_score(y_val, y_pred, zero_division=0)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    logger.info(f"Optimal threshold: {best_threshold:.3f} ({metric}={best_score:.4f})")
    return best_threshold, best_score


def optuna_objective(trial, X, y, cv, config):
    """Optuna objective function for logistic regression tuning."""
    
    param_space = config.PIPELINE_CONFIG["tuning"]["param_space"]
    lr_config = config.PIPELINE_CONFIG["logistic_regression"]
    
    # Sample hyperparameters
    C = trial.suggest_float("C", param_space["C"]["low"], param_space["C"]["high"], log=True)
    l1_ratio = trial.suggest_float("l1_ratio", param_space["l1_ratio"]["low"], param_space["l1_ratio"]["high"])
    
    model = LogisticRegression(
        C=C,
        l1_ratio=l1_ratio,
        penalty=lr_config["penalty"],
        solver=lr_config["solver"],
        class_weight=lr_config["class_weight"],
        max_iter=lr_config["max_iter"],
        random_state=lr_config["random_state"]
    )
    
    # Cross-validation
    metric = config.PIPELINE_CONFIG["tuning"]["metric"]
    if metric == "recall":
        scorer = make_scorer(recall_score)
    elif metric == "f1":
        scorer = make_scorer(f1_score)
    else:
        scorer = make_scorer(recall_score)
    
    scores = cross_val_score(model, X, y, cv=cv, scoring=scorer)
    
    return scores.mean()


def train_logistic_with_optuna(X_train, y_train, config):
    """
    Train logistic regression with Optuna hyperparameter optimization.
    
    Returns
    -------
    model : fitted LogisticRegression
        Best model found.
    study : optuna.Study
        Optuna study object.
    """
    if not OPTUNA_AVAILABLE:
        logger.warning("Optuna not installed. Using default hyperparameters.")
        lr_config = config.PIPELINE_CONFIG["logistic_regression"]
        model = LogisticRegression(
            C=1.0,
            l1_ratio=0.5,
            penalty=lr_config["penalty"],
            solver=lr_config["solver"],
            class_weight=lr_config["class_weight"],
            max_iter=lr_config["max_iter"],
            random_state=lr_config["random_state"]
        )
        model.fit(X_train, y_train)
        return model, None
    
    # Setup CV
    cv_config = config.PIPELINE_CONFIG["cv"]
    cv = StratifiedKFold(
        n_splits=cv_config["n_splits"],
        shuffle=cv_config["shuffle"],
        random_state=cv_config["random_state"]
    )
    
    # Run Optuna
    tuning_config = config.PIPELINE_CONFIG["tuning"]
    
    study = optuna.create_study(direction=tuning_config["direction"])
    study.optimize(
        lambda trial: optuna_objective(trial, X_train, y_train, cv, config),
        n_trials=tuning_config["n_trials"],
        show_progress_bar=True
    )
    
    logger.info(f"Best trial: {study.best_trial.params}")
    logger.info(f"Best CV {tuning_config['metric']}: {study.best_value:.4f}")
    
    # Train final model with best params
    lr_config = config.PIPELINE_CONFIG["logistic_regression"]
    best_params = study.best_trial.params
    
    model = LogisticRegression(
        C=best_params["C"],
        l1_ratio=best_params["l1_ratio"],
        penalty=lr_config["penalty"],
        solver=lr_config["solver"],
        class_weight=lr_config["class_weight"],
        max_iter=lr_config["max_iter"],
        random_state=lr_config["random_state"]
    )
    model.fit(X_train, y_train)
    
    # Save study results
    os.makedirs(config.METRICS_PATH, exist_ok=True)
    trials_df = study.trials_dataframe()
    trials_df.to_csv(os.path.join(config.METRICS_PATH, "optuna_study.csv"), index=False)
    logger.info("Saved Optuna study to optuna_study.csv")
    
    return model, study


def train_models(X_train, y_train, config=None):
    """
    Train all models with proper class balancing and hyperparameter tuning.
    
    Parameters
    ----------
    X_train : array-like
        Training features.
    y_train : array-like
        Training labels.
    config : Config, optional
        Configuration object. If None, uses defaults.
        
    Returns
    -------
    trained_models : dict
        Dictionary of trained models.
    optimal_thresholds : dict
        Optimal thresholds per model.
    optuna_study : optuna.Study or None
        Optuna study if used.
    """
    from src.config import Config
    if config is None:
        config = Config
    
    logger.info("Training models...")
    
    # Class imbalance ratio for XGBoost
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
    
    # 1. Train Logistic Regression with Optuna
    logger.info("Training Logistic Regression with Optuna optimization...")
    lr_model, optuna_study = train_logistic_with_optuna(X_train, y_train, config)
    
    # 2. Other models
    models = {
        "logistic_regression": lr_model,
        "xgboost": XGBClassifier(
            n_estimators=500,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=scale_pos_weight,
            random_state=config.RANDOM_STATE,
            eval_metric='logloss'
        ),
        "decision_tree": DecisionTreeClassifier(
            class_weight="balanced",
            random_state=config.RANDOM_STATE
        )
    }
    
    trained_models = {"logistic_regression": lr_model}
    
    for name, model in models.items():
        if name != "logistic_regression":
            logger.info(f"Training {name}...")
            model.fit(X_train, y_train)
            trained_models[name] = model
            logger.info(f"{name} trained.")
    
    # 3. Find optimal thresholds for both recall and F1
    logger.info("Finding optimal thresholds for recall and F1...")
    threshold_config = config.PIPELINE_CONFIG["threshold"]
    optimal_thresholds = {}
    optimal_thresholds_f1 = {}
    
    for name, model in trained_models.items():
        if hasattr(model, 'predict_proba'):
            # Use a portion of training data as validation for threshold
            val_size = int(0.2 * len(X_train))
            X_val, y_val = X_train[-val_size:], y_train[-val_size:]
            
            # Optimize for recall
            logger.info(f"{name} - Optimizing for recall:")
            opt_thresh_recall, _ = find_optimal_threshold(
                model, X_val, y_val,
                metric='recall',
                search_range=threshold_config["search_range"]
            )
            optimal_thresholds[name] = opt_thresh_recall
            
            # Optimize for F1
            logger.info(f"{name} - Optimizing for F1:")
            opt_thresh_f1, _ = find_optimal_threshold(
                model, X_val, y_val,
                metric='f1',
                search_range=threshold_config["search_range"]
            )
            optimal_thresholds_f1[name] = opt_thresh_f1
        else:
            optimal_thresholds[name] = 0.5
            optimal_thresholds_f1[name] = 0.5

    
    # 4. Create Monte Carlo wrapper for LR
    logger.info("Creating Monte Carlo Logistic Regression for uncertainty...")
    mc_config = config.PIPELINE_CONFIG["monte_carlo"]
    
    mc_lr = MonteCarloLogisticRegression(
        base_model=lr_model,
        n_simulations=mc_config["n_simulations"],
        n_jobs=mc_config["n_jobs"],
        random_state=config.RANDOM_STATE
    )
    mc_lr.fit(X_train, y_train)
    trained_models["monte_carlo_lr"] = mc_lr
    optimal_thresholds["monte_carlo_lr"] = optimal_thresholds.get("logistic_regression", 0.5)
    
    logger.info("All models trained successfully.")
    
    return trained_models, optimal_thresholds, optimal_thresholds_f1, optuna_study

