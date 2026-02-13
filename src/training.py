"""
Training module with Optuna hyperparameter optimization, 
stratified K-fold CV, and optimal threshold selection.
"""

import os
import logging
import numpy as np
import pandas as pd

try:
    import optuna
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import recall_score, f1_score, precision_score
from xgboost import XGBClassifier
from src.utils import setup_logger

logger = setup_logger(__name__)

def train_xgboost_with_optuna(X_train, y_train, config):
    """
    Train XGBoost Classifier using Optuna for hyperparameter optimization.

    Args:
        X_train (pd.DataFrame or np.ndarray): Training features.
        y_train (pd.Series or np.ndarray): Training labels.
        config (Config): Configuration object containing tuning parameters.

    Returns:
        XGBClassifier: The fitted best model.
    """
    if not OPTUNA_AVAILABLE:
        logger.warning("Optuna missing, using default XGBoost.")
        scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)
        model = XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=config.RANDOM_STATE)
        model.fit(X_train, y_train)
        return model

    # Tuning config
    xg_conf = config.PIPELINE_CONFIG["tuning"]["xgboost"]
    cv_config = config.PIPELINE_CONFIG["cv"]
    cv = StratifiedKFold(n_splits=cv_config["n_splits"], shuffle=True, random_state=config.RANDOM_STATE)
    scale_pos_weight = (y_train == 0).sum() / max((y_train == 1).sum(), 1)

    def objective(trial):
        param = {
            'n_estimators': 300,
            'max_depth': trial.suggest_int('max_depth', 3, 9),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'min_child_weight': 5,
            'scale_pos_weight': scale_pos_weight,
            'eval_metric': 'logloss',
            'n_jobs': 1,  # Parallelism handled by Optuna if needed, or outer loop
            'random_state': config.RANDOM_STATE
        }
        
        model = XGBClassifier(**param)
        # Optimize ROC-AUC to balance precision/recall capability
        score = cross_val_score(model, X_train, y_train, cv=cv, scoring='roc_auc', n_jobs=1).mean()
        return score

    logger.info("Tuning XGBoost with Optuna...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=xg_conf["n_trials"], show_progress_bar=True)
    
    best_params = study.best_trial.params
    # Add fixed params back
    best_params.update({
        'n_estimators': 300, 
        'min_child_weight': 5, 
        'scale_pos_weight': scale_pos_weight,
        'eval_metric': 'logloss', 
        'random_state': config.RANDOM_STATE
    })
    
    logger.info(f"Best XGB Params: depth={best_params['max_depth']}, lr={best_params['learning_rate']:.3f}, AUC={study.best_value:.4f}")
    
    final_model = XGBClassifier(**best_params)
    final_model.fit(X_train, y_train)
    return final_model


def train_dt_with_search(X_train, y_train, config):
    """
    Train Decision Tree Classifier using RandomizedSearchCV.

    Args:
        X_train (array-like): Training features.
        y_train (array-like): Training labels.
        config (Config): Configuration object.

    Returns:
        DecisionTreeClassifier: The fitted best estimator.
    """
    dt_conf = config.PIPELINE_CONFIG["tuning"]["decision_tree"]
    
    param_dist = dt_conf["param_space"]
    # Add Fixed params
    param_dist['class_weight'] = ['balanced', None]
    
    dt = DecisionTreeClassifier(random_state=config.RANDOM_STATE)
    
    search = RandomizedSearchCV(
        dt, 
        param_distributions=param_dist,
        n_iter=dt_conf["n_iter"],
        scoring='roc_auc',
        cv=5,
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )
    
    logger.info("Tuning Decision Tree...")
    search.fit(X_train, y_train)
    logger.info(f"Best DT Params: {search.best_params_}, AUC={search.best_score_:.4f}")
    
    return search.best_estimator_





def find_optimal_threshold(model, X_val, y_val, metric='recall', constraint_metric=None, constraint_value=None):
    """
    Find optimal classification threshold by maximizing a metric, optionally with a constraint.
    
    Parameters
    ----------
    model : fitted model
    X_val : array-like
    y_val : array-like
    metric : str ('recall', 'f1')
    constraint_metric : str ('precision'), optional
    constraint_value : float, optional
         Minimum value for constraint metric (e.g. 0.5 for Precision >= 0.5)
        
    Returns
    -------
    optimal_threshold : float
    best_score : float
    """
    # Search range from 0.3 to 0.7 (avoiding extremes)
    search_range = np.arange(0.3, 0.75, 0.01)
    
    probs = model.predict_proba(X_val)[:, 1]
    
    best_threshold = 0.5
    best_score = -1
    
    for threshold in search_range:
        y_pred = (probs >= threshold).astype(int)
        
        # Check constraint first
        if constraint_metric == 'precision':
            prec = precision_score(y_val, y_pred, zero_division=0)
            if prec < constraint_value:
                continue # Skip if constraint not met
                
        if metric == 'recall':
            score = recall_score(y_val, y_pred, zero_division=0)
        elif metric == 'f1':
            score = f1_score(y_val, y_pred, zero_division=0)
        else:
            score = recall_score(y_val, y_pred, zero_division=0)
        
        if score > best_score:
            best_score = score
            best_threshold = threshold
    
    if best_score == -1:
        logger.warning(f"No threshold constrained satisfaction found. Defaulting to 0.5")
        best_threshold = 0.5
        best_score = recall_score(y_val, (probs >= 0.5).astype(int))

    logger.info(f"Optimal threshold: {best_threshold:.3f} ({metric}={best_score:.4f} | precision>={constraint_value})")
    return best_threshold, best_score


def train_logistic_cv(X_train, y_train, config):
    """
    Train Logistic Regression with efficient Cross-Validation (L2 penalty).
    Equivalent to Grid Search but faster and cleaner.
    """
    lr_config = config.PIPELINE_CONFIG["logistic_regression"]
    
    # LogisticRegressionCV with L2 regularization
    # Automatically tries 10 values of C logarithmically (e.g. 1e-4 to 1e4)
    model = LogisticRegressionCV(
        Cs=20,  # Try 20 values for C
        cv=5, 
        penalty='l2',
        solver='lbfgs',
        scoring='roc_auc',  # Optimize ROC-AUC during CV
        class_weight='balanced',
        max_iter=3000,
        random_state=config.RANDOM_STATE,
        n_jobs=-1
    )
    
    model.fit(X_train, y_train)
    logger.info(f"LR CV Best C: {model.C_[0]:.4f}")
    logger.info(f"LR CV Best Score (ROC-AUC): {model.scores_[1].mean(axis=0).max():.4f}")
    
    return model, None








def train_models(X_train, y_train, config=None):
    """
    Train all models with proper class balancing and hyperparameter tuning.
    """
    from src.config import Config
    if config is None:
        config = Config
    
    logger.info("Training models...")
    
    # 1. Train Logistic Regression with Efficient CV (L2 Ridge)
    logger.info("Training Logistic Regression with CV (L2 Ridge)...")
    lr_model, _ = train_logistic_cv(X_train, y_train, config)
    
    # 2. Train XGBoost (Optuna Tuned)
    logger.info("Training XGBoost (Optuna)...")
    xgb_model = train_xgboost_with_optuna(X_train, y_train, config)
    
    # 3. Train Decision Tree (RandomizedSearch)
    logger.info("Training Decision Tree (RandomizedSearch)...")
    dt_model = train_dt_with_search(X_train, y_train, config)
    
    trained_models = {
        "logistic_regression": lr_model,
        "xgboost": xgb_model,
        "decision_tree": dt_model
    }
    
    # 4. Find optimal thresholds (Precision >= 0.5 constraint)
    logger.info("Finding optimal thresholds (Constraint: Precision >= 0.5)...")
    optimal_thresholds = {}
    optimal_thresholds_f1 = {}
    
    for name, model in trained_models.items():
        if hasattr(model, 'predict_proba'):
            val_size = int(0.2 * len(X_train))
            X_val, y_val = X_train[-val_size:], y_train[-val_size:]
            
            # Recall Optimization (Constraint: Precision >= 0.5)
            logger.info(f"{name} - Optimizing Recall (Precision >= 0.5):")
            opt_thresh_recall, _ = find_optimal_threshold(
                model, X_val, y_val,
                metric='recall',
                constraint_metric='precision',
                constraint_value=0.5
            )
            optimal_thresholds[name] = opt_thresh_recall
            
            # F1 Optimization
            logger.info(f"{name} - Optimizing F1:")
            opt_thresh_f1, _ = find_optimal_threshold(
                model, X_val, y_val,
                metric='f1'
            )
            optimal_thresholds_f1[name] = opt_thresh_f1
        else:
            optimal_thresholds[name] = 0.5
            optimal_thresholds_f1[name] = 0.5
            
    logger.info("All models trained successfully.")
    
    return trained_models, optimal_thresholds, optimal_thresholds_f1, None

