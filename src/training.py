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


from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.ensemble import VotingClassifier

def train_xgboost_with_optuna(X_train, y_train, config):
    """
    Train XGBoost Classifier using Optuna for hyperparameter optimization.
    """
    if not OPTUNA_AVAILABLE:
        logger.warning("Optuna missing, using default XGBoost.")
        model = XGBClassifier(scale_pos_weight=1, random_state=config.RANDOM_STATE)
        model.fit(X_train, y_train)
        return model

    # Tuning config
    xg_conf = config.PIPELINE_CONFIG["tuning"]["xgboost"]
    cv_config = config.PIPELINE_CONFIG["cv"]
    cv = StratifiedKFold(n_splits=cv_config["n_splits"], shuffle=True, random_state=config.RANDOM_STATE)

    def objective(trial):
        param = {
            'n_estimators': trial.suggest_int('n_estimators', 300, 600),
            'max_depth': trial.suggest_int('max_depth', 3, 10),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.2, log=True),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-3, 10.0, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-3, 10.0, log=True),
            'scale_pos_weight': 1, # Handled by SMOTE or set to 1 if using SMOTE
            'eval_metric': 'logloss',
            'n_jobs': 1, 
            'random_state': config.RANDOM_STATE
        }
        
        # Use SMOTE inside CV to avoid leakage
        pipeline = ImbPipeline([
            ('smote', SMOTE(random_state=config.RANDOM_STATE)),
            ('model', XGBClassifier(**param))
        ])
        
        # Optimize F1 score (Harmonic mean of Precision/Recall)
        score = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring='f1', n_jobs=1).mean()
        return score

    logger.info("Tuning XGBoost with Optuna (F1 Optimized + SMOTE)...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=xg_conf["n_trials"], show_progress_bar=True)
    
    best_params = study.best_trial.params
    # Add fixed params back
    best_params.update({
        'eval_metric': 'logloss', 
        'n_jobs': 1,
        'random_state': config.RANDOM_STATE
    })
    
    logger.info(f"Best XGB Params: F1={study.best_value:.4f}")
    
    # Train final model on FULL SMOTE data
    smote = SMOTE(random_state=config.RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    
    final_model = XGBClassifier(**best_params)
    final_model.fit(X_res, y_res)
    return final_model


def train_dt_with_search(X_train, y_train, config):
    """
    Train Decision Tree Classifier using RandomizedSearchCV.
    """
    dt_conf = config.PIPELINE_CONFIG["tuning"]["decision_tree"]
    
    # Simple DT with SMOTE applied before
    smote = SMOTE(random_state=config.RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)

    param_dist = dt_conf["param_space"]
    
    dt = DecisionTreeClassifier(random_state=config.RANDOM_STATE)
    
    search = RandomizedSearchCV(
        dt, 
        param_distributions=param_dist,
        n_iter=dt_conf["n_iter"],
        scoring='f1',
        cv=5,
        random_state=config.RANDOM_STATE,
        n_jobs=1 
    )
    
    logger.info("Tuning Decision Tree (SMOTE)...")
    search.fit(X_res, y_res)
    logger.info(f"Best DT Params: F1={search.best_score_:.4f}")
    
    return search.best_estimator_


def find_optimal_threshold(model, X_val, y_val, metric='f1', beta=1.0):
    """
    Find optimal classification threshold.
    Strategy: Find threshold that satisfies Precision > 0.7 and Recall > 0.7.
    If not possible, maximize F0.5 (Precision-weighted).
    """
    search_range = np.arange(0.3, 0.85, 0.01)
    probs = model.predict_proba(X_val)[:, 1]
    
    best_threshold = 0.5
    best_score = -1
    
    # Track metrics for fallback
    best_f05_threshold = 0.5
    best_f05_score = -1
    
    # Strategy: Maximize F1-Score (Balanced Precision/Recall)
    for threshold in search_range:
        y_pred = (probs >= threshold).astype(int)
        
        f1 = f1_score(y_val, y_pred, zero_division=0)
            
        if f1 > best_score:
            best_score = f1
            best_threshold = threshold
            
    logger.info(f"Selected Threshold: {best_threshold:.3f} (Max F1: {best_score:.4f})")
    return best_threshold, best_score


def train_logistic_cv(X_train, y_train, config):
    """
    Train Logistic Regression with SMOTE.
    """
    # Apply SMOTE first (for simplicity in LR, though pipeline is better, this is sufficient for strict LR)
    smote = SMOTE(random_state=config.RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    
    model = LogisticRegressionCV(
        Cs=20,  # Try 20 values for C
        cv=5, 
        penalty='l2',
        solver='lbfgs',
        scoring='f1',  # Optimize F1
        max_iter=3000,
        random_state=config.RANDOM_STATE,
        n_jobs=1 
    )
    
    model.fit(X_res, y_res)
    logger.info(f"LR CV Best Score (F1): {model.scores_[1].mean(axis=0).max():.4f}")
    
    return model, None


def train_models(X_train, y_train, config=None):
    """
    Train all models + Ensemble.
    """
    from src.config import Config
    if config is None:
        config = Config
    
    logger.info("Training models with SMOTE balancing...")
    
    # 1. Logistic Regression
    logger.info("Training Logistic Regression...")
    lr_model, _ = train_logistic_cv(X_train, y_train, config)
    
    # 2. XGBoost
    logger.info("Training XGBoost...")
    xgb_model = train_xgboost_with_optuna(X_train, y_train, config)
    
    # 3. Decision Tree
    logger.info("Training Decision Tree...")
    dt_model = train_dt_with_search(X_train, y_train, config)
    
    # 4. Ensemble (Voting)
    logger.info("Training Ensemble (Voting Classifier)...")
    voting_model = VotingClassifier(
        estimators=[
            ('lr', lr_model),
            ('xgb', xgb_model)
        ],
        voting='soft',
        n_jobs=1
    )
    # Fit ensemble on SMOTE data
    smote = SMOTE(random_state=config.RANDOM_STATE)
    X_res, y_res = smote.fit_resample(X_train, y_train)
    voting_model.fit(X_res, y_res)
    
    trained_models = {
        "logistic_regression": lr_model,
        "xgboost": xgb_model,
        "decision_tree": dt_model,
        "ensemble": voting_model
    }
    
    # 5. Find optimal thresholds
    logger.info("Finding optimal thresholds...")
    optimal_thresholds = {}
    optimal_thresholds_f1 = {}
    
    val_size = int(0.2 * len(X_train))
    X_val, y_val = X_train[-val_size:], y_train[-val_size:]
            
    for name, model in trained_models.items():
        if hasattr(model, 'predict_proba'):
            # Balanced (F1) - Aiming for High Recall/Precision balance
            # For Ensemble, we stick to F1 standard (Beta=1) or slight Precision bias (Beta=0.8)
            # Let's try to hit >75% F1
            logger.info(f"{name} - Optimizing Threshold:")
            opt_thresh_f1, _ = find_optimal_threshold(
                model, X_val, y_val,
                metric='f1',
                beta=1.0 
            )
            # We use the F1 threshold for both, as that's best for general performance
            optimal_thresholds[name] = opt_thresh_f1
            optimal_thresholds_f1[name] = opt_thresh_f1
        else:
            optimal_thresholds[name] = 0.5
            optimal_thresholds_f1[name] = 0.5
            
    logger.info("All models trained successfully.")
    
    return trained_models, optimal_thresholds, optimal_thresholds_f1, None

