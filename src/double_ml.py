"""
Double Machine Learning for Causal Inference.

Implements the DML framework to estimate causal treatment effects,
controlling for confounders using ML models.
"""

import numpy as np
import pandas as pd
import logging
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.base import clone

logger = logging.getLogger(__name__)


class DoubleMLChurn:
    """
    Double Machine Learning for estimating causal effect of treatment on churn.
    
    Uses cross-fitting to avoid overfitting bias in nuisance estimation.
    
    Parameters
    ----------
    treatment_col : str
        Name of treatment column (e.g., 'Contract_Month-to-month').
    n_folds : int
        Number of folds for cross-fitting.
    random_state : int
        Random seed.
    """
    
    def __init__(self, treatment_col, n_folds=5, random_state=42):
        self.treatment_col = treatment_col
        self.n_folds = n_folds
        self.random_state = random_state
        self.ate_ = None
        self.ate_se_ = None
        self.ate_ci_ = None
        
    def fit(self, X, y, feature_names):
        """
        Estimate Average Treatment Effect using Double ML.
        
        Parameters
        ----------
        X : array-like
            Feature matrix (including treatment).
        y : array-like
            Binary outcome (churn).
        feature_names : list
            Names of features.
            
        Returns
        -------
        self
        """
        # Convert to DataFrame for easier handling
        df = pd.DataFrame(X, columns=feature_names)
        
        # Find treatment column (handle one-hot encoded names)
        treatment_col = None
        for col in df.columns:
            if self.treatment_col in col:
                treatment_col = col
                break
        
        if treatment_col is None:
            logger.warning(f"Treatment column '{self.treatment_col}' not found. Using first column as treatment.")
            treatment_col = df.columns[0]
        
        # Extract treatment
        D = df[treatment_col].values
        
        # Control variables (everything except treatment)
        control_cols = [c for c in df.columns if c != treatment_col]
        W = df[control_cols].values
        
        logger.info(f"Running Double ML with treatment: {treatment_col}")
        logger.info(f"Control variables: {len(control_cols)}")
        
        # Cross-fitting
        kf = StratifiedKFold(n_splits=self.n_folds, shuffle=True, random_state=self.random_state)
        
        # Storage for residuals
        y_residuals = np.zeros(len(y))
        d_residuals = np.zeros(len(D))
        
        # Models for nuisance functions
        outcome_model = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=self.random_state
        )
        treatment_model = GradientBoostingClassifier(
            n_estimators=100, max_depth=3, random_state=self.random_state
        )
        
        # Cross-fitting loop
        for train_idx, test_idx in kf.split(W, y):
            W_train, W_test = W[train_idx], W[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            D_train, D_test = D[train_idx], D[test_idx]
            
            # Fit outcome model: E[Y | W]
            m_y = clone(outcome_model)
            m_y.fit(W_train, y_train)
            y_hat = m_y.predict_proba(W_test)[:, 1] if hasattr(m_y, 'predict_proba') else m_y.predict(W_test)
            
            # Fit treatment model: E[D | W] (propensity score)
            m_d = clone(treatment_model)
            m_d.fit(W_train, D_train)
            d_hat = m_d.predict_proba(W_test)[:, 1] if hasattr(m_d, 'predict_proba') else m_d.predict(W_test)
            
            # Clip propensity scores for stability
            d_hat = np.clip(d_hat, 0.05, 0.95)
            
            # Compute residuals
            y_residuals[test_idx] = y_test - y_hat
            d_residuals[test_idx] = D_test - d_hat
        
        # Final stage: regress Y residuals on D residuals
        # theta = E[y_resid * d_resid] / E[d_resid^2]
        self.ate_ = np.sum(y_residuals * d_residuals) / np.sum(d_residuals ** 2)
        
        # Standard error via sandwich formula
        n = len(y)
        psi = y_residuals - self.ate_ * d_residuals
        V = np.mean(psi ** 2) / (np.mean(d_residuals ** 2) ** 2)
        self.ate_se_ = np.sqrt(V / n)
        
        # 95% CI
        self.ate_ci_ = (
            self.ate_ - 1.96 * self.ate_se_,
            self.ate_ + 1.96 * self.ate_se_
        )
        
        logger.info(f"ATE estimate: {self.ate_:.4f}")
        logger.info(f"95% CI: [{self.ate_ci_[0]:.4f}, {self.ate_ci_[1]:.4f}]")
        
        return self
    
    def get_results(self):
        """Return results as DataFrame."""
        return pd.DataFrame({
            'Treatment': [self.treatment_col],
            'ATE': [self.ate_],
            'SE': [self.ate_se_],
            'CI_Lower': [self.ate_ci_[0]],
            'CI_Upper': [self.ate_ci_[1]],
            'Significant': [not (self.ate_ci_[0] <= 0 <= self.ate_ci_[1])]
        })
    
    def interpret(self):
        """Generate human-readable interpretation."""
        direction = "increases" if self.ate_ > 0 else "decreases"
        magnitude = abs(self.ate_) * 100
        significant = "statistically significant" if not (self.ate_ci_[0] <= 0 <= self.ate_ci_[1]) else "not statistically significant"
        
        interpretation = f"""
        Double Machine Learning Causal Analysis
        ========================================
        Treatment: {self.treatment_col}
        
        Estimated Average Treatment Effect (ATE): {self.ate_:.4f}
        Standard Error: {self.ate_se_:.4f}
        95% Confidence Interval: [{self.ate_ci_[0]:.4f}, {self.ate_ci_[1]:.4f}]
        
        Interpretation:
        The treatment '{self.treatment_col}' {direction} the probability of churn 
        by approximately {magnitude:.1f} percentage points. 
        This effect is {significant} at the 5% level.
        
        Note: This estimate controls for observable confounders using ML models.
        Unobserved confounding may still bias the estimate.
        """
        return interpretation


def run_double_ml(X, y, feature_names, config):
    """
    Run Double ML analysis using configuration settings.
    
    Parameters
    ----------
    X : array-like
        Feature matrix.
    y : array-like
        Target variable.
    feature_names : list
        Names of features.
    config : Config
        Configuration object.
        
    Returns
    -------
    DoubleMLChurn instance with results.
    """
    dml_config = config.PIPELINE_CONFIG["double_ml"]
    
    dml = DoubleMLChurn(
        treatment_col=dml_config["treatment_value"],  # This will match 'Month-to-month' in one-hot encoded col
        n_folds=dml_config["n_folds"],
        random_state=config.RANDOM_STATE
    )
    
    dml.fit(X, y, feature_names)
    
    return dml
