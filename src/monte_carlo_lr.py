"""
Monte Carlo Logistic Regression for Uncertainty Quantification.

Wraps a base logistic regression model and performs bootstrap resampling
to provide confidence intervals on predictions.
"""

import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin, clone
from sklearn.utils import resample
from joblib import Parallel, delayed


class MonteCarloLogisticRegression(BaseEstimator, ClassifierMixin):
    """
    Logistic Regression wrapper for Uncertainty Quantification via Bootstrap.
    
    Parameters
    ----------
    base_model : estimator
        Your optimized ElasticNet LogisticRegression pipeline.
    n_simulations : int, default=100
        Number of bootstrap simulations.
    n_jobs : int, default=-1
        Number of parallel jobs.
    random_state : int, default=42
        Random seed for reproducibility.
    """
    
    def __init__(self, base_model, n_simulations=100, n_jobs=-1, random_state=42):
        self.base_model = base_model
        self.n_simulations = n_simulations
        self.n_jobs = n_jobs
        self.random_state = random_state
        self.models_ = []
        self.classes_ = None
        
    def fit(self, X, y):
        """Fit multiple models on bootstrap samples."""
        self.classes_ = np.unique(y)
        
        def fit_single_bootstrap(i):
            # Resample data with stratification
            X_resampled, y_resampled = resample(
                X, y, 
                random_state=self.random_state + i, 
                stratify=y
            )
            # Clone and fit model
            model = clone(self.base_model)
            model.fit(X_resampled, y_resampled)
            return model

        # Run simulations in parallel
        self.models_ = Parallel(n_jobs=self.n_jobs)(
            delayed(fit_single_bootstrap)(i) for i in range(self.n_simulations)
        )
        return self

    def predict_proba(self, X):
        """Return mean probability across all bootstrap models."""
        all_probs = np.array([m.predict_proba(X)[:, 1] for m in self.models_]).T
        mean_prob = np.mean(all_probs, axis=1)
        return np.column_stack([1 - mean_prob, mean_prob])
    
    def predict(self, X, threshold=0.5):
        """Predict class labels using specified threshold."""
        probs = self.predict_proba(X)[:, 1]
        return (probs >= threshold).astype(int)

    def predict_proba_interval(self, X, alpha=0.05):
        """
        Returns Mean Probability and Confidence Interval.
        
        Parameters
        ----------
        X : array-like
            Features to predict.
        alpha : float, default=0.05
            Significance level (0.05 = 95% CI).
            
        Returns
        -------
        mean_prob : array
            Mean predicted probability.
        lower_bound : array
            Lower confidence bound.
        upper_bound : array
            Upper confidence bound.
        uncertainty : array
            Width of confidence interval.
        """
        # Get predictions from ALL models
        # Shape: (n_samples, n_simulations)
        all_probs = np.array([
            m.predict_proba(X)[:, 1] for m in self.models_
        ]).T 
        
        # 1. Robust point estimate (Mean)
        mean_prob = np.mean(all_probs, axis=1)
        
        # 2. Uncertainty interval (Percentiles)
        lower_bound = np.percentile(all_probs, 100 * (alpha / 2), axis=1)
        upper_bound = np.percentile(all_probs, 100 * (1 - alpha / 2), axis=1)
        
        # 3. Uncertainty score (interval width)
        uncertainty = upper_bound - lower_bound
        
        return mean_prob, lower_bound, upper_bound, uncertainty
    
    def get_coefficient_intervals(self, feature_names=None, alpha=0.05):
        """
        Get confidence intervals for model coefficients.
        
        Returns DataFrame with coefficient estimates and CIs.
        """
        import pandas as pd
        
        # Extract coefficients from all models
        all_coefs = []
        for model in self.models_:
            if hasattr(model, 'coef_'):
                all_coefs.append(model.coef_.flatten())
            elif hasattr(model, 'named_steps'):
                # Pipeline case
                for step_name, step in model.named_steps.items():
                    if hasattr(step, 'coef_'):
                        all_coefs.append(step.coef_.flatten())
                        break
        
        if not all_coefs:
            return None
            
        all_coefs = np.array(all_coefs)
        
        mean_coef = np.mean(all_coefs, axis=0)
        lower_coef = np.percentile(all_coefs, 100 * (alpha / 2), axis=0)
        upper_coef = np.percentile(all_coefs, 100 * (1 - alpha / 2), axis=0)
        
        if feature_names is None:
            feature_names = [f"feature_{i}" for i in range(len(mean_coef))]
        
        return pd.DataFrame({
            'feature': feature_names,
            'coef_mean': mean_coef,
            'coef_lower': lower_coef,
            'coef_upper': upper_coef,
            'significant': ~((lower_coef <= 0) & (upper_coef >= 0))
        })
