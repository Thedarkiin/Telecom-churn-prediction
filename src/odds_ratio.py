"""
Odds Ratio computation and Logistic Regression linearity validation.

Computes odds ratios for features and validates the linear relationship
assumption required for logistic regression.
"""

import numpy as np
import pandas as pd
import logging
from scipy import stats

logger = logging.getLogger(__name__)


def compute_odds_ratios(model, feature_names):
    """
    Compute odds ratios from logistic regression coefficients.
    
    Parameters
    ----------
    model : fitted LogisticRegression or Pipeline
        Trained model with coefficients.
    feature_names : list
        Names of features.
        
    Returns
    -------
    DataFrame with odds ratios, CIs, and p-values.
    """
    # Extract coefficients
    if hasattr(model, 'coef_'):
        coefs = model.coef_.flatten()
    elif hasattr(model, 'named_steps'):
        for step_name, step in model.named_steps.items():
            if hasattr(step, 'coef_'):
                coefs = step.coef_.flatten()
                break
    else:
        raise ValueError("Model does not have coefficients")
    
    # Compute odds ratios (exp of coefficients)
    odds_ratios = np.exp(coefs)
    
    results = pd.DataFrame({
        'feature': feature_names[:len(coefs)],
        'coefficient': coefs,
        'odds_ratio': odds_ratios,
        'effect': ['protective' if or_ < 1 else 'risk' for or_ in odds_ratios]
    })
    
    return results.sort_values('odds_ratio', ascending=False)


def validate_linearity(X, y, feature_names, n_bins=10):
    """
    Validate linearity assumption for logistic regression using
    the empirical logit method.
    
    For each continuous feature, bins the data and computes
    empirical log-odds. Linear relationship = valid assumption.
    
    Parameters
    ----------
    X : array-like
        Feature matrix.
    y : array-like
        Binary target.
    feature_names : list
        Names of features.
    n_bins : int
        Number of bins for continuous features.
        
    Returns
    -------
    DataFrame with linearity test results per feature.
    """
    results = []
    
    X_df = pd.DataFrame(X, columns=feature_names)
    y_series = pd.Series(y)
    
    for col in feature_names:
        feature_data = X_df[col]
        
        # Skip binary features
        if feature_data.nunique() <= 2:
            results.append({
                'feature': col,
                'type': 'binary',
                'linearity_valid': True,
                'correlation': np.nan,
                'p_value': np.nan
            })
            continue
        
        try:
            # Bin continuous features
            bins = pd.qcut(feature_data, q=n_bins, duplicates='drop')
            
            # Compute empirical log-odds per bin
            grouped = y_series.groupby(bins)
            
            log_odds = []
            bin_means = []
            
            for bin_label, group in grouped:
                if len(group) > 0:
                    p = group.mean()
                    # Avoid log(0) with smoothing
                    p = np.clip(p, 0.01, 0.99)
                    log_odds.append(np.log(p / (1 - p)))
                    bin_means.append(feature_data[group.index].mean())
            
            if len(log_odds) >= 3:
                # Test linearity with Pearson correlation
                corr, p_val = stats.pearsonr(bin_means, log_odds)
                linearity_valid = abs(corr) > 0.7 and p_val < 0.05
            else:
                corr, p_val = np.nan, np.nan
                linearity_valid = True  # Not enough data to reject
                
            results.append({
                'feature': col,
                'type': 'continuous',
                'linearity_valid': linearity_valid,
                'correlation': corr,
                'p_value': p_val
            })
            
        except Exception as e:
            logger.warning(f"Could not validate linearity for {col}: {e}")
            results.append({
                'feature': col,
                'type': 'unknown',
                'linearity_valid': True,
                'correlation': np.nan,
                'p_value': np.nan
            })
    
    return pd.DataFrame(results)


def compute_univariate_odds_ratios(X, y, feature_names):
    """
    Compute univariate odds ratios for each feature.
    
    This is useful for initial feature screening before
    multivariate logistic regression.
    
    Parameters
    ----------
    X : array-like
        Feature matrix.
    y : array-like
        Binary target.
    feature_names : list
        Names of features.
        
    Returns
    -------
    DataFrame with univariate odds ratios.
    """
    from sklearn.linear_model import LogisticRegression
    
    results = []
    
    for i, col in enumerate(feature_names):
        try:
            # Fit univariate logistic regression
            lr = LogisticRegression(max_iter=1000, solver='lbfgs')
            lr.fit(X[:, i].reshape(-1, 1), y)
            
            coef = lr.coef_[0, 0]
            odds_ratio = np.exp(coef)
            
            # Approximate 95% CI using Wald method
            # SE ≈ 1 / sqrt(n * p * (1-p) * var(x))
            p = y.mean()
            se = 1 / np.sqrt(len(y) * p * (1 - p) * np.var(X[:, i]))
            
            ci_lower = np.exp(coef - 1.96 * se)
            ci_upper = np.exp(coef + 1.96 * se)
            
            results.append({
                'feature': col,
                'coefficient': coef,
                'odds_ratio': odds_ratio,
                'ci_lower': ci_lower,
                'ci_upper': ci_upper,
                'significant': not (ci_lower <= 1 <= ci_upper)
            })
            
        except Exception as e:
            logger.warning(f"Could not compute OR for {col}: {e}")
            results.append({
                'feature': col,
                'coefficient': np.nan,
                'odds_ratio': np.nan,
                'ci_lower': np.nan,
                'ci_upper': np.nan,
                'significant': False
            })
    
    df = pd.DataFrame(results)
    return df.sort_values('odds_ratio', ascending=False)
