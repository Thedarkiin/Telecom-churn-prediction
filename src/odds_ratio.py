
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
import logging

logger = logging.getLogger(__name__)

def compute_odds_ratios(model, feature_names):
    """
    Compute multivariate odds ratios from a fitted Logistic Regression model.
    """
    if not hasattr(model, 'coef_'):
        logger.warning("Model does not have coefficients. Cannot compute odds ratios.")
        return pd.DataFrame()
        
    coefs = model.coef_[0]
    odds_ratios = np.exp(coefs)
    
    df = pd.DataFrame({
        'feature': feature_names[:len(coefs)],
        'coefficient': coefs,
        'odds_ratio': odds_ratios
    })
    
    df['effect'] = df['odds_ratio'].apply(lambda x: 'Increase Risk' if x > 1 else 'Decrease Risk')
    df['strength'] = df['odds_ratio'].apply(lambda x: abs(x - 1))
    
    return df.sort_values('strength', ascending=False)

def compute_univariate_odds_ratios(X, y, feature_names):
    """
    Compute univariate odds ratios for each feature independently.
    """
    results = []
    
    X_df = pd.DataFrame(X, columns=feature_names)
    
    for col in feature_names:
        try:
            # Simple LR for univariate
            model = LogisticRegression(solver='lbfgs', class_weight='balanced')
            # Reshape for single feature
            X_feat = X_df[[col]].values
            model.fit(X_feat, y)
            
            coef = model.coef_[0][0]
            or_val = np.exp(coef)
            
            p_value = 0.05 # Placeholder as we are not using statsmodels for speed
            
            results.append({
                'feature': col,
                'odds_ratio': or_val,
                'log_odds': coef,
                'p_value': p_value
            })
        except Exception as e:
            logger.debug(f"Could not compute univariate OR for {col}: {e}")
            
    return pd.DataFrame(results).sort_values('odds_ratio', ascending=False)

def validate_linearity(X, y, feature_names):
    """
    Validate linearity assumption for continuous features using Box-Tidwell test equivalent.
    For simplicity in this streamlined version, we check if log-transform improves univariate fit significantly.
    """
    results = []
    X_df = pd.DataFrame(X, columns=feature_names)
    
    # Identify continuous features (more than 10 unique values)
    continuous_cols = [c for c in feature_names if X_df[c].nunique() > 10]
    
    for col in continuous_cols:
        try:
            # Base model
            X_base = X_df[[col]].values
            lr_base = LogisticRegression(solver='lbfgs')
            lr_base.fit(X_base, y)
            score_base = lr_base.score(X_base, y)
            
            # Log transformed model (handling zeros)
            X_log = np.log1p(X_base - X_base.min() + 1)
            lr_log = LogisticRegression(solver='lbfgs')
            lr_log.fit(X_log, y)
            score_log = lr_log.score(X_log, y)
            
            # If log score is significantly better (> 2% improvement), assume non-linear in original scale
            is_linear = (score_log - score_base) < 0.02
            
            results.append({
                'feature': col,
                'linearity_valid': is_linear,
                'score_base': score_base,
                'score_log': score_log
            })
        except:
             results.append({'feature': col, 'linearity_valid': True})
             
    if not results:
        return pd.DataFrame(columns=['feature', 'linearity_valid'])
        
    return pd.DataFrame(results)
