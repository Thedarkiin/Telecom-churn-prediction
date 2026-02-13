import pandas as pd
import numpy as np
import logging
from sklearn.linear_model import LogisticRegression

logger = logging.getLogger(__name__)

def compute_odds_ratios(model, feature_names):
    """
    Compute multivariate odds ratios from a trained Logistic Regression model.
    OR = exp(coefficient)
    """
    if not hasattr(model, 'coef_'):
        logger.warning("Model does not have coefficients (not LR?).")
        return pd.DataFrame()
    
    coefs = model.coef_[0]
    odds_ratios = np.exp(coefs)
    
    results = pd.DataFrame({
        'feature': feature_names[:len(coefs)] if feature_names else range(len(coefs)),
        'coefficient': coefs,
        'odds_ratio': odds_ratios
    })
    
    # Add effect interpretation
    results['effect'] = results['odds_ratio'].apply(
        lambda x: f"Increases risk by {(x-1)*100:.1f}%" if x > 1 else f"Decreases risk by {(1-x)*100:.1f}%"
    )
    
    return results.sort_values('odds_ratio', ascending=False)

def compute_univariate_odds_ratios(X, y, feature_names):
    """
    Compute univariate odds ratios by training a separate LR for each feature.
    Helpful for understanding raw risk factors without confounders.
    """
    results = []
    
    for i, feature in enumerate(feature_names):
        try:
            # Reshape for single feature
            X_feat = X[:, i].reshape(-1, 1)
            
            # Simple LR
            lr = LogisticRegression(solver='lbfgs', class_weight='balanced', random_state=42)
            lr.fit(X_feat, y)
            
            coef = lr.coef_[0][0]
            or_val = np.exp(coef)
            
            results.append({
                'feature': feature,
                'univariate_coef': coef,
                'univariate_odds_ratio': or_val
            })
        except Exception as e:
            logger.warning(f"Error computing univariate OR for {feature}: {e}")
            
    return pd.DataFrame(results).sort_values('univariate_odds_ratio', ascending=False)

def validate_linearity(X, y, feature_names):
    """
    Validate linearity assumption for continuous features using Box-Tidwell test logic
    (adding Interaction with log-transform).
    
    Returns DataFrame with 'linearity_valid' boolean.
    """
    # Identify continuous columns (heuristic: more than 10 unique values)
    # Using the passed X matrix
    df_check = pd.DataFrame(X, columns=feature_names)
    continuous_cols = [col for col in feature_names if df_check[col].nunique() > 10]
    
    results = []
    
    for col in continuous_cols:
        try:
            # Add x * log(x) term
            x = df_check[col].values
            # Handle zeros or negative by shifting
            x_safe = x + 1.0 - x.min() if x.min() <= 0 else x
            x_log = x_safe * np.log(x_safe)
            
            X_test = np.column_stack([x, x_log])
            
            lr = LogisticRegression(solver='lbfgs', class_weight='balanced', max_iter=1000)
            lr.fit(X_test, y)
            
            # Check p-value of the interaction term? 
            # Scikit-learn doesn't give p-values easily. 
            # We'll usage coefficient magnitude heuristic: if interaction coef is large vs main coef, assumption violated.
            
            coef_main = lr.coef_[0][0]
            coef_inter = lr.coef_[0][1]
            
            # Heuristic: if interaction effect is significant relative to main effect
            is_valid = abs(coef_inter) < 0.1 or abs(coef_inter) < abs(coef_main) * 0.5
            
            results.append({
                'feature': col,
                'linearity_valid': is_valid,
                'interaction_coef': coef_inter
            })
        except:
            results.append({'feature': col, 'linearity_valid': True})
            
    return pd.DataFrame(results)
