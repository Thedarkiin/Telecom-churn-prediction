"""
Explainability methods for churn prediction models.

Provides SHAP and LIME explanations for model interpretability.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import logging

logger = logging.getLogger(__name__)


def explain_with_shap(model, X, feature_names, output_dir, max_display=20):
    """
    Generate SHAP explanations for the model.
    
    Parameters
    ----------
    model : fitted model
        The trained model to explain.
    X : array-like
        Feature matrix (use a sample for efficiency).
    feature_names : list
        Names of features.
    output_dir : str
        Directory to save plots.
    max_display : int
        Maximum features to display.
        
    Returns
    -------
    shap_values : array
        SHAP values for each prediction.
    """
    try:
        import shap
    except ImportError:
        logger.warning("SHAP not installed. Run: pip install shap")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create explainer based on model type
    if hasattr(model, 'predict_proba'):
        # Use SHAP LinearExplainer for logistic regression
        if hasattr(model, 'coef_'):
            explainer = shap.LinearExplainer(model, X)
        else:
            # Use KernelExplainer as fallback
            explainer = shap.KernelExplainer(
                lambda x: model.predict_proba(x)[:, 1], 
                shap.sample(X, min(100, len(X)))
            )
    else:
        explainer = shap.Explainer(model, X)
    
    # Compute SHAP values
    logger.info("Computing SHAP values...")
    shap_values = explainer.shap_values(X)
    
    # Handle different SHAP output formats
    if isinstance(shap_values, list):
        shap_values = shap_values[1]  # For binary classification, use positive class
    
    # 1. Summary plot (bar)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X, 
        feature_names=feature_names, 
        plot_type="bar",
        max_display=max_display,
        show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary_bar.png"), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved SHAP bar summary plot")
    
    # 2. Summary plot (beeswarm)
    plt.figure(figsize=(10, 8))
    shap.summary_plot(
        shap_values, X, 
        feature_names=feature_names,
        max_display=max_display,
        show=False
    )
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "shap_summary_beeswarm.png"), dpi=150, bbox_inches='tight')
    plt.close()
    logger.info("Saved SHAP beeswarm plot")
    
    # 3. Mean absolute SHAP values
    mean_shap = np.abs(shap_values).mean(axis=0)
    shap_importance = pd.DataFrame({
        'feature': feature_names,
        'mean_abs_shap': mean_shap
    }).sort_values('mean_abs_shap', ascending=False)
    
    shap_importance.to_csv(os.path.join(output_dir, "shap_importance.csv"), index=False)
    logger.info("Saved SHAP importance to CSV")
    
    return shap_values


def explain_with_lime(model, X_train, X_explain, feature_names, output_dir, n_samples=5):
    """
    Generate LIME explanations for individual predictions.
    
    Parameters
    ----------
    model : fitted model
        The trained model to explain.
    X_train : array-like
        Training data for LIME.
    X_explain : array-like
        Samples to explain.
    feature_names : list
        Names of features.
    output_dir : str
        Directory to save explanations.
    n_samples : int
        Number of samples to explain.
        
    Returns
    -------
    explanations : list
        LIME explanation objects.
    """
    try:
        from lime.lime_tabular import LimeTabularExplainer
    except ImportError:
        logger.warning("LIME not installed. Run: pip install lime")
        return None
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create LIME explainer
    explainer = LimeTabularExplainer(
        X_train,
        feature_names=feature_names,
        class_names=['No Churn', 'Churn'],
        mode='classification'
    )
    
    explanations = []
    
    for i in range(min(n_samples, len(X_explain))):
        logger.info(f"Generating LIME explanation for sample {i+1}/{n_samples}")
        
        # Get explanation
        exp = explainer.explain_instance(
            X_explain[i], 
            model.predict_proba,
            num_features=10
        )
        explanations.append(exp)
        
        # Save as HTML
        html_path = os.path.join(output_dir, f"lime_explanation_{i+1}.html")
        exp.save_to_file(html_path)
        
        # Save feature weights
        feature_weights = exp.as_list()
        weights_df = pd.DataFrame(feature_weights, columns=['feature_condition', 'weight'])
        weights_df.to_csv(os.path.join(output_dir, f"lime_weights_{i+1}.csv"), index=False)
    
    logger.info(f"Saved {n_samples} LIME explanations")
    
    return explanations


def plot_feature_importance_comparison(model, shap_values, feature_names, output_dir):
    """
    Compare model coefficients with SHAP importance.
    
    Parameters
    ----------
    model : fitted model
        Model with coefficients.
    shap_values : array
        SHAP values.
    feature_names : list
        Feature names.
    output_dir : str
        Output directory.
    """
    os.makedirs(output_dir, exist_ok=True)
    
    # Get model coefficients
    if hasattr(model, 'coef_'):
        coefs = np.abs(model.coef_.flatten())
    else:
        coefs = np.zeros(len(feature_names))
    
    # Mean absolute SHAP
    mean_shap = np.abs(shap_values).mean(axis=0)
    
    # Normalize for comparison
    coefs_norm = coefs / coefs.max() if coefs.max() > 0 else coefs
    shap_norm = mean_shap / mean_shap.max() if mean_shap.max() > 0 else mean_shap
    
    comparison_df = pd.DataFrame({
        'feature': feature_names,
        'coef_importance': coefs_norm,
        'shap_importance': shap_norm
    }).sort_values('shap_importance', ascending=False)
    
    # Plot comparison
    fig, ax = plt.subplots(figsize=(12, 8))
    x = np.arange(min(15, len(comparison_df)))
    width = 0.35
    
    top_features = comparison_df.head(15)
    
    ax.barh(x - width/2, top_features['coef_importance'], width, label='Coefficient', alpha=0.8)
    ax.barh(x + width/2, top_features['shap_importance'], width, label='SHAP', alpha=0.8)
    
    ax.set_yticks(x)
    ax.set_yticklabels(top_features['feature'])
    ax.set_xlabel('Normalized Importance')
    ax.set_title('Feature Importance: Coefficients vs SHAP')
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "importance_comparison.png"), dpi=150, bbox_inches='tight')
    plt.close()
    
    comparison_df.to_csv(os.path.join(output_dir, "importance_comparison.csv"), index=False)
    logger.info("Saved feature importance comparison")


def run_explainability(model, X_train, X_test, feature_names, config):
    """
    Run all explainability methods based on config.
    
    Parameters
    ----------
    model : fitted model
        Trained model.
    X_train : array-like
        Training features.
    X_test : array-like
        Test features.
    feature_names : list
        Feature names.
    config : Config
        Configuration object.
        
    Returns
    -------
    dict with SHAP values and LIME explanations.
    """
    output_dir = config.EXPLAINABILITY_PATH
    explain_config = config.PIPELINE_CONFIG["explainability"]
    
    results = {}
    
    # SHAP
    if explain_config.get("shap", True):
        logger.info("Running SHAP analysis...")
        # Use a sample of test data for efficiency
        sample_size = min(500, len(X_test))
        X_sample = X_test[:sample_size]
        
        shap_values = explain_with_shap(model, X_sample, feature_names, output_dir)
        results['shap_values'] = shap_values
        
        if shap_values is not None:
            plot_feature_importance_comparison(model, shap_values, feature_names, output_dir)
    
    # LIME
    if explain_config.get("lime", True):
        logger.info("Running LIME analysis...")
        n_lime = explain_config.get("n_samples_lime", 5)
        
        lime_explanations = explain_with_lime(
            model, X_train, X_test, feature_names, output_dir, n_samples=n_lime
        )
        results['lime_explanations'] = lime_explanations
    
    return results
