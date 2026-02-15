"""
Telecom Churn Prediction Pipeline

Complete ML pipeline with:
- Smart encoding (binary/one-hot)
- Optuna hyperparameter optimization
- Stratified K-fold cross-validation
- Class balancing
- Optimal threshold selection
- Double ML causal inference
- Odds ratio validation
"""

import os
import logging
import shutil
import pandas as pd
import warnings

warnings.filterwarnings('ignore')

from src.config import Config
from src.utils import setup_logger
from src.preprocessing import preprocess_data
from src.training import train_models
from src.evaluation import evaluate_models
from src.odds_ratio import compute_odds_ratios, validate_linearity, compute_univariate_odds_ratios
from src.double_ml import run_double_ml
from src.explainability import run_explainability


def clear_results():
    """Remove all previous results to keep only the latest run."""
    result_dirs = [
        Config.METRICS_PATH,
        Config.PREDICTIONS_PATH,
        Config.EXPLAINABILITY_PATH,
        Config.CAUSAL_PATH
    ]
    for folder in result_dirs:
        try:
            if os.path.exists(folder):
                shutil.rmtree(folder)
        except PermissionError:
            # On Windows, files may be locked - just clear contents
            for file in os.listdir(folder):
                try:
                    os.remove(os.path.join(folder, file))
                except:
                    pass
        os.makedirs(folder, exist_ok=True)
    
    # Also clear logs
    try:
        if os.path.exists("logs"):
            shutil.rmtree("logs")
    except PermissionError:
        pass
    os.makedirs("logs", exist_ok=True)


def main():
    # Clear previous results - only keep latest run
    clear_results()
    
    logger = setup_logger("logs/pipeline.log")
    logger.info("=" * 60)
    logger.info("Starting Enhanced Churn Prediction Pipeline")
    logger.info("=" * 60)
    logger.info("Cleared previous results - fresh run starting")
    
    # 1. PREPROCESSING
    logger.info("\n[STEP 1] PREPROCESSING")
    logger.info("-" * 40)
    X_train, X_test, y_train, y_test, feature_names = preprocess_data(Config)
    logger.info(f"Features: {len(feature_names)}")
    logger.info(f"Training samples: {len(X_train)}, Test samples: {len(X_test)}")
    
    # 2. ODDS RATIO VALIDATION
    logger.info("\n[STEP 2] ODDS RATIO & LINEARITY VALIDATION")
    logger.info("-" * 40)
    
    # Univariate odds ratios
    univariate_or = compute_univariate_odds_ratios(X_train, y_train, feature_names)
    univariate_or.to_csv(os.path.join(Config.METRICS_PATH, "univariate_odds_ratios.csv"), index=False)
    logger.info("Saved univariate odds ratios")
    
    # Linearity validation
    linearity_results = validate_linearity(X_train, y_train, feature_names)
    linearity_results.to_csv(os.path.join(Config.METRICS_PATH, "linearity_validation.csv"), index=False)
    
    non_linear = linearity_results[linearity_results['linearity_valid'] == False]
    if len(non_linear) > 0:
        logger.warning(f"Non-linear features detected: {non_linear['feature'].tolist()}")
    else:
        logger.info("Linearity assumption validated for all continuous features")
    
    # 3. TRAINING
    logger.info("\n[STEP 3] MODEL TRAINING")
    logger.info("-" * 40)
    models, optimal_thresholds_recall, optimal_thresholds_f1, optuna_study = train_models(X_train, y_train, Config)
    
    logger.info("Optimal thresholds (Recall-optimized):")
    for name, thresh in optimal_thresholds_recall.items():
        logger.info(f"  {name}: {thresh:.3f}")
    
    logger.info("Optimal thresholds (F1-optimized):")
    for name, thresh in optimal_thresholds_f1.items():
        logger.info(f"  {name}: {thresh:.3f}")
        
    # SAVE MODELS & ARTIFACTS
    import joblib
    os.makedirs('models', exist_ok=True)
    
    # Save Feature Names
    joblib.dump(feature_names, 'models/feature_names.pkl')
    logger.info(f"Saved feature names to models/feature_names.pkl")

    # Stats now saved in preprocessing.py for consistency (Raw Data)
    
    
    # Save Models (Dynamic)
    for name, model in models.items():
        filename = f'models/{name}_model.pkl'
        joblib.dump(model, filename)
        logger.info(f"Saved {name} to {filename}")
        
    # Save Logistic Regression specifically if needed for legacy lookups (though dynamic covers it)
    if "logistic_regression" in models:
        # Create symlink or just rely on dynamic name 'logistic_regression_model.pkl'
        # App might expect 'lr_model.pkl' - let's keep compatibility or update app
        joblib.dump(models["logistic_regression"], 'models/lr_model.pkl')

    
    # 4. EVALUATION - Recall-optimized thresholds
    logger.info("\n[STEP 4a] MODEL EVALUATION (Recall-optimized thresholds)")
    logger.info("-" * 40)
    metrics_df_recall = evaluate_models(models, X_test, y_test, optimal_thresholds_recall, feature_names)
    metrics_df_recall['threshold_type'] = 'recall'
    
    logger.info("\n[STEP 4b] MODEL EVALUATION (F1-optimized thresholds)")
    logger.info("-" * 40)
    metrics_df_f1 = evaluate_models(models, X_test, y_test, optimal_thresholds_f1, feature_names)
    metrics_df_f1['threshold_type'] = 'f1'
    
    metrics_df_f1 = evaluate_models(models, X_test, y_test, optimal_thresholds_f1, feature_names)
    metrics_df_f1['threshold_type'] = 'f1'
    
    # NEW: Overfitting Check
    from src.evaluation import plot_train_test_comparison
    plot_train_test_comparison(models, X_train, y_train, X_test, y_test, Config.METRICS_PATH)
    
    # Combine both results
    import pandas as pd
    metrics_df = pd.concat([metrics_df_recall, metrics_df_f1], ignore_index=True)
    metrics_df.to_csv(os.path.join(Config.METRICS_PATH, "all_metrics_combined.csv"), index=False)
    
    # Best model by recall
    best_recall = metrics_df_recall.loc[metrics_df_recall['recall'].idxmax()]
    logger.info(f"\nBest model (recall-optimized): {best_recall['model']} (recall={best_recall['recall']:.4f}, precision={best_recall['precision']:.4f})")
    
    # Best model by F1
    best_f1 = metrics_df_f1.loc[metrics_df_f1['f1'].idxmax()]
    logger.info(f"Best model (F1-optimized): {best_f1['model']} (f1={best_f1['f1']:.4f}, precision={best_f1['precision']:.4f}, recall={best_f1['recall']:.4f})")
    
    # 5. ODDS RATIOS FROM TRAINED MODEL
    logger.info("\n[STEP 5] MULTIVARIATE ODDS RATIOS")
    logger.info("-" * 40)
    
    lr_model = models.get("logistic_regression")
    if lr_model:
        multivariate_or = compute_odds_ratios(lr_model, feature_names)
        multivariate_or.to_csv(os.path.join(Config.METRICS_PATH, "multivariate_odds_ratios.csv"), index=False)
        logger.info("Top 10 risk factors:")
        for _, row in multivariate_or.head(10).iterrows():
            logger.info(f"  {row['feature']}: OR={row['odds_ratio']:.3f} ({row['effect']})")
    
    # 6. DOUBLE ML CAUSAL INFERENCE
    logger.info("\n[STEP 6] DOUBLE ML CAUSAL ANALYSIS")
    logger.info("-" * 40)
    
    try:
        dml = run_double_ml(X_train, y_train, feature_names, Config)
        dml_results = dml.get_results()
        dml_results.to_csv(os.path.join(Config.CAUSAL_PATH, "double_ml_results.csv"), index=False)
        
        interpretation_path = os.path.join(Config.CAUSAL_PATH, "double_ml_interpretation.txt")
        with open(interpretation_path, 'w') as f:
            f.write(dml.interpret())
        import json
        logger.info("Saved Double ML causal analysis")
        logger.info(f"ATE: {dml.ate_:.4f}, CI: [{dml.ate_ci_[0]:.4f}, {dml.ate_ci_[1]:.4f}]")
        
        # GENERATE JSON FOR APP
        # We create a simple rule: if ATE is negative and significant -> Action: "Encourage this!"
        # If ATE is positive and significant -> Action: "Avoid/Change this!"
        # This is strictly bound to the treatment variable (Contract_Two year)
        
        treatment = Config.PIPELINE_CONFIG["double_ml"]["treatment_variable"]
        is_significant = not (dml.ate_ci_[0] <= 0 <= dml.ate_ci_[1])
        effect_direction = "reduces" if dml.ate_ < 0 else "increases"
        
        dml_summary = {}
        
        # Add the main treatment insight
        if is_significant:
            desc = f"Switching to {treatment} strictly {effect_direction} churn risk by {abs(dml.ate_)*100:.1f}% on average."
            action = "Promote 2-year contracts actively." if dml.ate_ < 0 else "Investigate why 2-year contracts cause churn."
            
            # Map back to raw value for app lookup (Contract_Two year -> Contract_Two year)
            # The app looks up keys like "Contract_Two year", "InternetService_Fiber optic" etc.
            dml_summary[treatment] = {
                "description": desc,
                "action": action
            }
        
        with open('models/double_ml_summary.json', 'w') as f:
            json.dump(dml_summary, f, indent=4)
        logger.info("Saved Double ML summary for App (models/double_ml_summary.json)")

    except Exception as e:
        logger.warning(f"Double ML analysis failed: {e}")
    
    # 7. EXPLAINABILITY
    logger.info("\n[STEP 7] EXPLAINABILITY ANALYSIS")
    logger.info("-" * 40)
    
    try:
        lr_model = models.get("logistic_regression")
        if lr_model:
            explain_results = run_explainability(lr_model, X_train, X_test, feature_names, Config)
            logger.info("Saved SHAP and LIME explanations")
    except Exception as e:
        logger.warning(f"Explainability analysis error: {e}")
        logger.info("Install shap and lime: pip install shap lime")
    
    # 8. SUMMARY
    logger.info("\n" + "=" * 60)
    logger.info("PIPELINE COMPLETE")
    logger.info("=" * 60)
    logger.info(f"\nResults saved to:")
    logger.info(f"  Metrics:        {Config.METRICS_PATH}")
    logger.info(f"  Predictions:    {Config.PREDICTIONS_PATH}")
    logger.info(f"  Explainability: {Config.EXPLAINABILITY_PATH}")
    logger.info(f"  Causal:         {Config.CAUSAL_PATH}")
    
    # Print final metrics table
    logger.info("\n" + "-" * 60)
    logger.info("FINAL MODEL COMPARISON")
    logger.info("-" * 60)
    print(metrics_df.to_string(index=False))

    return models, metrics_df


if __name__ == "__main__":
    main()
