# Code Guide: Telecom Churn Prediction Pipeline

## Overview

This document explains how the churn prediction pipeline works, focusing on the **logistic regression model** and its supporting infrastructure for statistical rigor, interpretability, and uncertainty quantification.

##Architecture

```
Data → Preprocessing → Training → Evaluation → Analysis
         (src/preprocessing.py)   (src/training.py)   (src/evaluation.py)   (odds_ratio.py, double_ml.py, explainability.py, monte_carlo_lr.py)
```

---

## 1. Configuration (`src/config.py`)

Central configuration for the entire pipeline.

### Key Configurations:
- **Threshold search range**: `[0.35, 0.4, 0.45, 0.5, 0.55, 0.6, 0.65]` - Higher thresholds improve precision
- **Cross-validation**: Stratified K-Fold with 5 splits
- **Hyperparameter tuning**: Optuna with 50 trials, optimizing for recall
- **Logistic regression**: ElasticNet penalty (L1+L2), balanced class weights
- **Monte Carlo**: 100 simulations for uncertainty quantification

---

## 2. Data Preprocessing (`src/preprocessing.py`)

Transforms raw telecom data into model-ready features.

### Pipeline Steps:

1. **Column Dropping**: Remove `customerID` (not predictive)

2. **Deterministic Imputation**:
   ```python
   # TotalCharges = 0 when tenure == 0 (new customers)
   df.loc[df['tenure'] == 0, 'TotalCharges'] = 0
   ```

3. **Binary Encoding** (Yes/No → 1/0):
   - `Partner`, `Dependents`, `PhoneService`, `PaperlessBilling`, `gender`

4. **Service Binary Encoding** (handles missing services):
   - `OnlineSecurity`, `TechSupport`, etc.
   - Maps: `Yes→1`, `No→0`, `No internet service→0`

5. **One-Hot Encoding** (categorical features):
   - `InternetService`: DSL, Fiber optic, No
   - `Contract`: Month-to-month, One year, Two year
   - `PaymentMethod`: Electronic check, Mailed check, Bank transfer, Credit card

6. **KNN Imputation**: Fill remaining missing values using 5 nearest neighbors

7. **Log Transformation**: Apply log1p to `TotalCharges` and `MonthlyCharges` for normality

8. **Standardization**: Z-score normalization for numeric features

9. **Stratified Split**: 80/20 train/test split preserving churn rate

### Why This Matters:
- **Linearity for logistic regression**: Log transformations make relationships more linear
- **Proper encoding**: Binary/one-hot encoding creates interpretable coefficients
- **No data leakage**: Normalization fit only on train set

---

## 3. Model Training (`src/training.py`)

Trains logistic regression with hyperparameter optimization and threshold tuning.

### Training Process:

#### 3.1 Optuna Hyperparameter Optimization

Searches for best regularization parameters:

```python
# Search space
C (regularization):  0.001 to 100 (loguniform)
l1_ratio (ElasticNet mix): 0.0 to 1.0 (uniform)

# Objective: Maximize recall via 5-fold cross-validation
```

**Why ElasticNet?**
- Combines L1 (feature selection) + L2 (stability)
- Handles multicollinearity in telecom features (e.g., StreamingTV + StreamingMovies)

#### 3.2 Threshold Optimization

After training, finds optimal decision thresholds:

```python
# For each model, search [0.35, 0.4, ..., 0.65]
# Optimize separately for:
1. Recall: Maximize true churner detection
2. F1: Balance precision and recall
```

**Output**: Two thresholds per model (recall-optimized, F1-optimized)

#### 3.3 Monte Carlo Wrapper

Creates uncertainty-aware version of logistic regression:

```python
# Bootstrapping: Resample training data 100 times
# Train 100 logistic regression models
# Prediction = mean of 100 predictions ± confidence interval
```

**Benefits**:
- Quantifies model uncertainty (not just prediction uncertainty)
- Identifies high-uncertainty predictions for manual review
- Provides 95% confidence intervals for coefficients and predictions

---

## 4. Model Evaluation (`src/evaluation.py`)

Computes metrics at optimized thresholds and generates visualizations.

### Evaluation Flow:

1. **Get predictions** at optimal threshold:
   ```python
   probs = model.predict_proba(X_test)[:, 1]
   y_pred = (probs >= threshold).astype(int)
   ```

2. **Compute metrics**:
   - Accuracy, Precision, Recall, F1, ROC-AUC
   - Confusion matrix (True Positives, False Positives, etc.)

3. **Generate plots**:
   - Confusion matrix heatmap
   - ROC curve (TPR vs FPR)
   - Precision-Recall curve

4. **Monte Carlo specific**:
   - Uncertainty distribution
   - Coefficient confidence intervals

### Why Dual Threshold Evaluation?
- **Recall-optimized**: For retention campaigns (cast wide net)
- **F1-optimized**: For targeted interventions (cost-sensitive)

---

## 5. Statistical Validation (`src/odds_ratio.py`)

Validates logistic regression assumptions and interprets coefficients.

### 5.1 Univariate Odds Ratios

Fits simple logistic regression for each feature individually:

```python
# For each feature X:
logit(P(churn=1)) = β0 + β1*X
Odds Ratio = exp(β1)
```

**Interpretation**:
- OR = 2.5 → 150% increase in churn odds per unit increase
- OR = 0.6 → 40% decrease in churn odds per unit increase

### 5.2 Linearity Validation

Tests if continuous features have linear relationship with log-odds:

```python
# Bin continuous feature into quintiles
# Fit logistic regression on bins
# Check if trend is monotonic (p < 0.05)
```

**Fails if**: Non-monotonic relationship detected → feature needs transformation

### 5.3 Multivariate Odds Ratios

Extracts coefficients from trained model:

```python
# From full model with all features:
OR = exp(coefficient)
```

**Key difference from univariate**: Adjusts for confounding (e.g., tenure AND contract type together)

---

## 6. Causal Analysis (`src/double_ml.py`)

Estimates causal effect of `Contract` on churn using Double Machine Learning.

### Double ML Framework:

```python
# Step 1: Predict Y (churn) using all features except Contract
# Step 2: Predict Contract using all other features
# Step 3: Residualize to remove confounding
# Step 4: Estimate causal effect from residuals
```

**Result**: Average Treatment Effect (ATE) with confidence interval

**Why this matters**:
- Separates **correlation** (what predicts churn) from **causation** (what causes churn)
- Informs interventions: "If we convert month-to-month to yearly, churn decreases by X%"

---

## 7. Explainability (`src/explainability.py`)

Uses SHAP to explain individual predictions.

### SHAP (SHapley Additive exPlanations):

```python
# For each prediction:
# Calculate contribution of each feature to the output
# Based on game theory (Shapley values)
```

**Outputs**:
- `shap_summary_beeswarm.png`: Feature importance across all predictions
- `shap_summary_bar.png`: Average absolute impact per feature
- `shap_importance.csv`: Numerical importance scores

**Why SHAP over Lime?**
- Theoretically sound (satisfies consistency axiom)
- Handles feature interactions
- More reliable for logistic regression

---

## 8. Monte Carlo Uncertainty (`src/monte_carlo_lr.py`)

Bootstrapping-based uncertainty quantification for logistic regression.

### How it works:

```python
1. Resample training data with replacement (bootstrap)
2. Train logistic regression on bootstrap sample
3. Store model
4. Repeat 100 times

For prediction:
- Mean prediction = average of 100 models
- Confidence interval = 2.5th and 97.5th percentiles
- Uncertainty = CI width
```

###Outputs:

- **Coefficient intervals**: `monte_carlo_lr_coef_intervals.csv`
  - Mean, Lower CI, Upper CI for each feature
  - If CI crosses 0 → feature not statistically significant

- **Prediction uncertainty**: Per-sample uncertainty scores
  - High uncertainty (>0.3) → model unsure, manual review needed

---

## 9. Main Pipeline (`pipeline.py`)

Orchestrates the entire workflow.

### Execution Flow:

```
1. Load & Preprocess Data
   ↓
2. Validate Odds Ratios & Linearity
   ↓
3. Train Models (LR, XGBoost, Decision Tree)
   ├→ Optuna tuning for LR
   ├→ Threshold optimization (recall + F1)
   └→ Monte Carlo wrapper for LR
   ↓
4. Evaluate on Test Set
   ├→ Recall-optimized thresholds
   └→ F1-optimized thresholds
   ↓
5. Multivariate Odds Ratios (from trained LR)
   ↓
6. Double ML Causal Analysis
   ↓
7. SHAP Explainability
   ↓
8. Save all results to results/
```

### Key Design Decisions:

- **Logistic regression trained first**: Needed for odds ratio analysis
- **Dual threshold optimization**: Business needs both high recall and balanced F1
- **Monte Carlo after main training**: Reuses trained LR as base model
- **SHAP only for LR**: Focus on interpretable model

---

## 10. Results Structure

```
results/
├── metrics/
│   ├── all_metrics.csv              # Performance of all models
│   ├── all_metrics_combined.csv     # Both threshold strategies
│   ├── univariate_odds_ratios.csv   # Simple associations
│   ├── multivariate_odds_ratios.csv # Adjusted associations
│   ├── linearity_validation.csv     # Linearity test results
│   ├── monte_carlo_lr_coef_intervals.csv # Coefficient uncertainties
│   ├── optuna_study.csv             # Hyperparameter search history
│   └── *.png                        # Confusion matrices, ROC curves
├── predictions/
│   ├── logistic_regression_predictions.csv
│   └── monte_carlo_lr_uncertainty.csv  # With confidence intervals
├── explainability/
│   ├── shap_importance.csv
│   └── shap_summary_*.png
└── causal/
    ├── double_ml_results.csv
    └── double_ml_interpretation.txt
```

---

## Key Takeaways

### Why Logistic Regression?

1. **Coefficients = interpretable risk factors** (via odds ratios)
2. **Statistical tests** validate assumptions (linearity, significance)
3. **Uncertainty quantification** via Monte Carlo
4. **Causal inference** compatibility (Double ML)
5. **Regulatory compliance** (explainable AI for finance/telecom)

### What Makes This Pipeline Rigorous?

- **Stratified CV**: Prevents overfitting, preserves class balance
- **Separate threshold optimization**: Doesn't leak test data into training
- **Odds ratio validation**: Ensures logistic regression assumptions hold
- **Monte Carlo**: Quantifies epistemic uncertainty (model uncertainty)
- **SHAP**: Gold standard for local explanations

### Production Considerations:

- **Threshold flexibility**: Deploy both thresholds, let business choose
- **Uncertainty flagging**: Manual review for predictions with >0.3 uncertainty
- **Feature monitoring**: Track if linearity assumptions break over time
- **Model retraining**: Retrain when churn rate shifts >2%

---

## Running the Pipeline

```bash
# Full pipeline (takes ~5-10 minutes)
python pipeline.py

# Outputs:
# - results/metrics/all_metrics_combined.csv (key performance file)
# - results/metrics/*_confusion_matrix.png (visualizations)
# - results/explainability/shap_*.png (interpretability)
# - logs/pipeline.log (detailed execution log)
```

For questions or modifications, see the main README.md file.
