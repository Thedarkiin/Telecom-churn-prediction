[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/release/python/)
[![CI](https://github.com/Thedarkiin/telecom-project/workflows/CI/badge.svg)](https://github.com/Thedarkiin/telecom-project/actions)

# Telecom Churn Prediction Pipeline

## 🚀 Overview

A modular machine learning pipeline for predicting customer churn using the Kaggle Telco dataset (~7 000 customers, 21 columns).  
The aim is to identify which subscribers are likely to leave and highlight the key factors behind their decisions.

> **"A business problem tackled through a clean ML structure."**

A quick fact: Telecom companies save up to **7 times more** when retaining an existing customer than acquiring a new one.

---

## 💼 Business Problem & Solution

### The Challenge
Customer churn is a silent revenue killer. Identifying at-risk customers *after* they've decided to leave is too late. The challenge is to predict churn probability *before* it happens and understand *why*.

### The Solution
This project provides a robust, production-ready pipeline that:
1.  **Predicts Churn Probability**: Uses Logistic Regression (with 74% accuracy & 79% recall) to flag at-risk customers.
2.  **Identifies Key Drivers**: Uses Odds Ratios to quantify exactly how much risky behaviors (like month-to-month contracts) increase churn risk.
3.  **Validates Causality**: Uses Double Machine Learning to separate true causal drivers from mere correlations.
4.  **Quantifies Uncertainty**: Uses Monte Carlo simulations to tell you how confident the model is in its predictions.

---

## 📁 Project Structure & Components

```
churn/
├── data/              # Dataset location
├── logs/              # Execution logs
├── results/           # Generated metrics, plots, and predictions
├── src/               # Core source code
├── pipeline.py        # Main execution entry point
├── CODE_GUIDE.md      # Detailed technical documentation
└── README.md          # This file
```

### 🧩 Core Components

| Component | File | Description |
|-----------|------|-------------|
| **Pipeline Core** | `pipeline.py` | Orchestrates the entire flow: Loading → Cleaning → Training → Evaluation. |
| **Config** | `src/config.py` | Central control for hyperparameters, thresholds, and toggleable features (like Causal ML). |
| **Preprocessing** | `src/preprocessing.py` | Handles missing values, encodes categoricals, and creates features. Smartly avoids data leakage. |
| **Training** | `src/training.py` | Trains models using **Optuna** for hyperparameter tuning. Implements stratified Cross-Validation. |
| **Evaluation** | `src/evaluation.py` | Generates comprehensive reports: ROC-AUC, Precision-Recall, Confusion Matrices, and business metrics. |
| **Odds Ratio** | `src/odds_ratio.py` | Calculates univariate and multivariate odds ratios to interpret Logistic Regression coefficients for business users. |
| **Causal Inference** | `src/double_ml.py` | Implements **Double Machine Learning** to estimate the *causal* effect of treatments (e.g., contract type) on churn. |
| **Uncertainty** | `src/monte_carlo_lr.py` | Performs Monte Carlo simulations to provide confidence intervals for model coefficients and predictions. |

---

## 🔍 Model Performance

The pipeline trains three models but **focuses on Logistic Regression** for interpretability and statistical rigor.

| Model | Accuracy | Precision | Recall | F1‑score | ROC AUC | Threshold |
|-------|----------|-----------|--------|----------|---------|-----------|
| **Logistic Regression** | **0.7410** | **0.5051** | **0.7888** | **0.6178** | **0.8420** | 0.50 |
| XGBoost | 0.7530 | 0.5241 | 0.7567 | 0.6193 | 0.8376 | 0.50 |
| Decision Tree | 0.7374 | 0.5054 | 0.5027 | 0.5040 | 0.6629 | 0.50 |

**Why Logistic Regression?**
- **Interpretable**: Coefficients directly represent log-odds, convertible to odds ratios.
- **Statistical validation**: Linearity assumptions tested via univariate analysis.
- **Uncertainty quantification**: Monte Carlo simulation provides confidence intervals.
- **Causal inference ready**: Compatible with Double ML framework.

---

## 🎯 Logistic Regression Insights

### Top Churn Risk Factors (Multivariate Odds Ratios)

The logistic regression model identifies key risk factors through odds ratios:

**High Risk (OR > 2.0)**:
- **Month-to-month contract**: 4-5× higher churn risk vs. long-term contracts.
- **No online security**: 2-3× higher risk.
- **Fiber optic internet**: 2-2.5× higher risk (vs. DSL or no internet).
- **Electronic check payment**: 1.5-2× higher risk.

**Protective Factors (OR < 1.0)**:
- **Long tenure** (>24 months): 60-70% lower risk.
- **Two-year contract**: 80-90% lower risk vs. month-to-month.
- **Tech support subscription**: 40-50% lower risk.

---

## 🔬 Causal Inference (Double ML)

Correlation does not imply causation. To dig deeper, we implemented **Double Machine Learning**.

**The Findings:**
We analyzed the causal effect of **"Month-to-month contract"** on churn.
- **Odds Ratio**: ~4.0 (Highly correlated with churn).
- **Causal Effect (ATE)**: ~0.0014 (0.1%).
- **Significance**: Not statistically significant (p > 0.05).

**Interpretation:**
While customers with month-to-month contracts churn vastly more often, the *contract itself* might not be the sole cause. It is likely a proxy for other underlying factors (like tenure or customer commitment level). Simply switching a customer's contract without addressing underlying satisfaction might not significantly reduce their churn probability.

---

## 🛠 How to Run

> If you have Anaconda, I recommend creating a dedicated environment.

```bash
git clone https://github.com/Thedarkiin/telecom-project.git
cd churn
pip install -r requirements.txt
python pipeline.py
```

**Outputs generated:**
- `results/metrics/all_metrics.csv`
- `results/metrics/logistic_regression_confusion_matrix.png`, etc.
- `results/metrics/multivariate_odds_ratios.csv`

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or pull request to suggest improvements.

---

## 📎 About

Created by [Yassin Asermouh](https://www.linkedin.com/in/yassin-asermouh-984aa8249/).  
Built for learning, experimentation, and going beyond basic Jupyter notebooks.

**Data**: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
