[![Python](https://img.shields.io/badge/Python-3.8%2B-blue)](https://www.python.org/downloads/release/python/)
[![CI](https://github.com/Thedarkiin/telecom-project/workflows/CI/badge.svg)](https://github.com/Thedarkiin/telecom-project/actions)

# Telecom Churn Prediction Pipeline

## Overview

A modular machine learning pipeline for predicting customer churn using the Kaggle Telco dataset (~7 000 customers, 21 columns).  
The aim is to identify which subscribers are likely to leave and highlight the key factors behind their decisions.
A quick cool fact, telecom companies save up to 7 times more when keeping a subscribed customer rather than acquiring a new one.

> A business problem tackled through a clean ML structure.

---

## 📁 Project Structure

```
churn/
├── data/
│   ├── telecom_churn.csv      # Original dataset
│   └── cleaned_data.csv       # Preprocessed data
│
├── logs/
│   └── src.training           # Training logs
│
├── results/
│   ├── metrics/               # Model performance metrics
│   │   ├── all_metrics.csv
│   │   ├── univariate_odds_ratios.csv
│   │   ├── multivariate_odds_ratios.csv
│   │   ├── linearity_validation.csv
│   │   ├── optuna_study.csv
│   │   └── *_confusion_matrix.png, *_roc_curve.png, *_pr_curve.png
│   ├── predictions/           # Model predictions
│   │   ├── decision_tree_predictions.csv
│   │   ├── logistic_regression_predictions.csv
│   │   └── xgboost_predictions.csv
│   ├── explainability/        # SHAP analysis
│   │   ├── shap_importance.csv
│   │   ├── shap_summary_bar.png
│   │   ├── shap_summary_beeswarm.png
│   │   └── importance_comparison.*
│   └── causal/                # Causal inference results
│       ├── double_ml_results.csv
│       └── double_ml_interpretation.txt
│
├── src/
│   ├── __init__.py
│   ├── config.py              # Pipeline configuration (CV, tuning, etc.)
│   ├── utils.py               # Logging and utilities
│   ├── preprocessing.py       # Smart encoding & transformations
│   ├── training.py            # Optuna hyperparameter optimization
│   ├── evaluation.py          # Comprehensive metrics & visualizations
│   ├── odds_ratio.py          # Univariate/multivariate odds ratios
│   ├── double_ml.py           # Double ML causal inference
│   ├── explainability.py      # SHAP/LIME interpretability
│   └── monte_carlo_lr.py      # Monte Carlo uncertainty quantification
│
├── pipeline.py
├── diagram.svg
├── requirements.txt
└── README.md
```


---

## 🔍 Model Performance

The pipeline trains three models but **focuses on Logistic Regression** for interpretability and statistical rigor.

| Model               | Accuracy | Precision | Recall | F1‑score | ROC AUC | Threshold |
|--------------------|----------|-----------|--------|----------|---------|-----------|
| **Logistic Regression** | **0.7913** | **0.6299** | **0.5187** | **0.5689** | **0.7043** | 0.50 (F1) |
| XGBoost            | 0.7537   | 0.5252    | 0.7513 | 0.6183   | 0.7529  | 0.40      |
| Decision Tree      | 0.7260   | 0.4853    | 0.5294 | 0.5064   | 0.6633  | 0.45      |

**Why Logistic Regression?**
- **Interpretable**: Coefficients directly represent log-odds, convertible to odds ratios
- **Statistical validation**: Linearity assumptions tested via univariate analysis
- **Uncertainty quantification**: Monte Carlo simulation provides confidence intervals
- **Causal inference ready**: Compatible with Double ML framework

**Threshold Optimization**:
The pipeline computes optimal thresholds for both **recall** and **F1-score**:
- **Recall-optimized (0.35-0.40)**: Maximizes catching churners (fewer false negatives)
- **F1-optimized (0.50-0.55)**: Balances precision and recall (business optimal)

Higher thresholds reduce false positives (fewer non-churners incorrectly flagged), improving precision.

---

## 🎯 Logistic Regression Insights

### Top Churn Risk Factors (Multivariate Odds Ratios)

The logistic regression model identifies key risk factors through odds ratios:

**High Risk (OR > 2.0)**:
- **Month-to-month contract**: 4-5× higher churn risk vs. long-term contracts
- **No online security**: 2-3× higher risk
- **Fiber optic internet**: 2-2.5× higher risk (vs. DSL or no internet)
- **Electronic check payment**: 1.5-2× higher risk

**Protective Factors (OR < 1.0)**:
- **Long tenure** (>24 months): 60-70% lower risk
- **Two-year contract**: 80-90% lower risk vs. month-to-month
- **Tech support subscription**: 40-50% lower risk

---

## 💡 Why This Matters

Keeping a telecom customer costs less than acquiring a new one.  
This pipeline helps spot high-risk subscribers early and empowers the team to take action — whether it's offering discounts, improving support, or changing plans.

---

## 🛠 How to Run

>If you have anaconda do like me, i created a conda environnement so that i can install only the dependacies i want, aka needed for this specifc project.

```bash
git clone https://github.com/Thedarkiin/telecom-project.git
cd churn
pip install -r requirements.txt
python pipeline.py
```

**Outputs generated:**

- `results/metrics/all_metrics.csv`
- `results/metrics/feature_scores.csv`
- `results/metrics/xgboost_confusion_matrix.png`, etc.

---

## 📊 Visuals

### 🔹 Pipeline Architecture

![Pipeline Diagram](diagram.svg)

### 🔹 Confusion Matrix – XGBoost

![XGBoost Confusion Matrix](results/metrics/xgboost_confusion_matrix.png)



---

## 📝 Version Control

**Important**: The `results/` and `logs/` directories are gitignored as they contain generated files.
Only source code, configuration, and the original dataset (`data/telecom_churn.csv`) are tracked in version control.

To regenerate results:
```bash
python pipeline.py
```

---

## 🤝 Contributing

Contributions are welcome! Please open an issue or pull request to suggest improvements.

---

## 📎 About

Created by [Yassin Asermouh](https://www.linkedin.com/in/yassin-asermouh-984aa8249/).  
Built for learning, experimentation, and going beyond basic jupy notebooks.

**Data**: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)
