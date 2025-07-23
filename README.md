# Telecom Churn Prediction Pipeline

## Overview

A modular machine learning pipeline for predicting customer churn using the Kaggle Telco dataset (~7 000 customers, 21 columns).  
The aim is to identify which subscribers are likely to leave and highlight the key factors behind their decisions.

> A business problem tackled through a clean ML structure.

---

## 📁 Project Structure

```
churn_prediction_pipeline/
├── data/
│   ├── telecom_churn.csv
│   └── cleaned_data.csv
│
├── logs/
│   └── pipeline.log
│
├── results/
│   ├── metrics/
│   │   ├── all_metrics.csv
│   │   ├── xgboost_feature_importance.csv
│   │   └── feature_scores.csv
│   ├── predictions/
│   │   ├── decision_tree_predictions.csv
│   │   ├── logistic_regression_predictions.csv
│   │   └── xgboost_predictions.csv
│   └── plots/
│       ├── xgboost_confusion_matrix.png
│       ├── decision_tree_confusion_matrix.png
│       └── logistic_regression_confusion_matrix.png
│
├── src/
│   ├── __init__.py
│   ├── config.py              # Configuration variables (paths, params)
│   ├── utils.py               # Logging setup and helper methods
│   ├── preprocessing.py       # Cleaning, encoding, feature selection
│   ├── training.py            # Trains XGBoost, DecisionTree, LogisticRegression
│   ├── evaluation.py          # Metrics, plots, outputs
│   └── pipeline.py            # Full pipeline execution
│
├── diagram.svg                # Project flow diagram
├── requirements.txt
└── README.md
```

---

## 🔍 Models & Results

| Model                | Accuracy | Precision | Recall | F1‑score | ROC AUC |
|----------------------|----------|-----------|--------|----------|---------|
| Logistic Regression  | 0.7991   | 0.6584    | 0.5013 | 0.5692   | 0.7043  |
| XGBoost              | 0.7736   | 0.5882    | 0.4826 | 0.5302   | 0.7529  |
| Decision Tree        | 0.7296   | 0.4898    | 0.5174 | 0.5033   | 0.6633  |

> Logistic Regression is the best all-rounder, but XGBoost performs better in catching potential churners (recall).

---

## ⭐ Key Features Influencing Churn

| Rank | Feature         | Mutual Info |
|------|-----------------|-------------|
| 1    | Contract        | 0.0981      |
| 2    | Tenure          | 0.0838      |
| 3    | OnlineSecurity  | 0.0665      |
| 4    | TechSupport     | 0.0659      |
| 5    | OnlineBackup    | 0.0505      |

---

## 💡 Why This Matters

Keeping a telecom customer costs less than acquiring a new one.  
This pipeline helps spot high-risk subscribers early and empowers the team to take action — whether it's offering discounts, improving support, or changing plans.

---

## 🛠 How to Run

```bash
git clone https://github.com/YOUR_USERNAME/churn-prediction-pipeline.git
cd churn-prediction-pipeline
pip install -r requirements.txt
python src/pipeline.py
```

Outputs are saved in the `results/` folder as CSVs and plots.

---

## 📊 Visuals

### 🔹 Pipeline Architecture
> Diagram of the modular structure

![Pipeline Diagram](diagram.svg)

### 🔹 Confusion Matrix – XGBoost

![XGBoost Confusion Matrix](results/plots/xgboost_confusion_matrix.png)

### 🔹 Feature Importance

![Feature Scores](results/metrics/xgboost_feature_importance.csv)

---

## 🚧 (Optional) Next Steps

- Add SHAP explainability
- Wrap the model into a minimal API (e.g. FastAPI)
- Run the pipeline in a Docker container
- Schedule automatic runs via cron or Airflow

---

## 📎 About

Created by [Yassin Asermouh](https://www.linkedin.com/in/yassin-asermouh-984aa8249/).  
Built for learning, experimentation, and going beyond basic notebooks.

Data: [Kaggle - Telco Customer Churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn)

---
