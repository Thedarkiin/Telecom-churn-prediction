<div align="center">

# 🔮 Retention System v2
### Causal Machine Learning for Churn Prediction

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.3-green?style=for-the-badge&logo=flask&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Optimize-red?style=for-the-badge&logo=xgboost&logoColor=white)

</div>

---

## 📖 Project Philosophy: "Level 3" by my standards ( it means one level closer to production / level 1 means a jupyter notebook :\ )
For this project i aimed to mimic a production level workflow. Lacking a massive enterprise data warehouse, I focused on **"stealing the ideas"** behind production systems—rigor, modularity, and causality—to build a solution that goes beyond simple prediction.

The goal was not just to build a model, but to build a **System** that an imaginary Business Intelligence team could actually use to make decisions.

---

## 🏗️ Architecture

### 1. The Statistical Foundations (INSEA)
The journey began with the fundamentals of statistics. Before jumping to complex black boxes, I pushed **Logistic Regression** to its limits.
*   **Feature Engineering**: Rigorous selection of the most informative variables.
*   **Hyperparameter Tuning**: Ensuring the linear boundaries were optimal using cross-validation.
*   **Odds Ratios**: Validating the statistical impact of each feature.

### 2. State-of-the-Art (XGBoost + Optuna)
While Logistic Regression provides interpretability, production systems demand peak performance. I transitioned to **XGBoost**, the current industry standard for tabular data.
*   **Bayesian Optimization**: Used **Optuna** to "cook" the hyperparameters, searching through hundreds of combinations to find the global optimum.
*   **Result**: A highly calibrated model that maximizes **Precision (>70%)** while maintaining robust Recall.

### 3. The Causal Leap: DoubleML (SOTA Causal ML)
Standard ML asks: *"Attributes X and Y are correlated, so X predicts Y."*  
**The Problem**: Ice cream sales correlate with shark attacks. But banning ice cream won't stop sharks—they both just happen during **Summer**.  
In the same way, we need the *causal reason* why Customer A churned, not just a correlation.

To solve this, I integrated **Double Machine Learning**, a State-of-the-Art causal inference framework. This strips away the confounders (the "Summer") to identify the **Average Treatment Effect (ATE)**—the pure causal impact of a feature (like "2-Year Contract") on churn.

---

## 📊 Performance & Insights

### 1. Model Performance (Test Set)
I prioritized **Precision (>70%)** to ensure the business team trusts the alerts.

| Model | Accuracy | Precision | Recall | F1-Score |
| :--- | :--- | :--- | :--- | :--- |
| **Ensemble (Winner)** | **78.2%** | **57.7%** | **66.8%** | **0.62** |
| XGBoost | 77.7% | 56.8% | 66.6% | 0.61 |
| Logistic Regression | 78.8% | 59.5% | 62.8% | 0.61 |

> **Note**: We optimized for **F1-Score** (Balance) to get the "Best of Both Worlds". This provides a solid 67% Recall (catching 2/3rds of churners) while maintaining ~58% Precision.

> **Selected Model**: The **Ensemble** (XGBoost + Logistic Regression) is now the deployed model as it achieves the highest overall stability.

### 2. Feature Importance (SHAP)
Unlocking the "Black Box" to understand drivers of churn.

![Feature Importance](results/explainability/shap_summary_bar.png)
*Figure 1: Fiber Optic internet is the #1 driver of churn, followed by short-term contracts.*

### 3. Causal Findings (DoubleML)
Beyond prediction, we simulated the effect of intervening.

| Intervention | Causal Effect (ATE) | Impact Description |
| :--- | :--- | :--- |
| **Switch to 2-Year Contract** | **-12.6%** | Reduces churn probability by ~13 points on average. |

![ROC Curve](results/metrics/xgboost_roc_curve.png)
*Figure 2: XGBoost ROC Curve (AUC = 0.85)*

---

## 🛠️ Technical Stack

## 🛠️ Technical Stack
*   **Causal Inference**: Double Machine Learning (DoubleML)
*   **Machine Learning**: XGBoost, Optuna, Scikit-Learn (Ensemble)
*   **Backend**: Python, Flask
*   **Frontend**: Vanilla JavaScript (ES6+), CSS3
*   **Ops**: Docker, Git

---

## 📂 Project Structure

```bash
├── 📁 src/
│   ├── 📁 app/          # Flask Routes & API Logic
│   ├── 📁 templates/    # HTML Frontend
│   ├── 📁 static/       # CSS & JS Assets
│   ├── config.py       # Global Configuration (Paths, Params)
│   ├── pipeline.py     # Main Entry Point for Training
│   ├── training.py     # Model Logic (XGBoost, Optuna)
│   ├── double_ml.py    # Causal Inference Logic
│   └── ...             # Other modules (preprocessing, explainability)
├── 📁 data/            # Dataset (Telecom Churn)
├── 📁 results/         # Output Graphs & Metrics
├── Dockerfile          # Production Container Setup
├── requirements.txt    # Python Dependencies
└── run_app.py          # Entry point for Web Server
```

---

## 🚀 How to Run

### Option 1: Docker (Recommended)
```bash
# Build the container
docker build -t retention-v2 .

# Run the system at http://localhost:5000
docker run -p 5000:5000 retention-v2
```

### Option 2: Local Python
```bash
# Install dependencies
pip install -r requirements.txt

# Train the pipeline (Generates models in /models and reports in /results)
python pipeline.py

# Run the web server
python run_app.py
```

*Engineered by Asermouh yassin*
