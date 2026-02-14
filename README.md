<div align="center">

# 🔮 Retention System v2
### Causal Machine Learning for Churn Prediction

![Python](https://img.shields.io/badge/Python-3.12-blue?style=for-the-badge&logo=python&logoColor=white)
![Flask](https://img.shields.io/badge/Flask-2.3-green?style=for-the-badge&logo=flask&logoColor=white)
![Docker](https://img.shields.io/badge/Docker-Ready-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![XGBoost](https://img.shields.io/badge/XGBoost-Optimize-red?style=for-the-badge&logo=xgboost&logoColor=white)

</div>

---

## 📖 Project Philosophy: "Level 3" Engineering
This project was designed to mimic a high-stakes production workflow. Lacking a massive enterprise data warehouse, I focused on **"stealing the ideas"** behind production systems—rigor, modularity, and causality—to build a solution that goes beyond simple prediction.

The goal was not just to build a model, but to build a **System** that an imaginary Business Intelligence team could actually use to make decisions.

---

## 🏗️ Architecture

### 1. The Statistical Foundations (INSEA)
The journey began with the fundamentals of statistics. Before jumping to complex black boxes, I pushed **Logistic Regression** to its absolute limit.
*   **Feature Engineering**: Rigorous selection of the most informative variables.
*   **Hyperparameter Tuning**: Ensuring the linear boundaries were optimal using cross-validation.
*   **Odds Ratios**: Validating the statistical impact of each feature.

### 2. State-of-the-Art (XGBoost + Optuna)
While Logistic Regression provides interpretability, production systems demand peak performance. I transitioned to **XGBoost**, the current industry standard for tabular data.
*   **Bayesian Optimization**: Used **Optuna** to "cook" the hyperparameters, searching through hundreds of combinations to find the global optimum.
*   **Result**: A highly calibrated model that maximizes **Precision (>70%)** while maintaining robust Recall.

### 3. The Causal Leap: DoubleML
Standard ML asks: *"Attributes X and Y are correlated, so X predicts Y."*  
**Problem**: Ice cream sales correlate with shark attacks. Banning ice cream won't stop sharks.

To solve this, I integrated **Double Machine Learning**, a State-of-the-Art causal inference framework. This strips away the noise to identify the **Average Treatment Effect (ATE)**—the actual causal impact of a feature (like "2-Year Contract") on churn.

---

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
