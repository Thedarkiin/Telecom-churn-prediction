# Retention System v2
**Road to Causal Machine Learning**

## Project Philosophy: "Level 3" Engineering
This project was designed to mimic a high-stakes production workflow. Lacking a massive enterprise data warehouse, I focused on "stealing the ideas" behind production systems—rigor, modularity, and causality—to build a solution that goes beyond simple prediction.

The goal was not just to build a model, but to build a **System** that an imaginary Business Intelligence team could actually use to make decisions.

---

## 1. The Statistical Foundation (INSEA Roots)
The journey began with the fundamentals of statistics. Before jumping to complex black boxes, I pushed **Logistic Regression** to its absolute limit.
*   **Feature Engineering**: Rigorous selection of the most informative variables.
*   **Regularization**: Applying penalties to prevent overfitting.
*   **Hyperparameter Tuning**: Ensuring the linear boundaries were optimal.

This phase ensured the project was grounded in the "good habits" of a statistician: understanding the data distribution before trying to predict it.

## 2. The Move to State-of-the-Art (XGBoost + Optuna)
While Logistic Regression provides interpretability, production systems demand peak performance. I transitioned to **XGBoost**, the current industry standard for tabular data.
*   **Optimization**: I didn't just run the model; I "cooked" it using **Optuna** for Bayesian hyperparameter search.
*   **Result**: A highly calibrated model that maximizes Recall (catching churners) without sacrificing too much Precision.

## 3. The Causal Leap: Beyond Correlation
Standard Machine Learning asks: *"Attributes X and Y are correlated, so X predicts Y."*
But as the classic fallacy goes: **Ice cream sales correlate with shark attacks**. Does banning ice cream stop sharks? No. Both are caused by *Summer*.

In churn prediction, "High Price" might correlate with churn, but is it the *cause*? Or is it the poor service associated with that tier?
*   ** The Risk**: Acting on correlation creates erroneous business strategies that cost millions.
*   **The Solution (DoubleML)**: I integrated **Double Machine Learning**, a State-of-the-Art causal inference framework. This allows us to strip away the noise and identifying the *Average Treatment Effect* (ATE)—the actual causal impact of a feature on the target.

## 4. Production Readiness

### Monitoring & Drift
In the real world, human behavior changes. A model trained on 2020 data fails in 2024 because the distribution of features shifts (**Data Drift**).
*   **Strategy**: The system is designed to be monitored, with the understanding that retraining is necessary as the "population" evolves.

### Architecture & Modularity
The codebase isn't a single script; it's a modular system.
*   **Backend**: A Flask API serves the predictions, separating inference logic from the user interface.
*   **Modularity**: Every component (preprocessing, training, inference) is decoupled. If an error occurs, we know exactly where it is.
*   **Intervention**: To assist the business team, I added a **Behavioral Strategy** layer—translating math into "Economist suggestions" for retention (e.g., Nudging users via defaults).

### Deployment (Docker)
Finally, to prove this works on a production server:
*   **Containerization**: The entire application is wrapped in **Docker**. It runs identically on my laptop and on a cloud server in Ohio.

---

## Technical Stack
*   **Causal Inference**: DoubleML
*   **Machine Learning**: XGBoost, Optuna, Scikit-Learn
*   **Backend**: Python, Flask
*   **Frontend**: Vanilla JavaScript
*   **Ops**: Docker, Git

## How to Run
```bash
# Build the container
docker build -t retention-v2 .

# Run the system
docker run -p 5000:5000 retention-v2
```

*Engineered by [Asermouh yassin]*
