# 📘 Codebase Deep Dive: Retention System v2

This document provides a comprehensive, file-by-file explanation of the entire Retention System. It is designed to act as a "developer's manual" so you can understand every line of logic, from the statistical foundations to the production deployment configuration.

---

## 🏗️ Root Directory: The Orchestration Layer

### 1. `pipeline.py`
**Role:** The "Conductor" of the Machine Learning Orchestra.
This is the master script that runs the entire training process from start to finish. It doesn't serve the web app; it builds the intelligence.

*   **Key Functions:**
    *   `main()`: The entry point. It calls `preprocess_data`, `train_models`, `evaluate`, etc. sequentially.
    *   `clear_results()`: A utility to wipe the `results/` folder clean before a new run, ensuring no stale data remains.
*   **Logic Flow:**
    1.  **Preprocessing**: Clean data and handle missing values.
    2.  **Validation**: Check assumptions (e.g., "Are these variables actually linear?").
    3.  **Training**: Run XGBoost with Optuna to find the best model.
    4.  **Threshold Optimization**: Find the probability cutoff (e.g., 52%) that balances Precision and Recall.
    5.  **Causal Analysis**: Run DoubleML to find which features *cause* churn, not just predict it.
    6.  **Saving**: Dump the trained models (`.pkl` files) into `models/`.

### 2. `run_app.py`
**Role:** The Web Server Entrance.
This is a tiny script that acts as the "ignition key" for the Flask web application.
*   **Logic:** It imports the `create_app` function from `src.app` and runs it on `0.0.0.0` (accessible to the outside world, not just localhost) on port 5000. It's kept simple on purpose to separate "running" from "logic".

### 3. `Dockerfile`
**Role:** The Production Envelope.
This file creates a self-contained "virtual computer" (container) that holds your app.
*   **Line-by-Line Breakdown:**
    *   `FROM python:3.12-slim`: Starts with a minimal Python Linux installation.
    *   `WORKDIR /app`: Sets the working folder inside the container.
    *   `COPY . /app`: Copies your code from Windows into the Linux container.
    *   `RUN pip install...`: Installs all libraries listed in `requirements.txt`.
    *   `EXPOSE 5000`: Opens port 5000 so the world can talk to the Flask app.
    *   `CMD ["python", "run_app.py"]`: The command that runs when the container starts.

### 4. `docker-compose.yml`
**Role:** The Service Manager.
While `Dockerfile` builds the image, `docker-compose` runs it. It's like a configuration file for the runtime.
*   **Key Settings:**
    *   `volumes`: Maps your local `models/` folder to the container. This means if you re-train a model locally, the Docker app sees it instantly without rebuilding.
    *   `ports`: Connects your Windows port 5000 to the Container's port 5000.

### 5. `requirements.txt`
**Role:** The Ingredient List.
Lists every Python library needed.
*   `flask`: Web framework.
*   `xgboost`: The gradient boosting machine learning library.
*   `optuna`: The hyperparameter optimization engine.
*   `shap`: Game-theoretic explainability.
*   `DoubleML`: Causal inference library.

---

## 📁 src/: The Core Logic

### 6. `src/config.py`
**Role:** The Central Control Room.
Instead of hard-coding numbers like "0.2" or "42" throughout the code, we put them here.
*   **Key Sections:**
    *   `PIPELINE_CONFIG`: A massive dictionary controlling the ML behavior.
    *   `param_space`: Defines the range of values Optuna is allowed to search (e.g., "Try tree depths between 3 and 9").
    *   `threshold`: Defines the target metric (Precision/Recall). changing this one value changes how the model makes decisions globally.

### 7. `src/preprocessing.py`
**Role:** The Data Plumber.
Raw data is dirty. This file cleans it.
*   **Class `ChurnPreprocessor`**:
    *   `fit()`: Learns from the training data (e.g., "What is the average tenure?").
    *   `transform()`: Applies those learnings. Crucially, it fills missing values (Imputation) and converts text to numbers (Encoding).
    *   **Logic**: It treats "Tenure" and "MonthlyCharges" as continuous numbers (scaled) and "Plan Type" as categories (One-Hot Encoded).

### 8. `src/training.py`
**Role:** The Brain Builder.
This is where the magic happens.
*   **`train_xgboost_with_optuna()`**:
    *   It starts a "Study".
    *   It tries 20 different combinations of hyper-parameters (Learning Rate, Depth, etc.).
    *   For each combination, it evaluates performance using Cross-Validation (training on 4 chunks, testing on 1).
    *   It picks the winner and trains the final model.
*   **`find_optimal_threshold()`**:
    *   Most models output a probability (0.0 to 1.0).
    *   Standard ML picks 0.5 as the cutoff.
    *   This function scans 0.3, 0.31, 0.32... to find the cutoff that maximizes Precision (>70%), ensuring we don't spam innocent customers.

### 9. `src/double_ml.py`
**Role:** The Scientist.
This file implements Causal Inference.
*   **The Problem it Solves**: "Contract Type" predicts churn, but is it the contract itself, or the type of person who signs it?
*   **How it works**:
    1.  It builds a model to predict who signs a contract (Propensity).
    2.  It builds a model to predict churn (Outcome).
    3.  It subtracts the two to find the "Residual" — the pure causal effect remaining after removing bias.
*   **Output**: An "Average Treatment Effect" (ATE). If ATE is -0.15, the contract *physically causes* a 15% drop in churn.

### 10. `src/explainability.py`
**Role:** The Interpreter.
Black-box models (like XGBoost) are hard to understand. This file uses SHAP (SHapley Additive exPlanations).
*   **Logic**: It calculates how much each feature contributed to the prediction by simulating "what if this feature was missing?".
*   **Output**: The beeswarm plots you see in `results/explainability/`.

### 11. `src/evaluation.py`
**Role:** The Scorekeeper.
Generates the confusion matrices and ROC curves.
*   It calculates Precision (Accuracy of positive predictions), Recall (Coverage of actual positives), and F1-Score (Harmonic mean).

---

## 📁 src/app/: The Web Application

### 12. `src/app/routes.py`
**Role:** The Traffic Controller.
Handles HTTP requests from the browser.
*   **`@bp.route('/predict')`**:
    1.  Receives JSON data from the frontend.
    2.  Passes it to the `ModelLoader`.
    3.  Returns the probability and the "Causal Insight" (e.g., "Nudge this user to a 2-year contract").
*   **`ModelLoader` Class**: A Singleton that loads the heavy `.pkl` files once on startup, so we don't reload 500MB of models for every user request.

### 13. `src/templates/index.html`
**Role:** The Face.
The HTML structure of the user interface.
*   It uses a "Single Page Application" feel.
*   **Key Elements**:
    *   The Input Form (Left panel).
    *   The Result Card (Right panel, initially hidden).
    *   The "Compare" Slide-over (For the behavioral science nudges).

### 14. `src/static/js/app.js`
**Role:** The Nervous System.
Connects the HTML face to the Python brain.
*   **`predict()` function**:
    *   Grabs values from the HTML inputs.
    *   Packages them into JSON.
    *   Sends a `POST` request to `/predict`.
    *   Waits for the answer.
    *   Updates the DOM (Document Object Model) to show the "High Risk" red badge without reloading the page.

### 15. `src/static/css/style.css`
**Role:** The Makeup.
Makes it look Enterprise-grade.
*   **Design System**:
    *   Uses a restricted color palette (Enterprise Blue, Alert Red, Success Green).
    *   **Glassmorphism**: The cards have a slight transparency and blur (`backdrop-filter: blur(10px)`).
    *   **Flexbox/Grid**: Used for layout to ensure responsiveness on all screen sizes.

---

## 📁 results/: The Evidence

### 16. `results/metrics/all_metrics_combined.csv`
**Role:** The Report Card.
Contains the final scores of every model tried. This is what you checked to verify Precision > 70%.

### 17. `results/causal/double_ml_results.csv`
**Role:** The scientific proof.
Contains the raw ATE numbers and confidence intervals from the DoubleML analysis.

---

## 📁 models/: The Artifacts

### 18. `models/*.pkl`
**Role:** The Frozen Brains.
These are the serialized Python objects.
*   `xgboost_model.pkl`: The trained XGBoost model.
*   `preprocessor.pkl`: The exact rules used to clean the data (e.g., the mean value of "Tenure" used to fill N/A). It is CRITICAL this is saved, or the app wouldn't know how to process new data identically to the training data.
