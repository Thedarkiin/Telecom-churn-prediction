# 📘 Codebase Deep Dive: Retention System v2

This document provides a comprehensive, file-by-file explanation of the entire Retention System. It is designed to act as a "developer's manual" so you can understand every line of logic.

---

## 🏗️ Root Directory: The Orchestration Layer

### 1. `pipeline.py`
**Role:** The "Conductor" of the Machine Learning Orchestra.
*   **Purpose**: Runs the entire pipeline: Data Loading -> Preprocessing -> Training (XGBoost/Optuna) -> Evaluation -> Causal Analysis.
*   **Key Logic**: It orchestrates the flow. It uses `src.config` to decide *how* to run (e.g., which columns to drop) and calls functions from `src.training` and `src.double_ml`.

### 2. `run_app.py`
**Role:** The Web Server Entrance.
*   **Purpose**: Starts the Flask application.
*   **Code**: `create_app().run(host='0.0.0.0', port=5000)`. simpler is better here.

### 3. `Dockerfile`
**Role:** The Production Envelope.
*   **Purpose**: Wraps the Python code into a Linux container.
*   **Details**:
    *   `FROM python:3.12-slim`: Base image.
    *   `COPY . /app`: Moves code to container.
    *   `RUN pip install`: Installs dependencies.
    *   `CMD`: Explicitly runs `python run_app.py`.

### 4. `docker-compose.yml`
**Role:** The Service Manager.
*   **Purpose**: Runs the Docker container with specific configurations.
*   **Key Configuration**:
    *   `volumes: - ./models:/app/models`: **Crucial**. This maps your local `models` folder to the container. If you re-train the model locally, the potentially running Docker container sees the update immediately (if restart policy allows) or upon restart, without needing a full `docker build`.
    *   `ports: - "5000:5000"`: Exposes the internal web server to your `localhost:5000`.

### 5. `requirements.txt`
**Role:** The Ingredient List.
*   **Purpose**: Lists all dependencies (`flask`, `xgboost`, `optuna`, `shap`, `DoubleML`, `scikit-learn`, `pandas`, `numpy`).

### 6. `diagram.png`
**Role:** Visual Architecture.
*   **Purpose**: A high-level visual representation of the system architecture.
*   **Content**: Shows the flow from Raw Data -> Preprocessing -> Model Training -> API Serving -> End User. It illustrates how the "Offline Training" pipeline feeds into the "Online Serving" application.

### 7. `ARCHITECTURE.html`
**Role:** Interactive Architecture.
*   **Purpose**: An interactive HTML version of the system design, often generated for presentations or documentation sites.

---

## 📁 src/: The Core Logic

### 8. `src/config.py`
**Role:** The Central Control Room.
*   **Purpose**: Stores all "Magic Numbers" and configurations.
*   **Key Configs**:
    *   `PIPELINE_CONFIG`: Dictionary controlling the entire ML process.
    *   `xgboost.n_trials`: Controls how many Optuna trials to run.
    *   `threshold.metric`: Determines if we optimize for Precision, Recall, or F1.

### 9. `src/preprocessing.py`
**Role:** The Data Plumber.
*   **Class `ChurnPreprocessor`**:
    *   `fit()`: Learns encodings and imputation values from Training Data.
    *   `transform()`: Applies them to Test/Production data.
    *   **Logic**: Handles `Tenure` (Scaling), `Contract` (One-Hot Encoding), and `TotalCharges` (Imputation).

### 10. `src/training.py`
**Role:** The Brain Builder.
*   **Functions**:
    *   `train_xgboost_with_optuna()`: The heavy lifter. Runs a Bayesian Search to find the best hyperparameters.
    *   `find_optimal_threshold()`: Scans probabilities to find the cutoff that maximizes F1-Score (Balance).

### 11. `src/double_ml.py`
**Role:** The Causal Scientist.
*   **Purpose**: Uses the Double Machine Learning framework.
*   **Logic**: Predicts `T` (Treatment) and `Y` (Outcome) separately, then correlates the residuals to find the true causal effect (ATE).

### 12. `src/explainability.py`
**Role:** The Interpreter.
*   **Purpose**: Generates SHAP plots.
*   **Logic**: Computes marginal contributions of features to the prediction.

### 13. `src/evaluation.py`
**Role:** The Auditor.
*   **Purpose**: Calculates Accuracy, Precision, Recall, F1, and ROC-AUC. Generates the confusion matrix plots.

### 14. `src/odds_ratio.py`
**Role:** The Statistician (Legacy).
*   **Purpose**: Calculates Odds Ratios from Logistic Regression for improved interpretability in the early stages of the project.

### 15. `src/utils.py`
**Role:** The Helper.
*   **Purpose**: Shared utility functions like `setup_logger` to ensure consistent logging format across all files.

---

## 📁 src/app/: The Web Application

### 16. `src/app/routes.py`
**Role:** The Traffic Controller.
*   **Purpose**: Defines API endpoints.
*   **Endpoints**:
    *   `GET /`: Serves the HTML page.
    *   `POST /predict`: Accepts JSON, runs inference, returns JSON.

### 17. `src/app/app.py`
**Role:** The Factory.
*   **Purpose**: Contains the `create_app()` function that initializes Flask and registers the Blueprints.

### 18. `src/templates/index.html`
**Role:** The Face.
*   **Purpose**: The actual UI structure.
*   **Features**: Responsive form, hidden result section, "Compare" slider.

### 19. `src/static/js/app.js`
**Role:** The Nervous System.
*   **Purpose**: Handles user interaction.
*   **Logic**: Listens for "Submit", serializes form data, calls API, updates DOM with results (Red/Green badge).

### 20. `src/static/css/style.css`
**Role:** The Makeup.
*   **Purpose**: Styling.
*   **Features**: Glassmorphism, CSS Variables for theme colors, Flexbox layouts.

---

## 📁 results/ & models/

### 21. `results/`
**Role:** The Output.
*   Stores generated CSV reports (`metrics/`) and PNG charts (`explainability/`).

### 22. `models/`
**Role:** The Artifacts.
*   Stores the trained `.pkl` files. `preprocessor.pkl` is the most critical file for ensuring production data matches training data structure.
