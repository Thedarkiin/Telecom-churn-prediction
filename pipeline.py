import logging
import pandas as pd

from src.config import Config
from src.utils import setup_logger
from src.preprocessing import preprocess_data
from src.training import train_models
from src.evaluation import evaluate_models

def main():
    logger = setup_logger("logs/pipeline.log")
    logger.info("Starting churn prediction pipeline")

    df = pd.read_csv(Config.DATA_PATH)

    X_train, X_test, y_train, y_test = preprocess_data(Config)

    models = train_models(X_train, y_train)

    evaluate_models(models, X_test, y_test)

    logger.info("Pipeline execution completed")

if __name__ == "__main__":
    main()
