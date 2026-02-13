import os
import logging
import matplotlib.pyplot as plt

def setup_logger(name=__name__, log_file=None):
    """
    Set up logging configuration. Ensures log directory exists.
    Returns a logger instance.
    """
    # Use absolute path relative to project root (assuming src/utils.py is 2 levels deep from root)
    if log_file is None:
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        log_file = os.path.join(base_dir, "logs", "pipeline.log")

    log_dir = os.path.dirname(log_file)
    if log_dir and not os.path.exists(log_dir):
        os.makedirs(log_dir, exist_ok=True)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    
    # Avoid adding duplicate handlers
    if not logger.handlers:
        # File Handler
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
        logger.addHandler(file_handler)
        
        # Console Handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(logging.Formatter("%(message)s"))
        logger.addHandler(console_handler)

    return logger

def save_plot(fig, filename, folder):
    """
    Save matplotlib figure to folder with filename and close figure.
    """
    if not os.path.exists(folder):
        os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, filename)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logging.info(f"Plot saved to {path}")
