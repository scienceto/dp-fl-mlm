import json
import os
from datetime import datetime
import shutil
import logging

def setup_logger(name: str, log_file: str = None, level=logging.INFO) -> logging.Logger:
    """
    Sets up a logger with the specified name and optional file handler.
    
    Args:
        name (str): Name of the logger.
        log_file (str): If provided, logs will also be written to this file.
        level: Logging level (default is logging.INFO).

    Returns:
        logging.Logger: Configured logger instance.
    """
    formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(name)s | %(message)s')
    logger = logging.getLogger(name)
    logger.setLevel(level)

    if not logger.handlers:
        # Console handler
        ch = logging.StreamHandler()
        ch.setFormatter(formatter)
        logger.addHandler(ch)

        # File handler (optional)
        if log_file:
            os.makedirs(os.path.dirname(log_file), exist_ok=True)
            fh = logging.FileHandler(log_file)
            fh.setFormatter(formatter)
            logger.addHandler(fh)

    return logger

# Initialize logger
logger = setup_logger("utils")

def log_to_file(log_struct: dict, file_path: str) -> None:
    """
    Append a JSON record to a newline-delimited log file.

    Args:
        log_struct (dict): The dictionary to log.
        file_path (str): Destination log file path.
    """
    try:
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'a') as log_file:
            json.dump(log_struct, log_file)
            log_file.write('\n')
        logger.info(f"Logged entry to: {file_path}")
    except Exception as e:
        logger.error(f"Failed to log to file {file_path}: {e}")
        raise

def create_results_dir(base_dir: str = ".", overwrite: bool = False) -> str:
    """
    Create a timestamped results directory (format: results_YYYY-MM-DD_HH-MM).

    Args:
        base_dir (str): Where to create the results directory.
        overwrite (bool): Whether to overwrite an existing directory.

    Returns:
        str: Path to the created results directory.
    """
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    dir_name = f"results_{timestamp}"
    full_path = os.path.join(base_dir, dir_name)

    if os.path.exists(full_path):
        if overwrite:
            shutil.rmtree(full_path)
            logger.warning(f"Overwriting existing directory: {full_path}")
        else:
            logger.info(f"Results directory already exists, reusing: {full_path}")
            return full_path

    os.makedirs(full_path, exist_ok=True)
    logger.info(f"Created results directory: {full_path}")
    return full_path
