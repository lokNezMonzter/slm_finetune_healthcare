import os
import sys
import logging
from time import time


class CustomFormatter(logging.Formatter):
    """
    Adds colors to logs: Green (Success), Red (Error), Blue (Info), Yellow (Warning)
    """
    blue = "\x1b[34;20m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"
    log_format = "%(asctime)s - [%(name)s] - %(levelname)s - %(message)s"

    FORMATS = {
        logging.DEBUG: blue + log_format + reset,
        logging.INFO: green + log_format + reset, 
        logging.WARNING: yellow + log_format + reset,
        logging.ERROR: red + log_format + reset,
        logging.CRITICAL: bold_red + log_format + reset
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt, datefmt='%H:%M:%S')
        return formatter.format(record)


def setup_logger(logger_name="ClinicalPipeline"):
    logger = logging.getLogger(logger_name)

    # Prevent duplicate handlers if the logger is alrady initialized
    if logger.handlers:
        return logger
    
    logger.setLevel(logging.DEBUG)

    # Color formatter for terminal and plain text formatter for file
    color_formatter = CustomFormatter()
    plain_formatter = logging.Formatter(
        "%(asctime)s - [%(name)s] - %(levelname)s - %(message)s",
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Terminal Handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setFormatter(color_formatter)
    console_handler.setLevel(logging.INFO)

    # Permanent File Handler
    file_handler = logging.FileHandler(f"/workspace/logs/{logger_name}.log", mode="w")
    file_handler.setFormatter(plain_formatter)
    file_handler.setLevel(logging.DEBUG)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

    # SILENCE ALL LIBRARIES (Turn off OpenAI, Datasets, HTTPX, etc.)
    for package_name in logging.root.manager.loggerDict:
        if package_name != logger_name:
            logging.getLogger(package_name).setLevel(logging.ERROR)

    return logger
