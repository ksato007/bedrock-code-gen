import logging

def setup_logging(log_level):
    """
    Set up the logging configuration.

    Args:
        log_level (str): The desired log level (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR').
    """
    # Convert string log level to corresponding logging constant
    numeric_level = getattr(logging, log_level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {log_level}')

    # Configure the root logger
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Set the log level for all existing loggers
    for logger in logging.root.manager.loggerDict.values():
        if isinstance(logger, logging.Logger):
            logger.setLevel(numeric_level)

    # Ensure that new loggers will use this level by default
    logging.getLogger().setLevel(numeric_level)

logger = logging.getLogger(__name__)

def log(message, level='INFO'):
    """
    Log a message with the specified log level.

    Args:
        message (str): The message to be logged.
        level (str): The log level (e.g., 'DEBUG', 'INFO', 'WARNING', 'ERROR').
    """
    log_method = getattr(logger, level.lower())
    log_method(message)
