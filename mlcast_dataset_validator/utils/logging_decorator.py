from functools import wraps

from loguru import logger


def log_function_call(func):
    """Decorator to log function calls with their arguments."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        logger.info(f"Applying {func.__name__} with {kwargs}")
        report = func(*args, **kwargs)
        # Set module and function name on the result object
        for result in report.results:
            result.module = func.__module__
            result.function = func.__name__
        return report

    return wrapper
