"""Logging Configuration."""

import logging


def setup_logging() -> None:
    """Sets up the logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %I:%M:%S %p",
    )


def simulation_progress_logging(logger: logging, text: str, idx: int, log_freq: int) -> None:
    """Logs the progress of the simulation.

    Args:
        logger: Actual logger of the module.
        text: Text to be logged is in the format "{text}: {idx} DONE"
        idx: Actual index of the simulation.
        log_freq: The logging is going to be done every log_freq iterations.

    Returns:
        None
    """
    if idx % log_freq == 0:
        logger.info(f"{text}: {idx} DONE")
