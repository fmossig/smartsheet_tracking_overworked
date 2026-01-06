"""
Centralized logging configuration for Smartsheet tracker application.

This module provides a configurable logging setup that supports:
1. Environment variable configuration (LOG_LEVEL)
2. Command-line argument configuration (--log-level)
3. Default fallback to INFO level

Log Levels (in order of increasing severity):
- DEBUG: Detailed information for diagnosing problems
- INFO: Confirmation that things are working as expected
- WARNING: Something unexpected happened, but the software is still working
- ERROR: A more serious problem, the software couldn't perform some function

Usage:
    # In main scripts, add --log-level argument to parser:
    from logging_config import add_log_level_argument, configure_logging

    parser = argparse.ArgumentParser(description="My Script")
    add_log_level_argument(parser)
    args = parser.parse_args()

    # Configure logging with parsed args
    configure_logging(
        log_file="my_script.log",
        log_level=getattr(args, 'log_level', None)
    )

    # In utility modules, just get a logger:
    import logging
    logger = logging.getLogger(__name__)

Example:
    # Via environment variable:
    $ LOG_LEVEL=DEBUG python smartsheet_tracker.py

    # Via command-line argument:
    $ python smartsheet_tracker.py --log-level DEBUG

    # Command-line takes precedence over environment variable
"""

import os
import logging
import argparse
from typing import Optional

# Valid log levels
VALID_LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR']

# Default log level
DEFAULT_LOG_LEVEL = 'INFO'

# Environment variable name for log level
LOG_LEVEL_ENV_VAR = 'LOG_LEVEL'

# Default log formats
# Standard format for INFO and below - concise single line
STANDARD_LOG_FORMAT = '%(asctime)s - %(levelname)s - %(name)s - %(message)s'

# Detailed format for ERROR and above - includes more context
DETAILED_LOG_FORMAT = '%(asctime)s - %(levelname)s - %(name)s - %(funcName)s:%(lineno)d - %(message)s'

# Environment variable for detailed logging
DETAILED_LOGGING_ENV_VAR = 'DETAILED_LOGGING'


def get_log_level(cli_level: Optional[str] = None) -> int:
    """
    Determine the log level from CLI argument or environment variable.

    Priority order:
    1. Command-line argument (if provided)
    2. LOG_LEVEL environment variable
    3. Default (INFO)

    Args:
        cli_level: Log level specified via command-line argument.

    Returns:
        The logging level as an integer constant (e.g., logging.DEBUG).

    Raises:
        ValueError: If an invalid log level is specified.

    Example:
        >>> level = get_log_level('DEBUG')
        >>> level == logging.DEBUG
        True
    """
    # Determine the level string
    level_str = None

    # CLI argument takes precedence
    if cli_level:
        level_str = cli_level.upper()
    else:
        # Check environment variable
        env_level = os.getenv(LOG_LEVEL_ENV_VAR)
        if env_level:
            level_str = env_level.upper()

    # Use default if no level specified
    if not level_str:
        level_str = DEFAULT_LOG_LEVEL

    # Validate the level
    if level_str not in VALID_LOG_LEVELS:
        raise ValueError(
            f"Invalid log level: '{level_str}'. "
            f"Valid levels are: {', '.join(VALID_LOG_LEVELS)}"
        )

    # Convert to logging constant
    return getattr(logging, level_str)


class DetailedErrorFormatter(logging.Formatter):
    """
    Custom formatter that uses detailed format for ERROR and above.

    This formatter automatically switches to a more detailed format
    for ERROR and CRITICAL level messages to aid in debugging.
    """

    def __init__(
        self,
        standard_fmt: str = STANDARD_LOG_FORMAT,
        detailed_fmt: str = DETAILED_LOG_FORMAT,
        datefmt: Optional[str] = None
    ):
        """
        Initialize the formatter.

        Args:
            standard_fmt: Format for INFO and below.
            detailed_fmt: Format for ERROR and above.
            datefmt: Date format string.
        """
        super().__init__(fmt=standard_fmt, datefmt=datefmt)
        self.standard_fmt = standard_fmt
        self.detailed_fmt = detailed_fmt

    def format(self, record: logging.LogRecord) -> str:
        """Format the log record based on severity level."""
        # Use detailed format for ERROR and CRITICAL
        if record.levelno >= logging.ERROR:
            self._style._fmt = self.detailed_fmt
        else:
            self._style._fmt = self.standard_fmt

        return super().format(record)


def configure_logging(
    log_file: Optional[str] = None,
    log_level: Optional[str] = None,
    log_format: Optional[str] = None,
    use_detailed_formatter: bool = True
) -> logging.Logger:
    """
    Configure logging for the application with the specified settings.

    This function sets up the root logger with handlers for both console
    and file output (if log_file is specified). The log level is determined
    by the priority: CLI argument > environment variable > default.

    When use_detailed_formatter is True (default), ERROR and CRITICAL messages
    will automatically include additional context (function name, line number)
    to aid in debugging.

    Args:
        log_file: Optional path to log file. If None, only console logging.
        log_level: Optional log level from CLI argument. If None, checks
            environment variable or uses default.
        log_format: Format string for log messages. If None, uses smart
            formatting based on log level.
        use_detailed_formatter: If True, use DetailedErrorFormatter that
            switches format based on severity level.

    Returns:
        The configured root logger.

    Example:
        >>> logger = configure_logging(
        ...     log_file='app.log',
        ...     log_level='DEBUG'
        ... )
        >>> logger.debug("This will be logged")
        >>> logger.error("This will include function and line info")
    """
    # Get the appropriate log level
    level = get_log_level(log_level)

    # Clear any existing handlers on the root logger
    root_logger = logging.getLogger()
    root_logger.handlers.clear()
    root_logger.setLevel(level)

    # Determine formatter to use
    if log_format:
        # Custom format specified, use standard formatter
        formatter = logging.Formatter(log_format)
    elif use_detailed_formatter or os.getenv(DETAILED_LOGGING_ENV_VAR, '').lower() == 'true':
        # Use smart formatter that switches based on severity
        formatter = DetailedErrorFormatter()
    else:
        # Use standard format
        formatter = logging.Formatter(STANDARD_LOG_FORMAT)

    # Create and configure console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    root_logger.addHandler(console_handler)

    # Add file handler if log file specified
    if log_file:
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Log the configured level for transparency
    level_name = logging.getLevelName(level)
    source = "command-line" if log_level else (
        "environment variable" if os.getenv(LOG_LEVEL_ENV_VAR) else "default"
    )
    logging.debug(f"Logging configured: level={level_name} (from {source})")

    return root_logger


def add_log_level_argument(parser: argparse.ArgumentParser) -> None:
    """
    Add the --log-level argument to an ArgumentParser.

    This standardizes the log level argument across all scripts in the
    application, ensuring consistent behavior and help text.

    Args:
        parser: The ArgumentParser to add the argument to.

    Example:
        >>> parser = argparse.ArgumentParser()
        >>> add_log_level_argument(parser)
        >>> args = parser.parse_args(['--log-level', 'DEBUG'])
        >>> args.log_level
        'DEBUG'
    """
    parser.add_argument(
        '--log-level',
        type=str,
        choices=VALID_LOG_LEVELS,
        default=None,
        metavar='LEVEL',
        help=(
            f"Set logging verbosity level. "
            f"Choices: {', '.join(VALID_LOG_LEVELS)}. "
            f"Can also be set via {LOG_LEVEL_ENV_VAR} environment variable. "
            f"CLI argument takes precedence. Default: {DEFAULT_LOG_LEVEL}"
        )
    )


def get_module_logger(name: str) -> logging.Logger:
    """
    Get a logger for a specific module.

    This is a convenience function that simply wraps logging.getLogger().
    It's provided for consistency and potential future enhancements.

    Args:
        name: The module name, typically __name__.

    Returns:
        A Logger instance for the module.

    Example:
        >>> logger = get_module_logger(__name__)
        >>> logger.info("Module initialized")
    """
    return logging.getLogger(name)
