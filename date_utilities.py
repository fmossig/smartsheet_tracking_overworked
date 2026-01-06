"""
Date Utilities Module

Provides unified date parsing and normalization functions for the smartsheet-tracker
application. This module consolidates all date handling logic to ensure consistent
date parsing across the tracker and report modules.

Usage:
    from date_utilities import (
        parse_date,
        parse_timestamp,
        normalize_date_for_comparison,
        parse_date_argument,
        SUPPORTED_DATE_FORMATS,
    )
"""

import logging
from datetime import datetime, date
from typing import Optional, Union
import argparse

# Set up logging
logger = logging.getLogger(__name__)

# Supported date formats in order of preference
SUPPORTED_DATE_FORMATS = (
    '%Y-%m-%dT%H:%M:%S',  # ISO format with time
    '%Y-%m-%d',           # ISO format date only
    '%d.%m.%Y',           # European format (German)
    '%m/%d/%Y',           # US format
    '%Y/%m/%d',           # Alternative ISO format
    '%d/%m/%Y',           # UK format
)


def parse_date(value: Union[str, date, datetime, None]) -> Optional[date]:
    """
    Parse date from various formats into a date object.

    Handles datetime.date, datetime.datetime objects, and strings in multiple
    formats. Provides graceful degradation for invalid values by returning None
    instead of raising exceptions.

    Args:
        value: The date value to parse. Can be:
            - datetime.datetime object
            - datetime.date object
            - String in formats: 'YYYY-MM-DDTHH:MM:SS', 'YYYY-MM-DD',
              'DD.MM.YYYY', 'MM/DD/YYYY', 'YYYY/MM/DD', 'DD/MM/YYYY', or ISO format
            - None or empty string

    Returns:
        datetime.date: Parsed date object if successful.
        None: If the value is empty, None, or cannot be parsed.

    Raises:
        No exceptions are raised; invalid formats are logged as warnings
        and None is returned.

    Example:
        >>> parse_date("2025-01-06")
        datetime.date(2025, 1, 6)
        >>> parse_date(datetime(2025, 1, 6, 10, 30))
        datetime.date(2025, 1, 6)
        >>> parse_date("15.01.2025")
        datetime.date(2025, 1, 15)
        >>> parse_date("invalid")
        None  # Logs warning
    """
    if not value:
        return None

    # Accept native date/datetime objects directly
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, date):
        return value

    # Fall back to string parsing
    cleaned = str(value).strip()
    original_value = cleaned  # Keep original for logging

    # Clean up common trailing characters (e.g., accidental suffixes)
    if cleaned and not cleaned[-1].isdigit():
        cleaned = cleaned.rstrip('abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ')

    # Try various formats
    for fmt in SUPPORTED_DATE_FORMATS:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue

    try:
        # Try ISO format (catches many variations)
        return datetime.fromisoformat(cleaned).date()
    except Exception:
        # Log the invalid date format gracefully
        logger.warning(
            f"Invalid date format encountered: '{original_value}'. "
            f"Supported formats: {', '.join(SUPPORTED_DATE_FORMATS)} or ISO format. "
            f"Returning None and continuing processing."
        )
        return None


def parse_timestamp(timestamp_str: Union[str, None]) -> Optional[datetime]:
    """
    Parse timestamp string to datetime object.

    Args:
        timestamp_str: Timestamp string in format "YYYY-MM-DD HH:MM:SS"

    Returns:
        datetime object or None if parsing fails. Invalid timestamp formats
        are logged gracefully and processing continues without raising exceptions.

    Example:
        >>> parse_timestamp("2025-01-06 10:30:45")
        datetime.datetime(2025, 1, 6, 10, 30, 45)
        >>> parse_timestamp(None)
        None
    """
    if not timestamp_str:
        return None
    try:
        return datetime.strptime(str(timestamp_str).strip(), "%Y-%m-%d %H:%M:%S")
    except ValueError:
        # Log the invalid timestamp format gracefully
        logger.warning(
            f"Invalid timestamp format encountered: '{timestamp_str}'. "
            f"Expected format: 'YYYY-MM-DD HH:MM:SS'. "
            f"Returning None and continuing processing."
        )
        return None


def normalize_date_for_comparison(value: Union[str, date, datetime, None]) -> Optional[str]:
    """
    Normalize any date value to YYYY-MM-DD string for consistent comparison.

    Converts various date representations to a standardized ISO format string,
    enabling reliable comparison between values that may have different formats
    (e.g., comparing a datetime object with an ISO string).

    Args:
        value: The date value to normalize. Can be:
            - datetime.date object
            - datetime.datetime object (time component stripped)
            - ISO string with time ('2025-10-23T00:00:00')
            - Date string ('2025-10-23')
            - None

    Returns:
        str: Normalized date string in 'YYYY-MM-DD' format.
        None: If the input value is None or cannot be parsed.

    Example:
        >>> normalize_date_for_comparison(datetime.date(2025, 10, 23))
        '2025-10-23'
        >>> normalize_date_for_comparison('2025-10-23T00:00:00')
        '2025-10-23'
        >>> normalize_date_for_comparison(None)
        None
    """
    if value is None:
        return None

    # Try to parse as date first
    parsed = parse_date(value)
    if parsed:
        return parsed.isoformat()  # Always returns YYYY-MM-DD

    # Fallback to string representation
    return str(value).strip()


def parse_date_argument(date_str: str) -> date:
    """
    Parse a date string argument from command line input.

    This function is designed for use with argparse and raises
    argparse.ArgumentTypeError on invalid input for proper CLI error messages.

    Args:
        date_str: Date string to parse

    Returns:
        datetime.date: Parsed date object

    Raises:
        argparse.ArgumentTypeError: If the date string cannot be parsed

    Example:
        >>> parser.add_argument('--start-date', type=parse_date_argument)
    """
    if not date_str:
        raise argparse.ArgumentTypeError("Date string cannot be empty")

    cleaned = str(date_str).strip()

    # Try common CLI date formats
    cli_formats = ('%Y-%m-%d', '%d.%m.%Y', '%d/%m/%Y')
    for fmt in cli_formats:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue

    # Try ISO format as fallback
    try:
        return datetime.fromisoformat(cleaned).date()
    except Exception:
        pass

    raise argparse.ArgumentTypeError(
        f"Invalid date format: '{date_str}'. "
        f"Use YYYY-MM-DD, DD.MM.YYYY, or DD/MM/YYYY format."
    )


def parse_date_strict(value: Union[str, date, datetime]) -> date:
    """
    Parse date from various formats with strict validation.

    Similar to parse_date() but raises ValueError on invalid input
    instead of returning None. Use this when you need to ensure
    the date is valid and want to handle errors explicitly.

    Args:
        value: The date value to parse

    Returns:
        datetime.date: Parsed date object

    Raises:
        ValueError: If the value cannot be parsed as a date

    Example:
        >>> parse_date_strict("2025-01-06")
        datetime.date(2025, 1, 6)
        >>> parse_date_strict("invalid")
        ValueError: Cannot parse date from string: 'invalid'
    """
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if not isinstance(value, str):
        raise ValueError(f"Cannot parse date from type: {type(value)}")

    result = parse_date(value)
    if result is None:
        raise ValueError(f"Cannot parse date from string: '{value}'")
    return result


def format_report_timestamp(dt: Optional[datetime] = None) -> str:
    """
    Format a datetime for report generation timestamp display.

    Creates a human-readable timestamp string suitable for displaying
    on report cover pages and footers. Uses European format for dates
    with 24-hour time notation to match the project's locale conventions.

    Args:
        dt: The datetime to format. If None, uses current datetime.

    Returns:
        str: Formatted timestamp string in format "DD.MM.YYYY at HH:MM"

    Example:
        >>> format_report_timestamp(datetime(2025, 1, 6, 14, 35, 0))
        '06.01.2025 at 14:35'
        >>> format_report_timestamp()  # Uses current time
        '06.01.2025 at 14:35'  # Example output
    """
    if dt is None:
        dt = datetime.now()
    return dt.strftime('%d.%m.%Y at %H:%M')
