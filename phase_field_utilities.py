"""
Phase Field Utilities Module

Provides unified phase field definitions and parsing functions for the smartsheet-tracker
application. This module consolidates all phase field handling logic to ensure consistent
phase parsing across the tracker, report, and diagnostic modules.

This module eliminates code duplication by centralizing:
- PHASE_FIELDS: The definitive list of workflow phases and their column mappings
- EXPECTED_DATE_COLUMNS: List of date columns required for full tracking
- OPTIONAL_COLUMNS: List of columns that enhance tracking but are not required
- Column resolution functions for handling column name variations
- Phase availability detection for graceful degradation

Usage:
    from phase_field_utilities import (
        PHASE_FIELDS,
        EXPECTED_DATE_COLUMNS,
        OPTIONAL_COLUMNS,
        resolve_column_name,
        get_phase_columns,
        detect_missing_columns,
        get_available_phases,
        get_unavailable_phases,
    )
"""

import logging
from typing import Dict, List, Optional, Tuple, Any, Union

# Set up logging
logger = logging.getLogger(__name__)


# ============================================================================
# PHASE FIELD DEFINITIONS
# ============================================================================

# Fields to track - (date_column, user_column_variations, phase_number)
# user_column_variations is a tuple of possible column name variations to try
# These represent the workflow phases tracked in Smartsheet:
#   Phase 1: Kontrolle (Control check)
#   Phase 2: BE am (Processing date)
#   Phase 3: K am (Second control check)
#   Phase 4: C am (Completion date)
#   Phase 5: Reopen C2 am (Reopened completion date)
PHASE_FIELDS: List[Tuple[str, Tuple[str, ...], int]] = [
    ("Kontrolle", ("K von",), 1),
    ("BE am", ("BE von",), 2),
    ("K am", ("K2 von", "K 2 von"), 3),  # Bundle sheets use "K 2 von" (with space)
    ("C am", ("C von",), 4),
    ("Reopen C2 am", ("Reopen C2 von",), 5),
]

# Expected columns for validation (all date columns that should exist)
EXPECTED_DATE_COLUMNS: List[str] = ["Kontrolle", "BE am", "K am", "C am", "Reopen C2 am"]

# Optional columns that may or may not exist
OPTIONAL_COLUMNS: List[str] = ["Amazon"]

# Phase display names for reporting
PHASE_NAMES: Dict[str, str] = {
    "1": "Phase 1",
    "2": "Phase 2",
    "3": "Phase 3",
    "4": "Phase 4",
    "5": "Phase 5",
}


# ============================================================================
# COLUMN RESOLUTION FUNCTIONS
# ============================================================================

def resolve_column_name(
    col_map: Dict[str, int],
    column_variations: Union[Tuple[str, ...], str]
) -> Tuple[Optional[str], Optional[int]]:
    """
    Resolve a column name from a list of possible variations.

    Different Smartsheet sheets may use slightly different column names for the
    same logical field (e.g., "K2 von" vs "K 2 von"). This function tries each
    variation in order and returns the first match found.

    Args:
        col_map: Dictionary mapping column titles to column IDs.
            Example: {"Kontrolle": 123, "BE am": 456, "K2 von": 789}
        column_variations: Tuple of possible column name variations
            to try in order. Can also be a single string for backward compatibility.
            Example: ("K2 von", "K 2 von")

    Returns:
        Tuple of (resolved_column_name, column_id) if found.
        Returns (None, None) if no variation exists in col_map.

    Example:
        >>> col_map = {"K2 von": 789, "BE am": 456}
        >>> resolve_column_name(col_map, ("K2 von", "K 2 von"))
        ('K2 von', 789)
        >>> resolve_column_name(col_map, ("Missing",))
        (None, None)
        >>> resolve_column_name(col_map, "BE am")  # Single string also works
        ('BE am', 456)
    """
    if isinstance(column_variations, str):
        # Handle single string for backward compatibility
        column_variations = (column_variations,)

    for variation in column_variations:
        if variation in col_map:
            return variation, col_map[variation]

    return None, None


def get_phase_columns(
    col_map: Dict[str, int],
    date_col: str,
    user_col_variations: Tuple[str, ...]
) -> Tuple[Optional[int], Optional[str], Optional[int]]:
    """
    Get the date and user column IDs for a phase, handling column name variations.

    Retrieves the column IDs needed to track a specific workflow phase. Each phase
    has a date column (when the phase was completed) and a user column (who completed it).
    The user column may have multiple naming variations across different sheets.

    Args:
        col_map: Dictionary mapping column titles to column IDs.
            Example: {"Kontrolle": 123, "K von": 456}
        date_col: The exact name of the date column to find.
            Example: "Kontrolle", "BE am", "K am"
        user_col_variations: Tuple of possible user column name variations.
            Example: ("K von",), ("K2 von", "K 2 von")

    Returns:
        Tuple of (date_col_id, user_col_name, user_col_id) if both columns found.
        Returns (None, None, None) if either column is missing.

    Example:
        >>> col_map = {"Kontrolle": 123, "K von": 456, "BE am": 789, "BE von": 101}
        >>> get_phase_columns(col_map, "Kontrolle", ("K von",))
        (123, 'K von', 456)
        >>> get_phase_columns(col_map, "Missing", ("K von",))
        (None, None, None)
    """
    if date_col not in col_map:
        return None, None, None

    date_col_id = col_map[date_col]
    user_col_name, user_col_id = resolve_column_name(col_map, user_col_variations)

    if user_col_id is None:
        return None, None, None

    return date_col_id, user_col_name, user_col_id


# ============================================================================
# PHASE AVAILABILITY DETECTION
# ============================================================================

def detect_missing_columns(
    col_map: Dict[str, int],
    group_name: str,
    log_warnings: bool = True
) -> Dict[str, Any]:
    """
    Detect expected columns that are missing from a sheet.

    Analyzes a sheet's column structure to determine which of the expected
    phase columns are present and which are missing. This enables graceful
    degradation - the tracker can process sheets even if some columns are
    missing, simply skipping unavailable phases.

    Args:
        col_map: Dictionary mapping column titles to column IDs.
            Example: {"Kontrolle": 123, "K von": 456, "BE am": 789}
        group_name: Name of the sheet group for logging purposes.
            Example: "NA", "NF", "BUNDLE_FAN"
        log_warnings: Whether to log warnings for missing columns.
            Default is True. Set to False for silent validation.

    Returns:
        A dictionary containing column status information:
            - 'missing_date_columns' (list): Names of missing date columns
            - 'missing_user_columns' (list): Dicts with phase info for missing user columns
            - 'partial_phases' (list): Phases with date column but missing user column
            - 'available_phases' (list): Phase numbers (1-5) that can be tracked
            - 'unavailable_phases' (list): Phase numbers that cannot be tracked
            - 'has_amazon' (bool): Whether the optional Amazon column exists

    Example:
        >>> col_map = {"Kontrolle": 123, "K von": 456, "BE am": 789}
        >>> status = detect_missing_columns(col_map, "NA")
        >>> print(status['available_phases'])
        [1]  # Only phase 1 (Kontrolle/K von) is available
        >>> print(status['has_amazon'])
        False
    """
    result: Dict[str, Any] = {
        'missing_date_columns': [],
        'missing_user_columns': [],
        'partial_phases': [],
        'available_phases': [],
        'unavailable_phases': [],
        'has_amazon': 'Amazon' in col_map
    }

    for date_col, user_col_variations, phase_no in PHASE_FIELDS:
        # Check if date column exists
        date_col_exists = date_col in col_map

        # Check if any user column variation exists
        user_col_name, user_col_id = resolve_column_name(col_map, user_col_variations)
        user_col_exists = user_col_id is not None

        if date_col_exists and user_col_exists:
            # Phase is fully available
            result['available_phases'].append(phase_no)
        elif date_col_exists and not user_col_exists:
            # Partial phase - date exists but user column missing
            result['partial_phases'].append({
                'phase': phase_no,
                'date_col': date_col,
                'expected_user_cols': user_col_variations
            })
            result['missing_user_columns'].append({
                'phase': phase_no,
                'expected': user_col_variations
            })
            result['unavailable_phases'].append(phase_no)
        elif not date_col_exists:
            # Date column missing entirely
            result['missing_date_columns'].append(date_col)
            result['unavailable_phases'].append(phase_no)

    if log_warnings:
        _log_column_warnings(result, group_name)

    return result


def _log_column_warnings(result: Dict[str, Any], group_name: str) -> None:
    """
    Log warnings for missing columns.

    Internal helper function to handle logging of column availability issues.

    Args:
        result: The result dictionary from detect_missing_columns
        group_name: Name of the sheet group for logging purposes
    """
    # Log warnings for missing columns
    if result['missing_date_columns']:
        logger.warning(
            f"Sheet {group_name}: Missing date columns: {result['missing_date_columns']} - "
            f"Affected phases will be skipped"
        )

    if result['partial_phases']:
        for partial in result['partial_phases']:
            logger.warning(
                f"Sheet {group_name}: Phase {partial['phase']} has date column '{partial['date_col']}' "
                f"but missing user column (expected one of: {partial['expected_user_cols']}) - Phase will be skipped"
            )

    # Log summary
    total_phases = len(PHASE_FIELDS)
    available_count = len(result['available_phases'])

    if available_count < total_phases:
        logger.warning(
            f"Sheet {group_name}: Only {available_count}/{total_phases} phases available for tracking. "
            f"Available: {result['available_phases']}, Unavailable: {result['unavailable_phases']}"
        )
    else:
        logger.info(f"Sheet {group_name}: All {total_phases} phases available for tracking")

    if not result['has_amazon']:
        logger.info(
            f"Sheet {group_name}: Optional 'Amazon' marketplace column not found - "
            f"marketplace tracking disabled for this sheet"
        )


def get_available_phases(col_map: Dict[str, int]) -> List[int]:
    """
    Get list of phase numbers that are available for tracking.

    A phase is available if both its date column and user column exist
    in the sheet's column map.

    Args:
        col_map: Dictionary mapping column titles to column IDs.

    Returns:
        List of phase numbers (1-5) that can be tracked.

    Example:
        >>> col_map = {"Kontrolle": 123, "K von": 456, "BE am": 789, "BE von": 101}
        >>> get_available_phases(col_map)
        [1, 2]
    """
    result = detect_missing_columns(col_map, "", log_warnings=False)
    return result['available_phases']


def get_unavailable_phases(col_map: Dict[str, int]) -> List[int]:
    """
    Get list of phase numbers that are not available for tracking.

    A phase is unavailable if either its date column or user column
    is missing from the sheet's column map.

    Args:
        col_map: Dictionary mapping column titles to column IDs.

    Returns:
        List of phase numbers (1-5) that cannot be tracked.

    Example:
        >>> col_map = {"Kontrolle": 123, "K von": 456}
        >>> get_unavailable_phases(col_map)
        [2, 3, 4, 5]
    """
    result = detect_missing_columns(col_map, "", log_warnings=False)
    return result['unavailable_phases']


def get_phase_date_columns() -> List[str]:
    """
    Get list of all phase date column names.

    Returns:
        List of date column names in phase order.

    Example:
        >>> get_phase_date_columns()
        ['Kontrolle', 'BE am', 'K am', 'C am', 'Reopen C2 am']
    """
    return [date_col for date_col, _, _ in PHASE_FIELDS]


def get_phase_by_number(phase_number: int) -> Optional[Tuple[str, Tuple[str, ...], int]]:
    """
    Get phase field definition by phase number.

    Args:
        phase_number: The phase number (1-5)

    Returns:
        Tuple of (date_column, user_column_variations, phase_number)
        or None if phase number is invalid.

    Example:
        >>> get_phase_by_number(1)
        ('Kontrolle', ('K von',), 1)
        >>> get_phase_by_number(99)
        None
    """
    for phase_def in PHASE_FIELDS:
        if phase_def[2] == phase_number:
            return phase_def
    return None
