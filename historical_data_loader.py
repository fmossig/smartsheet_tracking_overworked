"""
Historical Data Loader Utility Module

Provides utility functions to load and parse historical change data from change_history.csv.
Supports filtering by date ranges and grouping by various dimensions (group, user, phase, marketplace).

Enhanced with DateRangeFilter support for custom date range filtering across all calculations
and visualizations.

Usage:
    from historical_data_loader import (
        load_change_history,
        load_with_date_range_filter,
        filter_by_date_range,
        filter_by_date_range_filter,
        filter_by_group,
        filter_by_user,
        filter_by_phase,
        filter_by_marketplace,
        group_by_dimension,
        get_change_summary,
        get_change_summary_for_range,
        get_user_activity,
        get_marketplace_distribution,
    )
"""

import os
import csv
import logging
from datetime import datetime, timedelta, date
from collections import defaultdict, Counter
from typing import List, Dict, Optional, Any, Union, Callable

from validation import (
    validate_csv_row,
    ValidationStats,
    create_validation_stats,
    is_empty_value,
    sanitize_value,
)
from date_utilities import (
    parse_date as parse_date_string,
    parse_timestamp,
)

# Import DateRangeFilter for custom date range support
# Use lazy import to avoid circular dependencies
_date_range_filter_module = None


def _get_date_range_filter_module():
    """Lazy load the date_range_filter module to avoid circular imports."""
    global _date_range_filter_module
    if _date_range_filter_module is None:
        from . import date_range_filter as drf
        _date_range_filter_module = drf
    return _date_range_filter_module

# Set up logging
logger = logging.getLogger(__name__)


# Directory containing tracking data
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "tracking_data")
CHANGES_FILE = os.path.join(DATA_DIR, "change_history.csv")

# CSV column names
COLUMNS = ["Timestamp", "Group", "RowID", "Phase", "DateField", "Date", "User", "Marketplace"]

# Valid dimensions for grouping
VALID_DIMENSIONS = {"group", "user", "phase", "marketplace", "date_field", "date"}

# Note: parse_timestamp and parse_date_string are now imported from date_utilities module

def load_change_history(
    start_date: Optional[Union[date, str]] = None,
    end_date: Optional[Union[date, str]] = None,
    file_path: Optional[str] = None,
    validate: bool = True,
    log_validation_stats: bool = True
) -> List[Dict[str, Any]]:
    """Load changes from the CSV file with optional date range filtering.

    Args:
        start_date: Start date for filtering (inclusive). Can be date object or "YYYY-MM-DD" string.
        end_date: End date for filtering (inclusive). Can be date object or "YYYY-MM-DD" string.
        file_path: Optional custom path to the CSV file. Defaults to tracking_data/change_history.csv.
        validate: Whether to validate rows and skip invalid ones. Default True.
        log_validation_stats: Whether to log validation statistics summary. Default True.

    Returns:
        List of change records as dictionaries with parsed dates.
        Each record contains:
            - Timestamp: Original timestamp string
            - ParsedTimestamp: datetime object
            - Group: Group code (NA, NF, NH, etc.)
            - RowID: Smartsheet row ID
            - Phase: Phase number (1-5)
            - DateField: Name of the date field that changed
            - Date: Original date string
            - ParsedDate: date object
            - User: User initials
            - Marketplace: Marketplace code

    Example:
        # Load all changes
        changes = load_change_history()

        # Load changes for a specific date range
        changes = load_change_history(
            start_date=date(2026, 1, 1),
            end_date=date(2026, 1, 31)
        )

        # Load with string dates
        changes = load_change_history(
            start_date="2026-01-01",
            end_date="2026-01-31"
        )

        # Load without validation (faster but may include invalid rows)
        changes = load_change_history(validate=False)
    """
    csv_path = file_path or CHANGES_FILE

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Changes file not found: {csv_path}")

    # Convert string dates to date objects if needed
    if isinstance(start_date, str):
        start_date = parse_date_string(start_date)
    if isinstance(end_date, str):
        end_date = parse_date_string(end_date)

    changes = []
    validation_stats = create_validation_stats() if validate else None

    # Track row number for error logging
    row_number = 0
    processing_errors = 0

    with open(csv_path, 'r', newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)

        for row in reader:
            row_number += 1
            # Row-level error isolation: wrap entire row processing in try-except
            try:
                # Get row identifiers for error logging
                row_group = row.get('Group', '')
                row_id = row.get('RowID', '')
                row_identifier = f"{row_group}:{row_id}" if row_group and row_id else f"line:{row_number}"

                # Validate the row if validation is enabled
                if validate:
                    try:
                        validation_result = validate_csv_row(row, log_issues=False)
                        validation_stats.record_result(validation_result)

                        if validation_result.skipped:
                            # Log at debug level to avoid spamming logs
                            logger.debug(
                                f"Skipping CSV row {row_identifier}: "
                                f"{validation_result.skip_reason}"
                            )
                            continue
                    except Exception as validation_error:
                        logger.warning(
                            f"CSV row {row_identifier}: Validation error - "
                            f"{type(validation_error).__name__}: {validation_error}. "
                            f"Skipping row and continuing."
                        )
                        if validation_stats:
                            validation_stats.record_processing_error(
                                row_id=row_id,
                                group=row_group,
                                error=validation_error,
                                error_category="csv_validation",
                                context={"row_number": row_number}
                            )
                        continue

                # Parse the timestamp - with error isolation
                try:
                    parsed_ts = parse_timestamp(row.get('Timestamp', ''))
                    if not parsed_ts:
                        if validate and validation_stats:
                            validation_stats.skipped_rows += 1
                            validation_stats.skipped_by_reason["invalid_timestamp"] = \
                                validation_stats.skipped_by_reason.get("invalid_timestamp", 0) + 1
                        logger.debug(f"Skipping row with invalid timestamp: {row.get('Timestamp', '')}")
                        continue  # Skip rows with invalid timestamps
                except Exception as ts_error:
                    logger.warning(
                        f"CSV row {row_identifier}: Timestamp parsing error - "
                        f"{type(ts_error).__name__}: {ts_error}. Skipping row."
                    )
                    if validate and validation_stats:
                        validation_stats.record_processing_error(
                            row_id=row_id,
                            group=row_group,
                            error=ts_error,
                            error_category="timestamp_parsing",
                            context={"row_number": row_number, "timestamp": row.get('Timestamp', '')}
                        )
                    continue

                ts_date = parsed_ts.date()

                # Apply date filter if specified
                if start_date and ts_date < start_date:
                    continue
                if end_date and ts_date > end_date:
                    continue

                # Parse and add the date field - with error isolation
                try:
                    parsed_date = parse_date_string(row.get('Date', ''))
                except Exception as date_error:
                    logger.warning(
                        f"CSV row {row_identifier}: Date parsing error - "
                        f"{type(date_error).__name__}: {date_error}. Using None for parsed date."
                    )
                    parsed_date = None

                # Sanitize optional fields to handle null-like values - with error isolation
                try:
                    user_value = sanitize_value(row.get('User', ''), 'User')
                    marketplace_value = sanitize_value(row.get('Marketplace', ''), 'Marketplace')
                except Exception as sanitize_error:
                    logger.warning(
                        f"CSV row {row_identifier}: Value sanitization error - "
                        f"{type(sanitize_error).__name__}: {sanitize_error}. Using raw values."
                    )
                    user_value = row.get('User', '')
                    marketplace_value = row.get('Marketplace', '')

                # Enhance the row with parsed values
                enhanced_row = {
                    'Timestamp': row.get('Timestamp', ''),
                    'ParsedTimestamp': parsed_ts,
                    'Group': row_group,
                    'RowID': row_id,
                    'Phase': row.get('Phase', ''),
                    'DateField': row.get('DateField', ''),
                    'Date': row.get('Date', ''),
                    'ParsedDate': parsed_date,
                    'User': user_value,
                    'Marketplace': marketplace_value,
                }

                changes.append(enhanced_row)

            except Exception as row_error:
                # Catch any unexpected errors at the row level
                processing_errors += 1
                logger.error(
                    f"CSV row {row_number}: Unexpected error during processing - "
                    f"{type(row_error).__name__}: {row_error}. "
                    f"Skipping this row and continuing with remaining rows."
                )
                if validate and validation_stats:
                    validation_stats.record_processing_error(
                        row_id=row.get('RowID', ''),
                        group=row.get('Group', ''),
                        error=row_error,
                        error_category="row_processing",
                        context={"row_number": row_number}
                    )
                continue

    # Log validation statistics if enabled
    if validate and log_validation_stats and validation_stats and validation_stats.total_rows > 0:
        logger.info(
            f"CSV Loading: Processed {validation_stats.total_rows} rows, "
            f"{validation_stats.valid_rows} valid, "
            f"{validation_stats.skipped_rows} skipped"
        )
        if validation_stats.skipped_by_reason:
            for reason, count in validation_stats.skipped_by_reason.items():
                logger.debug(f"  Skip reason '{reason}': {count} rows")

    # Log processing errors summary if any occurred
    if processing_errors > 0:
        logger.warning(
            f"CSV Loading: {processing_errors} rows encountered processing errors and were skipped. "
            f"Check logs for details."
        )

    return changes


def filter_by_date_range(
    changes: List[Dict[str, Any]],
    start_date: Optional[Union[date, str]] = None,
    end_date: Optional[Union[date, str]] = None
) -> List[Dict[str, Any]]:
    """Filter changes by timestamp date range.

    Args:
        changes: List of change records
        start_date: Start date for filtering (inclusive)
        end_date: End date for filtering (inclusive)

    Returns:
        Filtered list of changes
    """
    if start_date is None and end_date is None:
        return changes

    # Convert string dates if needed
    if isinstance(start_date, str):
        start_date = parse_date_string(start_date)
    if isinstance(end_date, str):
        end_date = parse_date_string(end_date)

    filtered = []
    for change in changes:
        ts = change.get('ParsedTimestamp')
        if not ts:
            continue
        ts_date = ts.date() if isinstance(ts, datetime) else ts

        if start_date and ts_date < start_date:
            continue
        if end_date and ts_date > end_date:
            continue

        filtered.append(change)

    return filtered


def filter_by_date_range_filter(
    changes: List[Dict[str, Any]],
    date_range_filter: Any  # DateRangeFilter type
) -> List[Dict[str, Any]]:
    """Filter changes using a DateRangeFilter object.

    This function provides integration with the DateRangeFilter class for
    consistent date range filtering across the application.

    Args:
        changes: List of change records
        date_range_filter: DateRangeFilter object specifying the range

    Returns:
        Filtered list of changes within the specified range
    """
    filtered = []
    for change in changes:
        ts = change.get('ParsedTimestamp')
        if not ts:
            continue
        ts_date = ts.date() if isinstance(ts, datetime) else ts

        if date_range_filter.contains_date(ts_date):
            filtered.append(change)

    return filtered


def load_with_date_range_filter(
    date_range_filter: Any,  # DateRangeFilter type
    file_path: Optional[str] = None,
    validate: bool = True,
    log_validation_stats: bool = True
) -> List[Dict[str, Any]]:
    """Load changes from CSV file using a DateRangeFilter.

    This is a convenience function that combines load_change_history with
    DateRangeFilter-based filtering.

    Args:
        date_range_filter: DateRangeFilter object specifying the date range
        file_path: Optional custom path to the CSV file
        validate: Whether to validate rows
        log_validation_stats: Whether to log validation statistics

    Returns:
        List of change records within the specified date range
    """
    return load_change_history(
        start_date=date_range_filter.start_date,
        end_date=date_range_filter.end_date,
        file_path=file_path,
        validate=validate,
        log_validation_stats=log_validation_stats
    )


def filter_by_group(changes: List[Dict[str, Any]], groups: Union[str, List[str]]) -> List[Dict[str, Any]]:
    """Filter changes by group code(s).

    Args:
        changes: List of change records
        groups: Single group code or list of group codes (e.g., "NF" or ["NF", "NA"])

    Returns:
        Filtered list of changes
    """
    if isinstance(groups, str):
        groups = [groups]
    groups_set = set(groups)
    return [c for c in changes if c.get('Group') in groups_set]


def filter_by_user(changes: List[Dict[str, Any]], users: Union[str, List[str]]) -> List[Dict[str, Any]]:
    """Filter changes by user initials.

    Args:
        changes: List of change records
        users: Single user or list of users (e.g., "DM" or ["DM", "JHU"])

    Returns:
        Filtered list of changes
    """
    if isinstance(users, str):
        users = [users]
    users_set = set(users)
    return [c for c in changes if c.get('User') in users_set]


def filter_by_phase(changes: List[Dict[str, Any]], phases: Union[int, str, List[Union[int, str]]]) -> List[Dict[str, Any]]:
    """Filter changes by phase number(s).

    Args:
        changes: List of change records
        phases: Single phase or list of phases (e.g., 1 or [1, 2] or "1" or ["1", "2"])

    Returns:
        Filtered list of changes
    """
    if isinstance(phases, (int, str)):
        phases = [phases]
    phases_set = set(str(p) for p in phases)
    return [c for c in changes if str(c.get('Phase')) in phases_set]


def filter_by_marketplace(changes: List[Dict[str, Any]], marketplaces: Union[str, List[str]]) -> List[Dict[str, Any]]:
    """Filter changes by marketplace code(s).

    Args:
        changes: List of change records
        marketplaces: Single marketplace or list (e.g., "com" or ["com", "de", "jp"])

    Returns:
        Filtered list of changes
    """
    if isinstance(marketplaces, str):
        marketplaces = [marketplaces]
    marketplaces_set = set(marketplaces)
    return [c for c in changes if c.get('Marketplace') in marketplaces_set]


def filter_changes(
    changes: List[Dict[str, Any]],
    groups: Optional[Union[str, List[str]]] = None,
    users: Optional[Union[str, List[str]]] = None,
    phases: Optional[Union[int, str, List[Union[int, str]]]] = None,
    marketplaces: Optional[Union[str, List[str]]] = None,
    start_date: Optional[Union[date, str]] = None,
    end_date: Optional[Union[date, str]] = None
) -> List[Dict[str, Any]]:
    """Apply multiple filters to changes in a single call.

    Args:
        changes: List of change records
        groups: Group code(s) to filter by
        users: User(s) to filter by
        phases: Phase number(s) to filter by
        marketplaces: Marketplace code(s) to filter by
        start_date: Start date for filtering
        end_date: End date for filtering

    Returns:
        Filtered list of changes matching all specified criteria

    Example:
        # Get all Phase 1 changes from user DM in the NF group
        filtered = filter_changes(
            changes,
            groups="NF",
            users="DM",
            phases=1
        )
    """
    result = changes

    if start_date or end_date:
        result = filter_by_date_range(result, start_date, end_date)
    if groups:
        result = filter_by_group(result, groups)
    if users:
        result = filter_by_user(result, users)
    if phases:
        result = filter_by_phase(result, phases)
    if marketplaces:
        result = filter_by_marketplace(result, marketplaces)

    return result


def group_by_dimension(
    changes: List[Dict[str, Any]],
    dimension: str
) -> Dict[str, List[Dict[str, Any]]]:
    """Group changes by a specified dimension.

    Args:
        changes: List of change records
        dimension: Dimension to group by. Valid values:
            - "group": Group by product group (NA, NF, etc.)
            - "user": Group by user initials
            - "phase": Group by phase number
            - "marketplace": Group by marketplace code
            - "date_field": Group by date field name
            - "date": Group by change date (YYYY-MM-DD)

    Returns:
        Dictionary mapping dimension values to lists of changes

    Example:
        # Group by user
        by_user = group_by_dimension(changes, "user")
        for user, user_changes in by_user.items():
            print(f"{user}: {len(user_changes)} changes")
    """
    dimension_map = {
        "group": "Group",
        "user": "User",
        "phase": "Phase",
        "marketplace": "Marketplace",
        "date_field": "DateField",
        "date": "Date",
    }

    if dimension.lower() not in dimension_map:
        raise ValueError(f"Invalid dimension: {dimension}. Valid dimensions: {list(dimension_map.keys())}")

    field = dimension_map[dimension.lower()]
    grouped = defaultdict(list)

    for change in changes:
        key = change.get(field, '')
        if key:  # Only include non-empty keys
            grouped[key].append(change)

    return dict(grouped)


def group_by_multiple_dimensions(
    changes: List[Dict[str, Any]],
    dimensions: List[str]
) -> Dict[tuple, List[Dict[str, Any]]]:
    """Group changes by multiple dimensions.

    Args:
        changes: List of change records
        dimensions: List of dimensions to group by (e.g., ["group", "phase"])

    Returns:
        Dictionary mapping tuples of dimension values to lists of changes

    Example:
        # Group by group and phase
        by_group_phase = group_by_multiple_dimensions(changes, ["group", "phase"])
        for (group, phase), group_changes in by_group_phase.items():
            print(f"{group} Phase {phase}: {len(group_changes)} changes")
    """
    dimension_map = {
        "group": "Group",
        "user": "User",
        "phase": "Phase",
        "marketplace": "Marketplace",
        "date_field": "DateField",
        "date": "Date",
    }

    for dim in dimensions:
        if dim.lower() not in dimension_map:
            raise ValueError(f"Invalid dimension: {dim}. Valid dimensions: {list(dimension_map.keys())}")

    fields = [dimension_map[dim.lower()] for dim in dimensions]
    grouped = defaultdict(list)

    for change in changes:
        key = tuple(change.get(field, '') for field in fields)
        if all(key):  # Only include if all keys are non-empty
            grouped[key].append(change)

    return dict(grouped)


def aggregate_by_dimension(
    changes: List[Dict[str, Any]],
    dimension: str
) -> Dict[str, int]:
    """Count changes by a specified dimension.

    Args:
        changes: List of change records
        dimension: Dimension to aggregate by

    Returns:
        Dictionary mapping dimension values to counts

    Example:
        # Count changes by user
        user_counts = aggregate_by_dimension(changes, "user")
        # Returns: {"DM": 45, "JHU": 67, "HI": 38}
    """
    grouped = group_by_dimension(changes, dimension)
    return {key: len(values) for key, values in grouped.items()}


def get_change_summary(changes: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Get a comprehensive summary of changes.

    Args:
        changes: List of change records

    Returns:
        Dictionary containing:
            - total_changes: Total number of changes
            - by_group: Counts by group
            - by_phase: Counts by phase
            - by_user: Counts by user
            - by_marketplace: Counts by marketplace
            - date_range: Tuple of (earliest_date, latest_date)
            - unique_rows: Number of unique row IDs

    Example:
        summary = get_change_summary(changes)
        print(f"Total changes: {summary['total_changes']}")
        print(f"By group: {summary['by_group']}")
    """
    if not changes:
        return {
            "total_changes": 0,
            "by_group": {},
            "by_phase": {},
            "by_user": {},
            "by_marketplace": {},
            "date_range": (None, None),
            "unique_rows": 0,
        }

    # Get date range
    dates = [c['ParsedTimestamp'].date() for c in changes if c.get('ParsedTimestamp')]
    date_range = (min(dates), max(dates)) if dates else (None, None)

    # Get unique rows
    unique_rows = len(set(c.get('RowID', '') for c in changes if c.get('RowID')))

    return {
        "total_changes": len(changes),
        "by_group": aggregate_by_dimension(changes, "group"),
        "by_phase": aggregate_by_dimension(changes, "phase"),
        "by_user": aggregate_by_dimension(changes, "user"),
        "by_marketplace": aggregate_by_dimension(changes, "marketplace"),
        "date_range": date_range,
        "unique_rows": unique_rows,
    }


def get_change_summary_for_range(
    date_range_filter: Any,  # DateRangeFilter type
    file_path: Optional[str] = None
) -> Dict[str, Any]:
    """Get a comprehensive summary of changes for a specific date range.

    This function combines loading, filtering, and summarizing into a single
    convenient call, ensuring all calculations respect the custom date range.

    Args:
        date_range_filter: DateRangeFilter object specifying the date range
        file_path: Optional custom path to the CSV file

    Returns:
        Dictionary containing:
            - total_changes: Total number of changes
            - by_group: Counts by group
            - by_phase: Counts by phase
            - by_user: Counts by user
            - by_marketplace: Counts by marketplace
            - date_range: The DateRangeFilter's date range
            - actual_date_range: Tuple of (earliest_date, latest_date) from data
            - unique_rows: Number of unique row IDs
            - daily_average: Average changes per day
            - filter_info: Information about the filter applied

    Example:
        from date_range_filter import create_date_range
        range_filter = create_date_range("2026-01-01", "2026-01-15")
        summary = get_change_summary_for_range(range_filter)
        print(f"Total changes: {summary['total_changes']}")
    """
    # Load changes within the date range
    changes = load_with_date_range_filter(date_range_filter, file_path=file_path)

    # Get base summary
    summary = get_change_summary(changes)

    # Enhance with date range filter information
    days_in_range = date_range_filter.days_in_range
    daily_average = summary["total_changes"] / days_in_range if days_in_range > 0 else 0

    summary["daily_average"] = round(daily_average, 2)
    summary["filter_info"] = {
        "start_date": date_range_filter.start_date.isoformat(),
        "end_date": date_range_filter.end_date.isoformat(),
        "label": date_range_filter.label,
        "preset": date_range_filter.preset.value if hasattr(date_range_filter.preset, 'value') else str(date_range_filter.preset),
        "days_in_range": days_in_range,
    }
    summary["actual_date_range"] = summary.pop("date_range")
    summary["date_range"] = (date_range_filter.start_date, date_range_filter.end_date)

    return summary


def get_user_activity(
    changes: List[Dict[str, Any]],
    user: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """Get activity breakdown by user.

    Args:
        changes: List of change records
        user: Optional specific user to get activity for

    Returns:
        Dictionary mapping users to their activity details:
            - total_changes: Number of changes by this user
            - by_group: Breakdown by group
            - by_phase: Breakdown by phase
            - by_marketplace: Breakdown by marketplace

    Example:
        activity = get_user_activity(changes)
        for user, stats in activity.items():
            print(f"{user}: {stats['total_changes']} changes")
    """
    if user:
        changes = filter_by_user(changes, user)

    by_user = group_by_dimension(changes, "user")

    result = {}
    for u, user_changes in by_user.items():
        result[u] = {
            "total_changes": len(user_changes),
            "by_group": aggregate_by_dimension(user_changes, "group"),
            "by_phase": aggregate_by_dimension(user_changes, "phase"),
            "by_marketplace": aggregate_by_dimension(user_changes, "marketplace"),
        }

    return result


def get_marketplace_distribution(
    changes: List[Dict[str, Any]],
    include_percentages: bool = True
) -> Dict[str, Dict[str, Any]]:
    """Get change distribution across marketplaces.

    Args:
        changes: List of change records
        include_percentages: Whether to include percentage calculations

    Returns:
        Dictionary mapping marketplace to:
            - count: Number of changes
            - percentage: Percentage of total (if include_percentages=True)
            - by_group: Breakdown by group
            - by_user: Breakdown by user
    """
    by_marketplace = group_by_dimension(changes, "marketplace")
    total = len(changes)

    result = {}
    for mp, mp_changes in by_marketplace.items():
        count = len(mp_changes)
        entry = {
            "count": count,
            "by_group": aggregate_by_dimension(mp_changes, "group"),
            "by_user": aggregate_by_dimension(mp_changes, "user"),
        }
        if include_percentages and total > 0:
            entry["percentage"] = round(count / total * 100, 2)
        result[mp] = entry

    return result


def get_phase_progression(
    changes: List[Dict[str, Any]],
    group: Optional[str] = None
) -> Dict[str, Dict[str, int]]:
    """Analyze phase progression (counts by phase for each group).

    Args:
        changes: List of change records
        group: Optional specific group to analyze

    Returns:
        Dictionary mapping groups to their phase counts

    Example:
        progression = get_phase_progression(changes)
        for group, phases in progression.items():
            print(f"{group}: Phase 1={phases.get('1', 0)}, Phase 2={phases.get('2', 0)}")
    """
    if group:
        changes = filter_by_group(changes, group)

    by_group = group_by_dimension(changes, "group")

    result = {}
    for g, group_changes in by_group.items():
        result[g] = aggregate_by_dimension(group_changes, "phase")

    return result


def get_daily_change_counts(
    changes: List[Dict[str, Any]],
    start_date: Optional[Union[date, str]] = None,
    end_date: Optional[Union[date, str]] = None
) -> Dict[str, int]:
    """Get change counts by date.

    Args:
        changes: List of change records
        start_date: Optional start date to include
        end_date: Optional end date to include

    Returns:
        Dictionary mapping date strings (YYYY-MM-DD) to counts
    """
    filtered = filter_by_date_range(changes, start_date, end_date)

    counts = defaultdict(int)
    for change in filtered:
        ts = change.get('ParsedTimestamp')
        if ts:
            date_str = ts.date().isoformat()
            counts[date_str] += 1

    # Sort by date
    return dict(sorted(counts.items()))


def get_unique_values(changes: List[Dict[str, Any]], dimension: str) -> List[str]:
    """Get unique values for a dimension.

    Args:
        changes: List of change records
        dimension: Dimension to get unique values for

    Returns:
        Sorted list of unique values
    """
    dimension_map = {
        "group": "Group",
        "user": "User",
        "phase": "Phase",
        "marketplace": "Marketplace",
        "date_field": "DateField",
    }

    if dimension.lower() not in dimension_map:
        raise ValueError(f"Invalid dimension: {dimension}")

    field = dimension_map[dimension.lower()]
    values = set(c.get(field, '') for c in changes if c.get(field))
    return sorted(values)


# Convenience functions for common use cases

def load_today() -> List[Dict[str, Any]]:
    """Load changes from today."""
    today = date.today()
    return load_change_history(start_date=today, end_date=today)


def load_last_n_days(days: int = 7) -> List[Dict[str, Any]]:
    """Load changes from the last N days.

    Args:
        days: Number of days to look back (default 7)
    """
    end = date.today()
    start = end - timedelta(days=days - 1)
    return load_change_history(start_date=start, end_date=end)


def load_this_week() -> List[Dict[str, Any]]:
    """Load changes from the current week (Monday to today)."""
    today = date.today()
    start_of_week = today - timedelta(days=today.weekday())
    return load_change_history(start_date=start_of_week, end_date=today)


def load_this_month() -> List[Dict[str, Any]]:
    """Load changes from the current month."""
    today = date.today()
    start_of_month = today.replace(day=1)
    return load_change_history(start_date=start_of_month, end_date=today)


if __name__ == "__main__":
    # Demo usage
    print("Historical Data Loader - Demo")
    print("=" * 50)

    try:
        # Load all changes
        changes = load_change_history()
        print(f"\nLoaded {len(changes)} total changes")

        if changes:
            # Get summary
            summary = get_change_summary(changes)
            print(f"\nSummary:")
            print(f"  Total changes: {summary['total_changes']}")
            print(f"  Date range: {summary['date_range'][0]} to {summary['date_range'][1]}")
            print(f"  Unique rows affected: {summary['unique_rows']}")

            print(f"\n  By Group:")
            for group, count in sorted(summary['by_group'].items()):
                print(f"    {group}: {count}")

            print(f"\n  By User:")
            for user, count in sorted(summary['by_user'].items(), key=lambda x: -x[1]):
                print(f"    {user}: {count}")

            print(f"\n  By Phase:")
            for phase, count in sorted(summary['by_phase'].items()):
                print(f"    Phase {phase}: {count}")

            # Example filtering
            print(f"\n\nFiltering Examples:")

            # Filter by user
            dm_changes = filter_by_user(changes, "DM")
            print(f"  DM's changes: {len(dm_changes)}")

            # Filter by group
            nf_changes = filter_by_group(changes, "NF")
            print(f"  NF group changes: {len(nf_changes)}")

            # Filter by phase
            phase1_changes = filter_by_phase(changes, 1)
            print(f"  Phase 1 changes: {len(phase1_changes)}")

            # Combined filter
            combined = filter_changes(changes, groups="NF", phases=1)
            print(f"  NF + Phase 1: {len(combined)}")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
