"""
Date Range Filter Module

Provides a unified interface for custom date range filtering across the smartsheet-tracker
application. This module centralizes date range validation, predefined range generation,
and filtering logic to ensure all calculations and visualizations respect custom ranges.

Usage:
    from date_range_filter import (
        DateRangeFilter,
        DateRangePreset,
        create_date_range,
        get_preset_range,
        validate_date_range,
        filter_data_by_range,
    )
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Any, Union, Tuple, Callable
from enum import Enum
import calendar

# Set up logging
logger = logging.getLogger(__name__)


class DateRangePreset(Enum):
    """Predefined date range presets for quick selection."""
    TODAY = "today"
    YESTERDAY = "yesterday"
    LAST_7_DAYS = "last_7_days"
    LAST_14_DAYS = "last_14_days"
    LAST_30_DAYS = "last_30_days"
    LAST_90_DAYS = "last_90_days"
    THIS_WEEK = "this_week"
    LAST_WEEK = "last_week"
    THIS_MONTH = "this_month"
    LAST_MONTH = "last_month"
    THIS_QUARTER = "this_quarter"
    LAST_QUARTER = "last_quarter"
    THIS_YEAR = "this_year"
    LAST_YEAR = "last_year"
    CUSTOM = "custom"


@dataclass
class DateRangeFilter:
    """A filter object for custom date range operations.

    This class encapsulates date range filtering logic and provides a unified
    interface for filtering change history data, calculating metrics, and
    generating visualizations within a specific date range.

    Attributes:
        start_date: Start date of the range (inclusive)
        end_date: End date of the range (inclusive)
        preset: The preset used to create this range (CUSTOM if manually specified)
        include_boundaries: Whether to include the boundary dates (default True)
        label: Human-readable label for the date range
    """
    start_date: date
    end_date: date
    preset: DateRangePreset = DateRangePreset.CUSTOM
    include_boundaries: bool = True
    label: Optional[str] = None

    def __post_init__(self):
        """Validate and normalize the date range after initialization."""
        # Convert string dates if provided
        if isinstance(self.start_date, str):
            self.start_date = parse_date_input(self.start_date)
        if isinstance(self.end_date, str):
            self.end_date = parse_date_input(self.end_date)

        # Ensure dates are date objects, not datetime
        if isinstance(self.start_date, datetime):
            self.start_date = self.start_date.date()
        if isinstance(self.end_date, datetime):
            self.end_date = self.end_date.date()

        # Validate the range
        if self.start_date > self.end_date:
            raise ValueError(
                f"Start date ({self.start_date}) must be before or equal to "
                f"end date ({self.end_date})"
            )

        # Generate label if not provided
        if self.label is None:
            self.label = self._generate_label()

    def _generate_label(self) -> str:
        """Generate a human-readable label for the date range."""
        if self.preset != DateRangePreset.CUSTOM:
            preset_labels = {
                DateRangePreset.TODAY: "Today",
                DateRangePreset.YESTERDAY: "Yesterday",
                DateRangePreset.LAST_7_DAYS: "Last 7 Days",
                DateRangePreset.LAST_14_DAYS: "Last 14 Days",
                DateRangePreset.LAST_30_DAYS: "Last 30 Days",
                DateRangePreset.LAST_90_DAYS: "Last 90 Days",
                DateRangePreset.THIS_WEEK: "This Week",
                DateRangePreset.LAST_WEEK: "Last Week",
                DateRangePreset.THIS_MONTH: "This Month",
                DateRangePreset.LAST_MONTH: "Last Month",
                DateRangePreset.THIS_QUARTER: "This Quarter",
                DateRangePreset.LAST_QUARTER: "Last Quarter",
                DateRangePreset.THIS_YEAR: "This Year",
                DateRangePreset.LAST_YEAR: "Last Year",
            }
            return preset_labels.get(self.preset, "Custom Range")

        # Generate custom label based on date range
        if self.start_date == self.end_date:
            return self.start_date.strftime("%B %d, %Y")
        elif self.start_date.year == self.end_date.year:
            if self.start_date.month == self.end_date.month:
                return f"{self.start_date.strftime('%B')} {self.start_date.day}-{self.end_date.day}, {self.start_date.year}"
            else:
                return f"{self.start_date.strftime('%B %d')} - {self.end_date.strftime('%B %d')}, {self.start_date.year}"
        else:
            return f"{self.start_date.strftime('%B %d, %Y')} - {self.end_date.strftime('%B %d, %Y')}"

    @property
    def days_in_range(self) -> int:
        """Get the number of days in the range (inclusive)."""
        return (self.end_date - self.start_date).days + 1

    @property
    def is_single_day(self) -> bool:
        """Check if the range covers a single day."""
        return self.start_date == self.end_date

    @property
    def weeks_in_range(self) -> float:
        """Get the approximate number of weeks in the range."""
        return self.days_in_range / 7.0

    @property
    def months_in_range(self) -> float:
        """Get the approximate number of months in the range."""
        return self.days_in_range / 30.0

    @property
    def filename_label(self) -> str:
        """Generate a filename-safe label for the date range.

        Returns a string suitable for use in filenames that clearly identifies
        the date range without special characters.

        Examples:
            - Preset "Last 7 Days": "last_7_days_2026-01-01_to_2026-01-07"
            - Custom single day: "2026-01-15"
            - Custom range same month: "2026-01_10-15"
            - Custom range different months: "2026-01-10_to_2026-02-15"

        Returns:
            A filename-safe string representation of the date range
        """
        # For preset ranges, include the preset name for clarity
        if self.preset != DateRangePreset.CUSTOM:
            preset_name = self.preset.value  # e.g., "last_7_days", "this_month"
            return f"{preset_name}_{self.start_date.isoformat()}_to_{self.end_date.isoformat()}"

        # For custom ranges, use a compact format
        if self.start_date == self.end_date:
            # Single day
            return self.start_date.isoformat()
        elif self.start_date.year == self.end_date.year:
            if self.start_date.month == self.end_date.month:
                # Same month: "2026-01_10-15"
                return f"{self.start_date.strftime('%Y-%m')}_{self.start_date.day:02d}-{self.end_date.day:02d}"
            else:
                # Different months: "2026-01-10_to_2026-02-15"
                return f"{self.start_date.isoformat()}_to_{self.end_date.isoformat()}"
        else:
            # Different years: "2025-12-01_to_2026-01-15"
            return f"{self.start_date.isoformat()}_to_{self.end_date.isoformat()}"

    @property
    def report_title(self) -> str:
        """Generate a title string for use in report headers.

        Returns a human-readable title that clearly identifies the date range
        type and period for use in report titles and headers.

        Examples:
            - Preset "Last 7 Days": "Last 7 Days Report"
            - Custom single day: "Daily Report - January 15, 2026"
            - Custom range: "Custom Period Report - January 10-15, 2026"

        Returns:
            A human-readable title string for the report
        """
        if self.preset != DateRangePreset.CUSTOM:
            # Use the preset label directly
            return f"{self.label} Report"

        # For custom ranges, indicate it's a custom period
        if self.is_single_day:
            return f"Daily Report - {self.label}"
        else:
            return f"Custom Period Report - {self.label}"

    @property
    def report_subtitle(self) -> str:
        """Generate a subtitle with detailed date range information.

        Returns a descriptive subtitle showing the exact date range
        and duration for use in report headers.

        Examples:
            - "January 10 - January 15, 2026 (6 days)"
            - "Last 7 Days: December 31, 2025 - January 6, 2026"

        Returns:
            A descriptive subtitle string with date range details
        """
        duration_str = f"{self.days_in_range} day{'s' if self.days_in_range != 1 else ''}"

        if self.preset != DateRangePreset.CUSTOM:
            # Show preset name with actual dates
            date_range = f"{self.start_date.strftime('%B %d, %Y')} - {self.end_date.strftime('%B %d, %Y')}"
            return f"{self.label}: {date_range} ({duration_str})"

        # For custom ranges, show the label with duration
        return f"{self.label} ({duration_str})"

    def contains_date(self, check_date: Union[date, datetime, str]) -> bool:
        """Check if a date falls within this range.

        Args:
            check_date: The date to check

        Returns:
            True if the date is within the range
        """
        if isinstance(check_date, str):
            check_date = parse_date_input(check_date)
        if isinstance(check_date, datetime):
            check_date = check_date.date()

        if self.include_boundaries:
            return self.start_date <= check_date <= self.end_date
        else:
            return self.start_date < check_date < self.end_date

    def overlaps_with(self, other: 'DateRangeFilter') -> bool:
        """Check if this range overlaps with another range.

        Args:
            other: Another DateRangeFilter to check against

        Returns:
            True if the ranges overlap
        """
        return self.start_date <= other.end_date and self.end_date >= other.start_date

    def get_previous_period(self) -> 'DateRangeFilter':
        """Get a DateRangeFilter for the previous period of the same duration.

        Returns:
            A new DateRangeFilter for the previous period
        """
        duration = self.days_in_range
        prev_end = self.start_date - timedelta(days=1)
        prev_start = prev_end - timedelta(days=duration - 1)

        return DateRangeFilter(
            start_date=prev_start,
            end_date=prev_end,
            preset=DateRangePreset.CUSTOM,
            label=f"Previous {self.label}" if self.label else None
        )

    def split_by_weeks(self) -> List['DateRangeFilter']:
        """Split the range into weekly segments.

        Returns:
            List of DateRangeFilter objects, one per week
        """
        ranges = []
        current_start = self.start_date

        while current_start <= self.end_date:
            # Find end of week (Sunday) or end of range
            days_to_sunday = 6 - current_start.weekday()
            week_end = current_start + timedelta(days=days_to_sunday)

            if week_end > self.end_date:
                week_end = self.end_date

            ranges.append(DateRangeFilter(
                start_date=current_start,
                end_date=week_end,
                preset=DateRangePreset.CUSTOM
            ))

            current_start = week_end + timedelta(days=1)

        return ranges

    def split_by_months(self) -> List['DateRangeFilter']:
        """Split the range into monthly segments.

        Returns:
            List of DateRangeFilter objects, one per month
        """
        ranges = []
        current_date = self.start_date

        while current_date <= self.end_date:
            # Find end of month or end of range
            _, days_in_month = calendar.monthrange(current_date.year, current_date.month)
            month_end = current_date.replace(day=days_in_month)

            if month_end > self.end_date:
                month_end = self.end_date

            ranges.append(DateRangeFilter(
                start_date=current_date,
                end_date=month_end,
                preset=DateRangePreset.CUSTOM
            ))

            # Move to first day of next month
            if month_end.month == 12:
                current_date = date(month_end.year + 1, 1, 1)
            else:
                current_date = date(month_end.year, month_end.month + 1, 1)

        return ranges

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "preset": self.preset.value,
            "label": self.label,
            "days_in_range": self.days_in_range,
            "include_boundaries": self.include_boundaries,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'DateRangeFilter':
        """Create a DateRangeFilter from a dictionary.

        Args:
            data: Dictionary with date range data

        Returns:
            New DateRangeFilter instance
        """
        preset = DateRangePreset(data.get("preset", "custom"))
        return cls(
            start_date=date.fromisoformat(data["start_date"]),
            end_date=date.fromisoformat(data["end_date"]),
            preset=preset,
            label=data.get("label"),
            include_boundaries=data.get("include_boundaries", True),
        )

    def __repr__(self) -> str:
        return f"DateRangeFilter({self.start_date} to {self.end_date}, label='{self.label}')"


def parse_date_input(date_input: Union[str, date, datetime]) -> date:
    """Parse various date input formats into a date object.

    Supports:
    - date object (returned as-is)
    - datetime object (converted to date)
    - String formats: YYYY-MM-DD, DD.MM.YYYY, MM/DD/YYYY, YYYY/MM/DD

    Args:
        date_input: The date input to parse

    Returns:
        A date object

    Raises:
        ValueError: If the input cannot be parsed
    """
    if isinstance(date_input, date) and not isinstance(date_input, datetime):
        return date_input
    if isinstance(date_input, datetime):
        return date_input.date()
    if not isinstance(date_input, str):
        raise ValueError(f"Cannot parse date from type: {type(date_input)}")

    cleaned = date_input.strip()

    # Try various formats
    formats = [
        '%Y-%m-%d',
        '%Y-%m-%dT%H:%M:%S',
        '%d.%m.%Y',
        '%m/%d/%Y',
        '%Y/%m/%d',
        '%d/%m/%Y',
    ]

    for fmt in formats:
        try:
            return datetime.strptime(cleaned, fmt).date()
        except ValueError:
            continue

    # Try ISO format as fallback
    try:
        return datetime.fromisoformat(cleaned).date()
    except Exception:
        pass

    raise ValueError(f"Cannot parse date from string: '{date_input}'")


def get_preset_range(
    preset: DateRangePreset,
    reference_date: Optional[date] = None
) -> DateRangeFilter:
    """Get a DateRangeFilter for a predefined preset.

    Args:
        preset: The preset to use
        reference_date: Reference date for calculations (defaults to today)

    Returns:
        DateRangeFilter for the specified preset

    Raises:
        ValueError: If preset is CUSTOM (use create_date_range instead)
    """
    if preset == DateRangePreset.CUSTOM:
        raise ValueError("Use create_date_range() for custom date ranges")

    if reference_date is None:
        reference_date = date.today()

    start_date: date
    end_date: date

    if preset == DateRangePreset.TODAY:
        start_date = end_date = reference_date

    elif preset == DateRangePreset.YESTERDAY:
        start_date = end_date = reference_date - timedelta(days=1)

    elif preset == DateRangePreset.LAST_7_DAYS:
        end_date = reference_date
        start_date = reference_date - timedelta(days=6)

    elif preset == DateRangePreset.LAST_14_DAYS:
        end_date = reference_date
        start_date = reference_date - timedelta(days=13)

    elif preset == DateRangePreset.LAST_30_DAYS:
        end_date = reference_date
        start_date = reference_date - timedelta(days=29)

    elif preset == DateRangePreset.LAST_90_DAYS:
        end_date = reference_date
        start_date = reference_date - timedelta(days=89)

    elif preset == DateRangePreset.THIS_WEEK:
        # Monday to today
        start_date = reference_date - timedelta(days=reference_date.weekday())
        end_date = reference_date

    elif preset == DateRangePreset.LAST_WEEK:
        # Previous Monday to Sunday
        this_monday = reference_date - timedelta(days=reference_date.weekday())
        end_date = this_monday - timedelta(days=1)
        start_date = end_date - timedelta(days=6)

    elif preset == DateRangePreset.THIS_MONTH:
        start_date = reference_date.replace(day=1)
        end_date = reference_date

    elif preset == DateRangePreset.LAST_MONTH:
        first_of_month = reference_date.replace(day=1)
        end_date = first_of_month - timedelta(days=1)
        start_date = end_date.replace(day=1)

    elif preset == DateRangePreset.THIS_QUARTER:
        quarter = (reference_date.month - 1) // 3
        start_date = date(reference_date.year, quarter * 3 + 1, 1)
        end_date = reference_date

    elif preset == DateRangePreset.LAST_QUARTER:
        quarter = (reference_date.month - 1) // 3
        if quarter == 0:
            start_date = date(reference_date.year - 1, 10, 1)
            end_date = date(reference_date.year - 1, 12, 31)
        else:
            start_date = date(reference_date.year, (quarter - 1) * 3 + 1, 1)
            end_month = quarter * 3
            _, end_day = calendar.monthrange(reference_date.year, end_month)
            end_date = date(reference_date.year, end_month, end_day)

    elif preset == DateRangePreset.THIS_YEAR:
        start_date = date(reference_date.year, 1, 1)
        end_date = reference_date

    elif preset == DateRangePreset.LAST_YEAR:
        start_date = date(reference_date.year - 1, 1, 1)
        end_date = date(reference_date.year - 1, 12, 31)

    else:
        raise ValueError(f"Unknown preset: {preset}")

    return DateRangeFilter(
        start_date=start_date,
        end_date=end_date,
        preset=preset
    )


def create_date_range(
    start_date: Union[date, str],
    end_date: Union[date, str],
    label: Optional[str] = None
) -> DateRangeFilter:
    """Create a custom DateRangeFilter.

    Args:
        start_date: Start date (date object or string)
        end_date: End date (date object or string)
        label: Optional custom label

    Returns:
        DateRangeFilter for the specified range
    """
    return DateRangeFilter(
        start_date=start_date,
        end_date=end_date,
        preset=DateRangePreset.CUSTOM,
        label=label
    )


def validate_date_range(
    start_date: Union[date, str],
    end_date: Union[date, str],
    max_days: Optional[int] = None
) -> Tuple[bool, Optional[str]]:
    """Validate a date range.

    Args:
        start_date: Start date to validate
        end_date: End date to validate
        max_days: Optional maximum number of days allowed

    Returns:
        Tuple of (is_valid, error_message)
    """
    try:
        if isinstance(start_date, str):
            start_date = parse_date_input(start_date)
        if isinstance(end_date, str):
            end_date = parse_date_input(end_date)
    except ValueError as e:
        return False, str(e)

    if start_date > end_date:
        return False, f"Start date ({start_date}) must be before or equal to end date ({end_date})"

    if max_days is not None:
        days_diff = (end_date - start_date).days + 1
        if days_diff > max_days:
            return False, f"Date range ({days_diff} days) exceeds maximum allowed ({max_days} days)"

    return True, None


def filter_data_by_range(
    data: List[Dict[str, Any]],
    date_range: DateRangeFilter,
    date_field: str = "ParsedTimestamp",
    fallback_field: str = "Timestamp"
) -> List[Dict[str, Any]]:
    """Filter a list of data records by date range.

    Args:
        data: List of data records (dictionaries)
        date_range: DateRangeFilter to apply
        date_field: Primary field name containing the date/datetime
        fallback_field: Fallback field name if primary is not found

    Returns:
        Filtered list of records within the date range
    """
    filtered = []

    for record in data:
        record_date = record.get(date_field)

        # Try fallback field if primary is not found
        if record_date is None:
            record_date = record.get(fallback_field)
            if record_date is None:
                continue

        # Parse the date if it's a string
        if isinstance(record_date, str):
            try:
                record_date = parse_date_input(record_date)
            except ValueError:
                continue
        elif isinstance(record_date, datetime):
            record_date = record_date.date()

        # Check if within range
        if date_range.contains_date(record_date):
            filtered.append(record)

    return filtered


def aggregate_by_date_range(
    data: List[Dict[str, Any]],
    date_range: DateRangeFilter,
    group_by: Optional[str] = None,
    date_field: str = "ParsedTimestamp"
) -> Dict[str, Any]:
    """Aggregate data within a date range.

    Args:
        data: List of data records
        date_range: DateRangeFilter to apply
        group_by: Optional field to group by
        date_field: Field name containing the date

    Returns:
        Dictionary with aggregated counts
    """
    filtered = filter_data_by_range(data, date_range, date_field)

    result = {
        "total": len(filtered),
        "date_range": date_range.to_dict(),
        "daily_average": len(filtered) / max(date_range.days_in_range, 1),
    }

    if group_by is not None:
        groups: Dict[str, int] = {}
        for record in filtered:
            key = record.get(group_by, "Unknown")
            groups[key] = groups.get(key, 0) + 1
        result["by_" + group_by.lower()] = groups

    return result


def get_date_range_options() -> List[Dict[str, str]]:
    """Get a list of preset options for UI selection.

    Returns:
        List of dictionaries with value and label for each preset
    """
    return [
        {"value": DateRangePreset.TODAY.value, "label": "Today"},
        {"value": DateRangePreset.YESTERDAY.value, "label": "Yesterday"},
        {"value": DateRangePreset.LAST_7_DAYS.value, "label": "Last 7 Days"},
        {"value": DateRangePreset.LAST_14_DAYS.value, "label": "Last 14 Days"},
        {"value": DateRangePreset.LAST_30_DAYS.value, "label": "Last 30 Days"},
        {"value": DateRangePreset.LAST_90_DAYS.value, "label": "Last 90 Days"},
        {"value": DateRangePreset.THIS_WEEK.value, "label": "This Week"},
        {"value": DateRangePreset.LAST_WEEK.value, "label": "Last Week"},
        {"value": DateRangePreset.THIS_MONTH.value, "label": "This Month"},
        {"value": DateRangePreset.LAST_MONTH.value, "label": "Last Month"},
        {"value": DateRangePreset.THIS_QUARTER.value, "label": "This Quarter"},
        {"value": DateRangePreset.LAST_QUARTER.value, "label": "Last Quarter"},
        {"value": DateRangePreset.THIS_YEAR.value, "label": "This Year"},
        {"value": DateRangePreset.LAST_YEAR.value, "label": "Last Year"},
        {"value": DateRangePreset.CUSTOM.value, "label": "Custom Range"},
    ]


if __name__ == "__main__":
    # Demo usage
    print("Date Range Filter - Demo")
    print("=" * 60)

    # Test preset ranges
    print("\nPreset Ranges:")
    for preset in [DateRangePreset.TODAY, DateRangePreset.LAST_7_DAYS,
                   DateRangePreset.THIS_WEEK, DateRangePreset.LAST_MONTH]:
        range_filter = get_preset_range(preset)
        print(f"  {range_filter.label}: {range_filter.start_date} to {range_filter.end_date} "
              f"({range_filter.days_in_range} days)")

    # Test custom range
    print("\nCustom Range:")
    custom = create_date_range("2026-01-01", "2026-01-15")
    print(f"  {custom.label}: {custom.start_date} to {custom.end_date}")
    print(f"  Days: {custom.days_in_range}")
    print(f"  Weeks: {custom.weeks_in_range:.1f}")

    # Test previous period
    print("\nPrevious Period:")
    prev = custom.get_previous_period()
    print(f"  {prev.label}: {prev.start_date} to {prev.end_date}")

    # Test date containment
    print("\nDate Containment:")
    test_date = date(2026, 1, 10)
    print(f"  {test_date} in {custom.label}: {custom.contains_date(test_date)}")

    # Test splitting
    print("\nSplit by Weeks:")
    for week_range in custom.split_by_weeks():
        print(f"  {week_range.label}")

    print("\n" + "=" * 60)
