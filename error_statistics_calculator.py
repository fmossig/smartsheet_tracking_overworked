"""
Error Statistics Calculator Module

Calculates aggregate error statistics including total errors, errors by category,
errors by sheet, error rate percentages, and tracks trends over time.

This module integrates with the ErrorCollector and DateRangeFilter to provide
comprehensive error analytics for reporting purposes.

Usage:
    from error_statistics_calculator import (
        ErrorStatistics,
        ErrorTrend,
        ErrorRateSeverity,
        calculate_error_statistics,
        calculate_error_rates,
        calculate_error_trends,
        get_error_statistics_summary,
        get_error_comparison,
    )
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Any, Union, Tuple
from enum import Enum
from collections import defaultdict

from error_collector import (
    ErrorCollector,
    CollectedError,
    ErrorSeverity,
    ErrorCategory,
    ErrorType,
    get_global_collector,
    CATEGORY_TO_TYPE_MAP,
)

from period_comparison_calculator import (
    TrendDirection,
    calculate_percent_change,
    get_trend_direction,
    format_percent_change,
)

# Set up logging
logger = logging.getLogger(__name__)


class ErrorRateSeverity(Enum):
    """Classification of error rates for status indicators."""
    LOW = "low"         # Error rate is acceptable (< 5%)
    MODERATE = "moderate"  # Error rate needs attention (5-15%)
    HIGH = "high"       # Error rate is concerning (15-30%)
    CRITICAL = "critical"  # Error rate is critical (> 30%)


@dataclass
class ErrorRateConfig:
    """Configuration for error rate thresholds.

    Attributes:
        low_threshold: Maximum percentage for LOW status (default 5%)
        moderate_threshold: Maximum percentage for MODERATE status (default 15%)
        high_threshold: Maximum percentage for HIGH status (default 30%)
    """
    low_threshold: float = 5.0
    moderate_threshold: float = 15.0
    high_threshold: float = 30.0


@dataclass
class CategoryStatistics:
    """Statistics for a specific error category.

    Attributes:
        category: The error category
        count: Number of errors in this category
        percentage: Percentage of total errors
        by_severity: Breakdown by severity level
        by_sheet: Breakdown by sheet ID
        by_group: Breakdown by group
    """
    category: ErrorCategory
    count: int
    percentage: float
    by_severity: Dict[str, int] = field(default_factory=dict)
    by_sheet: Dict[str, int] = field(default_factory=dict)
    by_group: Dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "category": self.category.value,
            "count": self.count,
            "percentage": round(self.percentage, 2),
            "by_severity": self.by_severity,
            "by_sheet": self.by_sheet,
            "by_group": self.by_group,
        }


@dataclass
class SheetStatistics:
    """Statistics for a specific sheet.

    Attributes:
        sheet_id: The sheet identifier
        sheet_name: Human-readable sheet name (if available)
        total_errors: Total number of errors for this sheet
        percentage: Percentage of total errors
        by_severity: Breakdown by severity level
        by_category: Breakdown by error category
        error_rate: Error rate percentage (if total rows known)
    """
    sheet_id: str
    sheet_name: Optional[str]
    total_errors: int
    percentage: float
    by_severity: Dict[str, int] = field(default_factory=dict)
    by_category: Dict[str, int] = field(default_factory=dict)
    error_rate: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "sheet_id": self.sheet_id,
            "sheet_name": self.sheet_name,
            "total_errors": self.total_errors,
            "percentage": round(self.percentage, 2),
            "by_severity": self.by_severity,
            "by_category": self.by_category,
            "error_rate": round(self.error_rate, 2) if self.error_rate else None,
        }


@dataclass
class ErrorStatistics:
    """Comprehensive error statistics for a time period.

    Attributes:
        start_time: Start of the analysis period
        end_time: End of the analysis period
        total_errors: Total number of errors
        total_warnings: Total number of warnings
        total_info: Total number of info messages
        total_critical: Total number of critical errors
        by_severity: Count by severity level
        by_category: Detailed statistics by category
        by_type: Count by high-level error type
        by_sheet: Detailed statistics by sheet
        by_group: Count by group
        error_rate: Overall error rate percentage (if denominator known)
        rate_severity: Classification of the error rate
    """
    start_time: Optional[datetime]
    end_time: Optional[datetime]
    total_errors: int
    total_warnings: int = 0
    total_info: int = 0
    total_critical: int = 0
    by_severity: Dict[str, int] = field(default_factory=dict)
    by_category: Dict[str, CategoryStatistics] = field(default_factory=dict)
    by_type: Dict[str, int] = field(default_factory=dict)
    by_sheet: Dict[str, SheetStatistics] = field(default_factory=dict)
    by_group: Dict[str, int] = field(default_factory=dict)
    error_rate: Optional[float] = None
    rate_severity: ErrorRateSeverity = ErrorRateSeverity.LOW

    @property
    def total_all(self) -> int:
        """Get total of all collected items (errors, warnings, info, critical)."""
        return sum(self.by_severity.values())

    @property
    def recoverable_count(self) -> int:
        """Estimate recoverable errors (non-critical)."""
        return self.total_all - self.total_critical

    @property
    def unique_categories(self) -> int:
        """Get count of unique error categories with errors."""
        return len([c for c in self.by_category.values() if c.count > 0])

    @property
    def unique_sheets(self) -> int:
        """Get count of unique sheets with errors."""
        return len(self.by_sheet)

    @property
    def most_common_category(self) -> Optional[str]:
        """Get the most common error category."""
        if not self.by_category:
            return None
        sorted_cats = sorted(
            self.by_category.values(),
            key=lambda x: x.count,
            reverse=True
        )
        return sorted_cats[0].category.value if sorted_cats else None

    @property
    def duration_hours(self) -> Optional[float]:
        """Get the duration of the analysis period in hours."""
        if self.start_time and self.end_time:
            delta = self.end_time - self.start_time
            return delta.total_seconds() / 3600
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "start_time": self.start_time.isoformat() if self.start_time else None,
            "end_time": self.end_time.isoformat() if self.end_time else None,
            "total_errors": self.total_errors,
            "total_warnings": self.total_warnings,
            "total_info": self.total_info,
            "total_critical": self.total_critical,
            "total_all": self.total_all,
            "by_severity": self.by_severity,
            "by_category": {k: v.to_dict() for k, v in self.by_category.items()},
            "by_type": self.by_type,
            "by_sheet": {k: v.to_dict() for k, v in self.by_sheet.items()},
            "by_group": self.by_group,
            "error_rate": round(self.error_rate, 2) if self.error_rate else None,
            "rate_severity": self.rate_severity.value,
            "unique_categories": self.unique_categories,
            "unique_sheets": self.unique_sheets,
            "most_common_category": self.most_common_category,
            "duration_hours": round(self.duration_hours, 2) if self.duration_hours else None,
        }


@dataclass
class ErrorTrend:
    """Trend data for error statistics over time.

    Attributes:
        current_period: Statistics for the current period
        previous_period: Statistics for the previous period
        absolute_change: Absolute change in total errors
        percent_change: Percentage change in total errors
        trend: Direction of the trend
        category_trends: Trends by category
        severity_trends: Trends by severity
        sheet_trends: Trends by sheet
    """
    current_period: ErrorStatistics
    previous_period: ErrorStatistics
    absolute_change: int = 0
    percent_change: float = 0.0
    trend: TrendDirection = TrendDirection.NO_DATA
    category_trends: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    severity_trends: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    sheet_trends: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate trend metrics after initialization."""
        if self.current_period and self.previous_period:
            self._calculate_trends()

    def _calculate_trends(self) -> None:
        """Calculate all trend metrics."""
        # Overall change
        curr_total = self.current_period.total_all
        prev_total = self.previous_period.total_all

        self.absolute_change = curr_total - prev_total
        self.percent_change = calculate_percent_change(curr_total, prev_total)
        self.trend = get_trend_direction(self.percent_change)

        # Severity trends
        for sev in ErrorSeverity:
            curr_val = self.current_period.by_severity.get(sev.value, 0)
            prev_val = self.previous_period.by_severity.get(sev.value, 0)
            pct_change = calculate_percent_change(curr_val, prev_val)

            self.severity_trends[sev.value] = {
                "current": curr_val,
                "previous": prev_val,
                "absolute_change": curr_val - prev_val,
                "percent_change": pct_change,
                "trend": get_trend_direction(pct_change).value,
            }

        # Category trends
        all_categories = set(self.current_period.by_category.keys()) | set(self.previous_period.by_category.keys())
        for cat in all_categories:
            curr_stats = self.current_period.by_category.get(cat)
            prev_stats = self.previous_period.by_category.get(cat)

            curr_val = curr_stats.count if curr_stats else 0
            prev_val = prev_stats.count if prev_stats else 0
            pct_change = calculate_percent_change(curr_val, prev_val)

            self.category_trends[cat] = {
                "current": curr_val,
                "previous": prev_val,
                "absolute_change": curr_val - prev_val,
                "percent_change": pct_change,
                "trend": get_trend_direction(pct_change).value,
            }

        # Sheet trends
        all_sheets = set(self.current_period.by_sheet.keys()) | set(self.previous_period.by_sheet.keys())
        for sheet in all_sheets:
            curr_stats = self.current_period.by_sheet.get(sheet)
            prev_stats = self.previous_period.by_sheet.get(sheet)

            curr_val = curr_stats.total_errors if curr_stats else 0
            prev_val = prev_stats.total_errors if prev_stats else 0
            pct_change = calculate_percent_change(curr_val, prev_val)

            self.sheet_trends[sheet] = {
                "current": curr_val,
                "previous": prev_val,
                "absolute_change": curr_val - prev_val,
                "percent_change": pct_change,
                "trend": get_trend_direction(pct_change).value,
            }

    @property
    def is_improving(self) -> bool:
        """Check if error trend is improving (decreasing)."""
        return self.trend == TrendDirection.DOWN

    @property
    def is_worsening(self) -> bool:
        """Check if error trend is worsening (increasing)."""
        return self.trend == TrendDirection.UP

    @property
    def top_increasing_categories(self) -> List[Tuple[str, float]]:
        """Get categories with the largest percentage increase."""
        increasing = [
            (cat, data["percent_change"])
            for cat, data in self.category_trends.items()
            if data["percent_change"] > 0
        ]
        return sorted(increasing, key=lambda x: x[1], reverse=True)[:5]

    @property
    def top_decreasing_categories(self) -> List[Tuple[str, float]]:
        """Get categories with the largest percentage decrease."""
        decreasing = [
            (cat, data["percent_change"])
            for cat, data in self.category_trends.items()
            if data["percent_change"] < 0
        ]
        return sorted(decreasing, key=lambda x: x[1])[:5]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "current_period": self.current_period.to_dict(),
            "previous_period": self.previous_period.to_dict(),
            "absolute_change": self.absolute_change,
            "percent_change": round(self.percent_change, 2),
            "trend": self.trend.value,
            "is_improving": self.is_improving,
            "is_worsening": self.is_worsening,
            "category_trends": self.category_trends,
            "severity_trends": self.severity_trends,
            "sheet_trends": self.sheet_trends,
            "top_increasing_categories": self.top_increasing_categories,
            "top_decreasing_categories": self.top_decreasing_categories,
        }


def get_error_rate_severity(
    error_rate: float,
    config: Optional[ErrorRateConfig] = None
) -> ErrorRateSeverity:
    """Determine the severity classification of an error rate.

    Args:
        error_rate: The error rate percentage
        config: Configuration for thresholds (uses defaults if not provided)

    Returns:
        ErrorRateSeverity classification
    """
    if config is None:
        config = ErrorRateConfig()

    if error_rate < config.low_threshold:
        return ErrorRateSeverity.LOW
    elif error_rate < config.moderate_threshold:
        return ErrorRateSeverity.MODERATE
    elif error_rate < config.high_threshold:
        return ErrorRateSeverity.HIGH
    else:
        return ErrorRateSeverity.CRITICAL


def calculate_error_statistics(
    errors: Optional[List[CollectedError]] = None,
    collector: Optional[ErrorCollector] = None,
    total_rows: Optional[int] = None,
    sheet_row_counts: Optional[Dict[str, int]] = None,
    rate_config: Optional[ErrorRateConfig] = None,
) -> ErrorStatistics:
    """Calculate comprehensive error statistics.

    Args:
        errors: List of collected errors (uses global collector if not provided)
        collector: ErrorCollector instance (uses global if not provided)
        total_rows: Total number of rows processed (for error rate calculation)
        sheet_row_counts: Row counts by sheet ID (for per-sheet error rates)
        rate_config: Configuration for error rate thresholds

    Returns:
        ErrorStatistics object with comprehensive statistics
    """
    # Get errors from collector if not provided
    if errors is None:
        if collector is None:
            collector = get_global_collector()
        errors = collector.get_all_errors()

    if not errors:
        return ErrorStatistics(
            start_time=None,
            end_time=None,
            total_errors=0,
        )

    # Initialize counters
    by_severity: Dict[str, int] = defaultdict(int)
    by_type: Dict[str, int] = defaultdict(int)
    by_group: Dict[str, int] = defaultdict(int)

    # Category tracking with detailed breakdown
    category_data: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "by_severity": defaultdict(int), "by_sheet": defaultdict(int), "by_group": defaultdict(int)}
    )

    # Sheet tracking with detailed breakdown
    sheet_data: Dict[str, Dict[str, Any]] = defaultdict(
        lambda: {"count": 0, "by_severity": defaultdict(int), "by_category": defaultdict(int), "name": None}
    )

    # Track timestamps
    timestamps = []

    for error in errors:
        timestamps.append(error.timestamp)

        # Severity counts
        by_severity[error.severity.value] += 1

        # Category detailed tracking
        cat_key = error.category.value
        category_data[cat_key]["count"] += 1
        category_data[cat_key]["by_severity"][error.severity.value] += 1

        # High-level type tracking
        error_type = error.get_high_level_type()
        by_type[error_type.value] += 1

        # Context-based tracking
        context = error.context or {}

        # Group tracking
        group = context.get("group")
        if group:
            by_group[group] += 1
            category_data[cat_key]["by_group"][group] += 1

        # Sheet tracking
        sheet_id = context.get("sheet_id")
        if sheet_id:
            sheet_key = str(sheet_id)
            sheet_data[sheet_key]["count"] += 1
            sheet_data[sheet_key]["by_severity"][error.severity.value] += 1
            sheet_data[sheet_key]["by_category"][cat_key] += 1
            category_data[cat_key]["by_sheet"][sheet_key] += 1

            # Try to get sheet name from context
            sheet_name = context.get("sheet_name")
            if sheet_name:
                sheet_data[sheet_key]["name"] = sheet_name

    # Calculate totals
    total_all = len(errors)
    total_errors = by_severity.get(ErrorSeverity.ERROR.value, 0)
    total_warnings = by_severity.get(ErrorSeverity.WARNING.value, 0)
    total_info = by_severity.get(ErrorSeverity.INFO.value, 0)
    total_critical = by_severity.get(ErrorSeverity.CRITICAL.value, 0)

    # Build category statistics
    by_category: Dict[str, CategoryStatistics] = {}
    for cat_key, data in category_data.items():
        try:
            category = ErrorCategory(cat_key)
        except ValueError:
            category = ErrorCategory.UNKNOWN

        by_category[cat_key] = CategoryStatistics(
            category=category,
            count=data["count"],
            percentage=(data["count"] / total_all * 100) if total_all > 0 else 0.0,
            by_severity=dict(data["by_severity"]),
            by_sheet=dict(data["by_sheet"]),
            by_group=dict(data["by_group"]),
        )

    # Build sheet statistics
    by_sheet: Dict[str, SheetStatistics] = {}
    for sheet_key, data in sheet_data.items():
        sheet_error_rate = None
        if sheet_row_counts and sheet_key in sheet_row_counts:
            row_count = sheet_row_counts[sheet_key]
            if row_count > 0:
                sheet_error_rate = (data["count"] / row_count) * 100

        by_sheet[sheet_key] = SheetStatistics(
            sheet_id=sheet_key,
            sheet_name=data["name"],
            total_errors=data["count"],
            percentage=(data["count"] / total_all * 100) if total_all > 0 else 0.0,
            by_severity=dict(data["by_severity"]),
            by_category=dict(data["by_category"]),
            error_rate=sheet_error_rate,
        )

    # Calculate overall error rate
    error_rate = None
    rate_severity = ErrorRateSeverity.LOW
    if total_rows and total_rows > 0:
        error_rate = (total_all / total_rows) * 100
        rate_severity = get_error_rate_severity(error_rate, rate_config)

    # Get time range
    start_time = min(timestamps) if timestamps else None
    end_time = max(timestamps) if timestamps else None

    return ErrorStatistics(
        start_time=start_time,
        end_time=end_time,
        total_errors=total_errors,
        total_warnings=total_warnings,
        total_info=total_info,
        total_critical=total_critical,
        by_severity=dict(by_severity),
        by_category=by_category,
        by_type=dict(by_type),
        by_sheet=by_sheet,
        by_group=dict(by_group),
        error_rate=error_rate,
        rate_severity=rate_severity,
    )


def calculate_error_rates(
    statistics: ErrorStatistics,
    total_rows: int,
    sheet_row_counts: Optional[Dict[str, int]] = None,
) -> Dict[str, Any]:
    """Calculate detailed error rate percentages.

    Args:
        statistics: Pre-calculated error statistics
        total_rows: Total number of rows processed
        sheet_row_counts: Row counts by sheet ID

    Returns:
        Dictionary with error rate calculations
    """
    if total_rows <= 0:
        return {
            "overall_rate": 0.0,
            "error_rate": 0.0,
            "warning_rate": 0.0,
            "critical_rate": 0.0,
            "by_category": {},
            "by_sheet": {},
            "rate_severity": ErrorRateSeverity.LOW.value,
        }

    overall_rate = (statistics.total_all / total_rows) * 100
    error_rate = (statistics.total_errors / total_rows) * 100
    warning_rate = (statistics.total_warnings / total_rows) * 100
    critical_rate = (statistics.total_critical / total_rows) * 100

    # Calculate rates by category
    category_rates = {}
    for cat_key, cat_stats in statistics.by_category.items():
        category_rates[cat_key] = {
            "count": cat_stats.count,
            "rate": (cat_stats.count / total_rows) * 100,
            "percentage_of_errors": cat_stats.percentage,
        }

    # Calculate rates by sheet
    sheet_rates = {}
    for sheet_key, sheet_stats in statistics.by_sheet.items():
        sheet_row_count = (sheet_row_counts or {}).get(sheet_key, 0)
        if sheet_row_count > 0:
            sheet_rate = (sheet_stats.total_errors / sheet_row_count) * 100
        else:
            sheet_rate = None

        sheet_rates[sheet_key] = {
            "count": sheet_stats.total_errors,
            "rate": sheet_rate,
            "percentage_of_errors": sheet_stats.percentage,
            "row_count": sheet_row_count,
        }

    rate_severity = get_error_rate_severity(overall_rate)

    return {
        "overall_rate": round(overall_rate, 2),
        "error_rate": round(error_rate, 2),
        "warning_rate": round(warning_rate, 2),
        "critical_rate": round(critical_rate, 2),
        "by_category": category_rates,
        "by_sheet": sheet_rates,
        "rate_severity": rate_severity.value,
        "total_rows": total_rows,
        "total_errors": statistics.total_all,
    }


def filter_errors_by_date_range(
    errors: List[CollectedError],
    start_date: date,
    end_date: date,
) -> List[CollectedError]:
    """Filter errors to a specific date range.

    Args:
        errors: List of collected errors
        start_date: Start date (inclusive)
        end_date: End date (inclusive)

    Returns:
        Filtered list of errors within the date range
    """
    filtered = []
    for error in errors:
        error_date = error.timestamp.date()
        if start_date <= error_date <= end_date:
            filtered.append(error)
    return filtered


def calculate_error_trends(
    current_start: date,
    current_end: date,
    previous_start: date,
    previous_end: date,
    errors: Optional[List[CollectedError]] = None,
    collector: Optional[ErrorCollector] = None,
    total_rows_current: Optional[int] = None,
    total_rows_previous: Optional[int] = None,
) -> ErrorTrend:
    """Calculate error trends between two time periods.

    Args:
        current_start: Start of current period
        current_end: End of current period
        previous_start: Start of previous period
        previous_end: End of previous period
        errors: List of errors (uses global collector if not provided)
        collector: ErrorCollector instance
        total_rows_current: Total rows in current period (for error rates)
        total_rows_previous: Total rows in previous period (for error rates)

    Returns:
        ErrorTrend object with comparison data
    """
    # Get all errors
    if errors is None:
        if collector is None:
            collector = get_global_collector()
        errors = collector.get_all_errors()

    # Filter errors by period
    current_errors = filter_errors_by_date_range(errors, current_start, current_end)
    previous_errors = filter_errors_by_date_range(errors, previous_start, previous_end)

    # Calculate statistics for each period
    current_stats = calculate_error_statistics(
        errors=current_errors,
        total_rows=total_rows_current,
    )
    previous_stats = calculate_error_statistics(
        errors=previous_errors,
        total_rows=total_rows_previous,
    )

    return ErrorTrend(
        current_period=current_stats,
        previous_period=previous_stats,
    )


def calculate_error_trends_for_range(
    date_range_filter: Any,  # DateRangeFilter type
    errors: Optional[List[CollectedError]] = None,
    collector: Optional[ErrorCollector] = None,
    total_rows_current: Optional[int] = None,
    total_rows_previous: Optional[int] = None,
) -> ErrorTrend:
    """Calculate error trends using a DateRangeFilter.

    This function integrates with DateRangeFilter to provide error trend
    analysis for any custom date range.

    Args:
        date_range_filter: DateRangeFilter object for the current period
        errors: List of errors (uses global collector if not provided)
        collector: ErrorCollector instance
        total_rows_current: Total rows in current period
        total_rows_previous: Total rows in previous period

    Returns:
        ErrorTrend object with comparison data
    """
    current_start = date_range_filter.start_date
    current_end = date_range_filter.end_date

    # Get previous period using DateRangeFilter's method
    previous_range = date_range_filter.get_previous_period()
    previous_start = previous_range.start_date
    previous_end = previous_range.end_date

    return calculate_error_trends(
        current_start=current_start,
        current_end=current_end,
        previous_start=previous_start,
        previous_end=previous_end,
        errors=errors,
        collector=collector,
        total_rows_current=total_rows_current,
        total_rows_previous=total_rows_previous,
    )


def get_error_statistics_summary(
    statistics: ErrorStatistics,
) -> str:
    """Get a human-readable summary of error statistics.

    Args:
        statistics: ErrorStatistics object

    Returns:
        Formatted summary string
    """
    lines = []
    lines.append("=" * 60)
    lines.append("ERROR STATISTICS SUMMARY")
    lines.append("=" * 60)

    # Time range
    if statistics.start_time and statistics.end_time:
        lines.append(f"Period: {statistics.start_time.strftime('%Y-%m-%d %H:%M')} to "
                    f"{statistics.end_time.strftime('%Y-%m-%d %H:%M')}")
        if statistics.duration_hours:
            lines.append(f"Duration: {statistics.duration_hours:.1f} hours")

    # Totals
    lines.append(f"\nTotal Items: {statistics.total_all}")
    lines.append(f"  Critical: {statistics.total_critical}")
    lines.append(f"  Errors: {statistics.total_errors}")
    lines.append(f"  Warnings: {statistics.total_warnings}")
    lines.append(f"  Info: {statistics.total_info}")

    # Error rate
    if statistics.error_rate is not None:
        lines.append(f"\nError Rate: {statistics.error_rate:.2f}% ({statistics.rate_severity.value})")

    # By severity
    lines.append("\nBy Severity:")
    for sev, count in sorted(statistics.by_severity.items(), key=lambda x: -x[1]):
        if count > 0:
            lines.append(f"  {sev.upper()}: {count}")

    # By type
    if statistics.by_type:
        type_counts = {k: v for k, v in statistics.by_type.items() if v > 0}
        if type_counts:
            lines.append("\nBy Type:")
            for type_val, count in sorted(type_counts.items(), key=lambda x: -x[1]):
                display_name = type_val.replace("_", " ").title()
                lines.append(f"  {display_name}: {count}")

    # Top categories
    if statistics.by_category:
        lines.append("\nTop Categories:")
        sorted_cats = sorted(
            statistics.by_category.values(),
            key=lambda x: x.count,
            reverse=True
        )[:5]
        for cat_stats in sorted_cats:
            if cat_stats.count > 0:
                lines.append(f"  {cat_stats.category.value}: {cat_stats.count} ({cat_stats.percentage:.1f}%)")

    # By group
    if statistics.by_group:
        lines.append("\nBy Group:")
        for group, count in sorted(statistics.by_group.items(), key=lambda x: -x[1]):
            lines.append(f"  {group}: {count}")

    # Top sheets
    if statistics.by_sheet:
        lines.append("\nTop Sheets (by error count):")
        sorted_sheets = sorted(
            statistics.by_sheet.values(),
            key=lambda x: x.total_errors,
            reverse=True
        )[:5]
        for sheet_stats in sorted_sheets:
            name = sheet_stats.sheet_name or sheet_stats.sheet_id
            rate_str = f" ({sheet_stats.error_rate:.2f}% rate)" if sheet_stats.error_rate else ""
            lines.append(f"  {name}: {sheet_stats.total_errors}{rate_str}")

    lines.append("=" * 60)

    return "\n".join(lines)


def get_error_comparison(
    trend: ErrorTrend,
) -> Dict[str, Any]:
    """Get a formatted comparison summary for error trends.

    Args:
        trend: ErrorTrend object

    Returns:
        Dictionary with formatted comparison data
    """
    current = trend.current_period
    previous = trend.previous_period

    # Top movers
    top_increasing = []
    for cat, pct in trend.top_increasing_categories[:3]:
        top_increasing.append({
            "category": cat,
            "change": format_percent_change(pct),
            "current": trend.category_trends[cat]["current"],
            "previous": trend.category_trends[cat]["previous"],
        })

    top_decreasing = []
    for cat, pct in trend.top_decreasing_categories[:3]:
        top_decreasing.append({
            "category": cat,
            "change": format_percent_change(pct),
            "current": trend.category_trends[cat]["current"],
            "previous": trend.category_trends[cat]["previous"],
        })

    return {
        "summary": {
            "current_total": current.total_all,
            "previous_total": previous.total_all,
            "absolute_change": trend.absolute_change,
            "percent_change": format_percent_change(trend.percent_change),
            "trend": trend.trend.value,
            "is_improving": trend.is_improving,
            "is_worsening": trend.is_worsening,
        },
        "current_period": {
            "start": current.start_time.isoformat() if current.start_time else None,
            "end": current.end_time.isoformat() if current.end_time else None,
            "errors": current.total_errors,
            "warnings": current.total_warnings,
            "critical": current.total_critical,
        },
        "previous_period": {
            "start": previous.start_time.isoformat() if previous.start_time else None,
            "end": previous.end_time.isoformat() if previous.end_time else None,
            "errors": previous.total_errors,
            "warnings": previous.total_warnings,
            "critical": previous.total_critical,
        },
        "severity_comparison": trend.severity_trends,
        "top_increasing_categories": top_increasing,
        "top_decreasing_categories": top_decreasing,
    }


def format_error_trend_report(
    trend: ErrorTrend,
) -> str:
    """Format error trends as a text report.

    Args:
        trend: ErrorTrend object

    Returns:
        Formatted text report
    """
    lines = []
    lines.append("=" * 60)
    lines.append("ERROR TREND REPORT")
    lines.append("=" * 60)

    current = trend.current_period
    previous = trend.previous_period

    # Overall trend
    trend_icon = {"up": "+", "down": "-", "flat": "=", "no_data": "?"}[trend.trend.value]
    lines.append(f"\nOverall Trend: {trend.trend.value.upper()} {trend_icon}")
    lines.append(f"Change: {trend.absolute_change} errors ({format_percent_change(trend.percent_change)})")

    if trend.is_improving:
        lines.append("Status: IMPROVING - Error count decreased")
    elif trend.is_worsening:
        lines.append("Status: WORSENING - Error count increased")
    else:
        lines.append("Status: STABLE - Error count relatively unchanged")

    # Period comparison
    lines.append("\n" + "-" * 60)
    lines.append("Period Comparison")
    lines.append("-" * 60)

    lines.append(f"\nCurrent Period:")
    lines.append(f"  Total: {current.total_all}")
    lines.append(f"  Critical: {current.total_critical}")
    lines.append(f"  Errors: {current.total_errors}")
    lines.append(f"  Warnings: {current.total_warnings}")

    lines.append(f"\nPrevious Period:")
    lines.append(f"  Total: {previous.total_all}")
    lines.append(f"  Critical: {previous.total_critical}")
    lines.append(f"  Errors: {previous.total_errors}")
    lines.append(f"  Warnings: {previous.total_warnings}")

    # Severity trends
    lines.append("\n" + "-" * 60)
    lines.append("Severity Trends")
    lines.append("-" * 60)

    for sev, data in trend.severity_trends.items():
        if data["current"] > 0 or data["previous"] > 0:
            trend_str = format_percent_change(data["percent_change"])
            lines.append(f"  {sev.upper()}: {data['previous']} -> {data['current']} ({trend_str})")

    # Top movers
    if trend.top_increasing_categories:
        lines.append("\n" + "-" * 60)
        lines.append("Categories with Largest Increase")
        lines.append("-" * 60)
        for cat, pct in trend.top_increasing_categories:
            data = trend.category_trends[cat]
            lines.append(f"  {cat}: {data['previous']} -> {data['current']} ({format_percent_change(pct)})")

    if trend.top_decreasing_categories:
        lines.append("\n" + "-" * 60)
        lines.append("Categories with Largest Decrease")
        lines.append("-" * 60)
        for cat, pct in trend.top_decreasing_categories:
            data = trend.category_trends[cat]
            lines.append(f"  {cat}: {data['previous']} -> {data['current']} ({format_percent_change(pct)})")

    lines.append("\n" + "=" * 60)

    return "\n".join(lines)


def calculate_daily_error_counts(
    errors: List[CollectedError],
    start_date: date,
    end_date: date,
) -> Dict[str, int]:
    """Calculate error counts by day.

    Args:
        errors: List of collected errors
        start_date: Start date (inclusive)
        end_date: End date (inclusive)

    Returns:
        Dictionary mapping date strings to error counts
    """
    daily_counts: Dict[str, int] = defaultdict(int)

    # Initialize all days in range with 0
    current = start_date
    while current <= end_date:
        daily_counts[current.isoformat()] = 0
        current += timedelta(days=1)

    # Count errors by day
    for error in errors:
        error_date = error.timestamp.date()
        if start_date <= error_date <= end_date:
            daily_counts[error_date.isoformat()] += 1

    return dict(daily_counts)


def calculate_hourly_distribution(
    errors: List[CollectedError],
) -> Dict[int, int]:
    """Calculate error distribution by hour of day.

    Args:
        errors: List of collected errors

    Returns:
        Dictionary mapping hour (0-23) to error count
    """
    hourly_counts: Dict[int, int] = defaultdict(int)

    # Initialize all hours with 0
    for hour in range(24):
        hourly_counts[hour] = 0

    for error in errors:
        hour = error.timestamp.hour
        hourly_counts[hour] += 1

    return dict(hourly_counts)


def get_error_hotspots(
    statistics: ErrorStatistics,
    top_n: int = 5,
) -> Dict[str, Any]:
    """Identify error hotspots in the data.

    Args:
        statistics: ErrorStatistics object
        top_n: Number of top items to return for each dimension

    Returns:
        Dictionary with hotspot information
    """
    # Top categories
    top_categories = sorted(
        statistics.by_category.values(),
        key=lambda x: x.count,
        reverse=True
    )[:top_n]

    # Top sheets
    top_sheets = sorted(
        statistics.by_sheet.values(),
        key=lambda x: x.total_errors,
        reverse=True
    )[:top_n]

    # Top groups
    top_groups = sorted(
        statistics.by_group.items(),
        key=lambda x: x[1],
        reverse=True
    )[:top_n]

    return {
        "top_categories": [
            {
                "category": c.category.value,
                "count": c.count,
                "percentage": c.percentage,
            }
            for c in top_categories
        ],
        "top_sheets": [
            {
                "sheet_id": s.sheet_id,
                "sheet_name": s.sheet_name,
                "count": s.total_errors,
                "percentage": s.percentage,
            }
            for s in top_sheets
        ],
        "top_groups": [
            {
                "group": g[0],
                "count": g[1],
            }
            for g in top_groups
        ],
    }


if __name__ == "__main__":
    # Demo usage with sample data
    print("Error Statistics Calculator - Demo")
    print("=" * 60)

    from error_collector import ErrorCollector, ErrorSeverity, ErrorCategory

    # Create a sample collector with test errors
    collector = ErrorCollector(log_on_collect=False)

    # Generate sample errors
    import random
    base_time = datetime.now()
    categories = list(ErrorCategory)
    severities = [ErrorSeverity.INFO, ErrorSeverity.WARNING, ErrorSeverity.ERROR, ErrorSeverity.CRITICAL]
    groups = ["NA", "NF", "NH", "NM", "NP"]
    sheet_ids = ["123456", "234567", "345678", "456789"]

    for i in range(100):
        days_ago = random.randint(0, 30)
        hours_ago = random.randint(0, 24)

        category = random.choice(categories)
        severity = random.choice(severities)
        group = random.choice(groups)
        sheet_id = random.choice(sheet_ids)

        collector.collect(
            severity=severity,
            category=category,
            message=f"Sample error {i+1} for testing",
            context={
                "group": group,
                "sheet_id": sheet_id,
            }
        )

    # Calculate statistics
    print("\n--- Calculating Statistics ---")
    stats = calculate_error_statistics(collector=collector, total_rows=1000)
    print(get_error_statistics_summary(stats))

    # Calculate error rates
    print("\n--- Calculating Error Rates ---")
    rates = calculate_error_rates(stats, total_rows=1000)
    print(f"Overall Error Rate: {rates['overall_rate']}%")
    print(f"Error Rate: {rates['error_rate']}%")
    print(f"Warning Rate: {rates['warning_rate']}%")
    print(f"Rate Severity: {rates['rate_severity']}")

    # Calculate trends
    print("\n--- Calculating Trends ---")
    today = date.today()
    trend = calculate_error_trends(
        current_start=today - timedelta(days=7),
        current_end=today,
        previous_start=today - timedelta(days=14),
        previous_end=today - timedelta(days=8),
        collector=collector,
    )
    print(format_error_trend_report(trend))

    # Get hotspots
    print("\n--- Error Hotspots ---")
    hotspots = get_error_hotspots(stats)
    print("Top Categories:")
    for cat in hotspots["top_categories"]:
        print(f"  {cat['category']}: {cat['count']} ({cat['percentage']:.1f}%)")

    print("\nTop Groups:")
    for grp in hotspots["top_groups"]:
        print(f"  {grp['group']}: {grp['count']}")

    print("\n" + "=" * 60)
