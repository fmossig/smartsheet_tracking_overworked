"""
Period Comparison Calculator Module

Provides utility functions to compare current period metrics against previous periods
(week-over-week, month-over-month). Calculates percentage changes and trends.

Enhanced with DateRangeFilter support for custom date range comparisons.

Usage:
    from period_comparison_calculator import (
        PeriodMetrics,
        PeriodComparison,
        TrendDirection,
        calculate_week_over_week,
        calculate_month_over_month,
        calculate_period_comparison,
        calculate_custom_range_comparison,
        get_comparison_summary,
        get_trend_direction,
    )
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Any, Union, Tuple
from enum import Enum

from historical_data_loader import (
    load_change_history,
    filter_by_date_range,
    aggregate_by_dimension,
    get_change_summary,
    group_by_dimension,
)

# Set up logging
logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    """Direction of trend compared to previous period."""
    UP = "up"
    DOWN = "down"
    FLAT = "flat"
    NO_DATA = "no_data"


@dataclass
class PeriodMetrics:
    """Metrics for a specific time period.

    Attributes:
        start_date: Start of the period (inclusive)
        end_date: End of the period (inclusive)
        total_changes: Total number of changes in the period
        by_group: Changes broken down by group
        by_phase: Changes broken down by phase
        by_user: Changes broken down by user
        by_marketplace: Changes broken down by marketplace
        unique_rows: Number of unique row IDs affected
        daily_average: Average changes per day
    """
    start_date: date
    end_date: date
    total_changes: int
    by_group: Dict[str, int] = field(default_factory=dict)
    by_phase: Dict[str, int] = field(default_factory=dict)
    by_user: Dict[str, int] = field(default_factory=dict)
    by_marketplace: Dict[str, int] = field(default_factory=dict)
    unique_rows: int = 0
    daily_average: float = 0.0

    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        if self.total_changes > 0 and self.start_date and self.end_date:
            days_in_period = (self.end_date - self.start_date).days + 1
            if days_in_period > 0:
                self.daily_average = round(self.total_changes / days_in_period, 2)


@dataclass
class PeriodComparison:
    """Comparison between two periods (current vs previous).

    Attributes:
        current_period: Metrics for the current period
        previous_period: Metrics for the previous period
        absolute_change: Absolute change in total (current - previous)
        percent_change: Percentage change ((current - previous) / previous * 100)
        trend: Direction of the trend (UP, DOWN, FLAT)
        by_group: Comparison by group with percent changes
        by_phase: Comparison by phase with percent changes
        by_user: Comparison by user with percent changes
        by_marketplace: Comparison by marketplace with percent changes
    """
    current_period: PeriodMetrics
    previous_period: PeriodMetrics
    absolute_change: int = 0
    percent_change: float = 0.0
    trend: TrendDirection = TrendDirection.NO_DATA
    by_group: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_phase: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_user: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    by_marketplace: Dict[str, Dict[str, Any]] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate comparison metrics after initialization."""
        if self.current_period and self.previous_period:
            self._calculate_comparison()

    def _calculate_comparison(self) -> None:
        """Calculate all comparison metrics."""
        # Calculate total change
        self.absolute_change = self.current_period.total_changes - self.previous_period.total_changes
        self.percent_change = calculate_percent_change(
            self.current_period.total_changes,
            self.previous_period.total_changes
        )
        self.trend = get_trend_direction(self.percent_change)

        # Calculate dimensional comparisons
        self.by_group = self._compare_dimension(
            self.current_period.by_group,
            self.previous_period.by_group
        )
        self.by_phase = self._compare_dimension(
            self.current_period.by_phase,
            self.previous_period.by_phase
        )
        self.by_user = self._compare_dimension(
            self.current_period.by_user,
            self.previous_period.by_user
        )
        self.by_marketplace = self._compare_dimension(
            self.current_period.by_marketplace,
            self.previous_period.by_marketplace
        )

    def _compare_dimension(
        self,
        current: Dict[str, int],
        previous: Dict[str, int]
    ) -> Dict[str, Dict[str, Any]]:
        """Compare a dimension between current and previous periods.

        Returns:
            Dictionary with each key containing:
                - current: Current period value
                - previous: Previous period value
                - absolute_change: Difference
                - percent_change: Percentage change
                - trend: Trend direction
        """
        all_keys = set(current.keys()) | set(previous.keys())
        result = {}

        for key in all_keys:
            curr_val = current.get(key, 0)
            prev_val = previous.get(key, 0)
            pct_change = calculate_percent_change(curr_val, prev_val)

            result[key] = {
                "current": curr_val,
                "previous": prev_val,
                "absolute_change": curr_val - prev_val,
                "percent_change": pct_change,
                "trend": get_trend_direction(pct_change).value,
            }

        return result


def calculate_percent_change(current: int, previous: int) -> float:
    """Calculate percentage change between two values.

    Args:
        current: Current period value
        previous: Previous period value

    Returns:
        Percentage change rounded to 2 decimal places.
        Returns 0.0 if previous is 0 and current is 0.
        Returns 100.0 if previous is 0 and current > 0.
        Returns -100.0 if previous is 0 and current < 0 (shouldn't happen with counts).

    Example:
        >>> calculate_percent_change(150, 100)
        50.0
        >>> calculate_percent_change(80, 100)
        -20.0
        >>> calculate_percent_change(100, 0)
        100.0
    """
    if previous == 0:
        if current == 0:
            return 0.0
        return 100.0 if current > 0 else -100.0

    return round(((current - previous) / previous) * 100, 2)


def get_trend_direction(percent_change: float, threshold: float = 1.0) -> TrendDirection:
    """Determine trend direction based on percentage change.

    Args:
        percent_change: The percentage change value
        threshold: Minimum absolute change to be considered UP or DOWN (default 1%)

    Returns:
        TrendDirection enum value (UP, DOWN, or FLAT)

    Example:
        >>> get_trend_direction(15.5)
        TrendDirection.UP
        >>> get_trend_direction(-8.2)
        TrendDirection.DOWN
        >>> get_trend_direction(0.5)
        TrendDirection.FLAT
    """
    if percent_change >= threshold:
        return TrendDirection.UP
    elif percent_change <= -threshold:
        return TrendDirection.DOWN
    else:
        return TrendDirection.FLAT


def get_period_metrics(
    changes: List[Dict[str, Any]],
    start_date: date,
    end_date: date
) -> PeriodMetrics:
    """Calculate metrics for a specific period.

    Args:
        changes: List of change records (from load_change_history)
        start_date: Start of period (inclusive)
        end_date: End of period (inclusive)

    Returns:
        PeriodMetrics object with calculated values
    """
    # Filter changes to the period
    period_changes = filter_by_date_range(changes, start_date, end_date)

    # Get summary
    summary = get_change_summary(period_changes)

    return PeriodMetrics(
        start_date=start_date,
        end_date=end_date,
        total_changes=summary["total_changes"],
        by_group=summary["by_group"],
        by_phase=summary["by_phase"],
        by_user=summary["by_user"],
        by_marketplace=summary["by_marketplace"],
        unique_rows=summary["unique_rows"],
    )


def calculate_week_over_week(
    reference_date: Optional[date] = None,
    changes: Optional[List[Dict[str, Any]]] = None
) -> PeriodComparison:
    """Calculate week-over-week comparison.

    Compares the current week (Monday to reference_date) with the previous week.

    Args:
        reference_date: The reference date (defaults to today)
        changes: Optional pre-loaded changes (will load if not provided)

    Returns:
        PeriodComparison object with week-over-week metrics

    Example:
        >>> wow = calculate_week_over_week()
        >>> print(f"WoW change: {wow.percent_change}%")
        >>> print(f"Trend: {wow.trend.value}")
    """
    if reference_date is None:
        reference_date = date.today()

    # Calculate current week (Monday to reference_date)
    days_since_monday = reference_date.weekday()
    current_week_start = reference_date - timedelta(days=days_since_monday)
    current_week_end = reference_date

    # Calculate previous week (full week before current week's Monday)
    previous_week_end = current_week_start - timedelta(days=1)
    previous_week_start = previous_week_end - timedelta(days=6)

    # Load changes if not provided
    if changes is None:
        # Load enough data to cover both periods
        all_start = previous_week_start
        changes = load_change_history(start_date=all_start, end_date=reference_date)

    # Calculate metrics for each period
    current_metrics = get_period_metrics(changes, current_week_start, current_week_end)
    previous_metrics = get_period_metrics(changes, previous_week_start, previous_week_end)

    return PeriodComparison(
        current_period=current_metrics,
        previous_period=previous_metrics,
    )


def calculate_month_over_month(
    reference_date: Optional[date] = None,
    changes: Optional[List[Dict[str, Any]]] = None
) -> PeriodComparison:
    """Calculate month-over-month comparison.

    Compares the current month (1st to reference_date) with the previous month.

    Args:
        reference_date: The reference date (defaults to today)
        changes: Optional pre-loaded changes (will load if not provided)

    Returns:
        PeriodComparison object with month-over-month metrics

    Example:
        >>> mom = calculate_month_over_month()
        >>> print(f"MoM change: {mom.percent_change}%")
        >>> print(f"Trend: {mom.trend.value}")
    """
    if reference_date is None:
        reference_date = date.today()

    # Calculate current month (1st to reference_date)
    current_month_start = reference_date.replace(day=1)
    current_month_end = reference_date

    # Calculate previous month
    previous_month_end = current_month_start - timedelta(days=1)
    previous_month_start = previous_month_end.replace(day=1)

    # Load changes if not provided
    if changes is None:
        all_start = previous_month_start
        changes = load_change_history(start_date=all_start, end_date=reference_date)

    # Calculate metrics for each period
    current_metrics = get_period_metrics(changes, current_month_start, current_month_end)
    previous_metrics = get_period_metrics(changes, previous_month_start, previous_month_end)

    return PeriodComparison(
        current_period=current_metrics,
        previous_period=previous_metrics,
    )


def calculate_period_comparison(
    current_start: date,
    current_end: date,
    previous_start: date,
    previous_end: date,
    changes: Optional[List[Dict[str, Any]]] = None
) -> PeriodComparison:
    """Calculate comparison between two custom periods.

    Args:
        current_start: Start of current period
        current_end: End of current period
        previous_start: Start of previous period
        previous_end: End of previous period
        changes: Optional pre-loaded changes (will load if not provided)

    Returns:
        PeriodComparison object with comparison metrics

    Example:
        >>> comparison = calculate_period_comparison(
        ...     current_start=date(2026, 1, 1),
        ...     current_end=date(2026, 1, 15),
        ...     previous_start=date(2025, 12, 16),
        ...     previous_end=date(2025, 12, 31)
        ... )
    """
    # Load changes if not provided
    if changes is None:
        all_start = min(current_start, previous_start)
        all_end = max(current_end, previous_end)
        changes = load_change_history(start_date=all_start, end_date=all_end)

    # Calculate metrics for each period
    current_metrics = get_period_metrics(changes, current_start, current_end)
    previous_metrics = get_period_metrics(changes, previous_start, previous_end)

    return PeriodComparison(
        current_period=current_metrics,
        previous_period=previous_metrics,
    )


def get_comparison_summary(comparison: PeriodComparison) -> Dict[str, Any]:
    """Get a formatted summary of a period comparison.

    Args:
        comparison: PeriodComparison object

    Returns:
        Dictionary with formatted comparison summary

    Example:
        >>> wow = calculate_week_over_week()
        >>> summary = get_comparison_summary(wow)
        >>> print(summary)
    """
    current = comparison.current_period
    previous = comparison.previous_period

    # Get top movers (biggest changes by absolute value)
    top_group_changes = sorted(
        comparison.by_group.items(),
        key=lambda x: abs(x[1]["absolute_change"]),
        reverse=True
    )[:3]

    top_user_changes = sorted(
        comparison.by_user.items(),
        key=lambda x: abs(x[1]["absolute_change"]),
        reverse=True
    )[:3]

    return {
        "period": {
            "current": {
                "start": current.start_date.isoformat(),
                "end": current.end_date.isoformat(),
                "total": current.total_changes,
                "daily_average": current.daily_average,
            },
            "previous": {
                "start": previous.start_date.isoformat(),
                "end": previous.end_date.isoformat(),
                "total": previous.total_changes,
                "daily_average": previous.daily_average,
            },
        },
        "comparison": {
            "absolute_change": comparison.absolute_change,
            "percent_change": comparison.percent_change,
            "trend": comparison.trend.value,
        },
        "top_group_changes": [
            {"group": g, **data} for g, data in top_group_changes
        ],
        "top_user_changes": [
            {"user": u, **data} for u, data in top_user_changes
        ],
        "breakdown": {
            "by_group": comparison.by_group,
            "by_phase": comparison.by_phase,
            "by_user": comparison.by_user,
            "by_marketplace": comparison.by_marketplace,
        },
    }


def calculate_rolling_comparison(
    days: int = 7,
    reference_date: Optional[date] = None,
    changes: Optional[List[Dict[str, Any]]] = None
) -> PeriodComparison:
    """Calculate comparison for rolling N-day periods.

    Compares the last N days with the N days before that.

    Args:
        days: Number of days in each period (default 7)
        reference_date: End date for current period (defaults to today)
        changes: Optional pre-loaded changes (will load if not provided)

    Returns:
        PeriodComparison object with rolling comparison metrics

    Example:
        >>> # Compare last 7 days with previous 7 days
        >>> rolling = calculate_rolling_comparison(days=7)
        >>>
        >>> # Compare last 30 days with previous 30 days
        >>> rolling30 = calculate_rolling_comparison(days=30)
    """
    if reference_date is None:
        reference_date = date.today()

    # Current period: last N days ending at reference_date
    current_end = reference_date
    current_start = reference_date - timedelta(days=days - 1)

    # Previous period: N days before current period
    previous_end = current_start - timedelta(days=1)
    previous_start = previous_end - timedelta(days=days - 1)

    return calculate_period_comparison(
        current_start=current_start,
        current_end=current_end,
        previous_start=previous_start,
        previous_end=previous_end,
        changes=changes,
    )


def calculate_custom_range_comparison(
    date_range_filter: Any,  # DateRangeFilter type
    compare_to_previous: bool = True,
    changes: Optional[List[Dict[str, Any]]] = None
) -> PeriodComparison:
    """Calculate comparison for a custom date range filter.

    This function integrates with DateRangeFilter to provide period comparisons
    for any custom date range. It automatically calculates the previous period
    based on the duration of the custom range.

    Args:
        date_range_filter: DateRangeFilter object specifying the current period
        compare_to_previous: If True, compare with the previous period of same duration
        changes: Optional pre-loaded changes (will load if not provided)

    Returns:
        PeriodComparison object with comparison metrics

    Example:
        >>> from date_range_filter import create_date_range
        >>> custom_range = create_date_range("2026-01-01", "2026-01-15")
        >>> comparison = calculate_custom_range_comparison(custom_range)
        >>> print(f"Change: {comparison.percent_change}%")
    """
    current_start = date_range_filter.start_date
    current_end = date_range_filter.end_date

    if compare_to_previous:
        # Get the previous period using DateRangeFilter's method
        previous_range = date_range_filter.get_previous_period()
        previous_start = previous_range.start_date
        previous_end = previous_range.end_date
    else:
        # No comparison - use same period for both
        previous_start = current_start
        previous_end = current_end

    return calculate_period_comparison(
        current_start=current_start,
        current_end=current_end,
        previous_start=previous_start,
        previous_end=previous_end,
        changes=changes,
    )


def calculate_multi_period_comparison(
    date_range_filter: Any,  # DateRangeFilter type
    num_periods: int = 3,
    changes: Optional[List[Dict[str, Any]]] = None
) -> List[PeriodComparison]:
    """Calculate comparisons across multiple consecutive periods.

    Useful for analyzing trends over time with a custom date range as the base.

    Args:
        date_range_filter: DateRangeFilter for the most recent period
        num_periods: Number of periods to compare (default 3)
        changes: Optional pre-loaded changes

    Returns:
        List of PeriodComparison objects, from most recent to oldest

    Example:
        >>> from date_range_filter import get_preset_range, DateRangePreset
        >>> last_week = get_preset_range(DateRangePreset.LAST_WEEK)
        >>> comparisons = calculate_multi_period_comparison(last_week, num_periods=4)
        >>> for comp in comparisons:
        ...     print(f"{comp.current_period.start_date}: {comp.current_period.total_changes}")
    """
    comparisons = []
    current_range = date_range_filter

    for i in range(num_periods):
        if i == 0:
            # First period - compare with previous
            comparison = calculate_custom_range_comparison(
                current_range,
                compare_to_previous=True,
                changes=changes
            )
        else:
            # Subsequent periods - calculate for the period
            comparison = calculate_custom_range_comparison(
                current_range,
                compare_to_previous=True,
                changes=changes
            )

        comparisons.append(comparison)

        # Move to the previous period for next iteration
        current_range = current_range.get_previous_period()

    return comparisons


def get_trend_indicator(trend: TrendDirection) -> str:
    """Get a visual indicator for a trend direction.

    Args:
        trend: TrendDirection enum value

    Returns:
        String indicator for the trend
    """
    indicators = {
        TrendDirection.UP: "+",
        TrendDirection.DOWN: "-",
        TrendDirection.FLAT: "=",
        TrendDirection.NO_DATA: "?",
    }
    return indicators.get(trend, "?")


def format_percent_change(percent_change: float, include_sign: bool = True) -> str:
    """Format a percentage change for display.

    Args:
        percent_change: The percentage change value
        include_sign: Whether to include + for positive values

    Returns:
        Formatted string (e.g., "+15.5%", "-8.2%", "0.0%")
    """
    if include_sign and percent_change > 0:
        return f"+{percent_change:.1f}%"
    return f"{percent_change:.1f}%"


def get_dimensional_trends(
    comparison: PeriodComparison,
    dimension: str = "group"
) -> List[Dict[str, Any]]:
    """Get sorted trends for a specific dimension.

    Args:
        comparison: PeriodComparison object
        dimension: Dimension to analyze ("group", "phase", "user", "marketplace")

    Returns:
        List of dimensional changes sorted by percent change (descending)

    Example:
        >>> wow = calculate_week_over_week()
        >>> group_trends = get_dimensional_trends(wow, "group")
        >>> for trend in group_trends:
        ...     print(f"{trend['key']}: {trend['percent_change']}%")
    """
    dimension_map = {
        "group": comparison.by_group,
        "phase": comparison.by_phase,
        "user": comparison.by_user,
        "marketplace": comparison.by_marketplace,
    }

    if dimension.lower() not in dimension_map:
        raise ValueError(f"Invalid dimension: {dimension}. Valid: {list(dimension_map.keys())}")

    dim_data = dimension_map[dimension.lower()]

    trends = []
    for key, data in dim_data.items():
        trends.append({
            "key": key,
            "current": data["current"],
            "previous": data["previous"],
            "absolute_change": data["absolute_change"],
            "percent_change": data["percent_change"],
            "trend": data["trend"],
            "formatted_change": format_percent_change(data["percent_change"]),
        })

    # Sort by percent change (descending)
    return sorted(trends, key=lambda x: x["percent_change"], reverse=True)


if __name__ == "__main__":
    # Demo usage
    print("Period Comparison Calculator - Demo")
    print("=" * 60)

    try:
        # Load all changes
        changes = load_change_history()
        print(f"\nLoaded {len(changes)} total changes")

        if changes:
            # Week-over-Week comparison
            print("\n" + "-" * 60)
            print("Week-over-Week Comparison")
            print("-" * 60)

            wow = calculate_week_over_week(changes=changes)
            print(f"Current week: {wow.current_period.start_date} to {wow.current_period.end_date}")
            print(f"  Total changes: {wow.current_period.total_changes}")
            print(f"  Daily average: {wow.current_period.daily_average}")
            print(f"Previous week: {wow.previous_period.start_date} to {wow.previous_period.end_date}")
            print(f"  Total changes: {wow.previous_period.total_changes}")
            print(f"  Daily average: {wow.previous_period.daily_average}")
            print(f"\nChange: {wow.absolute_change} ({format_percent_change(wow.percent_change)})")
            print(f"Trend: {wow.trend.value} {get_trend_indicator(wow.trend)}")

            # Show group trends
            print("\nGroup Trends (WoW):")
            group_trends = get_dimensional_trends(wow, "group")
            for trend in group_trends[:5]:
                indicator = get_trend_indicator(TrendDirection(trend["trend"]))
                print(f"  {trend['key']}: {trend['current']} vs {trend['previous']} "
                      f"({trend['formatted_change']}) {indicator}")

            # Month-over-Month comparison
            print("\n" + "-" * 60)
            print("Month-over-Month Comparison")
            print("-" * 60)

            mom = calculate_month_over_month(changes=changes)
            print(f"Current month: {mom.current_period.start_date} to {mom.current_period.end_date}")
            print(f"  Total changes: {mom.current_period.total_changes}")
            print(f"Previous month: {mom.previous_period.start_date} to {mom.previous_period.end_date}")
            print(f"  Total changes: {mom.previous_period.total_changes}")
            print(f"\nChange: {mom.absolute_change} ({format_percent_change(mom.percent_change)})")
            print(f"Trend: {mom.trend.value} {get_trend_indicator(mom.trend)}")

            # Rolling 7-day comparison
            print("\n" + "-" * 60)
            print("Rolling 7-Day Comparison")
            print("-" * 60)

            rolling = calculate_rolling_comparison(days=7, changes=changes)
            print(f"Last 7 days: {rolling.current_period.start_date} to {rolling.current_period.end_date}")
            print(f"  Total: {rolling.current_period.total_changes}")
            print(f"Previous 7 days: {rolling.previous_period.start_date} to {rolling.previous_period.end_date}")
            print(f"  Total: {rolling.previous_period.total_changes}")
            print(f"\nChange: {rolling.absolute_change} ({format_percent_change(rolling.percent_change)})")
            print(f"Trend: {rolling.trend.value} {get_trend_indicator(rolling.trend)}")

            # Get full summary
            print("\n" + "-" * 60)
            print("Full Comparison Summary (WoW)")
            print("-" * 60)
            summary = get_comparison_summary(wow)
            print(f"Top Group Changes:")
            for gc in summary["top_group_changes"]:
                print(f"  {gc['group']}: {gc['absolute_change']} ({gc['percent_change']}%)")

    except FileNotFoundError as e:
        print(f"Error: {e}")
    except Exception as e:
        print(f"Unexpected error: {e}")
        import traceback
        traceback.print_exc()
