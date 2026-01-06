"""
Completion Rate Calculator Module

Calculates completion rates for items reaching final phases (Phase 4) versus items
started (Phase 1). Provides breakdown by group and time period with trend analysis.

This module builds on the phase progression funnel calculator to provide focused
completion rate analytics for workflow efficiency monitoring.

Usage:
    from completion_rate_calculator import (
        CompletionRateMetrics,
        GroupCompletionRate,
        TimePeriodCompletion,
        CompletionRateComparison,
        calculate_completion_rate,
        calculate_completion_by_group,
        calculate_completion_by_time_period,
        calculate_completion_for_range,
        compare_completion_periods,
        get_completion_summary,
        get_completion_visualization_data,
        format_completion_report,
    )
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Any, Union, Tuple
from enum import Enum
from collections import defaultdict

from period_comparison_calculator import (
    TrendDirection,
    calculate_percent_change,
    get_trend_direction,
    format_percent_change,
)

# Set up logging
logger = logging.getLogger(__name__)


# Constants for phase definitions
STARTING_PHASE = 1  # Phase 1: Kontrolle - items enter the workflow
COMPLETION_PHASE = 4  # Phase 4: C am - items complete the main workflow


class CompletionTrend(Enum):
    """Trend direction for completion rate changes."""
    IMPROVING = "improving"
    DECLINING = "declining"
    STABLE = "stable"
    NO_DATA = "no_data"


@dataclass
class CompletionRateMetrics:
    """Core metrics for completion rate calculation.

    Attributes:
        items_started: Number of items that entered Phase 1 (started)
        items_completed: Number of items that reached Phase 4 (completed)
        completion_rate: Percentage of started items that completed (0-100)
        date_range_start: Start date of the analysis period
        date_range_end: End date of the analysis period
        average_days_to_complete: Average number of days to complete (if calculable)
        unique_users: Number of unique users who processed completed items
    """
    items_started: int = 0
    items_completed: int = 0
    completion_rate: float = 0.0
    date_range_start: Optional[date] = None
    date_range_end: Optional[date] = None
    average_days_to_complete: Optional[float] = None
    unique_users: int = 0

    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        if self.items_started > 0:
            self.completion_rate = round(
                (self.items_completed / self.items_started) * 100, 2
            )

    @property
    def items_incomplete(self) -> int:
        """Get the number of items that started but didn't complete."""
        return max(0, self.items_started - self.items_completed)

    @property
    def incompletion_rate(self) -> float:
        """Get the incompletion rate (inverse of completion rate)."""
        return round(100.0 - self.completion_rate, 2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "items_started": self.items_started,
            "items_completed": self.items_completed,
            "items_incomplete": self.items_incomplete,
            "completion_rate": self.completion_rate,
            "incompletion_rate": self.incompletion_rate,
            "date_range_start": self.date_range_start.isoformat() if self.date_range_start else None,
            "date_range_end": self.date_range_end.isoformat() if self.date_range_end else None,
            "average_days_to_complete": self.average_days_to_complete,
            "unique_users": self.unique_users,
        }


@dataclass
class GroupCompletionRate:
    """Completion rate metrics for a specific group.

    Attributes:
        group: Group identifier (NA, NF, NH, NM, NP, NT, NV, BUNDLE_FAN, BUNDLE_COOLER)
        metrics: CompletionRateMetrics for this group
        rank: Rank among all groups (1 = highest completion rate)
        performance_level: Performance classification (excellent, good, average, poor)
    """
    group: str
    metrics: CompletionRateMetrics
    rank: int = 0
    performance_level: str = "unknown"

    def __post_init__(self):
        """Calculate performance level after initialization."""
        if self.performance_level == "unknown":
            self.performance_level = self._calculate_performance_level()

    def _calculate_performance_level(self) -> str:
        """Determine performance level based on completion rate."""
        rate = self.metrics.completion_rate
        if rate >= 80:
            return "excellent"
        elif rate >= 60:
            return "good"
        elif rate >= 40:
            return "average"
        else:
            return "poor"

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "group": self.group,
            "metrics": self.metrics.to_dict(),
            "rank": self.rank,
            "performance_level": self.performance_level,
        }


@dataclass
class TimePeriodCompletion:
    """Completion rate metrics for a specific time period.

    Attributes:
        period_label: Human-readable label for the period (e.g., "Week 1", "January")
        period_start: Start date of the period
        period_end: End date of the period
        metrics: CompletionRateMetrics for this period
        change_from_previous: Percentage point change from previous period
        trend: Trend direction compared to previous period
    """
    period_label: str
    period_start: date
    period_end: date
    metrics: CompletionRateMetrics
    change_from_previous: float = 0.0
    trend: TrendDirection = TrendDirection.NO_DATA

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "period_label": self.period_label,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "metrics": self.metrics.to_dict(),
            "change_from_previous": self.change_from_previous,
            "trend": self.trend.value,
        }


@dataclass
class CompletionRateComparison:
    """Comparison of completion rates between two periods.

    Attributes:
        current_period: Metrics for the current period
        previous_period: Metrics for the previous period
        rate_change: Change in completion rate (percentage points)
        percent_change: Percentage change in completion rate
        items_started_change: Change in number of items started
        items_completed_change: Change in number of items completed
        trend: Overall trend direction
        analysis_notes: List of notable observations
    """
    current_period: CompletionRateMetrics
    previous_period: CompletionRateMetrics
    rate_change: float = 0.0
    percent_change: float = 0.0
    items_started_change: int = 0
    items_completed_change: int = 0
    trend: CompletionTrend = CompletionTrend.NO_DATA
    analysis_notes: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Calculate comparison metrics after initialization."""
        self._calculate_comparison()

    def _calculate_comparison(self) -> None:
        """Calculate all comparison metrics."""
        # Calculate rate change (percentage points)
        self.rate_change = round(
            self.current_period.completion_rate - self.previous_period.completion_rate, 2
        )

        # Calculate percent change
        if self.previous_period.completion_rate > 0:
            self.percent_change = calculate_percent_change(
                self.current_period.completion_rate,
                self.previous_period.completion_rate
            )
        else:
            self.percent_change = 100.0 if self.current_period.completion_rate > 0 else 0.0

        # Calculate item changes
        self.items_started_change = (
            self.current_period.items_started - self.previous_period.items_started
        )
        self.items_completed_change = (
            self.current_period.items_completed - self.previous_period.items_completed
        )

        # Determine trend
        if self.rate_change >= 5:
            self.trend = CompletionTrend.IMPROVING
        elif self.rate_change <= -5:
            self.trend = CompletionTrend.DECLINING
        else:
            self.trend = CompletionTrend.STABLE

        # Generate analysis notes
        self._generate_analysis_notes()

    def _generate_analysis_notes(self) -> None:
        """Generate notable observations about the comparison."""
        notes = []

        # Significant rate change
        if abs(self.rate_change) >= 10:
            direction = "increased" if self.rate_change > 0 else "decreased"
            notes.append(
                f"Completion rate {direction} significantly by {abs(self.rate_change):.1f} percentage points"
            )

        # Volume changes
        if self.items_started_change != 0:
            direction = "increased" if self.items_started_change > 0 else "decreased"
            notes.append(
                f"Items started {direction} by {abs(self.items_started_change)}"
            )

        # Efficiency observation
        if self.items_completed_change > 0 and self.items_started_change <= 0:
            notes.append("Improved efficiency: more completions with same or fewer starts")
        elif self.items_completed_change < 0 and self.items_started_change >= 0:
            notes.append("Decreased efficiency: fewer completions despite same or more starts")

        self.analysis_notes = notes

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "current_period": self.current_period.to_dict(),
            "previous_period": self.previous_period.to_dict(),
            "rate_change": self.rate_change,
            "percent_change": self.percent_change,
            "items_started_change": self.items_started_change,
            "items_completed_change": self.items_completed_change,
            "trend": self.trend.value,
            "analysis_notes": self.analysis_notes,
        }


def _extract_phase_data(
    changes: List[Dict[str, Any]],
    group: Optional[str] = None
) -> Tuple[set, set, Dict[str, set]]:
    """Extract phase data from change records.

    Args:
        changes: List of change records from historical data
        group: Optional group to filter by

    Returns:
        Tuple of (started_rows, completed_rows, users_by_row)
    """
    # Filter by group if specified
    if group:
        changes = [c for c in changes if c.get("Group") == group]

    started_rows: set = set()
    completed_rows: set = set()
    users_by_row: Dict[str, set] = defaultdict(set)

    for change in changes:
        row_id = change.get("RowID")
        phase = change.get("Phase")
        user = change.get("User")

        if not row_id or not phase:
            continue

        try:
            phase_num = int(phase)
        except (ValueError, TypeError):
            continue

        if phase_num == STARTING_PHASE:
            started_rows.add(row_id)
        elif phase_num == COMPLETION_PHASE:
            completed_rows.add(row_id)
            if user:
                users_by_row[row_id].add(user)

    return started_rows, completed_rows, users_by_row


def calculate_completion_rate(
    changes: List[Dict[str, Any]],
    group: Optional[str] = None
) -> CompletionRateMetrics:
    """Calculate completion rate from change history data.

    This function analyzes change records to determine how many items
    started (reached Phase 1) and how many completed (reached Phase 4).

    Args:
        changes: List of change records from historical data
        group: Optional group to filter by. If None, analyzes all groups

    Returns:
        CompletionRateMetrics object with completion analysis

    Example:
        >>> from historical_data_loader import load_change_history
        >>> changes = load_change_history()
        >>> metrics = calculate_completion_rate(changes)
        >>> print(f"Completion rate: {metrics.completion_rate}%")
    """
    if not changes:
        return CompletionRateMetrics()

    started_rows, completed_rows, users_by_row = _extract_phase_data(changes, group)

    # Get date range from changes
    dates = [c.get('ParsedTimestamp') or c.get('ParsedDate') for c in changes]
    dates = [d.date() if hasattr(d, 'date') else d for d in dates if d]
    date_range_start = min(dates) if dates else None
    date_range_end = max(dates) if dates else None

    # Count unique users who processed completed items
    unique_users = set()
    for row_users in users_by_row.values():
        unique_users.update(row_users)

    return CompletionRateMetrics(
        items_started=len(started_rows),
        items_completed=len(completed_rows),
        date_range_start=date_range_start,
        date_range_end=date_range_end,
        unique_users=len(unique_users),
    )


def calculate_completion_for_range(
    date_range_filter: Any,  # DateRangeFilter type
    group: Optional[str] = None,
    changes: Optional[List[Dict[str, Any]]] = None
) -> CompletionRateMetrics:
    """Calculate completion rate for a specific date range.

    This function integrates with DateRangeFilter to ensure completion
    calculations respect the custom date range parameters.

    Args:
        date_range_filter: DateRangeFilter object specifying the date range
        group: Optional group to filter by
        changes: Optional pre-loaded changes (will load if not provided)

    Returns:
        CompletionRateMetrics object with completion analysis for the range

    Example:
        >>> from date_range_filter import create_date_range
        >>> custom_range = create_date_range("2026-01-01", "2026-01-15")
        >>> metrics = calculate_completion_for_range(custom_range)
        >>> print(f"Completion: {metrics.completion_rate}%")
    """
    # Load changes if not provided
    if changes is None:
        from historical_data_loader import load_change_history
        changes = load_change_history(
            start_date=date_range_filter.start_date,
            end_date=date_range_filter.end_date
        )
    else:
        # Filter existing changes by date range
        from historical_data_loader import filter_by_date_range_filter
        changes = filter_by_date_range_filter(changes, date_range_filter)

    metrics = calculate_completion_rate(changes, group)
    metrics.date_range_start = date_range_filter.start_date
    metrics.date_range_end = date_range_filter.end_date

    return metrics


def calculate_completion_by_group(
    changes: List[Dict[str, Any]]
) -> Dict[str, GroupCompletionRate]:
    """Calculate completion rates for each group.

    Args:
        changes: List of change records from historical data

    Returns:
        Dictionary mapping group names to GroupCompletionRate objects

    Example:
        >>> from historical_data_loader import load_change_history
        >>> changes = load_change_history()
        >>> by_group = calculate_completion_by_group(changes)
        >>> for group, data in by_group.items():
        ...     print(f"{group}: {data.metrics.completion_rate}%")
    """
    # Get unique groups
    groups = set(c.get("Group") for c in changes if c.get("Group"))

    # Calculate metrics for each group
    group_metrics = {}
    for group in groups:
        metrics = calculate_completion_rate(changes, group)
        group_metrics[group] = GroupCompletionRate(
            group=group,
            metrics=metrics,
        )

    # Assign ranks (1 = highest completion rate)
    sorted_groups = sorted(
        group_metrics.values(),
        key=lambda x: x.metrics.completion_rate,
        reverse=True
    )
    for rank, group_data in enumerate(sorted_groups, start=1):
        group_data.rank = rank

    return group_metrics


def calculate_completion_by_group_for_range(
    date_range_filter: Any,  # DateRangeFilter type
    changes: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, GroupCompletionRate]:
    """Calculate completion rates by group for a specific date range.

    Args:
        date_range_filter: DateRangeFilter object specifying the date range
        changes: Optional pre-loaded changes (will load if not provided)

    Returns:
        Dictionary mapping group names to GroupCompletionRate objects
    """
    # Load changes if not provided
    if changes is None:
        from historical_data_loader import load_change_history
        changes = load_change_history(
            start_date=date_range_filter.start_date,
            end_date=date_range_filter.end_date
        )
    else:
        from historical_data_loader import filter_by_date_range_filter
        changes = filter_by_date_range_filter(changes, date_range_filter)

    return calculate_completion_by_group(changes)


def calculate_completion_by_time_period(
    changes: List[Dict[str, Any]],
    period_type: str = "week",
    group: Optional[str] = None
) -> List[TimePeriodCompletion]:
    """Calculate completion rates broken down by time period.

    Args:
        changes: List of change records from historical data
        period_type: Type of period ("day", "week", or "month")
        group: Optional group to filter by

    Returns:
        List of TimePeriodCompletion objects, sorted by period start date

    Example:
        >>> from historical_data_loader import load_change_history
        >>> changes = load_change_history()
        >>> weekly = calculate_completion_by_time_period(changes, "week")
        >>> for period in weekly:
        ...     print(f"{period.period_label}: {period.metrics.completion_rate}%")
    """
    if not changes:
        return []

    # Filter by group if specified
    if group:
        changes = [c for c in changes if c.get("Group") == group]

    if not changes:
        return []

    # Get date range
    dates = [c.get('ParsedTimestamp') or c.get('ParsedDate') for c in changes]
    dates = [d.date() if hasattr(d, 'date') else d for d in dates if d]
    if not dates:
        return []

    min_date = min(dates)
    max_date = max(dates)

    # Generate periods
    periods = _generate_periods(min_date, max_date, period_type)

    # Calculate completion for each period
    results = []
    previous_rate = None

    for period_start, period_end, period_label in periods:
        # Filter changes for this period
        period_changes = [
            c for c in changes
            if _change_in_period(c, period_start, period_end)
        ]

        metrics = calculate_completion_rate(period_changes, group=None)
        metrics.date_range_start = period_start
        metrics.date_range_end = period_end

        # Calculate change from previous period
        change_from_previous = 0.0
        trend = TrendDirection.NO_DATA
        if previous_rate is not None:
            change_from_previous = round(metrics.completion_rate - previous_rate, 2)
            trend = get_trend_direction(change_from_previous, threshold=2.0)

        period_completion = TimePeriodCompletion(
            period_label=period_label,
            period_start=period_start,
            period_end=period_end,
            metrics=metrics,
            change_from_previous=change_from_previous,
            trend=trend,
        )
        results.append(period_completion)
        previous_rate = metrics.completion_rate

    return results


def _generate_periods(
    start_date: date,
    end_date: date,
    period_type: str
) -> List[Tuple[date, date, str]]:
    """Generate time periods between start and end dates.

    Args:
        start_date: Start of the date range
        end_date: End of the date range
        period_type: Type of period ("day", "week", or "month")

    Returns:
        List of tuples (period_start, period_end, label)
    """
    periods = []
    current = start_date

    if period_type == "day":
        while current <= end_date:
            label = current.strftime("%Y-%m-%d")
            periods.append((current, current, label))
            current += timedelta(days=1)

    elif period_type == "week":
        # Start from the Monday of the start_date's week
        week_start = current - timedelta(days=current.weekday())
        week_num = 1

        while week_start <= end_date:
            week_end = min(week_start + timedelta(days=6), end_date)
            label = f"Week {week_num} ({week_start.strftime('%m/%d')})"
            periods.append((max(week_start, start_date), week_end, label))
            week_start += timedelta(days=7)
            week_num += 1

    elif period_type == "month":
        while current <= end_date:
            # Get last day of month
            if current.month == 12:
                next_month = date(current.year + 1, 1, 1)
            else:
                next_month = date(current.year, current.month + 1, 1)
            month_end = min(next_month - timedelta(days=1), end_date)

            label = current.strftime("%B %Y")
            periods.append((current, month_end, label))

            current = next_month

    else:
        raise ValueError(f"Invalid period_type: {period_type}. Use 'day', 'week', or 'month'.")

    return periods


def _change_in_period(change: Dict[str, Any], start: date, end: date) -> bool:
    """Check if a change falls within a date period."""
    ts = change.get('ParsedTimestamp') or change.get('ParsedDate')
    if not ts:
        return False
    change_date = ts.date() if hasattr(ts, 'date') else ts
    return start <= change_date <= end


def compare_completion_periods(
    current_changes: List[Dict[str, Any]],
    previous_changes: List[Dict[str, Any]],
    group: Optional[str] = None
) -> CompletionRateComparison:
    """Compare completion rates between two periods.

    Args:
        current_changes: Change records for the current period
        previous_changes: Change records for the previous period
        group: Optional group to filter by

    Returns:
        CompletionRateComparison object with comparison analysis

    Example:
        >>> from historical_data_loader import load_change_history
        >>> from datetime import date, timedelta
        >>> current = load_change_history(start_date=date.today() - timedelta(days=7))
        >>> previous = load_change_history(
        ...     start_date=date.today() - timedelta(days=14),
        ...     end_date=date.today() - timedelta(days=8)
        ... )
        >>> comparison = compare_completion_periods(current, previous)
        >>> print(f"Trend: {comparison.trend.value}")
    """
    current_metrics = calculate_completion_rate(current_changes, group)
    previous_metrics = calculate_completion_rate(previous_changes, group)

    return CompletionRateComparison(
        current_period=current_metrics,
        previous_period=previous_metrics,
    )


def compare_completion_for_range(
    date_range_filter: Any,  # DateRangeFilter type
    group: Optional[str] = None,
    changes: Optional[List[Dict[str, Any]]] = None
) -> CompletionRateComparison:
    """Compare completion rates for a date range against its previous period.

    Args:
        date_range_filter: DateRangeFilter object specifying the current period
        group: Optional group to filter by
        changes: Optional pre-loaded changes (will load if not provided)

    Returns:
        CompletionRateComparison object with comparison analysis

    Example:
        >>> from date_range_filter import get_preset_range, DateRangePreset
        >>> last_week = get_preset_range(DateRangePreset.LAST_WEEK)
        >>> comparison = compare_completion_for_range(last_week)
        >>> print(f"Rate change: {comparison.rate_change} percentage points")
    """
    # Get previous period
    previous_range = date_range_filter.get_previous_period()

    # Load changes if not provided
    if changes is None:
        from historical_data_loader import load_change_history
        # Load changes covering both periods
        all_start = min(date_range_filter.start_date, previous_range.start_date)
        all_end = max(date_range_filter.end_date, previous_range.end_date)
        changes = load_change_history(start_date=all_start, end_date=all_end)

    # Filter changes for each period
    from historical_data_loader import filter_by_date_range_filter
    current_changes = filter_by_date_range_filter(changes, date_range_filter)
    previous_changes = filter_by_date_range_filter(changes, previous_range)

    return compare_completion_periods(current_changes, previous_changes, group)


def get_completion_summary(
    changes: List[Dict[str, Any]],
    include_by_group: bool = True,
    include_time_periods: bool = True,
    period_type: str = "week"
) -> Dict[str, Any]:
    """Get a comprehensive summary of completion rates.

    Args:
        changes: List of change records from historical data
        include_by_group: Whether to include group breakdown
        include_time_periods: Whether to include time period breakdown
        period_type: Type of time period for breakdown ("day", "week", "month")

    Returns:
        Dictionary with comprehensive completion analysis

    Example:
        >>> from historical_data_loader import load_change_history
        >>> changes = load_change_history()
        >>> summary = get_completion_summary(changes)
        >>> print(f"Overall: {summary['overall']['completion_rate']}%")
    """
    # Calculate overall metrics
    overall = calculate_completion_rate(changes)

    summary = {
        "overall": overall.to_dict(),
        "date_range": {
            "start": overall.date_range_start.isoformat() if overall.date_range_start else None,
            "end": overall.date_range_end.isoformat() if overall.date_range_end else None,
        },
    }

    # Add group breakdown
    if include_by_group:
        by_group = calculate_completion_by_group(changes)
        summary["by_group"] = {
            group: data.to_dict() for group, data in by_group.items()
        }

        # Add group rankings
        sorted_groups = sorted(
            by_group.items(),
            key=lambda x: x[1].metrics.completion_rate,
            reverse=True
        )
        summary["group_rankings"] = [
            {"group": g, "completion_rate": d.metrics.completion_rate, "rank": d.rank}
            for g, d in sorted_groups
        ]

        # Identify best and worst performers
        if sorted_groups:
            summary["best_group"] = {
                "group": sorted_groups[0][0],
                "completion_rate": sorted_groups[0][1].metrics.completion_rate,
            }
            summary["worst_group"] = {
                "group": sorted_groups[-1][0],
                "completion_rate": sorted_groups[-1][1].metrics.completion_rate,
            }

    # Add time period breakdown
    if include_time_periods:
        time_periods = calculate_completion_by_time_period(changes, period_type)
        summary["by_time_period"] = [p.to_dict() for p in time_periods]

        # Add trend analysis
        if len(time_periods) >= 2:
            first_rate = time_periods[0].metrics.completion_rate
            last_rate = time_periods[-1].metrics.completion_rate
            overall_trend = last_rate - first_rate

            if overall_trend >= 5:
                trend_label = "improving"
            elif overall_trend <= -5:
                trend_label = "declining"
            else:
                trend_label = "stable"

            summary["trend_analysis"] = {
                "overall_change": round(overall_trend, 2),
                "trend": trend_label,
                "periods_analyzed": len(time_periods),
            }

    return summary


def get_completion_visualization_data(
    changes: List[Dict[str, Any]],
    group: Optional[str] = None
) -> Dict[str, Any]:
    """Get data formatted for completion rate visualization.

    This function prepares data in a format suitable for rendering
    charts in PDF reports or other visualizations.

    Args:
        changes: List of change records
        group: Optional group to filter by

    Returns:
        Dictionary with visualization-ready data

    Example:
        >>> changes = load_change_history()
        >>> viz_data = get_completion_visualization_data(changes)
        >>> # Use viz_data to render a completion chart
    """
    # Calculate overall metrics
    overall = calculate_completion_rate(changes, group)

    # Calculate by group
    by_group = calculate_completion_by_group(changes) if group is None else {}

    # Prepare group data for bar chart
    group_data = []
    if by_group:
        for g, data in sorted(by_group.items(), key=lambda x: x[1].rank):
            rate = data.metrics.completion_rate
            # Assign color based on performance level
            if rate >= 80:
                color = "#1B5E20"  # Green
            elif rate >= 60:
                color = "#8B6914"  # Yellow/Gold
            elif rate >= 40:
                color = "#E65100"  # Orange
            else:
                color = "#B71C1C"  # Red

            group_data.append({
                "group": g,
                "completion_rate": rate,
                "items_started": data.metrics.items_started,
                "items_completed": data.metrics.items_completed,
                "performance_level": data.performance_level,
                "color": color,
            })

    return {
        "overall": {
            "completion_rate": overall.completion_rate,
            "items_started": overall.items_started,
            "items_completed": overall.items_completed,
            "items_incomplete": overall.items_incomplete,
        },
        "by_group": group_data,
        "donut_data": {
            "completed": overall.items_completed,
            "incomplete": overall.items_incomplete,
            "labels": ["Completed", "Incomplete"],
            "colors": ["#1B5E20", "#B71C1C"],
        },
        "date_range": {
            "start": overall.date_range_start.isoformat() if overall.date_range_start else None,
            "end": overall.date_range_end.isoformat() if overall.date_range_end else None,
        },
    }


def format_completion_report(
    changes: List[Dict[str, Any]],
    include_by_group: bool = True
) -> str:
    """Format completion rate metrics as a text report.

    Args:
        changes: List of change records from historical data
        include_by_group: Whether to include group breakdown

    Returns:
        Formatted text report string
    """
    overall = calculate_completion_rate(changes)

    lines = []
    lines.append("=" * 70)
    lines.append("COMPLETION RATE REPORT")
    lines.append("=" * 70)

    # Header
    if overall.date_range_start and overall.date_range_end:
        lines.append(f"\nPeriod: {overall.date_range_start} to {overall.date_range_end}")

    # Overview
    lines.append("\n" + "-" * 70)
    lines.append("OVERVIEW")
    lines.append("-" * 70)
    lines.append(f"  Items Started (Phase 1): {overall.items_started}")
    lines.append(f"  Items Completed (Phase 4): {overall.items_completed}")
    lines.append(f"  Items Incomplete: {overall.items_incomplete}")
    lines.append(f"  Completion Rate: {overall.completion_rate}%")
    lines.append(f"  Unique Users (Completers): {overall.unique_users}")

    # Performance indicator
    if overall.completion_rate >= 80:
        indicator = "EXCELLENT"
    elif overall.completion_rate >= 60:
        indicator = "GOOD"
    elif overall.completion_rate >= 40:
        indicator = "AVERAGE"
    else:
        indicator = "NEEDS IMPROVEMENT"
    lines.append(f"\n  Performance: {indicator}")

    # Group breakdown
    if include_by_group:
        by_group = calculate_completion_by_group(changes)

        if by_group:
            lines.append("\n" + "-" * 70)
            lines.append("BREAKDOWN BY GROUP")
            lines.append("-" * 70)
            lines.append(f"{'Rank':<6} {'Group':<15} {'Started':>10} {'Completed':>12} {'Rate':>10} {'Level':<12}")
            lines.append("-" * 70)

            sorted_groups = sorted(by_group.values(), key=lambda x: x.rank)
            for group_data in sorted_groups:
                m = group_data.metrics
                lines.append(
                    f"{group_data.rank:<6} "
                    f"{group_data.group:<15} "
                    f"{m.items_started:>10} "
                    f"{m.items_completed:>12} "
                    f"{m.completion_rate:>9.1f}% "
                    f"{group_data.performance_level:<12}"
                )

    lines.append("\n" + "=" * 70)

    return "\n".join(lines)


if __name__ == "__main__":
    # Demo usage
    print("Completion Rate Calculator - Demo")
    print("=" * 70)

    # Create sample data
    import random
    sample_changes = []
    groups = ["NA", "NF", "NH", "NM", "NP", "NT", "NV"]

    for i in range(200):
        group = random.choice(groups)
        row_id = f"row_{i}"

        # Phase 1 - all items start
        sample_changes.append({
            "Group": group,
            "RowID": row_id,
            "Phase": "1",
            "User": random.choice(["DM", "JHU", "HI"]),
            "ParsedTimestamp": date.today() - timedelta(days=random.randint(0, 30)),
        })

        # Phase 4 - some items complete (varying by group)
        completion_prob = 0.5 + (0.1 * groups.index(group) / len(groups))
        if random.random() < completion_prob:
            sample_changes.append({
                "Group": group,
                "RowID": row_id,
                "Phase": "4",
                "User": random.choice(["DM", "JHU", "HI"]),
                "ParsedTimestamp": date.today() - timedelta(days=random.randint(0, 30)),
            })

    # Calculate and print report
    print(format_completion_report(sample_changes))

    # Print summary
    print("\nJSON Summary:")
    summary = get_completion_summary(sample_changes, period_type="week")
    import json
    print(json.dumps(summary, indent=2, default=str))

    # Print visualization data
    print("\nVisualization Data:")
    viz_data = get_completion_visualization_data(sample_changes)
    print(json.dumps(viz_data, indent=2))
