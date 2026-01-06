"""
Average Items Per Period Calculator Module

Calculates average items processed per day and per week for each user.
Supports filtering by date ranges and groups.

This module follows the established patterns from user_productivity_calculator.py
and completion_rate_calculator.py for consistency with the existing codebase.

Usage:
    from average_items_per_period_calculator import (
        AverageItemsMetrics,
        UserAverageItems,
        GroupAverageBreakdown,
        WeeklyBreakdown,
        calculate_average_items_per_user,
        calculate_average_items_all_users,
        calculate_average_items_by_group,
        calculate_average_items_for_range,
        calculate_weekly_breakdown,
        get_average_items_summary,
        get_average_items_visualization_data,
        format_average_items_report,
    )
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Any, Union, Tuple
from enum import Enum
from collections import defaultdict

from constants import PerformanceThreshold

# Set up logging
logger = logging.getLogger(__name__)


class PerformanceLevel(Enum):
    """Performance level classification based on average items per day.

    Thresholds are defined in constants.PerformanceThreshold.
    """
    EXCEPTIONAL = "exceptional"  # >= PerformanceThreshold.EXCEPTIONAL items/day
    HIGH = "high"  # >= PerformanceThreshold.HIGH items/day
    AVERAGE = "average"  # >= PerformanceThreshold.AVERAGE items/day
    LOW = "low"  # >= PerformanceThreshold.LOW items/day
    MINIMAL = "minimal"  # < PerformanceThreshold.LOW items/day


@dataclass
class WeeklyBreakdown:
    """Breakdown of items processed for a specific week.

    Attributes:
        week_number: Week number (1-based) within the analysis period
        week_label: Human-readable label (e.g., "Week 1 (01/06)")
        week_start: Start date of the week (Monday)
        week_end: End date of the week (Sunday or end of range)
        total_items: Total items processed in this week
        items_per_day: Average items per day for this week
        active_days: Number of days with activity in this week
    """
    week_number: int
    week_label: str
    week_start: date
    week_end: date
    total_items: int = 0
    items_per_day: float = 0.0
    active_days: int = 0

    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        days_in_week = (self.week_end - self.week_start).days + 1
        if self.total_items > 0 and days_in_week > 0:
            self.items_per_day = self.total_items / days_in_week

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "week_number": self.week_number,
            "week_label": self.week_label,
            "week_start": self.week_start.isoformat(),
            "week_end": self.week_end.isoformat(),
            "total_items": self.total_items,
            "items_per_day": round(self.items_per_day, 2),
            "active_days": self.active_days,
        }


@dataclass
class GroupAverageBreakdown:
    """Average items metrics for a specific group.

    Attributes:
        group: Group code (NA, NF, NH, etc.)
        total_items: Total items processed in this group
        items_per_day: Average items per day for this group
        items_per_week: Average items per week for this group
        percentage_of_total: What percentage of all items this group represents
    """
    group: str
    total_items: int = 0
    items_per_day: float = 0.0
    items_per_week: float = 0.0
    percentage_of_total: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "group": self.group,
            "total_items": self.total_items,
            "items_per_day": round(self.items_per_day, 2),
            "items_per_week": round(self.items_per_week, 2),
            "percentage_of_total": round(self.percentage_of_total, 2),
        }


@dataclass
class UserAverageItems:
    """Average items per period metrics for a specific user.

    Attributes:
        user: User identifier (initials like "JHU", "DM")
        total_items: Total items processed by this user
        items_per_day: Average items processed per day
        items_per_week: Average items processed per week
        active_days: Number of days with at least one activity
        active_weeks: Number of weeks with at least one activity
        date_range_start: Start date of the analysis period
        date_range_end: End date of the analysis period
        performance_level: Classification of performance level
        weekly_breakdown: Per-week breakdown of items
        group_breakdown: Per-group breakdown of items
    """
    user: str
    total_items: int = 0
    items_per_day: float = 0.0
    items_per_week: float = 0.0
    active_days: int = 0
    active_weeks: int = 0
    date_range_start: Optional[date] = None
    date_range_end: Optional[date] = None
    performance_level: PerformanceLevel = PerformanceLevel.AVERAGE
    weekly_breakdown: List[WeeklyBreakdown] = field(default_factory=list)
    group_breakdown: Dict[str, GroupAverageBreakdown] = field(default_factory=dict)

    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        if self.performance_level == PerformanceLevel.AVERAGE and self.items_per_day > 0:
            self.performance_level = self._calculate_performance_level()

    def _calculate_performance_level(self) -> PerformanceLevel:
        """Determine performance level based on items per day."""
        if self.items_per_day >= PerformanceThreshold.EXCEPTIONAL:
            return PerformanceLevel.EXCEPTIONAL
        elif self.items_per_day >= PerformanceThreshold.HIGH:
            return PerformanceLevel.HIGH
        elif self.items_per_day >= PerformanceThreshold.AVERAGE:
            return PerformanceLevel.AVERAGE
        elif self.items_per_day >= PerformanceThreshold.LOW:
            return PerformanceLevel.LOW
        else:
            return PerformanceLevel.MINIMAL

    @property
    def days_in_range(self) -> int:
        """Get the number of days in the analysis range."""
        if self.date_range_start and self.date_range_end:
            return (self.date_range_end - self.date_range_start).days + 1
        return 0

    @property
    def weeks_in_range(self) -> float:
        """Get the number of weeks in the analysis range."""
        return self.days_in_range / 7.0

    @property
    def activity_ratio(self) -> float:
        """Get the ratio of active days to total days in range (0-1)."""
        if self.days_in_range > 0:
            return self.active_days / self.days_in_range
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "user": self.user,
            "total_items": self.total_items,
            "items_per_day": round(self.items_per_day, 2),
            "items_per_week": round(self.items_per_week, 2),
            "active_days": self.active_days,
            "active_weeks": self.active_weeks,
            "date_range_start": self.date_range_start.isoformat() if self.date_range_start else None,
            "date_range_end": self.date_range_end.isoformat() if self.date_range_end else None,
            "days_in_range": self.days_in_range,
            "weeks_in_range": round(self.weeks_in_range, 2),
            "activity_ratio": round(self.activity_ratio, 2),
            "performance_level": self.performance_level.value,
            "weekly_breakdown": [w.to_dict() for w in self.weekly_breakdown],
            "group_breakdown": {g: b.to_dict() for g, b in self.group_breakdown.items()},
        }


@dataclass
class AverageItemsMetrics:
    """Overall average items metrics across all users or for a specific scope.

    Attributes:
        total_items: Total items processed
        total_users: Number of users in the analysis
        overall_items_per_day: Average items per day across all users combined
        overall_items_per_week: Average items per week across all users combined
        avg_items_per_user_per_day: Average items per day per user
        avg_items_per_user_per_week: Average items per week per user
        date_range_start: Start date of the analysis period
        date_range_end: End date of the analysis period
        users: Dictionary mapping users to their individual metrics
        groups: Dictionary mapping groups to their aggregate metrics
    """
    total_items: int = 0
    total_users: int = 0
    overall_items_per_day: float = 0.0
    overall_items_per_week: float = 0.0
    avg_items_per_user_per_day: float = 0.0
    avg_items_per_user_per_week: float = 0.0
    date_range_start: Optional[date] = None
    date_range_end: Optional[date] = None
    users: Dict[str, UserAverageItems] = field(default_factory=dict)
    groups: Dict[str, GroupAverageBreakdown] = field(default_factory=dict)

    @property
    def days_in_range(self) -> int:
        """Get the number of days in the analysis range."""
        if self.date_range_start and self.date_range_end:
            return (self.date_range_end - self.date_range_start).days + 1
        return 0

    @property
    def weeks_in_range(self) -> float:
        """Get the number of weeks in the analysis range."""
        return self.days_in_range / 7.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_items": self.total_items,
            "total_users": self.total_users,
            "overall_items_per_day": round(self.overall_items_per_day, 2),
            "overall_items_per_week": round(self.overall_items_per_week, 2),
            "avg_items_per_user_per_day": round(self.avg_items_per_user_per_day, 2),
            "avg_items_per_user_per_week": round(self.avg_items_per_user_per_week, 2),
            "date_range_start": self.date_range_start.isoformat() if self.date_range_start else None,
            "date_range_end": self.date_range_end.isoformat() if self.date_range_end else None,
            "days_in_range": self.days_in_range,
            "weeks_in_range": round(self.weeks_in_range, 2),
            "users": {u: m.to_dict() for u, m in self.users.items()},
            "groups": {g: b.to_dict() for g, b in self.groups.items()},
        }


def _get_date_range(changes: List[Dict[str, Any]]) -> Tuple[Optional[date], Optional[date]]:
    """Extract the date range from a list of changes.

    Args:
        changes: List of change records

    Returns:
        Tuple of (start_date, end_date)
    """
    dates = []
    for c in changes:
        ts = c.get('ParsedTimestamp') or c.get('ParsedDate')
        if ts:
            d = ts.date() if hasattr(ts, 'date') else ts
            dates.append(d)

    if dates:
        return min(dates), max(dates)
    return None, None


def _generate_week_periods(
    start_date: date,
    end_date: date
) -> List[Tuple[date, date, str, int]]:
    """Generate week periods between start and end dates.

    Weeks start on Monday (ISO standard).

    Args:
        start_date: Start of the date range
        end_date: End of the date range

    Returns:
        List of tuples (week_start, week_end, label, week_number)
    """
    periods = []
    # Start from the Monday of the start_date's week
    week_start = start_date - timedelta(days=start_date.weekday())
    week_num = 1

    while week_start <= end_date:
        week_end = week_start + timedelta(days=6)  # Sunday
        # Clamp to the actual range
        actual_start = max(week_start, start_date)
        actual_end = min(week_end, end_date)
        label = f"Week {week_num} ({actual_start.strftime('%m/%d')})"
        periods.append((actual_start, actual_end, label, week_num))
        week_start += timedelta(days=7)
        week_num += 1

    return periods


def _extract_user_activity(
    changes: List[Dict[str, Any]],
    user: Optional[str] = None
) -> Tuple[Dict[str, List[Dict]], Dict[str, set], Dict[str, set], Dict[str, set]]:
    """Extract user activity data from changes.

    Args:
        changes: List of change records
        user: Optional specific user to filter for

    Returns:
        Tuple of:
        - user_changes: Dict mapping users to their change records
        - user_groups: Dict mapping users to groups they worked in
        - user_active_dates: Dict mapping users to dates they were active
        - user_active_weeks: Dict mapping users to week start dates they were active
    """
    user_changes: Dict[str, List[Dict]] = defaultdict(list)
    user_groups: Dict[str, set] = defaultdict(set)
    user_active_dates: Dict[str, set] = defaultdict(set)
    user_active_weeks: Dict[str, set] = defaultdict(set)

    for change in changes:
        change_user = change.get("User")
        if not change_user:
            continue

        # Filter for specific user if requested
        if user and change_user != user:
            continue

        user_changes[change_user].append(change)

        group = change.get("Group")
        if group:
            user_groups[change_user].add(group)

        # Extract date for active day/week tracking
        ts = change.get('ParsedTimestamp') or change.get('ParsedDate')
        if ts:
            activity_date = ts.date() if hasattr(ts, 'date') else ts
            user_active_dates[change_user].add(activity_date)
            # Week starts on Monday
            week_start = activity_date - timedelta(days=activity_date.weekday())
            user_active_weeks[change_user].add(week_start)

    return user_changes, user_groups, user_active_dates, user_active_weeks


def calculate_weekly_breakdown(
    changes: List[Dict[str, Any]],
    user: Optional[str] = None
) -> List[WeeklyBreakdown]:
    """Calculate weekly breakdown of items processed.

    Args:
        changes: List of change records
        user: Optional user to filter for

    Returns:
        List of WeeklyBreakdown objects, one per week in the range
    """
    if not changes:
        return []

    # Filter by user if specified
    if user:
        changes = [c for c in changes if c.get("User") == user]

    if not changes:
        return []

    # Get date range
    start_date, end_date = _get_date_range(changes)
    if not start_date or not end_date:
        return []

    # Generate week periods
    week_periods = _generate_week_periods(start_date, end_date)

    # Count items per week
    weekly_items: Dict[int, int] = defaultdict(int)
    weekly_active_days: Dict[int, set] = defaultdict(set)

    for change in changes:
        ts = change.get('ParsedTimestamp') or change.get('ParsedDate')
        if not ts:
            continue
        change_date = ts.date() if hasattr(ts, 'date') else ts

        # Find which week this change belongs to
        for week_start, week_end, label, week_num in week_periods:
            if week_start <= change_date <= week_end:
                weekly_items[week_num] += 1
                weekly_active_days[week_num].add(change_date)
                break

    # Build WeeklyBreakdown objects
    result = []
    for week_start, week_end, label, week_num in week_periods:
        breakdown = WeeklyBreakdown(
            week_number=week_num,
            week_label=label,
            week_start=week_start,
            week_end=week_end,
            total_items=weekly_items.get(week_num, 0),
            active_days=len(weekly_active_days.get(week_num, set())),
        )
        result.append(breakdown)

    return result


def calculate_average_items_per_user(
    changes: List[Dict[str, Any]],
    user: str,
    include_weekly_breakdown: bool = True,
    include_group_breakdown: bool = True
) -> UserAverageItems:
    """Calculate average items per day and per week for a specific user.

    Args:
        changes: List of change records from historical data
        user: User identifier to calculate metrics for
        include_weekly_breakdown: Whether to include per-week breakdown
        include_group_breakdown: Whether to include per-group breakdown

    Returns:
        UserAverageItems object with comprehensive metrics

    Example:
        >>> from historical_data_loader import load_change_history
        >>> changes = load_change_history()
        >>> metrics = calculate_average_items_per_user(changes, "JHU")
        >>> print(f"Items per day: {metrics.items_per_day}")
        >>> print(f"Items per week: {metrics.items_per_week}")
    """
    if not changes:
        return UserAverageItems(user=user)

    # Extract user activity
    user_changes, user_groups, user_active_dates, user_active_weeks = _extract_user_activity(
        changes, user
    )

    if user not in user_changes:
        return UserAverageItems(user=user)

    user_records = user_changes[user]

    # Get date range
    start_date, end_date = _get_date_range(changes)
    days_in_range = (end_date - start_date).days + 1 if start_date and end_date else 1
    weeks_in_range = days_in_range / 7.0

    # Calculate metrics
    total_items = len(user_records)
    items_per_day = total_items / max(days_in_range, 1)
    items_per_week = total_items / max(weeks_in_range, 1)

    # Active days and weeks
    active_days = len(user_active_dates[user])
    active_weeks = len(user_active_weeks[user])

    # Build result
    result = UserAverageItems(
        user=user,
        total_items=total_items,
        items_per_day=items_per_day,
        items_per_week=items_per_week,
        active_days=active_days,
        active_weeks=active_weeks,
        date_range_start=start_date,
        date_range_end=end_date,
    )

    # Weekly breakdown
    if include_weekly_breakdown:
        result.weekly_breakdown = calculate_weekly_breakdown(changes, user)

    # Group breakdown
    if include_group_breakdown:
        group_items: Dict[str, int] = defaultdict(int)
        for record in user_records:
            group = record.get("Group")
            if group:
                group_items[group] += 1

        for group, count in group_items.items():
            result.group_breakdown[group] = GroupAverageBreakdown(
                group=group,
                total_items=count,
                items_per_day=count / max(days_in_range, 1),
                items_per_week=count / max(weeks_in_range, 1),
                percentage_of_total=(count / total_items * 100) if total_items > 0 else 0.0,
            )

    return result


def calculate_average_items_all_users(
    changes: List[Dict[str, Any]],
    include_weekly_breakdown: bool = True,
    include_group_breakdown: bool = True
) -> Dict[str, UserAverageItems]:
    """Calculate average items metrics for all users in the data.

    Args:
        changes: List of change records from historical data
        include_weekly_breakdown: Whether to include per-week breakdown
        include_group_breakdown: Whether to include per-group breakdown

    Returns:
        Dictionary mapping user identifiers to UserAverageItems objects

    Example:
        >>> from historical_data_loader import load_change_history
        >>> changes = load_change_history()
        >>> all_metrics = calculate_average_items_all_users(changes)
        >>> for user, metrics in all_metrics.items():
        ...     print(f"{user}: {metrics.items_per_day:.1f} items/day, {metrics.items_per_week:.1f} items/week")
    """
    if not changes:
        return {}

    # Get all unique users
    users = set(c.get("User") for c in changes if c.get("User"))

    result = {}
    for user in users:
        result[user] = calculate_average_items_per_user(
            changes,
            user,
            include_weekly_breakdown=include_weekly_breakdown,
            include_group_breakdown=include_group_breakdown
        )

    return result


def calculate_average_items_by_group(
    changes: List[Dict[str, Any]],
    groups: Optional[Union[str, List[str]]] = None
) -> Dict[str, AverageItemsMetrics]:
    """Calculate average items metrics grouped by product group.

    Args:
        changes: List of change records
        groups: Optional specific group(s) to filter for

    Returns:
        Dictionary mapping group codes to AverageItemsMetrics

    Example:
        >>> from historical_data_loader import load_change_history
        >>> changes = load_change_history()
        >>> by_group = calculate_average_items_by_group(changes)
        >>> for group, metrics in by_group.items():
        ...     print(f"{group}: {metrics.overall_items_per_day:.1f} items/day")
    """
    if not changes:
        return {}

    # Filter by groups if specified
    if groups:
        if isinstance(groups, str):
            groups = [groups]
        groups_set = set(groups)
        changes = [c for c in changes if c.get("Group") in groups_set]

    if not changes:
        return {}

    # Get date range
    start_date, end_date = _get_date_range(changes)
    days_in_range = (end_date - start_date).days + 1 if start_date and end_date else 1
    weeks_in_range = days_in_range / 7.0

    # Group changes by group code
    by_group: Dict[str, List[Dict]] = defaultdict(list)
    for change in changes:
        group = change.get("Group")
        if group:
            by_group[group].append(change)

    result = {}
    for group, group_changes in by_group.items():
        # Get users in this group
        user_metrics = calculate_average_items_all_users(
            group_changes,
            include_weekly_breakdown=False,
            include_group_breakdown=False
        )

        total_items = len(group_changes)

        metrics = AverageItemsMetrics(
            total_items=total_items,
            total_users=len(user_metrics),
            overall_items_per_day=total_items / max(days_in_range, 1),
            overall_items_per_week=total_items / max(weeks_in_range, 1),
            avg_items_per_user_per_day=sum(m.items_per_day for m in user_metrics.values()) / max(len(user_metrics), 1),
            avg_items_per_user_per_week=sum(m.items_per_week for m in user_metrics.values()) / max(len(user_metrics), 1),
            date_range_start=start_date,
            date_range_end=end_date,
            users=user_metrics,
        )
        result[group] = metrics

    return result


def calculate_average_items_for_range(
    date_range_filter: Any,  # DateRangeFilter type
    user: Optional[str] = None,
    groups: Optional[Union[str, List[str]]] = None,
    changes: Optional[List[Dict[str, Any]]] = None
) -> Union[UserAverageItems, AverageItemsMetrics]:
    """Calculate average items metrics for a specific date range.

    Args:
        date_range_filter: DateRangeFilter object specifying the date range
        user: Optional specific user to analyze
        groups: Optional group(s) to filter by
        changes: Optional pre-loaded changes (will load if not provided)

    Returns:
        If user specified: UserAverageItems for that user
        If user is None: AverageItemsMetrics with all users

    Example:
        >>> from date_range_filter import get_preset_range, DateRangePreset
        >>> last_week = get_preset_range(DateRangePreset.LAST_WEEK)
        >>> metrics = calculate_average_items_for_range(last_week, user="JHU")
        >>> print(f"Items per day: {metrics.items_per_day}")
        >>> print(f"Items per week: {metrics.items_per_week}")
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

    # Filter by groups if specified
    if groups:
        if isinstance(groups, str):
            groups = [groups]
        groups_set = set(groups)
        changes = [c for c in changes if c.get("Group") in groups_set]

    if user:
        metrics = calculate_average_items_per_user(changes, user)
        metrics.date_range_start = date_range_filter.start_date
        metrics.date_range_end = date_range_filter.end_date
        return metrics
    else:
        return get_average_items_summary(
            changes,
            date_range_start=date_range_filter.start_date,
            date_range_end=date_range_filter.end_date
        )


def get_average_items_summary(
    changes: List[Dict[str, Any]],
    date_range_start: Optional[date] = None,
    date_range_end: Optional[date] = None
) -> AverageItemsMetrics:
    """Get comprehensive summary of average items metrics.

    Args:
        changes: List of change records from historical data
        date_range_start: Optional override for start date
        date_range_end: Optional override for end date

    Returns:
        AverageItemsMetrics object with comprehensive analysis

    Example:
        >>> from historical_data_loader import load_change_history
        >>> changes = load_change_history()
        >>> summary = get_average_items_summary(changes)
        >>> print(f"Total items: {summary.total_items}")
        >>> print(f"Overall items/day: {summary.overall_items_per_day:.1f}")
    """
    if not changes:
        return AverageItemsMetrics()

    # Get date range
    if date_range_start and date_range_end:
        start_date, end_date = date_range_start, date_range_end
    else:
        start_date, end_date = _get_date_range(changes)

    days_in_range = (end_date - start_date).days + 1 if start_date and end_date else 1
    weeks_in_range = days_in_range / 7.0

    # Calculate per-user metrics
    user_metrics = calculate_average_items_all_users(changes)

    # Calculate per-group metrics
    group_items: Dict[str, int] = defaultdict(int)
    for change in changes:
        group = change.get("Group")
        if group:
            group_items[group] += 1

    total_items = len(changes)
    total_users = len(user_metrics)

    group_breakdown = {}
    for group, count in group_items.items():
        group_breakdown[group] = GroupAverageBreakdown(
            group=group,
            total_items=count,
            items_per_day=count / max(days_in_range, 1),
            items_per_week=count / max(weeks_in_range, 1),
            percentage_of_total=(count / total_items * 100) if total_items > 0 else 0.0,
        )

    return AverageItemsMetrics(
        total_items=total_items,
        total_users=total_users,
        overall_items_per_day=total_items / max(days_in_range, 1),
        overall_items_per_week=total_items / max(weeks_in_range, 1),
        avg_items_per_user_per_day=sum(m.items_per_day for m in user_metrics.values()) / max(total_users, 1),
        avg_items_per_user_per_week=sum(m.items_per_week for m in user_metrics.values()) / max(total_users, 1),
        date_range_start=start_date,
        date_range_end=end_date,
        users=user_metrics,
        groups=group_breakdown,
    )


def get_average_items_visualization_data(
    changes: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Get data formatted for average items visualization.

    This function prepares data in a format suitable for rendering
    charts in PDF reports or other visualizations.

    Args:
        changes: List of change records

    Returns:
        Dictionary with visualization-ready data

    Example:
        >>> changes = load_change_history()
        >>> viz_data = get_average_items_visualization_data(changes)
        >>> # Use viz_data to render average items charts
    """
    summary = get_average_items_summary(changes)

    # Prepare bar chart data for items per day (sorted by items_per_day descending)
    sorted_users = sorted(
        summary.users.items(),
        key=lambda x: x[1].items_per_day,
        reverse=True
    )

    bar_chart_data = []
    for rank, (user, metrics) in enumerate(sorted_users, start=1):
        level = metrics.performance_level
        # Assign colors based on performance level
        color_map = {
            PerformanceLevel.EXCEPTIONAL: "#1B5E20",  # Green
            PerformanceLevel.HIGH: "#388E3C",  # Light Green
            PerformanceLevel.AVERAGE: "#8B6914",  # Yellow/Gold
            PerformanceLevel.LOW: "#E65100",  # Orange
            PerformanceLevel.MINIMAL: "#B71C1C",  # Red
        }

        bar_chart_data.append({
            "user": user,
            "items_per_day": metrics.items_per_day,
            "items_per_week": metrics.items_per_week,
            "total_items": metrics.total_items,
            "rank": rank,
            "performance_level": level.value,
            "color": color_map.get(level, "#757575"),
        })

    # Prepare weekly trend data
    weekly_breakdown = calculate_weekly_breakdown(changes)
    weekly_trend_data = [w.to_dict() for w in weekly_breakdown]

    # Prepare group distribution data
    group_distribution = [
        {
            "group": group,
            "total_items": breakdown.total_items,
            "items_per_day": breakdown.items_per_day,
            "items_per_week": breakdown.items_per_week,
            "percentage": breakdown.percentage_of_total,
        }
        for group, breakdown in sorted(summary.groups.items())
    ]

    return {
        "bar_chart_data": bar_chart_data,
        "weekly_trend_data": weekly_trend_data,
        "group_distribution": group_distribution,
        "summary": {
            "total_users": summary.total_users,
            "total_items_processed": summary.total_items,
            "overall_items_per_day": round(summary.overall_items_per_day, 2),
            "overall_items_per_week": round(summary.overall_items_per_week, 2),
            "avg_items_per_user_per_day": round(summary.avg_items_per_user_per_day, 2),
            "avg_items_per_user_per_week": round(summary.avg_items_per_user_per_week, 2),
            "days_in_range": summary.days_in_range,
            "weeks_in_range": round(summary.weeks_in_range, 2),
        },
    }


def format_average_items_report(
    changes: List[Dict[str, Any]],
    include_weekly_breakdown: bool = True,
    include_group_breakdown: bool = True
) -> str:
    """Format average items metrics as a text report.

    Args:
        changes: List of change records from historical data
        include_weekly_breakdown: Whether to include per-week breakdown
        include_group_breakdown: Whether to include per-group breakdown

    Returns:
        Formatted text report string
    """
    summary = get_average_items_summary(changes)

    lines = []
    lines.append("=" * 80)
    lines.append("AVERAGE ITEMS PER PERIOD REPORT")
    lines.append("=" * 80)

    if summary.total_items == 0:
        lines.append("\nNo data available.")
        return "\n".join(lines)

    # Date range info
    if summary.date_range_start and summary.date_range_end:
        lines.append(f"\nPeriod: {summary.date_range_start} to {summary.date_range_end}")
        lines.append(f"Days in period: {summary.days_in_range}")
        lines.append(f"Weeks in period: {summary.weeks_in_range:.1f}")

    # Summary
    lines.append("\n" + "-" * 80)
    lines.append("OVERALL SUMMARY")
    lines.append("-" * 80)
    lines.append(f"  Total Items Processed: {summary.total_items}")
    lines.append(f"  Total Users: {summary.total_users}")
    lines.append(f"  Overall Items/Day: {summary.overall_items_per_day:.2f}")
    lines.append(f"  Overall Items/Week: {summary.overall_items_per_week:.2f}")
    lines.append(f"  Avg Items/User/Day: {summary.avg_items_per_user_per_day:.2f}")
    lines.append(f"  Avg Items/User/Week: {summary.avg_items_per_user_per_week:.2f}")

    # User rankings
    lines.append("\n" + "-" * 80)
    lines.append("USER RANKINGS (by Items/Day)")
    lines.append("-" * 80)
    lines.append(f"{'Rank':<6} {'User':<10} {'Items/Day':>12} {'Items/Week':>12} {'Total':>10} {'Level':<15}")
    lines.append("-" * 80)

    sorted_users = sorted(
        summary.users.items(),
        key=lambda x: x[1].items_per_day,
        reverse=True
    )

    for rank, (user, metrics) in enumerate(sorted_users, start=1):
        lines.append(
            f"{rank:<6} "
            f"{user:<10} "
            f"{metrics.items_per_day:>12.2f} "
            f"{metrics.items_per_week:>12.2f} "
            f"{metrics.total_items:>10} "
            f"{metrics.performance_level.value:<15}"
        )

    # Weekly breakdown
    if include_weekly_breakdown:
        weekly_breakdown = calculate_weekly_breakdown(changes)
        if weekly_breakdown:
            lines.append("\n" + "-" * 80)
            lines.append("WEEKLY BREAKDOWN")
            lines.append("-" * 80)
            lines.append(f"{'Week':<25} {'Total Items':>12} {'Items/Day':>12} {'Active Days':>12}")
            lines.append("-" * 80)

            for week in weekly_breakdown:
                lines.append(
                    f"{week.week_label:<25} "
                    f"{week.total_items:>12} "
                    f"{week.items_per_day:>12.2f} "
                    f"{week.active_days:>12}"
                )

    # Group breakdown
    if include_group_breakdown:
        lines.append("\n" + "-" * 80)
        lines.append("BY GROUP")
        lines.append("-" * 80)
        lines.append(f"{'Group':<15} {'Total Items':>12} {'Items/Day':>12} {'Items/Week':>12} {'% of Total':>12}")
        lines.append("-" * 80)

        for group in sorted(summary.groups.keys()):
            breakdown = summary.groups[group]
            lines.append(
                f"{group:<15} "
                f"{breakdown.total_items:>12} "
                f"{breakdown.items_per_day:>12.2f} "
                f"{breakdown.items_per_week:>12.2f} "
                f"{breakdown.percentage_of_total:>11.1f}%"
            )

    lines.append("\n" + "=" * 80)

    return "\n".join(lines)


if __name__ == "__main__":
    # Demo usage
    print("Average Items Per Period Calculator - Demo")
    print("=" * 80)

    # Create sample data
    import random
    sample_changes = []
    groups = ["NA", "NF", "NH", "NM", "NP", "NT", "NV"]
    users = ["DM", "JHU", "HI", "MK", "AS"]

    for i in range(500):
        group = random.choice(groups)
        user = random.choice(users)
        phase = random.randint(1, 5)
        row_id = f"row_{i % 100}"

        sample_changes.append({
            "Group": group,
            "RowID": row_id,
            "Phase": str(phase),
            "User": user,
            "ParsedTimestamp": datetime.now() - timedelta(days=random.randint(0, 30)),
        })

    # Calculate and print report
    print(format_average_items_report(sample_changes))

    # Print summary
    print("\nJSON Summary (excerpt):")
    summary = get_average_items_summary(sample_changes)
    import json
    print(json.dumps({
        "total_items": summary.total_items,
        "total_users": summary.total_users,
        "overall_items_per_day": summary.overall_items_per_day,
        "overall_items_per_week": summary.overall_items_per_week,
        "avg_items_per_user_per_day": summary.avg_items_per_user_per_day,
        "avg_items_per_user_per_week": summary.avg_items_per_user_per_week,
    }, indent=2, default=str))

    # Print visualization data
    print("\nVisualization Data (summary):")
    viz_data = get_average_items_visualization_data(sample_changes)
    print(json.dumps(viz_data["summary"], indent=2))
