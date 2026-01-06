"""
User Productivity Calculator Module

Calculates per-user productivity metrics including:
- Items per day (throughput)
- Items per phase (phase-specific productivity)
- Average completion time (efficiency)
- Per-group breakdown (detailed analysis by product group)

This module follows the established patterns from completion_rate_calculator.py
and integrates with the historical_data_loader for data access.

Usage:
    from user_productivity_calculator import (
        UserProductivityMetrics,
        GroupProductivityBreakdown,
        PhaseProductivity,
        calculate_user_productivity,
        calculate_user_productivity_by_group,
        calculate_all_users_productivity,
        calculate_productivity_for_range,
        get_productivity_summary,
        get_productivity_visualization_data,
        format_productivity_report,
    )
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Any, Union, Tuple
from enum import Enum
from collections import defaultdict

# Set up logging
logger = logging.getLogger(__name__)


# Constants for phase definitions (aligned with completion_rate_calculator.py)
STARTING_PHASE = 1  # Phase 1: Kontrolle - items enter the workflow
COMPLETION_PHASE = 4  # Phase 4: C am - items complete the main workflow
ALL_PHASES = [1, 2, 3, 4, 5]  # All phases in the workflow


class ProductivityLevel(Enum):
    """Productivity level classification."""
    EXCEPTIONAL = "exceptional"  # Top tier productivity
    HIGH = "high"
    AVERAGE = "average"
    LOW = "low"
    MINIMAL = "minimal"


@dataclass
class PhaseProductivity:
    """Productivity metrics for a specific phase.

    Attributes:
        phase: Phase number (1-5)
        phase_name: Human-readable phase name
        items_count: Number of items processed in this phase
        items_per_day: Average items processed per day in this phase
        percentage_of_total: What percentage of user's work this phase represents
    """
    phase: int
    phase_name: str
    items_count: int = 0
    items_per_day: float = 0.0
    percentage_of_total: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "phase": self.phase,
            "phase_name": self.phase_name,
            "items_count": self.items_count,
            "items_per_day": round(self.items_per_day, 2),
            "percentage_of_total": round(self.percentage_of_total, 2),
        }


@dataclass
class UserProductivityMetrics:
    """Core per-user productivity metrics.

    Attributes:
        user: User identifier (initials like "JHU", "DM")
        total_items: Total number of items processed
        items_per_day: Average items processed per day
        items_per_phase: Detailed productivity by phase
        avg_completion_time_days: Average days from Phase 1 to Phase 4 (if available)
        unique_groups: Number of different groups the user processed items for
        unique_rows: Number of unique row IDs processed
        date_range_start: Start date of the analysis period
        date_range_end: End date of the analysis period
        productivity_level: Classification of productivity level
        active_days: Number of days with at least one activity
    """
    user: str
    total_items: int = 0
    items_per_day: float = 0.0
    items_per_phase: Dict[int, PhaseProductivity] = field(default_factory=dict)
    avg_completion_time_days: Optional[float] = None
    unique_groups: int = 0
    unique_rows: int = 0
    date_range_start: Optional[date] = None
    date_range_end: Optional[date] = None
    productivity_level: ProductivityLevel = ProductivityLevel.AVERAGE
    active_days: int = 0

    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        if self.productivity_level == ProductivityLevel.AVERAGE and self.items_per_day > 0:
            self.productivity_level = self._calculate_productivity_level()

    def _calculate_productivity_level(self) -> ProductivityLevel:
        """Determine productivity level based on items per day."""
        if self.items_per_day >= 50:
            return ProductivityLevel.EXCEPTIONAL
        elif self.items_per_day >= 25:
            return ProductivityLevel.HIGH
        elif self.items_per_day >= 10:
            return ProductivityLevel.AVERAGE
        elif self.items_per_day >= 3:
            return ProductivityLevel.LOW
        else:
            return ProductivityLevel.MINIMAL

    @property
    def days_in_range(self) -> int:
        """Get the number of days in the analysis range."""
        if self.date_range_start and self.date_range_end:
            return (self.date_range_end - self.date_range_start).days + 1
        return 0

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
            "items_per_phase": {
                str(phase): metrics.to_dict()
                for phase, metrics in self.items_per_phase.items()
            },
            "avg_completion_time_days": round(self.avg_completion_time_days, 2) if self.avg_completion_time_days else None,
            "unique_groups": self.unique_groups,
            "unique_rows": self.unique_rows,
            "date_range_start": self.date_range_start.isoformat() if self.date_range_start else None,
            "date_range_end": self.date_range_end.isoformat() if self.date_range_end else None,
            "days_in_range": self.days_in_range,
            "active_days": self.active_days,
            "activity_ratio": round(self.activity_ratio, 2),
            "productivity_level": self.productivity_level.value,
        }


@dataclass
class GroupProductivityBreakdown:
    """Productivity metrics for a user within a specific group.

    Attributes:
        user: User identifier
        group: Group code (NA, NF, NH, etc.)
        items_count: Total items processed in this group
        items_per_day: Average items per day for this group
        phases_handled: List of phase numbers handled
        items_per_phase: Count of items per phase
        percentage_of_user_total: What percentage of user's total work is in this group
    """
    user: str
    group: str
    items_count: int = 0
    items_per_day: float = 0.0
    phases_handled: List[int] = field(default_factory=list)
    items_per_phase: Dict[int, int] = field(default_factory=dict)
    percentage_of_user_total: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "user": self.user,
            "group": self.group,
            "items_count": self.items_count,
            "items_per_day": round(self.items_per_day, 2),
            "phases_handled": sorted(self.phases_handled),
            "items_per_phase": {str(k): v for k, v in self.items_per_phase.items()},
            "percentage_of_user_total": round(self.percentage_of_user_total, 2),
        }


@dataclass
class UserRanking:
    """Ranking information for a user's productivity.

    Attributes:
        user: User identifier
        rank: Rank position (1 = highest productivity)
        metrics: The user's productivity metrics
        items_behind_leader: How many items/day behind the leader
    """
    user: str
    rank: int
    metrics: UserProductivityMetrics
    items_behind_leader: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "user": self.user,
            "rank": self.rank,
            "metrics": self.metrics.to_dict(),
            "items_behind_leader": round(self.items_behind_leader, 2),
        }


# Phase name mapping
PHASE_NAMES = {
    1: "Kontrolle (Control)",
    2: "BE (Processing)",
    3: "K2 (Second Control)",
    4: "C (Completion)",
    5: "Reopen C2 (Reopen)",
}


def _get_phase_name(phase: int) -> str:
    """Get human-readable name for a phase number."""
    return PHASE_NAMES.get(phase, f"Phase {phase}")


def _extract_user_data(
    changes: List[Dict[str, Any]],
    user: Optional[str] = None
) -> Tuple[Dict[str, List[Dict]], Dict[str, set], Dict[str, set], Dict[str, set]]:
    """Extract and organize change data by user.

    Args:
        changes: List of change records from historical data
        user: Optional specific user to filter for

    Returns:
        Tuple of:
        - user_changes: Dict mapping users to their change records
        - user_rows: Dict mapping users to unique row IDs they touched
        - user_groups: Dict mapping users to groups they worked in
        - user_active_dates: Dict mapping users to dates they were active
    """
    user_changes: Dict[str, List[Dict]] = defaultdict(list)
    user_rows: Dict[str, set] = defaultdict(set)
    user_groups: Dict[str, set] = defaultdict(set)
    user_active_dates: Dict[str, set] = defaultdict(set)

    for change in changes:
        change_user = change.get("User")
        if not change_user:
            continue

        # Filter for specific user if requested
        if user and change_user != user:
            continue

        user_changes[change_user].append(change)

        row_id = change.get("RowID")
        if row_id:
            user_rows[change_user].add(row_id)

        group = change.get("Group")
        if group:
            user_groups[change_user].add(group)

        # Extract date
        ts = change.get('ParsedTimestamp') or change.get('ParsedDate')
        if ts:
            activity_date = ts.date() if hasattr(ts, 'date') else ts
            user_active_dates[change_user].add(activity_date)

    return user_changes, user_rows, user_groups, user_active_dates


def _calculate_phase_productivity(
    changes: List[Dict[str, Any]],
    days_in_range: int
) -> Dict[int, PhaseProductivity]:
    """Calculate productivity metrics for each phase.

    Args:
        changes: List of change records for a user
        days_in_range: Number of days in the analysis period

    Returns:
        Dict mapping phase numbers to PhaseProductivity objects
    """
    # Count items per phase
    phase_counts: Dict[int, int] = defaultdict(int)

    for change in changes:
        phase = change.get("Phase")
        if phase:
            try:
                phase_num = int(phase)
                phase_counts[phase_num] += 1
            except (ValueError, TypeError):
                continue

    total_items = sum(phase_counts.values())

    # Build PhaseProductivity for each phase
    result = {}
    for phase in ALL_PHASES:
        count = phase_counts.get(phase, 0)
        items_per_day = count / max(days_in_range, 1)
        percentage = (count / total_items * 100) if total_items > 0 else 0.0

        result[phase] = PhaseProductivity(
            phase=phase,
            phase_name=_get_phase_name(phase),
            items_count=count,
            items_per_day=items_per_day,
            percentage_of_total=percentage,
        )

    return result


def _calculate_completion_times(
    changes: List[Dict[str, Any]]
) -> Optional[float]:
    """Calculate average completion time from Phase 1 to Phase 4.

    This looks at items where the user processed both Phase 1 and Phase 4
    and calculates the average time difference.

    Args:
        changes: List of change records for a user

    Returns:
        Average completion time in days, or None if not calculable
    """
    # Group changes by RowID
    row_phases: Dict[str, Dict[int, date]] = defaultdict(dict)

    for change in changes:
        row_id = change.get("RowID")
        phase = change.get("Phase")
        if not row_id or not phase:
            continue

        try:
            phase_num = int(phase)
        except (ValueError, TypeError):
            continue

        # Get the date of this phase
        ts = change.get('ParsedTimestamp') or change.get('ParsedDate')
        if ts:
            phase_date = ts.date() if hasattr(ts, 'date') else ts
            # Store earliest date for each phase
            if phase_num not in row_phases[row_id] or phase_date < row_phases[row_id][phase_num]:
                row_phases[row_id][phase_num] = phase_date

    # Calculate completion times for rows with both Phase 1 and Phase 4
    completion_times = []
    for row_id, phases in row_phases.items():
        if STARTING_PHASE in phases and COMPLETION_PHASE in phases:
            start = phases[STARTING_PHASE]
            end = phases[COMPLETION_PHASE]
            if end >= start:
                days = (end - start).days
                completion_times.append(days)

    if completion_times:
        return sum(completion_times) / len(completion_times)
    return None


def calculate_user_productivity(
    changes: List[Dict[str, Any]],
    user: str
) -> UserProductivityMetrics:
    """Calculate productivity metrics for a specific user.

    Args:
        changes: List of change records from historical data
        user: User identifier to calculate metrics for

    Returns:
        UserProductivityMetrics object with comprehensive productivity analysis

    Example:
        >>> from historical_data_loader import load_change_history
        >>> changes = load_change_history()
        >>> metrics = calculate_user_productivity(changes, "JHU")
        >>> print(f"Items per day: {metrics.items_per_day}")
    """
    if not changes:
        return UserProductivityMetrics(user=user)

    # Extract data for this user
    user_changes, user_rows, user_groups, user_active_dates = _extract_user_data(
        changes, user
    )

    if user not in user_changes:
        return UserProductivityMetrics(user=user)

    user_records = user_changes[user]

    # Get date range
    dates = [c.get('ParsedTimestamp') or c.get('ParsedDate') for c in changes]
    dates = [d.date() if hasattr(d, 'date') else d for d in dates if d]
    date_range_start = min(dates) if dates else None
    date_range_end = max(dates) if dates else None
    days_in_range = (date_range_end - date_range_start).days + 1 if date_range_start and date_range_end else 1

    # Calculate metrics
    total_items = len(user_records)
    items_per_day = total_items / max(days_in_range, 1)

    # Phase productivity
    items_per_phase = _calculate_phase_productivity(user_records, days_in_range)

    # Completion time
    avg_completion_time = _calculate_completion_times(user_records)

    # Active days
    active_days = len(user_active_dates[user])

    return UserProductivityMetrics(
        user=user,
        total_items=total_items,
        items_per_day=items_per_day,
        items_per_phase=items_per_phase,
        avg_completion_time_days=avg_completion_time,
        unique_groups=len(user_groups[user]),
        unique_rows=len(user_rows[user]),
        date_range_start=date_range_start,
        date_range_end=date_range_end,
        active_days=active_days,
    )


def calculate_user_productivity_by_group(
    changes: List[Dict[str, Any]],
    user: str
) -> Dict[str, GroupProductivityBreakdown]:
    """Calculate per-group productivity breakdown for a specific user.

    Args:
        changes: List of change records from historical data
        user: User identifier to analyze

    Returns:
        Dictionary mapping group codes to GroupProductivityBreakdown objects

    Example:
        >>> from historical_data_loader import load_change_history
        >>> changes = load_change_history()
        >>> by_group = calculate_user_productivity_by_group(changes, "DM")
        >>> for group, data in by_group.items():
        ...     print(f"{group}: {data.items_count} items")
    """
    if not changes:
        return {}

    # Filter for specific user
    user_changes = [c for c in changes if c.get("User") == user]

    if not user_changes:
        return {}

    # Get date range for items_per_day calculation
    dates = [c.get('ParsedTimestamp') or c.get('ParsedDate') for c in changes]
    dates = [d.date() if hasattr(d, 'date') else d for d in dates if d]
    days_in_range = 1
    if dates:
        date_range_start = min(dates)
        date_range_end = max(dates)
        days_in_range = max((date_range_end - date_range_start).days + 1, 1)

    # Group by group code
    by_group: Dict[str, List[Dict]] = defaultdict(list)
    for change in user_changes:
        group = change.get("Group")
        if group:
            by_group[group].append(change)

    total_user_items = len(user_changes)
    result = {}

    for group, group_changes in by_group.items():
        # Count items per phase
        phase_counts: Dict[int, int] = defaultdict(int)
        phases_handled: set = set()

        for change in group_changes:
            phase = change.get("Phase")
            if phase:
                try:
                    phase_num = int(phase)
                    phase_counts[phase_num] += 1
                    phases_handled.add(phase_num)
                except (ValueError, TypeError):
                    continue

        items_count = len(group_changes)

        result[group] = GroupProductivityBreakdown(
            user=user,
            group=group,
            items_count=items_count,
            items_per_day=items_count / days_in_range,
            phases_handled=sorted(phases_handled),
            items_per_phase=dict(phase_counts),
            percentage_of_user_total=(items_count / total_user_items * 100) if total_user_items > 0 else 0.0,
        )

    return result


def calculate_all_users_productivity(
    changes: List[Dict[str, Any]]
) -> Dict[str, UserProductivityMetrics]:
    """Calculate productivity metrics for all users in the data.

    Args:
        changes: List of change records from historical data

    Returns:
        Dictionary mapping user identifiers to UserProductivityMetrics objects

    Example:
        >>> from historical_data_loader import load_change_history
        >>> changes = load_change_history()
        >>> all_metrics = calculate_all_users_productivity(changes)
        >>> for user, metrics in all_metrics.items():
        ...     print(f"{user}: {metrics.items_per_day:.1f} items/day")
    """
    if not changes:
        return {}

    # Get all unique users
    users = set(c.get("User") for c in changes if c.get("User"))

    result = {}
    for user in users:
        result[user] = calculate_user_productivity(changes, user)

    return result


def calculate_productivity_for_range(
    date_range_filter: Any,  # DateRangeFilter type
    user: Optional[str] = None,
    changes: Optional[List[Dict[str, Any]]] = None
) -> Union[UserProductivityMetrics, Dict[str, UserProductivityMetrics]]:
    """Calculate productivity metrics for a specific date range.

    Args:
        date_range_filter: DateRangeFilter object specifying the date range
        user: Optional specific user to analyze. If None, analyzes all users
        changes: Optional pre-loaded changes (will load if not provided)

    Returns:
        If user specified: UserProductivityMetrics for that user
        If user is None: Dict mapping users to their metrics

    Example:
        >>> from date_range_filter import get_preset_range, DateRangePreset
        >>> last_week = get_preset_range(DateRangePreset.LAST_WEEK)
        >>> metrics = calculate_productivity_for_range(last_week, user="JHU")
        >>> print(f"Items per day: {metrics.items_per_day}")
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

    if user:
        metrics = calculate_user_productivity(changes, user)
        metrics.date_range_start = date_range_filter.start_date
        metrics.date_range_end = date_range_filter.end_date
        return metrics
    else:
        all_metrics = calculate_all_users_productivity(changes)
        # Update date ranges to match filter
        for m in all_metrics.values():
            m.date_range_start = date_range_filter.start_date
            m.date_range_end = date_range_filter.end_date
        return all_metrics


def get_user_rankings(
    changes: List[Dict[str, Any]]
) -> List[UserRanking]:
    """Get ranked list of users by productivity.

    Args:
        changes: List of change records from historical data

    Returns:
        List of UserRanking objects, sorted by items_per_day (highest first)

    Example:
        >>> changes = load_change_history()
        >>> rankings = get_user_rankings(changes)
        >>> for r in rankings:
        ...     print(f"#{r.rank} {r.user}: {r.metrics.items_per_day:.1f} items/day")
    """
    all_metrics = calculate_all_users_productivity(changes)

    if not all_metrics:
        return []

    # Sort by items_per_day (descending)
    sorted_users = sorted(
        all_metrics.items(),
        key=lambda x: x[1].items_per_day,
        reverse=True
    )

    leader_rate = sorted_users[0][1].items_per_day if sorted_users else 0

    rankings = []
    for rank, (user, metrics) in enumerate(sorted_users, start=1):
        rankings.append(UserRanking(
            user=user,
            rank=rank,
            metrics=metrics,
            items_behind_leader=leader_rate - metrics.items_per_day,
        ))

    return rankings


def get_productivity_summary(
    changes: List[Dict[str, Any]],
    include_rankings: bool = True,
    include_by_group: bool = True
) -> Dict[str, Any]:
    """Get a comprehensive summary of user productivity.

    Args:
        changes: List of change records from historical data
        include_rankings: Whether to include user rankings
        include_by_group: Whether to include per-group breakdown for each user

    Returns:
        Dictionary with comprehensive productivity analysis

    Example:
        >>> changes = load_change_history()
        >>> summary = get_productivity_summary(changes)
        >>> print(f"Top performer: {summary['top_performer']['user']}")
    """
    all_metrics = calculate_all_users_productivity(changes)

    if not all_metrics:
        return {
            "total_users": 0,
            "users": {},
            "rankings": [],
            "top_performer": None,
            "average_items_per_day": 0,
        }

    summary: Dict[str, Any] = {
        "total_users": len(all_metrics),
        "users": {user: metrics.to_dict() for user, metrics in all_metrics.items()},
    }

    # Calculate averages
    avg_items_per_day = sum(m.items_per_day for m in all_metrics.values()) / len(all_metrics)
    summary["average_items_per_day"] = round(avg_items_per_day, 2)

    # Add rankings
    if include_rankings:
        rankings = get_user_rankings(changes)
        summary["rankings"] = [r.to_dict() for r in rankings]

        if rankings:
            summary["top_performer"] = {
                "user": rankings[0].user,
                "items_per_day": rankings[0].metrics.items_per_day,
                "productivity_level": rankings[0].metrics.productivity_level.value,
            }
            summary["lowest_performer"] = {
                "user": rankings[-1].user,
                "items_per_day": rankings[-1].metrics.items_per_day,
                "productivity_level": rankings[-1].metrics.productivity_level.value,
            }

    # Add per-group breakdown
    if include_by_group:
        by_group_data = {}
        for user in all_metrics.keys():
            by_group = calculate_user_productivity_by_group(changes, user)
            by_group_data[user] = {
                group: breakdown.to_dict()
                for group, breakdown in by_group.items()
            }
        summary["by_group"] = by_group_data

    # Get date range from the data
    dates = [c.get('ParsedTimestamp') or c.get('ParsedDate') for c in changes]
    dates = [d.date() if hasattr(d, 'date') else d for d in dates if d]
    if dates:
        summary["date_range"] = {
            "start": min(dates).isoformat(),
            "end": max(dates).isoformat(),
        }

    return summary


def get_productivity_visualization_data(
    changes: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Get data formatted for productivity visualization.

    This function prepares data in a format suitable for rendering
    charts in PDF reports or other visualizations.

    Args:
        changes: List of change records

    Returns:
        Dictionary with visualization-ready data

    Example:
        >>> changes = load_change_history()
        >>> viz_data = get_productivity_visualization_data(changes)
        >>> # Use viz_data to render productivity charts
    """
    all_metrics = calculate_all_users_productivity(changes)
    rankings = get_user_rankings(changes)

    # Prepare bar chart data for items per day
    bar_chart_data = []
    for ranking in rankings:
        level = ranking.metrics.productivity_level
        # Assign colors based on productivity level
        color_map = {
            ProductivityLevel.EXCEPTIONAL: "#1B5E20",  # Green
            ProductivityLevel.HIGH: "#388E3C",  # Light Green
            ProductivityLevel.AVERAGE: "#8B6914",  # Yellow/Gold
            ProductivityLevel.LOW: "#E65100",  # Orange
            ProductivityLevel.MINIMAL: "#B71C1C",  # Red
        }

        bar_chart_data.append({
            "user": ranking.user,
            "items_per_day": ranking.metrics.items_per_day,
            "total_items": ranking.metrics.total_items,
            "rank": ranking.rank,
            "productivity_level": level.value,
            "color": color_map.get(level, "#757575"),
        })

    # Prepare phase distribution data (aggregate across all users)
    phase_totals: Dict[int, int] = defaultdict(int)
    for metrics in all_metrics.values():
        for phase, phase_data in metrics.items_per_phase.items():
            phase_totals[phase] += phase_data.items_count

    phase_distribution = [
        {
            "phase": phase,
            "phase_name": _get_phase_name(phase),
            "total_items": count,
        }
        for phase, count in sorted(phase_totals.items())
    ]

    # Calculate summary stats
    total_items = sum(m.total_items for m in all_metrics.values())
    avg_items_per_day = sum(m.items_per_day for m in all_metrics.values()) / max(len(all_metrics), 1)

    return {
        "bar_chart_data": bar_chart_data,
        "phase_distribution": phase_distribution,
        "summary": {
            "total_users": len(all_metrics),
            "total_items_processed": total_items,
            "average_items_per_day": round(avg_items_per_day, 2),
        },
    }


def format_productivity_report(
    changes: List[Dict[str, Any]],
    include_by_group: bool = True
) -> str:
    """Format productivity metrics as a text report.

    Args:
        changes: List of change records from historical data
        include_by_group: Whether to include per-group breakdown

    Returns:
        Formatted text report string
    """
    all_metrics = calculate_all_users_productivity(changes)
    rankings = get_user_rankings(changes)

    lines = []
    lines.append("=" * 80)
    lines.append("USER PRODUCTIVITY REPORT")
    lines.append("=" * 80)

    if not all_metrics:
        lines.append("\nNo user data available.")
        return "\n".join(lines)

    # Get date range
    sample_metrics = next(iter(all_metrics.values()))
    if sample_metrics.date_range_start and sample_metrics.date_range_end:
        lines.append(f"\nPeriod: {sample_metrics.date_range_start} to {sample_metrics.date_range_end}")
        lines.append(f"Days in period: {sample_metrics.days_in_range}")

    # Summary
    lines.append("\n" + "-" * 80)
    lines.append("SUMMARY")
    lines.append("-" * 80)

    total_items = sum(m.total_items for m in all_metrics.values())
    avg_items_per_day = sum(m.items_per_day for m in all_metrics.values()) / len(all_metrics)

    lines.append(f"  Total Users: {len(all_metrics)}")
    lines.append(f"  Total Items Processed: {total_items}")
    lines.append(f"  Average Items/Day (per user): {avg_items_per_day:.2f}")

    # Rankings
    lines.append("\n" + "-" * 80)
    lines.append("USER RANKINGS (by Items/Day)")
    lines.append("-" * 80)
    lines.append(f"{'Rank':<6} {'User':<10} {'Items/Day':>12} {'Total Items':>12} {'Level':<15} {'Active Days':>12}")
    lines.append("-" * 80)

    for ranking in rankings:
        m = ranking.metrics
        lines.append(
            f"{ranking.rank:<6} "
            f"{ranking.user:<10} "
            f"{m.items_per_day:>12.2f} "
            f"{m.total_items:>12} "
            f"{m.productivity_level.value:<15} "
            f"{m.active_days:>12}"
        )

    # Phase breakdown for each user
    lines.append("\n" + "-" * 80)
    lines.append("ITEMS PER PHASE (by User)")
    lines.append("-" * 80)
    lines.append(f"{'User':<10} {'Phase 1':>10} {'Phase 2':>10} {'Phase 3':>10} {'Phase 4':>10} {'Phase 5':>10}")
    lines.append("-" * 80)

    for user in sorted(all_metrics.keys()):
        m = all_metrics[user]
        phase_counts = [
            m.items_per_phase.get(p, PhaseProductivity(p, "")).items_count
            for p in ALL_PHASES
        ]
        lines.append(
            f"{user:<10} "
            + " ".join(f"{count:>10}" for count in phase_counts)
        )

    # Per-group breakdown
    if include_by_group:
        lines.append("\n" + "-" * 80)
        lines.append("PER-GROUP BREAKDOWN")
        lines.append("-" * 80)

        for user in sorted(all_metrics.keys()):
            by_group = calculate_user_productivity_by_group(changes, user)
            if by_group:
                lines.append(f"\n  {user}:")
                for group in sorted(by_group.keys()):
                    breakdown = by_group[group]
                    lines.append(
                        f"    {group}: {breakdown.items_count} items "
                        f"({breakdown.percentage_of_user_total:.1f}% of total), "
                        f"Phases: {breakdown.phases_handled}"
                    )

    lines.append("\n" + "=" * 80)

    return "\n".join(lines)


if __name__ == "__main__":
    # Demo usage
    print("User Productivity Calculator - Demo")
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
        row_id = f"row_{i % 100}"  # Some rows processed multiple times

        sample_changes.append({
            "Group": group,
            "RowID": row_id,
            "Phase": str(phase),
            "User": user,
            "ParsedTimestamp": datetime.now() - timedelta(days=random.randint(0, 30)),
        })

    # Calculate and print report
    print(format_productivity_report(sample_changes))

    # Print summary
    print("\nJSON Summary (excerpt):")
    summary = get_productivity_summary(sample_changes, include_by_group=False)
    import json
    print(json.dumps({
        "total_users": summary["total_users"],
        "average_items_per_day": summary["average_items_per_day"],
        "top_performer": summary.get("top_performer"),
        "lowest_performer": summary.get("lowest_performer"),
    }, indent=2, default=str))

    # Print visualization data
    print("\nVisualization Data (excerpt):")
    viz_data = get_productivity_visualization_data(sample_changes)
    print(json.dumps(viz_data["summary"], indent=2))
