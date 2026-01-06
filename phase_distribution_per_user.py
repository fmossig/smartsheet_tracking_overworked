"""
Phase Distribution Per User Calculator Module

Calculates which phases each user works in most frequently and generates
distribution data for visualization showing user specialization patterns.

This module analyzes user activity across workflow phases to identify:
- Primary phase focus for each user
- Specialization vs generalist patterns
- Distribution evenness (via Gini coefficient)
- Phase concentration patterns

Usage:
    from phase_distribution_per_user import (
        SpecializationLevel,
        PhaseConcentration,
        PhaseDistribution,
        UserPhaseDistribution,
        PhaseDistributionSummary,
        calculate_user_phase_distribution,
        calculate_all_users_phase_distribution,
        calculate_phase_distribution_for_range,
        get_phase_distribution_visualization_data,
        format_phase_distribution_report,
        get_specialization_insights,
    )
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
from collections import defaultdict

from period_comparison_calculator import (
    TrendDirection,
    calculate_percent_change,
    get_trend_direction,
)

# Set up logging
logger = logging.getLogger(__name__)


# Constants for phase definitions (aligned with other modules)
ALL_PHASES = [1, 2, 3, 4, 5]

# Phase name mapping
PHASE_NAMES = {
    1: "Kontrolle (Control)",
    2: "BE (Processing)",
    3: "K2 (Second Control)",
    4: "C (Completion)",
    5: "Reopen C2 (Reopen)",
}


class SpecializationLevel(Enum):
    """Classification of user specialization based on phase distribution.

    SPECIALIST: User focuses heavily on 1-2 phases (>60% in top phase)
    FOCUSED: User has a clear primary phase (45-60% in top phase)
    BALANCED: User works across phases relatively evenly (30-45% in top phase)
    GENERALIST: User has very even distribution (<30% in any single phase)
    """
    SPECIALIST = "specialist"
    FOCUSED = "focused"
    BALANCED = "balanced"
    GENERALIST = "generalist"


class PhaseConcentration(Enum):
    """Classification of overall phase concentration for a user.

    Based on Gini coefficient:
    - Higher Gini = more concentrated (work focused on fewer phases)
    - Lower Gini = more distributed (work spread across phases)
    """
    HIGHLY_CONCENTRATED = "highly_concentrated"  # Gini > 0.6
    CONCENTRATED = "concentrated"                 # Gini 0.4-0.6
    MODERATE = "moderate"                         # Gini 0.2-0.4
    DISTRIBUTED = "distributed"                   # Gini 0.1-0.2
    HIGHLY_DISTRIBUTED = "highly_distributed"     # Gini < 0.1


def _get_phase_name(phase: int) -> str:
    """Get human-readable name for a phase number."""
    return PHASE_NAMES.get(phase, f"Phase {phase}")


def _calculate_gini_coefficient(values: List[float]) -> float:
    """Calculate Gini coefficient for distribution inequality.

    The Gini coefficient measures inequality of distribution:
    - 0 = perfect equality (all values equal)
    - 1 = perfect inequality (one value has everything)

    Args:
        values: List of numeric values representing distribution

    Returns:
        Gini coefficient between 0 and 1
    """
    if not values or len(values) < 2:
        return 0.0

    # Filter out zero values for meaningful calculation
    non_zero_values = [v for v in values if v > 0]
    if len(non_zero_values) < 2:
        return 1.0 if sum(values) > 0 else 0.0

    # Sort values
    sorted_values = sorted(non_zero_values)
    n = len(sorted_values)

    # Calculate Gini coefficient using the standard formula
    cumulative = 0.0
    for i, value in enumerate(sorted_values):
        cumulative += (2 * (i + 1) - n - 1) * value

    total = sum(sorted_values)
    if total == 0:
        return 0.0

    gini = cumulative / (n * total)
    return max(0.0, min(1.0, gini))  # Clamp to [0, 1]


def _classify_specialization_level(top_phase_percentage: float) -> SpecializationLevel:
    """Classify user specialization level based on top phase percentage.

    Args:
        top_phase_percentage: Percentage of work in the user's top phase

    Returns:
        SpecializationLevel enum value
    """
    if top_phase_percentage >= 60:
        return SpecializationLevel.SPECIALIST
    elif top_phase_percentage >= 45:
        return SpecializationLevel.FOCUSED
    elif top_phase_percentage >= 30:
        return SpecializationLevel.BALANCED
    else:
        return SpecializationLevel.GENERALIST


def _classify_concentration_level(gini: float) -> PhaseConcentration:
    """Classify phase concentration level based on Gini coefficient.

    Args:
        gini: Gini coefficient (0-1)

    Returns:
        PhaseConcentration enum value
    """
    if gini > 0.6:
        return PhaseConcentration.HIGHLY_CONCENTRATED
    elif gini > 0.4:
        return PhaseConcentration.CONCENTRATED
    elif gini > 0.2:
        return PhaseConcentration.MODERATE
    elif gini > 0.1:
        return PhaseConcentration.DISTRIBUTED
    else:
        return PhaseConcentration.HIGHLY_DISTRIBUTED


@dataclass
class PhaseDistribution:
    """Distribution metrics for a specific phase for a user.

    Attributes:
        phase: Phase number (1-5)
        phase_name: Human-readable phase name
        count: Number of items processed in this phase
        percentage: Percentage of user's total work in this phase
        rank: Rank among all phases for this user (1 = most work)
        is_primary: Whether this is the user's primary phase
    """
    phase: int
    phase_name: str
    count: int = 0
    percentage: float = 0.0
    rank: int = 0
    is_primary: bool = False

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "phase": self.phase,
            "phase_name": self.phase_name,
            "count": self.count,
            "percentage": round(self.percentage, 2),
            "rank": self.rank,
            "is_primary": self.is_primary,
        }


@dataclass
class UserPhaseDistribution:
    """Complete phase distribution profile for a user.

    Attributes:
        user: User identifier (initials)
        total_items: Total items processed across all phases
        phases: Dictionary mapping phase numbers to PhaseDistribution objects
        primary_phase: The phase with the most work
        primary_phase_percentage: Percentage of work in primary phase
        secondary_phase: The phase with second-most work
        specialization_level: Classification of specialization
        concentration_level: Classification of phase concentration
        gini_coefficient: Gini coefficient for distribution inequality
        phases_active: Number of phases the user has worked in
        date_range_start: Start date of analysis period
        date_range_end: End date of analysis period
        previous_period_total: Total from previous period (for comparison)
        total_percent_change: Percentage change vs previous period
        trend: Trend direction vs previous period
    """
    user: str
    total_items: int = 0
    phases: Dict[int, PhaseDistribution] = field(default_factory=dict)
    primary_phase: Optional[int] = None
    primary_phase_percentage: float = 0.0
    secondary_phase: Optional[int] = None
    specialization_level: SpecializationLevel = SpecializationLevel.BALANCED
    concentration_level: PhaseConcentration = PhaseConcentration.MODERATE
    gini_coefficient: float = 0.0
    phases_active: int = 0
    date_range_start: Optional[date] = None
    date_range_end: Optional[date] = None
    previous_period_total: Optional[int] = None
    total_percent_change: Optional[float] = None
    trend: TrendDirection = TrendDirection.NO_DATA

    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        if self.phases and self.total_items > 0:
            # Calculate phases_active
            self.phases_active = sum(1 for p in self.phases.values() if p.count > 0)

            # Calculate Gini coefficient
            counts = [p.count for p in self.phases.values()]
            self.gini_coefficient = _calculate_gini_coefficient(counts)

            # Determine concentration level
            self.concentration_level = _classify_concentration_level(self.gini_coefficient)

            # Find primary and secondary phases
            sorted_phases = sorted(
                [(p.phase, p.count) for p in self.phases.values()],
                key=lambda x: x[1],
                reverse=True
            )

            if sorted_phases and sorted_phases[0][1] > 0:
                self.primary_phase = sorted_phases[0][0]
                self.primary_phase_percentage = (sorted_phases[0][1] / self.total_items) * 100

                if len(sorted_phases) > 1 and sorted_phases[1][1] > 0:
                    self.secondary_phase = sorted_phases[1][0]

            # Determine specialization level
            self.specialization_level = _classify_specialization_level(
                self.primary_phase_percentage
            )

    @property
    def days_in_range(self) -> int:
        """Get the number of days in the analysis range."""
        if self.date_range_start and self.date_range_end:
            return (self.date_range_end - self.date_range_start).days + 1
        return 0

    @property
    def items_per_day(self) -> float:
        """Get average items processed per day."""
        if self.days_in_range > 0:
            return self.total_items / self.days_in_range
        return 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "user": self.user,
            "total_items": self.total_items,
            "phases": {str(k): v.to_dict() for k, v in self.phases.items()},
            "primary_phase": self.primary_phase,
            "primary_phase_name": _get_phase_name(self.primary_phase) if self.primary_phase else None,
            "primary_phase_percentage": round(self.primary_phase_percentage, 2),
            "secondary_phase": self.secondary_phase,
            "secondary_phase_name": _get_phase_name(self.secondary_phase) if self.secondary_phase else None,
            "specialization_level": self.specialization_level.value,
            "concentration_level": self.concentration_level.value,
            "gini_coefficient": round(self.gini_coefficient, 3),
            "phases_active": self.phases_active,
            "date_range_start": self.date_range_start.isoformat() if self.date_range_start else None,
            "date_range_end": self.date_range_end.isoformat() if self.date_range_end else None,
            "days_in_range": self.days_in_range,
            "items_per_day": round(self.items_per_day, 2),
            "previous_period_total": self.previous_period_total,
            "total_percent_change": round(self.total_percent_change, 2) if self.total_percent_change is not None else None,
            "trend": self.trend.value,
        }


@dataclass
class PhaseDistributionSummary:
    """Summary of phase distribution across all users.

    Attributes:
        users: Dictionary mapping user identifiers to their distributions
        total_users: Number of users analyzed
        total_items: Total items processed across all users
        overall_phase_distribution: Aggregate phase distribution
        specialists: List of users classified as specialists
        generalists: List of users classified as generalists
        phase_leaders: Dictionary mapping phases to the user with most items
        average_gini: Average Gini coefficient across users
        date_range_start: Start date of analysis period
        date_range_end: End date of analysis period
    """
    users: Dict[str, UserPhaseDistribution] = field(default_factory=dict)
    total_users: int = 0
    total_items: int = 0
    overall_phase_distribution: Dict[int, int] = field(default_factory=dict)
    specialists: List[str] = field(default_factory=list)
    generalists: List[str] = field(default_factory=list)
    phase_leaders: Dict[int, Tuple[str, int]] = field(default_factory=dict)
    average_gini: float = 0.0
    date_range_start: Optional[date] = None
    date_range_end: Optional[date] = None

    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        if self.users:
            self.total_users = len(self.users)
            self.total_items = sum(u.total_items for u in self.users.values())

            # Calculate average Gini
            ginis = [u.gini_coefficient for u in self.users.values() if u.total_items > 0]
            self.average_gini = sum(ginis) / len(ginis) if ginis else 0.0

            # Identify specialists and generalists
            self.specialists = [
                user for user, dist in self.users.items()
                if dist.specialization_level == SpecializationLevel.SPECIALIST
            ]
            self.generalists = [
                user for user, dist in self.users.items()
                if dist.specialization_level == SpecializationLevel.GENERALIST
            ]

            # Calculate overall phase distribution
            for user_dist in self.users.values():
                for phase, phase_dist in user_dist.phases.items():
                    self.overall_phase_distribution[phase] = (
                        self.overall_phase_distribution.get(phase, 0) + phase_dist.count
                    )

            # Find phase leaders
            for phase in ALL_PHASES:
                max_user = None
                max_count = 0
                for user, user_dist in self.users.items():
                    phase_count = user_dist.phases.get(phase, PhaseDistribution(phase, "")).count
                    if phase_count > max_count:
                        max_count = phase_count
                        max_user = user
                if max_user:
                    self.phase_leaders[phase] = (max_user, max_count)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "users": {k: v.to_dict() for k, v in self.users.items()},
            "total_users": self.total_users,
            "total_items": self.total_items,
            "overall_phase_distribution": {
                str(k): v for k, v in self.overall_phase_distribution.items()
            },
            "specialists": self.specialists,
            "generalists": self.generalists,
            "phase_leaders": {
                str(k): {"user": v[0], "count": v[1]}
                for k, v in self.phase_leaders.items()
            },
            "average_gini": round(self.average_gini, 3),
            "date_range_start": self.date_range_start.isoformat() if self.date_range_start else None,
            "date_range_end": self.date_range_end.isoformat() if self.date_range_end else None,
        }


def calculate_user_phase_distribution(
    changes: List[Dict[str, Any]],
    user: str
) -> UserPhaseDistribution:
    """Calculate phase distribution for a specific user.

    Args:
        changes: List of change records from historical data
        user: User identifier to calculate distribution for

    Returns:
        UserPhaseDistribution object with comprehensive phase analysis

    Example:
        >>> from historical_data_loader import load_change_history
        >>> changes = load_change_history()
        >>> dist = calculate_user_phase_distribution(changes, "JHU")
        >>> print(f"Primary phase: {dist.primary_phase}")
        >>> print(f"Specialization: {dist.specialization_level.value}")
    """
    if not changes:
        return UserPhaseDistribution(user=user)

    # Filter for specific user
    user_changes = [c for c in changes if c.get("User") == user]

    if not user_changes:
        return UserPhaseDistribution(user=user)

    # Count items per phase
    phase_counts: Dict[int, int] = defaultdict(int)

    for change in user_changes:
        phase = change.get("Phase")
        if phase:
            try:
                phase_num = int(phase)
                phase_counts[phase_num] += 1
            except (ValueError, TypeError):
                continue

    total_items = sum(phase_counts.values())

    if total_items == 0:
        return UserPhaseDistribution(user=user)

    # Build PhaseDistribution for each phase
    phase_distributions: Dict[int, PhaseDistribution] = {}

    # Sort phases by count for ranking
    sorted_phases = sorted(
        [(phase, count) for phase, count in phase_counts.items()],
        key=lambda x: x[1],
        reverse=True
    )
    phase_ranks = {phase: rank + 1 for rank, (phase, _) in enumerate(sorted_phases)}

    for phase in ALL_PHASES:
        count = phase_counts.get(phase, 0)
        percentage = (count / total_items * 100) if total_items > 0 else 0.0
        rank = phase_ranks.get(phase, len(ALL_PHASES))
        is_primary = (rank == 1 and count > 0)

        phase_distributions[phase] = PhaseDistribution(
            phase=phase,
            phase_name=_get_phase_name(phase),
            count=count,
            percentage=percentage,
            rank=rank,
            is_primary=is_primary,
        )

    # Get date range from data
    dates = [c.get('ParsedTimestamp') or c.get('ParsedDate') for c in changes]
    dates = [d.date() if hasattr(d, 'date') else d for d in dates if d]
    date_range_start = min(dates) if dates else None
    date_range_end = max(dates) if dates else None

    return UserPhaseDistribution(
        user=user,
        total_items=total_items,
        phases=phase_distributions,
        date_range_start=date_range_start,
        date_range_end=date_range_end,
    )


def calculate_all_users_phase_distribution(
    changes: List[Dict[str, Any]]
) -> Dict[str, UserPhaseDistribution]:
    """Calculate phase distribution for all users in the data.

    Args:
        changes: List of change records from historical data

    Returns:
        Dictionary mapping user identifiers to their phase distributions

    Example:
        >>> from historical_data_loader import load_change_history
        >>> changes = load_change_history()
        >>> all_dist = calculate_all_users_phase_distribution(changes)
        >>> for user, dist in all_dist.items():
        ...     print(f"{user}: {dist.specialization_level.value}")
    """
    if not changes:
        return {}

    # Get all unique users
    users = set(c.get("User") for c in changes if c.get("User"))

    result = {}
    for user in users:
        result[user] = calculate_user_phase_distribution(changes, user)

    return result


def calculate_phase_distribution_for_range(
    date_range_filter: Any,  # DateRangeFilter type
    user: Optional[str] = None,
    changes: Optional[List[Dict[str, Any]]] = None
) -> PhaseDistributionSummary:
    """Calculate phase distribution for a specific date range.

    Args:
        date_range_filter: DateRangeFilter object specifying the date range
        user: Optional specific user to analyze. If None, analyzes all users
        changes: Optional pre-loaded changes (will load if not provided)

    Returns:
        PhaseDistributionSummary with all user distributions

    Example:
        >>> from date_range_filter import get_preset_range, DateRangePreset
        >>> last_week = get_preset_range(DateRangePreset.LAST_WEEK)
        >>> summary = calculate_phase_distribution_for_range(last_week)
        >>> print(f"Specialists: {summary.specialists}")
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
        user_dist = calculate_user_phase_distribution(changes, user)
        users = {user: user_dist}
    else:
        users = calculate_all_users_phase_distribution(changes)

    summary = PhaseDistributionSummary(
        users=users,
        date_range_start=date_range_filter.start_date,
        date_range_end=date_range_filter.end_date,
    )

    return summary


def get_phase_distribution_summary(
    changes: List[Dict[str, Any]]
) -> PhaseDistributionSummary:
    """Get a comprehensive summary of phase distribution across all users.

    Args:
        changes: List of change records from historical data

    Returns:
        PhaseDistributionSummary with aggregate analysis

    Example:
        >>> from historical_data_loader import load_change_history
        >>> changes = load_change_history()
        >>> summary = get_phase_distribution_summary(changes)
        >>> print(f"Total users: {summary.total_users}")
        >>> print(f"Specialists: {summary.specialists}")
    """
    users = calculate_all_users_phase_distribution(changes)

    # Get date range from data
    dates = [c.get('ParsedTimestamp') or c.get('ParsedDate') for c in changes]
    dates = [d.date() if hasattr(d, 'date') else d for d in dates if d]
    date_range_start = min(dates) if dates else None
    date_range_end = max(dates) if dates else None

    return PhaseDistributionSummary(
        users=users,
        date_range_start=date_range_start,
        date_range_end=date_range_end,
    )


def get_phase_distribution_visualization_data(
    changes: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Get data formatted for phase distribution visualization.

    This function prepares data in a format suitable for rendering
    charts in PDF reports or other visualizations.

    Args:
        changes: List of change records

    Returns:
        Dictionary with visualization-ready data including:
        - user_phase_heatmap: Data for user x phase heatmap
        - specialization_pie: Data for specialization level pie chart
        - phase_totals_bar: Data for phase totals bar chart
        - user_rankings: Users ranked by specialization

    Example:
        >>> from historical_data_loader import load_change_history
        >>> changes = load_change_history()
        >>> viz_data = get_phase_distribution_visualization_data(changes)
        >>> # Use viz_data to render charts
    """
    summary = get_phase_distribution_summary(changes)

    # Prepare heatmap data (users x phases)
    heatmap_data = []
    for user, dist in sorted(summary.users.items()):
        user_row = {
            "user": user,
            "phases": {},
            "primary_phase": dist.primary_phase,
            "specialization": dist.specialization_level.value,
        }
        for phase in ALL_PHASES:
            phase_dist = dist.phases.get(phase)
            if phase_dist:
                user_row["phases"][phase] = {
                    "count": phase_dist.count,
                    "percentage": phase_dist.percentage,
                }
            else:
                user_row["phases"][phase] = {"count": 0, "percentage": 0}
        heatmap_data.append(user_row)

    # Prepare specialization pie chart data
    specialization_counts = defaultdict(int)
    for dist in summary.users.values():
        specialization_counts[dist.specialization_level.value] += 1

    specialization_colors = {
        SpecializationLevel.SPECIALIST.value: "#1B5E20",     # Dark green
        SpecializationLevel.FOCUSED.value: "#388E3C",        # Green
        SpecializationLevel.BALANCED.value: "#8B6914",       # Gold
        SpecializationLevel.GENERALIST.value: "#1565C0",     # Blue
    }

    specialization_pie = [
        {
            "label": level.replace("_", " ").title(),
            "value": count,
            "color": specialization_colors.get(level, "#757575"),
        }
        for level, count in specialization_counts.items()
    ]

    # Prepare phase totals bar chart data
    phase_colors = {
        1: "#4CAF50",  # Green
        2: "#2196F3",  # Blue
        3: "#FF9800",  # Orange
        4: "#9C27B0",  # Purple
        5: "#F44336",  # Red
    }

    phase_totals_bar = [
        {
            "phase": phase,
            "phase_name": _get_phase_name(phase),
            "total_items": summary.overall_phase_distribution.get(phase, 0),
            "color": phase_colors.get(phase, "#757575"),
        }
        for phase in ALL_PHASES
    ]

    # Prepare user rankings by total items
    user_rankings = sorted(
        [
            {
                "user": user,
                "total_items": dist.total_items,
                "primary_phase": dist.primary_phase,
                "primary_phase_name": _get_phase_name(dist.primary_phase) if dist.primary_phase else None,
                "primary_phase_percentage": dist.primary_phase_percentage,
                "specialization_level": dist.specialization_level.value,
                "gini": dist.gini_coefficient,
            }
            for user, dist in summary.users.items()
        ],
        key=lambda x: x["total_items"],
        reverse=True
    )

    # Add rank
    for rank, user_data in enumerate(user_rankings, 1):
        user_data["rank"] = rank

    return {
        "user_phase_heatmap": heatmap_data,
        "specialization_pie": specialization_pie,
        "phase_totals_bar": phase_totals_bar,
        "user_rankings": user_rankings,
        "summary": {
            "total_users": summary.total_users,
            "total_items": summary.total_items,
            "specialists_count": len(summary.specialists),
            "generalists_count": len(summary.generalists),
            "average_gini": round(summary.average_gini, 3),
        },
        "phase_leaders": {
            phase: {
                "user": leader[0],
                "count": leader[1],
                "phase_name": _get_phase_name(phase),
            }
            for phase, leader in summary.phase_leaders.items()
        },
    }


def get_specialization_insights(
    changes: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Generate insights about user specialization patterns.

    Analyzes the distribution data to provide actionable insights
    about workload distribution and specialization patterns.

    Args:
        changes: List of change records

    Returns:
        Dictionary containing insights and recommendations
    """
    summary = get_phase_distribution_summary(changes)

    if not summary.users:
        return {"insights": [], "recommendations": []}

    insights = []
    recommendations = []

    # Insight: Overall team specialization balance
    specialist_ratio = len(summary.specialists) / summary.total_users if summary.total_users > 0 else 0
    if specialist_ratio > 0.5:
        insights.append({
            "type": "warning",
            "title": "High Specialization",
            "message": f"{len(summary.specialists)} of {summary.total_users} users are specialists. "
                      "This may create bottlenecks if specialists are unavailable.",
        })
        recommendations.append({
            "priority": "medium",
            "action": "Consider cross-training specialists to increase team flexibility.",
        })
    elif specialist_ratio < 0.2 and summary.total_users >= 3:
        insights.append({
            "type": "info",
            "title": "Generalist Team",
            "message": "Most team members work across multiple phases. "
                      "This provides good flexibility but may indicate lack of deep expertise.",
        })

    # Insight: Phase coverage
    phases_with_activity = len([p for p in ALL_PHASES if summary.overall_phase_distribution.get(p, 0) > 0])
    if phases_with_activity < len(ALL_PHASES):
        inactive_phases = [p for p in ALL_PHASES if summary.overall_phase_distribution.get(p, 0) == 0]
        insights.append({
            "type": "info",
            "title": "Inactive Phases",
            "message": f"Phase(s) {inactive_phases} have no activity in this period.",
        })

    # Insight: Phase imbalance
    if summary.overall_phase_distribution:
        phase_values = [summary.overall_phase_distribution.get(p, 0) for p in ALL_PHASES]
        if phase_values and max(phase_values) > 0:
            max_phase = max(phase_values)
            min_nonzero = min(v for v in phase_values if v > 0)
            if max_phase > min_nonzero * 5:  # 5x difference
                max_phase_num = [p for p, v in summary.overall_phase_distribution.items() if v == max_phase][0]
                insights.append({
                    "type": "warning",
                    "title": "Phase Imbalance",
                    "message": f"Phase {max_phase_num} ({_get_phase_name(max_phase_num)}) has significantly more "
                              f"activity than other phases. This may indicate a bottleneck downstream.",
                })

    # Insight: Single points of failure
    for phase, (user, count) in summary.phase_leaders.items():
        total_phase = summary.overall_phase_distribution.get(phase, 0)
        if total_phase > 0:
            user_share = (count / total_phase) * 100
            if user_share > 70:
                insights.append({
                    "type": "warning",
                    "title": f"Single Point of Failure: Phase {phase}",
                    "message": f"User {user} handles {user_share:.0f}% of Phase {phase} ({_get_phase_name(phase)}). "
                              "This creates risk if they are unavailable.",
                })
                recommendations.append({
                    "priority": "high",
                    "action": f"Cross-train additional users on Phase {phase} to reduce dependency on {user}.",
                })

    return {
        "insights": insights,
        "recommendations": recommendations,
        "statistics": {
            "total_users": summary.total_users,
            "specialists": summary.specialists,
            "generalists": summary.generalists,
            "average_gini": round(summary.average_gini, 3),
            "phase_coverage": f"{phases_with_activity}/{len(ALL_PHASES)}",
        },
    }


def format_phase_distribution_report(
    changes: List[Dict[str, Any]]
) -> str:
    """Format phase distribution metrics as a text report.

    Args:
        changes: List of change records from historical data

    Returns:
        Formatted text report string
    """
    summary = get_phase_distribution_summary(changes)

    lines = []
    lines.append("=" * 90)
    lines.append("USER PHASE DISTRIBUTION REPORT")
    lines.append("=" * 90)

    if not summary.users:
        lines.append("\nNo user data available.")
        return "\n".join(lines)

    # Date range
    if summary.date_range_start and summary.date_range_end:
        lines.append(f"\nPeriod: {summary.date_range_start} to {summary.date_range_end}")

    # Summary section
    lines.append("\n" + "-" * 90)
    lines.append("SUMMARY")
    lines.append("-" * 90)
    lines.append(f"  Total Users: {summary.total_users}")
    lines.append(f"  Total Items Processed: {summary.total_items}")
    lines.append(f"  Average Gini Coefficient: {summary.average_gini:.3f}")
    lines.append(f"  Specialists: {len(summary.specialists)} - {summary.specialists}")
    lines.append(f"  Generalists: {len(summary.generalists)} - {summary.generalists}")

    # Overall phase distribution
    lines.append("\n" + "-" * 90)
    lines.append("OVERALL PHASE DISTRIBUTION")
    lines.append("-" * 90)

    total = sum(summary.overall_phase_distribution.values())
    for phase in ALL_PHASES:
        count = summary.overall_phase_distribution.get(phase, 0)
        pct = (count / total * 100) if total > 0 else 0
        bar = "#" * int(pct / 2)  # Simple bar visualization
        leader = summary.phase_leaders.get(phase, ("N/A", 0))
        lines.append(
            f"  Phase {phase} ({_get_phase_name(phase)[:15]:<15}): "
            f"{count:>6} ({pct:>5.1f}%) {bar:<25} [Leader: {leader[0]}]"
        )

    # User details
    lines.append("\n" + "-" * 90)
    lines.append("USER DETAILS")
    lines.append("-" * 90)
    lines.append(f"{'User':<10} {'Total':>8} {'Primary':>8} {'%':>6} {'Gini':>6} {'Specialization':<15} {'Phases Active':>14}")
    lines.append("-" * 90)

    # Sort users by total items
    sorted_users = sorted(
        summary.users.items(),
        key=lambda x: x[1].total_items,
        reverse=True
    )

    for user, dist in sorted_users:
        lines.append(
            f"{user:<10} "
            f"{dist.total_items:>8} "
            f"{dist.primary_phase or 'N/A':>8} "
            f"{dist.primary_phase_percentage:>5.1f}% "
            f"{dist.gini_coefficient:>6.3f} "
            f"{dist.specialization_level.value:<15} "
            f"{dist.phases_active:>14}"
        )

    # Phase breakdown per user
    lines.append("\n" + "-" * 90)
    lines.append("PHASE BREAKDOWN PER USER")
    lines.append("-" * 90)
    lines.append(f"{'User':<10} {'Phase 1':>10} {'Phase 2':>10} {'Phase 3':>10} {'Phase 4':>10} {'Phase 5':>10}")
    lines.append("-" * 90)

    for user, dist in sorted_users:
        phase_strs = []
        for phase in ALL_PHASES:
            phase_dist = dist.phases.get(phase)
            if phase_dist and phase_dist.count > 0:
                phase_strs.append(f"{phase_dist.count:>4}({phase_dist.percentage:>4.0f}%)")
            else:
                phase_strs.append(f"{'0':>4}({'0':>4}%)")

        lines.append(f"{user:<10} " + " ".join(f"{s:>10}" for s in phase_strs))

    lines.append("\n" + "=" * 90)

    return "\n".join(lines)


if __name__ == "__main__":
    # Demo usage
    import random

    print("Phase Distribution Per User Calculator - Demo")
    print("=" * 80)

    # Create sample data
    sample_changes = []
    groups = ["NA", "NF", "NH", "NM", "NP", "NT", "NV"]
    users = ["DM", "JHU", "HI", "MK", "AS"]

    # Create realistic distribution patterns
    # DM: Specialist in Phase 1
    # JHU: Focused on Phase 2-3
    # HI: Generalist
    # MK: Specialist in Phase 4
    # AS: Balanced

    user_phase_weights = {
        "DM": {1: 70, 2: 15, 3: 5, 4: 5, 5: 5},
        "JHU": {1: 10, 2: 40, 3: 35, 4: 10, 5: 5},
        "HI": {1: 20, 2: 20, 3: 20, 4: 20, 5: 20},
        "MK": {1: 5, 2: 10, 3: 10, 4: 65, 5: 10},
        "AS": {1: 25, 2: 25, 3: 25, 4: 15, 5: 10},
    }

    for i in range(500):
        group = random.choice(groups)
        user = random.choice(users)

        # Select phase based on user weights
        weights = user_phase_weights[user]
        phases = list(weights.keys())
        probabilities = [weights[p] / 100 for p in phases]
        phase = random.choices(phases, weights=probabilities)[0]

        row_id = f"row_{i % 100}"

        sample_changes.append({
            "Group": group,
            "RowID": row_id,
            "Phase": str(phase),
            "User": user,
            "ParsedTimestamp": datetime.now() - timedelta(days=random.randint(0, 30)),
        })

    # Calculate and print report
    print(format_phase_distribution_report(sample_changes))

    # Print insights
    print("\n" + "=" * 80)
    print("INSIGHTS & RECOMMENDATIONS")
    print("=" * 80)
    insights = get_specialization_insights(sample_changes)

    for insight in insights["insights"]:
        print(f"\n[{insight['type'].upper()}] {insight['title']}")
        print(f"  {insight['message']}")

    print("\nRecommendations:")
    for rec in insights["recommendations"]:
        print(f"  [{rec['priority'].upper()}] {rec['action']}")

    # Print visualization data summary
    print("\n" + "=" * 80)
    print("VISUALIZATION DATA (excerpt)")
    print("=" * 80)
    viz_data = get_phase_distribution_visualization_data(sample_changes)
    import json
    print(json.dumps(viz_data["summary"], indent=2))
    print("\nPhase Leaders:")
    print(json.dumps(viz_data["phase_leaders"], indent=2, default=str))
