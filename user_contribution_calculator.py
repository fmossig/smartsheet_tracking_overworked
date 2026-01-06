"""
User Contribution Percentage Calculator Module

Calculates each user's contribution as a percentage of total team activity.
Provides detailed attribution breakdowns by group and phase.

This module follows the established patterns from user_productivity_calculator.py
and integrates with the historical_data_loader for data access.

Usage:
    from user_contribution_calculator import (
        UserGroupContribution,
        UserPhaseContribution,
        UserTeamContribution,
        TeamContributionSummary,
        calculate_user_group_contribution,
        calculate_user_phase_contribution,
        calculate_user_team_contribution,
        calculate_all_user_contributions,
        get_contribution_summary,
        get_contribution_visualization_data,
        format_contribution_report,
    )
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Any, Union, Tuple
from collections import defaultdict

# Set up logging
logger = logging.getLogger(__name__)


# Constants for phase definitions (aligned with other calculators)
ALL_PHASES = [1, 2, 3, 4, 5]

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


@dataclass
class UserPhaseContribution:
    """User's contribution within a specific phase.

    Attributes:
        user: User identifier (initials like "JHU", "DM")
        phase: Phase number (1-5)
        phase_name: Human-readable phase name
        items_count: Number of items processed in this phase by the user
        percentage_of_phase_total: User's percentage of all activity in this phase
        percentage_of_user_total: What percentage of user's total activity is in this phase
    """
    user: str
    phase: int
    phase_name: str
    items_count: int = 0
    percentage_of_phase_total: float = 0.0
    percentage_of_user_total: float = 0.0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "user": self.user,
            "phase": self.phase,
            "phase_name": self.phase_name,
            "items_count": self.items_count,
            "percentage_of_phase_total": round(self.percentage_of_phase_total, 2),
            "percentage_of_user_total": round(self.percentage_of_user_total, 2),
        }


@dataclass
class UserGroupContribution:
    """User's contribution within a specific group.

    Attributes:
        user: User identifier (initials like "JHU", "DM")
        group: Group code (NA, NF, NH, etc.)
        items_count: Number of items processed in this group by the user
        percentage_of_group_total: User's percentage of all activity in this group
        percentage_of_user_total: What percentage of user's total activity is in this group
        percentage_of_team_total: User's contribution in this group as percentage of all team activity
        items_per_phase: Count of items per phase within this group
        phase_contributions: Detailed phase breakdown with percentages
    """
    user: str
    group: str
    items_count: int = 0
    percentage_of_group_total: float = 0.0
    percentage_of_user_total: float = 0.0
    percentage_of_team_total: float = 0.0
    items_per_phase: Dict[int, int] = field(default_factory=dict)
    phase_contributions: List['UserPhaseContribution'] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "user": self.user,
            "group": self.group,
            "items_count": self.items_count,
            "percentage_of_group_total": round(self.percentage_of_group_total, 2),
            "percentage_of_user_total": round(self.percentage_of_user_total, 2),
            "percentage_of_team_total": round(self.percentage_of_team_total, 2),
            "items_per_phase": {str(k): v for k, v in self.items_per_phase.items()},
            "phase_contributions": [pc.to_dict() for pc in self.phase_contributions],
        }


@dataclass
class UserTeamContribution:
    """User's overall contribution to team activity.

    Attributes:
        user: User identifier (initials like "JHU", "DM")
        total_items: Total number of items processed by this user
        percentage_of_team_total: User's percentage of all team activity
        by_group: Breakdown of contribution by group
        by_phase: Breakdown of contribution by phase
        rank: User's rank among all team members (1 = highest contributor)
        unique_groups: Number of different groups the user contributed to
        unique_rows: Number of unique row IDs processed
        date_range_start: Start date of the analysis period
        date_range_end: End date of the analysis period
    """
    user: str
    total_items: int = 0
    percentage_of_team_total: float = 0.0
    by_group: Dict[str, UserGroupContribution] = field(default_factory=dict)
    by_phase: Dict[int, UserPhaseContribution] = field(default_factory=dict)
    rank: int = 0
    unique_groups: int = 0
    unique_rows: int = 0
    date_range_start: Optional[date] = None
    date_range_end: Optional[date] = None

    @property
    def days_in_range(self) -> int:
        """Get the number of days in the analysis range."""
        if self.date_range_start and self.date_range_end:
            return (self.date_range_end - self.date_range_start).days + 1
        return 0

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "user": self.user,
            "total_items": self.total_items,
            "percentage_of_team_total": round(self.percentage_of_team_total, 2),
            "by_group": {
                group: contrib.to_dict()
                for group, contrib in self.by_group.items()
            },
            "by_phase": {
                str(phase): contrib.to_dict()
                for phase, contrib in self.by_phase.items()
            },
            "rank": self.rank,
            "unique_groups": self.unique_groups,
            "unique_rows": self.unique_rows,
            "date_range_start": self.date_range_start.isoformat() if self.date_range_start else None,
            "date_range_end": self.date_range_end.isoformat() if self.date_range_end else None,
            "days_in_range": self.days_in_range,
        }


@dataclass
class TeamContributionSummary:
    """Summary of team contributions with rankings and statistics.

    Attributes:
        total_team_activity: Total number of changes across all users
        total_users: Number of users with activity
        user_contributions: Dictionary mapping users to their contribution metrics
        by_group_totals: Total activity per group
        by_phase_totals: Total activity per phase
        top_contributors: List of top N contributors
        date_range_start: Start date of the analysis period
        date_range_end: End date of the analysis period
    """
    total_team_activity: int = 0
    total_users: int = 0
    user_contributions: Dict[str, UserTeamContribution] = field(default_factory=dict)
    by_group_totals: Dict[str, int] = field(default_factory=dict)
    by_phase_totals: Dict[int, int] = field(default_factory=dict)
    top_contributors: List[Dict[str, Any]] = field(default_factory=list)
    date_range_start: Optional[date] = None
    date_range_end: Optional[date] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_team_activity": self.total_team_activity,
            "total_users": self.total_users,
            "user_contributions": {
                user: contrib.to_dict()
                for user, contrib in self.user_contributions.items()
            },
            "by_group_totals": self.by_group_totals,
            "by_phase_totals": {str(k): v for k, v in self.by_phase_totals.items()},
            "top_contributors": self.top_contributors,
            "date_range_start": self.date_range_start.isoformat() if self.date_range_start else None,
            "date_range_end": self.date_range_end.isoformat() if self.date_range_end else None,
        }


def _extract_totals(changes: List[Dict[str, Any]]) -> Tuple[int, Dict[str, int], Dict[int, int], Dict[str, Dict[int, int]]]:
    """Extract total counts by various dimensions.

    Args:
        changes: List of change records

    Returns:
        Tuple of:
        - total_count: Total number of changes
        - by_group: Dict mapping groups to counts
        - by_phase: Dict mapping phases to counts
        - by_group_phase: Dict mapping groups to phase counts
    """
    total_count = len(changes)
    by_group: Dict[str, int] = defaultdict(int)
    by_phase: Dict[int, int] = defaultdict(int)
    by_group_phase: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))

    for change in changes:
        group = change.get("Group")
        phase = change.get("Phase")

        if group:
            by_group[group] += 1

        if phase:
            try:
                phase_num = int(phase)
                by_phase[phase_num] += 1
                if group:
                    by_group_phase[group][phase_num] += 1
            except (ValueError, TypeError):
                pass

    return total_count, dict(by_group), dict(by_phase), {k: dict(v) for k, v in by_group_phase.items()}


def _extract_user_data(
    changes: List[Dict[str, Any]]
) -> Dict[str, Dict[str, Any]]:
    """Extract user-specific data from changes.

    Args:
        changes: List of change records

    Returns:
        Dictionary mapping users to their activity data:
        - changes: List of their change records
        - by_group: Dict mapping groups to counts
        - by_phase: Dict mapping phases to counts
        - by_group_phase: Dict mapping groups to phase counts
        - unique_rows: Set of unique row IDs
    """
    user_data: Dict[str, Dict[str, Any]] = defaultdict(lambda: {
        "changes": [],
        "by_group": defaultdict(int),
        "by_phase": defaultdict(int),
        "by_group_phase": defaultdict(lambda: defaultdict(int)),
        "unique_rows": set(),
    })

    for change in changes:
        user = change.get("User")
        if not user:
            continue

        user_data[user]["changes"].append(change)

        group = change.get("Group")
        phase = change.get("Phase")
        row_id = change.get("RowID")

        if group:
            user_data[user]["by_group"][group] += 1

        if phase:
            try:
                phase_num = int(phase)
                user_data[user]["by_phase"][phase_num] += 1
                if group:
                    user_data[user]["by_group_phase"][group][phase_num] += 1
            except (ValueError, TypeError):
                pass

        if row_id:
            user_data[user]["unique_rows"].add(row_id)

    return dict(user_data)


def calculate_user_group_contribution(
    changes: List[Dict[str, Any]],
    user: str,
    group: str
) -> UserGroupContribution:
    """Calculate a user's contribution within a specific group.

    Args:
        changes: List of change records from historical data
        user: User identifier to analyze
        group: Group code to analyze

    Returns:
        UserGroupContribution object with contribution metrics

    Example:
        >>> from historical_data_loader import load_change_history
        >>> changes = load_change_history()
        >>> contribution = calculate_user_group_contribution(changes, "JHU", "NA")
        >>> print(f"Contribution: {contribution.percentage_of_group_total}%")
    """
    if not changes:
        return UserGroupContribution(user=user, group=group)

    # Get total counts
    total_count, by_group_totals, by_phase_totals, by_group_phase = _extract_totals(changes)
    group_total = by_group_totals.get(group, 0)

    # Get user's data
    user_data = _extract_user_data(changes)
    if user not in user_data:
        return UserGroupContribution(user=user, group=group)

    user_info = user_data[user]
    user_total = len(user_info["changes"])
    user_group_count = user_info["by_group"].get(group, 0)
    user_group_phases = user_info["by_group_phase"].get(group, {})

    # Calculate percentages
    pct_of_group = (user_group_count / group_total * 100) if group_total > 0 else 0.0
    pct_of_user = (user_group_count / user_total * 100) if user_total > 0 else 0.0
    pct_of_team = (user_group_count / total_count * 100) if total_count > 0 else 0.0

    # Build phase contributions within this group
    group_phase_totals = by_group_phase.get(group, {})
    phase_contributions = []
    for phase in ALL_PHASES:
        user_phase_count = user_group_phases.get(phase, 0)
        phase_total = group_phase_totals.get(phase, 0)
        pct_of_phase = (user_phase_count / phase_total * 100) if phase_total > 0 else 0.0
        pct_of_user_total = (user_phase_count / user_group_count * 100) if user_group_count > 0 else 0.0

        if user_phase_count > 0:
            phase_contributions.append(UserPhaseContribution(
                user=user,
                phase=phase,
                phase_name=_get_phase_name(phase),
                items_count=user_phase_count,
                percentage_of_phase_total=pct_of_phase,
                percentage_of_user_total=pct_of_user_total,
            ))

    return UserGroupContribution(
        user=user,
        group=group,
        items_count=user_group_count,
        percentage_of_group_total=pct_of_group,
        percentage_of_user_total=pct_of_user,
        percentage_of_team_total=pct_of_team,
        items_per_phase=dict(user_group_phases),
        phase_contributions=phase_contributions,
    )


def calculate_user_phase_contribution(
    changes: List[Dict[str, Any]],
    user: str,
    phase: int
) -> UserPhaseContribution:
    """Calculate a user's contribution within a specific phase.

    Args:
        changes: List of change records from historical data
        user: User identifier to analyze
        phase: Phase number to analyze (1-5)

    Returns:
        UserPhaseContribution object with contribution metrics

    Example:
        >>> from historical_data_loader import load_change_history
        >>> changes = load_change_history()
        >>> contribution = calculate_user_phase_contribution(changes, "DM", 1)
        >>> print(f"Phase 1 contribution: {contribution.percentage_of_phase_total}%")
    """
    if not changes:
        return UserPhaseContribution(
            user=user,
            phase=phase,
            phase_name=_get_phase_name(phase)
        )

    # Get total counts
    total_count, _, by_phase_totals, _ = _extract_totals(changes)
    phase_total = by_phase_totals.get(phase, 0)

    # Get user's data
    user_data = _extract_user_data(changes)
    if user not in user_data:
        return UserPhaseContribution(
            user=user,
            phase=phase,
            phase_name=_get_phase_name(phase)
        )

    user_info = user_data[user]
    user_total = len(user_info["changes"])
    user_phase_count = user_info["by_phase"].get(phase, 0)

    # Calculate percentages
    pct_of_phase = (user_phase_count / phase_total * 100) if phase_total > 0 else 0.0
    pct_of_user = (user_phase_count / user_total * 100) if user_total > 0 else 0.0

    return UserPhaseContribution(
        user=user,
        phase=phase,
        phase_name=_get_phase_name(phase),
        items_count=user_phase_count,
        percentage_of_phase_total=pct_of_phase,
        percentage_of_user_total=pct_of_user,
    )


def calculate_user_team_contribution(
    changes: List[Dict[str, Any]],
    user: str
) -> UserTeamContribution:
    """Calculate a user's overall contribution to team activity.

    Args:
        changes: List of change records from historical data
        user: User identifier to analyze

    Returns:
        UserTeamContribution object with comprehensive contribution metrics

    Example:
        >>> from historical_data_loader import load_change_history
        >>> changes = load_change_history()
        >>> contribution = calculate_user_team_contribution(changes, "JHU")
        >>> print(f"Team contribution: {contribution.percentage_of_team_total}%")
    """
    if not changes:
        return UserTeamContribution(user=user)

    # Get total counts
    total_count, by_group_totals, by_phase_totals, by_group_phase = _extract_totals(changes)

    # Get user's data
    user_data = _extract_user_data(changes)
    if user not in user_data:
        return UserTeamContribution(user=user)

    user_info = user_data[user]
    user_total = len(user_info["changes"])

    # Calculate date range
    dates = [c.get('ParsedTimestamp') or c.get('ParsedDate') for c in changes]
    dates = [d.date() if hasattr(d, 'date') else d for d in dates if d]
    date_range_start = min(dates) if dates else None
    date_range_end = max(dates) if dates else None

    # Calculate team percentage
    pct_of_team = (user_total / total_count * 100) if total_count > 0 else 0.0

    # Calculate rank among all users
    all_user_counts = [(u, len(d["changes"])) for u, d in user_data.items()]
    all_user_counts.sort(key=lambda x: x[1], reverse=True)
    rank = next((i + 1 for i, (u, _) in enumerate(all_user_counts) if u == user), 0)

    # Build group contributions
    by_group = {}
    for group in user_info["by_group"].keys():
        by_group[group] = calculate_user_group_contribution(changes, user, group)

    # Build phase contributions
    by_phase = {}
    for phase in ALL_PHASES:
        phase_contrib = calculate_user_phase_contribution(changes, user, phase)
        if phase_contrib.items_count > 0:
            by_phase[phase] = phase_contrib

    return UserTeamContribution(
        user=user,
        total_items=user_total,
        percentage_of_team_total=pct_of_team,
        by_group=by_group,
        by_phase=by_phase,
        rank=rank,
        unique_groups=len(user_info["by_group"]),
        unique_rows=len(user_info["unique_rows"]),
        date_range_start=date_range_start,
        date_range_end=date_range_end,
    )


def calculate_all_user_contributions(
    changes: List[Dict[str, Any]]
) -> Dict[str, UserTeamContribution]:
    """Calculate contribution metrics for all users in the data.

    Args:
        changes: List of change records from historical data

    Returns:
        Dictionary mapping user identifiers to UserTeamContribution objects

    Example:
        >>> from historical_data_loader import load_change_history
        >>> changes = load_change_history()
        >>> all_contributions = calculate_all_user_contributions(changes)
        >>> for user, contrib in all_contributions.items():
        ...     print(f"{user}: {contrib.percentage_of_team_total:.1f}%")
    """
    if not changes:
        return {}

    # Get all unique users
    users = set(c.get("User") for c in changes if c.get("User"))

    result = {}
    for user in users:
        result[user] = calculate_user_team_contribution(changes, user)

    return result


def get_contribution_summary(
    changes: List[Dict[str, Any]],
    top_n: int = 5
) -> TeamContributionSummary:
    """Get a comprehensive summary of team contributions.

    Args:
        changes: List of change records from historical data
        top_n: Number of top contributors to include in summary

    Returns:
        TeamContributionSummary object with comprehensive analysis

    Example:
        >>> from historical_data_loader import load_change_history
        >>> changes = load_change_history()
        >>> summary = get_contribution_summary(changes)
        >>> print(f"Total team activity: {summary.total_team_activity}")
        >>> for contributor in summary.top_contributors:
        ...     print(f"{contributor['user']}: {contributor['percentage']}%")
    """
    if not changes:
        return TeamContributionSummary()

    # Get total counts
    total_count, by_group_totals, by_phase_totals, _ = _extract_totals(changes)

    # Get all user contributions
    user_contributions = calculate_all_user_contributions(changes)

    # Calculate date range
    dates = [c.get('ParsedTimestamp') or c.get('ParsedDate') for c in changes]
    dates = [d.date() if hasattr(d, 'date') else d for d in dates if d]
    date_range_start = min(dates) if dates else None
    date_range_end = max(dates) if dates else None

    # Build top contributors list
    sorted_contributors = sorted(
        user_contributions.values(),
        key=lambda x: x.total_items,
        reverse=True
    )

    top_contributors = []
    for contrib in sorted_contributors[:top_n]:
        top_contributors.append({
            "user": contrib.user,
            "total_items": contrib.total_items,
            "percentage": round(contrib.percentage_of_team_total, 2),
            "rank": contrib.rank,
            "unique_groups": contrib.unique_groups,
        })

    return TeamContributionSummary(
        total_team_activity=total_count,
        total_users=len(user_contributions),
        user_contributions=user_contributions,
        by_group_totals=by_group_totals,
        by_phase_totals=by_phase_totals,
        top_contributors=top_contributors,
        date_range_start=date_range_start,
        date_range_end=date_range_end,
    )


def get_contribution_visualization_data(
    changes: List[Dict[str, Any]]
) -> Dict[str, Any]:
    """Get data formatted for contribution visualization.

    This function prepares data in a format suitable for rendering
    charts in PDF reports or other visualizations.

    Args:
        changes: List of change records

    Returns:
        Dictionary with visualization-ready data

    Example:
        >>> changes = load_change_history()
        >>> viz_data = get_contribution_visualization_data(changes)
        >>> # Use viz_data to render contribution charts
    """
    summary = get_contribution_summary(changes)

    # Prepare pie chart data for user contributions
    pie_chart_data = []
    for user, contrib in summary.user_contributions.items():
        pie_chart_data.append({
            "user": user,
            "value": contrib.total_items,
            "percentage": round(contrib.percentage_of_team_total, 2),
            "rank": contrib.rank,
        })

    # Sort by percentage descending
    pie_chart_data.sort(key=lambda x: x["percentage"], reverse=True)

    # Prepare group breakdown data
    group_breakdown = []
    for group, total in sorted(summary.by_group_totals.items()):
        group_users = []
        for user, contrib in summary.user_contributions.items():
            if group in contrib.by_group:
                group_contrib = contrib.by_group[group]
                group_users.append({
                    "user": user,
                    "items": group_contrib.items_count,
                    "percentage": round(group_contrib.percentage_of_group_total, 2),
                })

        group_users.sort(key=lambda x: x["percentage"], reverse=True)

        group_breakdown.append({
            "group": group,
            "total_items": total,
            "user_contributions": group_users,
        })

    # Prepare phase breakdown data
    phase_breakdown = []
    for phase in ALL_PHASES:
        phase_total = summary.by_phase_totals.get(phase, 0)
        if phase_total == 0:
            continue

        phase_users = []
        for user, contrib in summary.user_contributions.items():
            if phase in contrib.by_phase:
                phase_contrib = contrib.by_phase[phase]
                phase_users.append({
                    "user": user,
                    "items": phase_contrib.items_count,
                    "percentage": round(phase_contrib.percentage_of_phase_total, 2),
                })

        phase_users.sort(key=lambda x: x["percentage"], reverse=True)

        phase_breakdown.append({
            "phase": phase,
            "phase_name": _get_phase_name(phase),
            "total_items": phase_total,
            "user_contributions": phase_users,
        })

    return {
        "pie_chart_data": pie_chart_data,
        "group_breakdown": group_breakdown,
        "phase_breakdown": phase_breakdown,
        "summary": {
            "total_team_activity": summary.total_team_activity,
            "total_users": summary.total_users,
            "top_contributors": summary.top_contributors,
        },
    }


def format_contribution_report(
    changes: List[Dict[str, Any]],
    include_by_group: bool = True,
    include_by_phase: bool = True
) -> str:
    """Format contribution metrics as a text report.

    Args:
        changes: List of change records from historical data
        include_by_group: Whether to include per-group breakdown
        include_by_phase: Whether to include per-phase breakdown

    Returns:
        Formatted text report string
    """
    summary = get_contribution_summary(changes)

    lines = []
    lines.append("=" * 80)
    lines.append("USER CONTRIBUTION PERCENTAGE REPORT")
    lines.append("=" * 80)

    if summary.total_team_activity == 0:
        lines.append("\nNo activity data available.")
        return "\n".join(lines)

    # Date range
    if summary.date_range_start and summary.date_range_end:
        lines.append(f"\nPeriod: {summary.date_range_start} to {summary.date_range_end}")

    # Overall summary
    lines.append("\n" + "-" * 80)
    lines.append("TEAM SUMMARY")
    lines.append("-" * 80)
    lines.append(f"  Total Team Activity: {summary.total_team_activity} items")
    lines.append(f"  Total Users: {summary.total_users}")

    # User rankings with contribution percentages
    lines.append("\n" + "-" * 80)
    lines.append("USER CONTRIBUTION RANKINGS")
    lines.append("-" * 80)
    lines.append(f"{'Rank':<6} {'User':<10} {'Items':>10} {'% of Team':>12} {'Groups':>8} {'Unique Rows':>12}")
    lines.append("-" * 80)

    sorted_contribs = sorted(
        summary.user_contributions.values(),
        key=lambda x: x.percentage_of_team_total,
        reverse=True
    )

    for contrib in sorted_contribs:
        lines.append(
            f"{contrib.rank:<6} "
            f"{contrib.user:<10} "
            f"{contrib.total_items:>10} "
            f"{contrib.percentage_of_team_total:>11.2f}% "
            f"{contrib.unique_groups:>8} "
            f"{contrib.unique_rows:>12}"
        )

    # Per-group breakdown
    if include_by_group:
        lines.append("\n" + "-" * 80)
        lines.append("CONTRIBUTION BY GROUP")
        lines.append("-" * 80)

        for group, total in sorted(summary.by_group_totals.items()):
            lines.append(f"\n  {group} (Total: {total} items):")

            group_contribs = []
            for user, contrib in summary.user_contributions.items():
                if group in contrib.by_group:
                    gc = contrib.by_group[group]
                    group_contribs.append((user, gc))

            group_contribs.sort(key=lambda x: x[1].percentage_of_group_total, reverse=True)

            for user, gc in group_contribs:
                lines.append(
                    f"    {user}: {gc.items_count} items "
                    f"({gc.percentage_of_group_total:.1f}% of group, "
                    f"{gc.percentage_of_team_total:.1f}% of team)"
                )

    # Per-phase breakdown
    if include_by_phase:
        lines.append("\n" + "-" * 80)
        lines.append("CONTRIBUTION BY PHASE")
        lines.append("-" * 80)

        for phase in ALL_PHASES:
            phase_total = summary.by_phase_totals.get(phase, 0)
            if phase_total == 0:
                continue

            lines.append(f"\n  Phase {phase} - {_get_phase_name(phase)} (Total: {phase_total} items):")

            phase_contribs = []
            for user, contrib in summary.user_contributions.items():
                if phase in contrib.by_phase:
                    pc = contrib.by_phase[phase]
                    phase_contribs.append((user, pc))

            phase_contribs.sort(key=lambda x: x[1].percentage_of_phase_total, reverse=True)

            for user, pc in phase_contribs:
                lines.append(
                    f"    {user}: {pc.items_count} items "
                    f"({pc.percentage_of_phase_total:.1f}% of phase, "
                    f"{pc.percentage_of_user_total:.1f}% of user's total)"
                )

    lines.append("\n" + "=" * 80)

    return "\n".join(lines)


def calculate_contribution_for_range(
    date_range_filter: Any,  # DateRangeFilter type
    user: Optional[str] = None,
    changes: Optional[List[Dict[str, Any]]] = None
) -> Union[UserTeamContribution, Dict[str, UserTeamContribution]]:
    """Calculate contribution metrics for a specific date range.

    Args:
        date_range_filter: DateRangeFilter object specifying the date range
        user: Optional specific user to analyze. If None, analyzes all users
        changes: Optional pre-loaded changes (will load if not provided)

    Returns:
        If user specified: UserTeamContribution for that user
        If user is None: Dict mapping users to their contributions

    Example:
        >>> from date_range_filter import get_preset_range, DateRangePreset
        >>> last_week = get_preset_range(DateRangePreset.LAST_WEEK)
        >>> contribution = calculate_contribution_for_range(last_week, user="JHU")
        >>> print(f"Team contribution: {contribution.percentage_of_team_total}%")
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
        contribution = calculate_user_team_contribution(changes, user)
        contribution.date_range_start = date_range_filter.start_date
        contribution.date_range_end = date_range_filter.end_date
        return contribution
    else:
        all_contributions = calculate_all_user_contributions(changes)
        # Update date ranges to match filter
        for c in all_contributions.values():
            c.date_range_start = date_range_filter.start_date
            c.date_range_end = date_range_filter.end_date
        return all_contributions


if __name__ == "__main__":
    # Demo usage
    print("User Contribution Percentage Calculator - Demo")
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
    print(format_contribution_report(sample_changes))

    # Print summary
    print("\nJSON Summary (excerpt):")
    summary = get_contribution_summary(sample_changes)
    import json
    print(json.dumps({
        "total_team_activity": summary.total_team_activity,
        "total_users": summary.total_users,
        "top_contributors": summary.top_contributors,
    }, indent=2, default=str))

    # Print visualization data
    print("\nVisualization Data (excerpt):")
    viz_data = get_contribution_visualization_data(sample_changes)
    print(json.dumps(viz_data["summary"], indent=2))
