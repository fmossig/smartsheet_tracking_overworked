"""
Phase Progression Funnel Calculator Module

Calculates how many items move through each phase in sequence to visualize the
processing funnel. Identifies bottlenecks and drop-off points at each phase transition.

This module provides metrics for understanding workflow efficiency:
- Total items entering each phase
- Conversion rates between phases
- Drop-off analysis at each transition
- Bottleneck identification

Enhanced with DateRangeFilter support for custom date range analysis.

Usage:
    from phase_progression_funnel_calculator import (
        FunnelPhase,
        FunnelMetrics,
        BottleneckInfo,
        FunnelComparison,
        calculate_funnel_metrics,
        calculate_funnel_for_range,
        identify_bottlenecks,
        compare_funnel_periods,
        get_funnel_summary,
        format_funnel_report,
    )
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Any, Union, Tuple
from enum import Enum
from collections import defaultdict

from phase_field_utilities import (
    PHASE_FIELDS,
    PHASE_NAMES,
    get_phase_by_number,
)

# Set up logging
logger = logging.getLogger(__name__)


class BottleneckSeverity(Enum):
    """Severity levels for bottlenecks."""
    CRITICAL = "critical"    # >50% drop-off
    HIGH = "high"            # 30-50% drop-off
    MEDIUM = "medium"        # 15-30% drop-off
    LOW = "low"              # <15% drop-off
    NONE = "none"            # No bottleneck


@dataclass
class FunnelPhase:
    """Metrics for a single phase in the funnel.

    Attributes:
        phase_number: The phase number (1-5)
        phase_name: Human-readable name for the phase
        date_column: The date column associated with this phase
        items_entered: Number of items that have reached this phase
        items_completed: Number of items that have completed this phase (moved to next)
        completion_rate: Percentage of items that completed this phase
        drop_off_count: Number of items that didn't progress from previous phase
        drop_off_rate: Percentage of items lost from previous phase
        average_time_in_phase: Average time spent in this phase (if calculable)
        unique_users: Number of unique users who processed items in this phase
    """
    phase_number: int
    phase_name: str
    date_column: str
    items_entered: int = 0
    items_completed: int = 0
    completion_rate: float = 0.0
    drop_off_count: int = 0
    drop_off_rate: float = 0.0
    average_time_in_phase: Optional[float] = None  # in days
    unique_users: int = 0

    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        if self.items_entered > 0:
            self.completion_rate = round((self.items_completed / self.items_entered) * 100, 2)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "phase_number": self.phase_number,
            "phase_name": self.phase_name,
            "date_column": self.date_column,
            "items_entered": self.items_entered,
            "items_completed": self.items_completed,
            "completion_rate": self.completion_rate,
            "drop_off_count": self.drop_off_count,
            "drop_off_rate": self.drop_off_rate,
            "average_time_in_phase": self.average_time_in_phase,
            "unique_users": self.unique_users,
        }


@dataclass
class BottleneckInfo:
    """Information about a bottleneck in the funnel.

    Attributes:
        phase_number: The phase where the bottleneck occurs
        phase_name: Human-readable name of the phase
        severity: Severity level of the bottleneck
        drop_off_rate: Percentage of items lost at this point
        items_lost: Absolute number of items lost
        previous_phase_items: Number of items from the previous phase
        suggested_actions: List of suggested actions to address the bottleneck
    """
    phase_number: int
    phase_name: str
    severity: BottleneckSeverity
    drop_off_rate: float
    items_lost: int
    previous_phase_items: int
    suggested_actions: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Generate suggested actions based on severity."""
        if not self.suggested_actions:
            self.suggested_actions = self._generate_suggestions()

    def _generate_suggestions(self) -> List[str]:
        """Generate action suggestions based on bottleneck severity."""
        suggestions = []

        if self.severity == BottleneckSeverity.CRITICAL:
            suggestions = [
                f"Immediately investigate Phase {self.phase_number} processing",
                "Review resource allocation for this phase",
                "Check for systemic issues blocking progression",
                "Consider adding more team members to this phase",
            ]
        elif self.severity == BottleneckSeverity.HIGH:
            suggestions = [
                f"Review Phase {self.phase_number} workflow for inefficiencies",
                "Analyze common blockers preventing progression",
                "Consider process optimization for this phase",
            ]
        elif self.severity == BottleneckSeverity.MEDIUM:
            suggestions = [
                f"Monitor Phase {self.phase_number} performance",
                "Identify patterns in stalled items",
            ]
        elif self.severity == BottleneckSeverity.LOW:
            suggestions = [
                "Continue monitoring this phase",
            ]

        return suggestions

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "phase_number": self.phase_number,
            "phase_name": self.phase_name,
            "severity": self.severity.value,
            "drop_off_rate": self.drop_off_rate,
            "items_lost": self.items_lost,
            "previous_phase_items": self.previous_phase_items,
            "suggested_actions": self.suggested_actions,
        }


@dataclass
class FunnelMetrics:
    """Complete funnel metrics for a group or overall analysis.

    Attributes:
        group: Group identifier (or "ALL" for overall analysis)
        date_range_start: Start date of the analysis period
        date_range_end: End date of the analysis period
        total_items_started: Number of items that entered Phase 1
        total_items_completed: Number of items that completed all phases
        overall_completion_rate: Percentage of items completing all phases
        phases: List of FunnelPhase objects with per-phase metrics
        bottlenecks: List of identified bottlenecks
        biggest_bottleneck: The phase with the highest drop-off
        funnel_efficiency: Overall efficiency score (0-100)
    """
    group: str
    date_range_start: Optional[date] = None
    date_range_end: Optional[date] = None
    total_items_started: int = 0
    total_items_completed: int = 0
    overall_completion_rate: float = 0.0
    phases: List[FunnelPhase] = field(default_factory=list)
    bottlenecks: List[BottleneckInfo] = field(default_factory=list)
    biggest_bottleneck: Optional[int] = None
    funnel_efficiency: float = 0.0

    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        if self.total_items_started > 0:
            self.overall_completion_rate = round(
                (self.total_items_completed / self.total_items_started) * 100, 2
            )
            self._calculate_efficiency()

    def _calculate_efficiency(self) -> None:
        """Calculate overall funnel efficiency score.

        Efficiency is a weighted score based on:
        - Overall completion rate (40%)
        - Average phase completion rate (30%)
        - Absence of critical bottlenecks (30%)
        """
        if not self.phases:
            self.funnel_efficiency = 0.0
            return

        # Component 1: Overall completion rate (40%)
        completion_component = self.overall_completion_rate * 0.4

        # Component 2: Average phase completion rate (30%)
        avg_phase_completion = sum(p.completion_rate for p in self.phases) / len(self.phases)
        phase_component = avg_phase_completion * 0.3

        # Component 3: Bottleneck penalty (30%)
        # Starts at 100, reduced based on bottleneck severity
        bottleneck_score = 100.0
        for bn in self.bottlenecks:
            if bn.severity == BottleneckSeverity.CRITICAL:
                bottleneck_score -= 40
            elif bn.severity == BottleneckSeverity.HIGH:
                bottleneck_score -= 20
            elif bn.severity == BottleneckSeverity.MEDIUM:
                bottleneck_score -= 10
            elif bn.severity == BottleneckSeverity.LOW:
                bottleneck_score -= 5
        bottleneck_score = max(0, bottleneck_score)
        bottleneck_component = bottleneck_score * 0.3

        self.funnel_efficiency = round(
            completion_component + phase_component + bottleneck_component, 1
        )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "group": self.group,
            "date_range_start": self.date_range_start.isoformat() if self.date_range_start else None,
            "date_range_end": self.date_range_end.isoformat() if self.date_range_end else None,
            "total_items_started": self.total_items_started,
            "total_items_completed": self.total_items_completed,
            "overall_completion_rate": self.overall_completion_rate,
            "phases": [p.to_dict() for p in self.phases],
            "bottlenecks": [b.to_dict() for b in self.bottlenecks],
            "biggest_bottleneck": self.biggest_bottleneck,
            "funnel_efficiency": self.funnel_efficiency,
        }


@dataclass
class FunnelComparison:
    """Comparison of funnel metrics between two periods.

    Attributes:
        current_period: FunnelMetrics for the current period
        previous_period: FunnelMetrics for the previous period
        completion_rate_change: Change in overall completion rate
        efficiency_change: Change in funnel efficiency
        phase_changes: Changes in each phase's metrics
        new_bottlenecks: Bottlenecks that appeared in current period
        resolved_bottlenecks: Bottlenecks that were resolved from previous period
        trend: Overall trend direction
    """
    current_period: FunnelMetrics
    previous_period: FunnelMetrics
    completion_rate_change: float = 0.0
    efficiency_change: float = 0.0
    phase_changes: List[Dict[str, Any]] = field(default_factory=list)
    new_bottlenecks: List[int] = field(default_factory=list)
    resolved_bottlenecks: List[int] = field(default_factory=list)
    trend: str = "flat"

    def __post_init__(self):
        """Calculate comparison metrics after initialization."""
        self._calculate_changes()

    def _calculate_changes(self) -> None:
        """Calculate all comparison metrics."""
        # Calculate overall changes
        self.completion_rate_change = round(
            self.current_period.overall_completion_rate -
            self.previous_period.overall_completion_rate, 2
        )
        self.efficiency_change = round(
            self.current_period.funnel_efficiency -
            self.previous_period.funnel_efficiency, 2
        )

        # Determine trend
        if self.efficiency_change >= 5:
            self.trend = "improving"
        elif self.efficiency_change <= -5:
            self.trend = "declining"
        else:
            self.trend = "stable"

        # Calculate phase-by-phase changes
        self.phase_changes = []
        current_phases = {p.phase_number: p for p in self.current_period.phases}
        previous_phases = {p.phase_number: p for p in self.previous_period.phases}

        all_phases = set(current_phases.keys()) | set(previous_phases.keys())
        for phase_num in sorted(all_phases):
            curr = current_phases.get(phase_num)
            prev = previous_phases.get(phase_num)

            if curr and prev:
                self.phase_changes.append({
                    "phase_number": phase_num,
                    "items_entered_change": curr.items_entered - prev.items_entered,
                    "completion_rate_change": round(curr.completion_rate - prev.completion_rate, 2),
                    "drop_off_change": round(curr.drop_off_rate - prev.drop_off_rate, 2),
                })

        # Identify new and resolved bottlenecks
        current_bottleneck_phases = {b.phase_number for b in self.current_period.bottlenecks
                                      if b.severity in [BottleneckSeverity.CRITICAL, BottleneckSeverity.HIGH]}
        previous_bottleneck_phases = {b.phase_number for b in self.previous_period.bottlenecks
                                       if b.severity in [BottleneckSeverity.CRITICAL, BottleneckSeverity.HIGH]}

        self.new_bottlenecks = list(current_bottleneck_phases - previous_bottleneck_phases)
        self.resolved_bottlenecks = list(previous_bottleneck_phases - current_bottleneck_phases)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "current_period": self.current_period.to_dict(),
            "previous_period": self.previous_period.to_dict(),
            "completion_rate_change": self.completion_rate_change,
            "efficiency_change": self.efficiency_change,
            "phase_changes": self.phase_changes,
            "new_bottlenecks": self.new_bottlenecks,
            "resolved_bottlenecks": self.resolved_bottlenecks,
            "trend": self.trend,
        }


def _get_severity(drop_off_rate: float) -> BottleneckSeverity:
    """Determine bottleneck severity based on drop-off rate."""
    if drop_off_rate >= 50:
        return BottleneckSeverity.CRITICAL
    elif drop_off_rate >= 30:
        return BottleneckSeverity.HIGH
    elif drop_off_rate >= 15:
        return BottleneckSeverity.MEDIUM
    elif drop_off_rate > 0:
        return BottleneckSeverity.LOW
    else:
        return BottleneckSeverity.NONE


def calculate_funnel_metrics(
    changes: List[Dict[str, Any]],
    group: Optional[str] = None,
    include_bottlenecks: bool = True
) -> FunnelMetrics:
    """Calculate funnel metrics from change history data.

    This function analyzes change records to determine how items progress
    through each phase of the workflow, identifying where items drop off
    and calculating conversion rates.

    Args:
        changes: List of change records from historical data
        group: Optional group to filter by. If None, analyzes all groups
        include_bottlenecks: Whether to identify and include bottleneck analysis

    Returns:
        FunnelMetrics object with complete funnel analysis

    Example:
        >>> from historical_data_loader import load_change_history
        >>> changes = load_change_history()
        >>> funnel = calculate_funnel_metrics(changes)
        >>> print(f"Overall completion: {funnel.overall_completion_rate}%")
    """
    # Filter by group if specified
    if group:
        changes = [c for c in changes if c.get("Group") == group]

    if not changes:
        return FunnelMetrics(group=group or "ALL")

    # Get date range from changes
    dates = [c.get('ParsedTimestamp') or c.get('ParsedDate') for c in changes]
    dates = [d.date() if hasattr(d, 'date') else d for d in dates if d]
    date_range_start = min(dates) if dates else None
    date_range_end = max(dates) if dates else None

    # Track unique row IDs by phase
    # A row is "entered" in a phase if it has any change record for that phase
    rows_by_phase: Dict[int, set] = defaultdict(set)
    users_by_phase: Dict[int, set] = defaultdict(set)

    for change in changes:
        phase = change.get("Phase")
        row_id = change.get("RowID")
        user = change.get("User")

        if phase and row_id:
            try:
                phase_num = int(phase)
                rows_by_phase[phase_num].add(row_id)
                if user:
                    users_by_phase[phase_num].add(user)
            except (ValueError, TypeError):
                continue

    # Build phase metrics
    phases = []
    previous_items = 0

    for date_col, user_cols, phase_num in PHASE_FIELDS:
        items_entered = len(rows_by_phase.get(phase_num, set()))
        unique_users = len(users_by_phase.get(phase_num, set()))

        # For first phase, there's no drop-off from previous
        if phase_num == 1:
            drop_off_count = 0
            drop_off_rate = 0.0
        else:
            # Items that didn't progress from previous phase
            drop_off_count = max(0, previous_items - items_entered)
            drop_off_rate = round(
                (drop_off_count / previous_items * 100) if previous_items > 0 else 0, 2
            )

        # Items completed = items in next phase (or current for last phase)
        next_phase_num = phase_num + 1
        if next_phase_num <= 5:
            items_completed = len(rows_by_phase.get(next_phase_num, set()))
        else:
            items_completed = items_entered  # Last phase - all entered are "completed"

        phase = FunnelPhase(
            phase_number=phase_num,
            phase_name=PHASE_NAMES.get(str(phase_num), f"Phase {phase_num}"),
            date_column=date_col,
            items_entered=items_entered,
            items_completed=items_completed,
            drop_off_count=drop_off_count,
            drop_off_rate=drop_off_rate,
            unique_users=unique_users,
        )
        phases.append(phase)
        previous_items = items_entered

    # Calculate overall metrics
    total_started = len(rows_by_phase.get(1, set()))
    # Items that completed Phase 4 (the main completion phase before Reopen)
    total_completed = len(rows_by_phase.get(4, set()))

    # Identify bottlenecks
    bottlenecks = []
    biggest_bottleneck = None
    max_drop_off = 0.0

    if include_bottlenecks:
        for phase in phases:
            if phase.phase_number > 1:  # Can't have bottleneck at phase 1
                severity = _get_severity(phase.drop_off_rate)
                if severity != BottleneckSeverity.NONE:
                    bn = BottleneckInfo(
                        phase_number=phase.phase_number,
                        phase_name=phase.phase_name,
                        severity=severity,
                        drop_off_rate=phase.drop_off_rate,
                        items_lost=phase.drop_off_count,
                        previous_phase_items=phases[phase.phase_number - 2].items_entered,
                    )
                    bottlenecks.append(bn)

                    if phase.drop_off_rate > max_drop_off:
                        max_drop_off = phase.drop_off_rate
                        biggest_bottleneck = phase.phase_number

    return FunnelMetrics(
        group=group or "ALL",
        date_range_start=date_range_start,
        date_range_end=date_range_end,
        total_items_started=total_started,
        total_items_completed=total_completed,
        phases=phases,
        bottlenecks=bottlenecks,
        biggest_bottleneck=biggest_bottleneck,
    )


def calculate_funnel_for_range(
    date_range_filter: Any,  # DateRangeFilter type
    group: Optional[str] = None,
    changes: Optional[List[Dict[str, Any]]] = None
) -> FunnelMetrics:
    """Calculate funnel metrics for a specific date range.

    This function integrates with DateRangeFilter to ensure funnel
    calculations respect the custom date range parameters.

    Args:
        date_range_filter: DateRangeFilter object specifying the date range
        group: Optional group to filter by
        changes: Optional pre-loaded changes (will load if not provided)

    Returns:
        FunnelMetrics object with funnel analysis for the specified range

    Example:
        >>> from date_range_filter import create_date_range
        >>> custom_range = create_date_range("2026-01-01", "2026-01-15")
        >>> funnel = calculate_funnel_for_range(custom_range)
        >>> print(f"Efficiency: {funnel.funnel_efficiency}%")
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

    metrics = calculate_funnel_metrics(changes, group)
    metrics.date_range_start = date_range_filter.start_date
    metrics.date_range_end = date_range_filter.end_date

    return metrics


def calculate_funnel_by_group(
    changes: List[Dict[str, Any]]
) -> Dict[str, FunnelMetrics]:
    """Calculate funnel metrics for each group.

    Args:
        changes: List of change records from historical data

    Returns:
        Dictionary mapping group names to FunnelMetrics objects

    Example:
        >>> from historical_data_loader import load_change_history
        >>> changes = load_change_history()
        >>> by_group = calculate_funnel_by_group(changes)
        >>> for group, funnel in by_group.items():
        ...     print(f"{group}: {funnel.overall_completion_rate}% completion")
    """
    # Get unique groups
    groups = set(c.get("Group") for c in changes if c.get("Group"))

    result = {}
    for group in groups:
        result[group] = calculate_funnel_metrics(changes, group)

    return result


def identify_bottlenecks(
    funnel: FunnelMetrics,
    min_severity: BottleneckSeverity = BottleneckSeverity.LOW
) -> List[BottleneckInfo]:
    """Identify and return bottlenecks from funnel metrics.

    Args:
        funnel: FunnelMetrics object to analyze
        min_severity: Minimum severity level to include

    Returns:
        List of BottleneckInfo objects, sorted by severity

    Example:
        >>> funnel = calculate_funnel_metrics(changes)
        >>> critical = identify_bottlenecks(funnel, BottleneckSeverity.HIGH)
        >>> for bn in critical:
        ...     print(f"Phase {bn.phase_number}: {bn.drop_off_rate}% drop-off")
    """
    severity_order = {
        BottleneckSeverity.CRITICAL: 0,
        BottleneckSeverity.HIGH: 1,
        BottleneckSeverity.MEDIUM: 2,
        BottleneckSeverity.LOW: 3,
        BottleneckSeverity.NONE: 4,
    }

    # Filter by minimum severity
    filtered = [
        bn for bn in funnel.bottlenecks
        if severity_order.get(bn.severity, 4) <= severity_order.get(min_severity, 4)
    ]

    # Sort by severity (most severe first)
    return sorted(filtered, key=lambda x: severity_order.get(x.severity, 4))


def compare_funnel_periods(
    current_changes: List[Dict[str, Any]],
    previous_changes: List[Dict[str, Any]],
    group: Optional[str] = None
) -> FunnelComparison:
    """Compare funnel metrics between two periods.

    Args:
        current_changes: Change records for the current period
        previous_changes: Change records for the previous period
        group: Optional group to filter by

    Returns:
        FunnelComparison object with comparison analysis

    Example:
        >>> from historical_data_loader import load_change_history
        >>> from datetime import date, timedelta
        >>> current = load_change_history(start_date=date.today() - timedelta(days=7))
        >>> previous = load_change_history(
        ...     start_date=date.today() - timedelta(days=14),
        ...     end_date=date.today() - timedelta(days=8)
        ... )
        >>> comparison = compare_funnel_periods(current, previous)
        >>> print(f"Trend: {comparison.trend}")
    """
    current_funnel = calculate_funnel_metrics(current_changes, group)
    previous_funnel = calculate_funnel_metrics(previous_changes, group)

    return FunnelComparison(
        current_period=current_funnel,
        previous_period=previous_funnel,
    )


def compare_funnel_for_range(
    date_range_filter: Any,  # DateRangeFilter type
    group: Optional[str] = None,
    changes: Optional[List[Dict[str, Any]]] = None
) -> FunnelComparison:
    """Compare funnel metrics for a date range against its previous period.

    Args:
        date_range_filter: DateRangeFilter object specifying the current period
        group: Optional group to filter by
        changes: Optional pre-loaded changes (will load if not provided)

    Returns:
        FunnelComparison object with comparison analysis

    Example:
        >>> from date_range_filter import get_preset_range, DateRangePreset
        >>> last_week = get_preset_range(DateRangePreset.LAST_WEEK)
        >>> comparison = compare_funnel_for_range(last_week)
        >>> print(f"Efficiency change: {comparison.efficiency_change}%")
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

    return compare_funnel_periods(current_changes, previous_changes, group)


def get_funnel_summary(funnel: FunnelMetrics) -> Dict[str, Any]:
    """Get a formatted summary of funnel metrics.

    Args:
        funnel: FunnelMetrics object to summarize

    Returns:
        Dictionary with summary statistics

    Example:
        >>> funnel = calculate_funnel_metrics(changes)
        >>> summary = get_funnel_summary(funnel)
        >>> print(summary)
    """
    # Phase with best completion rate
    best_phase = None
    best_completion = 0.0
    for phase in funnel.phases:
        if phase.completion_rate > best_completion:
            best_completion = phase.completion_rate
            best_phase = phase.phase_number

    # Phase with worst drop-off
    worst_phase = funnel.biggest_bottleneck
    worst_drop_off = 0.0
    if worst_phase:
        for phase in funnel.phases:
            if phase.phase_number == worst_phase:
                worst_drop_off = phase.drop_off_rate
                break

    return {
        "group": funnel.group,
        "date_range": {
            "start": funnel.date_range_start.isoformat() if funnel.date_range_start else None,
            "end": funnel.date_range_end.isoformat() if funnel.date_range_end else None,
        },
        "overview": {
            "total_started": funnel.total_items_started,
            "total_completed": funnel.total_items_completed,
            "overall_completion_rate": funnel.overall_completion_rate,
            "funnel_efficiency": funnel.funnel_efficiency,
        },
        "best_performing_phase": {
            "phase": best_phase,
            "completion_rate": best_completion,
        },
        "biggest_bottleneck": {
            "phase": worst_phase,
            "drop_off_rate": worst_drop_off,
        },
        "bottleneck_count": {
            "critical": sum(1 for b in funnel.bottlenecks if b.severity == BottleneckSeverity.CRITICAL),
            "high": sum(1 for b in funnel.bottlenecks if b.severity == BottleneckSeverity.HIGH),
            "medium": sum(1 for b in funnel.bottlenecks if b.severity == BottleneckSeverity.MEDIUM),
            "low": sum(1 for b in funnel.bottlenecks if b.severity == BottleneckSeverity.LOW),
        },
        "phase_summary": [
            {
                "phase": p.phase_number,
                "name": p.phase_name,
                "entered": p.items_entered,
                "completion_rate": p.completion_rate,
                "drop_off_rate": p.drop_off_rate,
            }
            for p in funnel.phases
        ],
    }


def get_funnel_visualization_data(funnel: FunnelMetrics) -> Dict[str, Any]:
    """Get data formatted for funnel visualization.

    This function prepares data in a format suitable for rendering
    funnel charts in PDF reports or other visualizations.

    Args:
        funnel: FunnelMetrics object to visualize

    Returns:
        Dictionary with visualization-ready data

    Example:
        >>> funnel = calculate_funnel_metrics(changes)
        >>> viz_data = get_funnel_visualization_data(funnel)
        >>> # Use viz_data to render a funnel chart
    """
    if not funnel.phases:
        return {
            "phases": [],
            "max_value": 0,
            "widths": [],
            "labels": [],
            "colors": [],
        }

    # Calculate relative widths (first phase = 100%)
    max_items = max(p.items_entered for p in funnel.phases) if funnel.phases else 1
    if max_items == 0:
        max_items = 1

    phases_data = []
    widths = []
    labels = []
    colors = []

    # Color scheme based on completion rate
    for phase in funnel.phases:
        width = (phase.items_entered / max_items) * 100 if max_items > 0 else 0
        widths.append(width)

        label = f"Phase {phase.phase_number}: {phase.items_entered} items"
        labels.append(label)

        # Color based on completion rate
        if phase.completion_rate >= 80:
            color = "#1B5E20"  # Green
        elif phase.completion_rate >= 60:
            color = "#8B6914"  # Yellow/Gold
        elif phase.completion_rate >= 40:
            color = "#E65100"  # Orange
        else:
            color = "#B71C1C"  # Red

        colors.append(color)

        phases_data.append({
            "phase_number": phase.phase_number,
            "phase_name": phase.phase_name,
            "items_entered": phase.items_entered,
            "completion_rate": phase.completion_rate,
            "drop_off_rate": phase.drop_off_rate,
            "width_percent": round(width, 1),
            "color": color,
        })

    return {
        "group": funnel.group,
        "phases": phases_data,
        "max_value": max_items,
        "widths": widths,
        "labels": labels,
        "colors": colors,
        "efficiency": funnel.funnel_efficiency,
        "overall_completion": funnel.overall_completion_rate,
    }


def format_funnel_report(funnel: FunnelMetrics) -> str:
    """Format funnel metrics as a text report.

    Args:
        funnel: FunnelMetrics object to format

    Returns:
        Formatted text report string
    """
    lines = []
    lines.append("=" * 70)
    lines.append("PHASE PROGRESSION FUNNEL REPORT")
    lines.append("=" * 70)

    # Header
    lines.append(f"\nGroup: {funnel.group}")
    if funnel.date_range_start and funnel.date_range_end:
        lines.append(f"Period: {funnel.date_range_start} to {funnel.date_range_end}")

    # Overview
    lines.append("\n" + "-" * 70)
    lines.append("OVERVIEW")
    lines.append("-" * 70)
    lines.append(f"  Items Started (Phase 1): {funnel.total_items_started}")
    lines.append(f"  Items Completed (Phase 4): {funnel.total_items_completed}")
    lines.append(f"  Overall Completion Rate: {funnel.overall_completion_rate}%")
    lines.append(f"  Funnel Efficiency Score: {funnel.funnel_efficiency}/100")

    # Phase breakdown
    lines.append("\n" + "-" * 70)
    lines.append("PHASE BREAKDOWN")
    lines.append("-" * 70)
    lines.append(f"{'Phase':<20} {'Entered':>10} {'Completed':>12} {'Drop-off':>12} {'Rate':>10}")
    lines.append("-" * 70)

    for phase in funnel.phases:
        lines.append(
            f"{phase.phase_name:<20} "
            f"{phase.items_entered:>10} "
            f"{phase.items_completed:>12} "
            f"{phase.drop_off_count:>12} "
            f"{phase.drop_off_rate:>9.1f}%"
        )

    # Bottleneck analysis
    if funnel.bottlenecks:
        lines.append("\n" + "-" * 70)
        lines.append("BOTTLENECK ANALYSIS")
        lines.append("-" * 70)

        for bn in sorted(funnel.bottlenecks, key=lambda x: x.drop_off_rate, reverse=True):
            severity_icon = {
                BottleneckSeverity.CRITICAL: "[!!!]",
                BottleneckSeverity.HIGH: "[!!] ",
                BottleneckSeverity.MEDIUM: "[!]  ",
                BottleneckSeverity.LOW: "[-]  ",
            }.get(bn.severity, "[?]  ")

            lines.append(f"\n{severity_icon} {bn.phase_name} ({bn.severity.value.upper()})")
            lines.append(f"       Drop-off: {bn.drop_off_rate}% ({bn.items_lost} items)")
            lines.append("       Suggested Actions:")
            for action in bn.suggested_actions[:2]:  # Show top 2 actions
                lines.append(f"         • {action}")
    else:
        lines.append("\n  No significant bottlenecks detected.")

    # Biggest bottleneck highlight
    if funnel.biggest_bottleneck:
        lines.append("\n" + "-" * 70)
        lines.append(f"BIGGEST BOTTLENECK: Phase {funnel.biggest_bottleneck}")
        lines.append("-" * 70)

    lines.append("\n" + "=" * 70)

    return "\n".join(lines)


if __name__ == "__main__":
    # Demo usage
    print("Phase Progression Funnel Calculator - Demo")
    print("=" * 70)

    # Create sample data
    sample_changes = []
    for i in range(100):
        # Simulate decreasing items per phase (funnel shape)
        phase_probs = [1.0, 0.85, 0.70, 0.60, 0.10]  # Probability of reaching each phase
        import random
        for phase_num in range(1, 6):
            if random.random() < phase_probs[phase_num - 1]:
                sample_changes.append({
                    "Group": ["NA", "NF", "NH"][i % 3],
                    "RowID": f"row_{i}",
                    "Phase": str(phase_num),
                    "User": ["DM", "JHU", "HI"][i % 3],
                    "ParsedTimestamp": date.today(),
                })

    # Calculate funnel metrics
    funnel = calculate_funnel_metrics(sample_changes)

    # Print report
    print(format_funnel_report(funnel))

    # Print summary
    print("\nJSON Summary:")
    summary = get_funnel_summary(funnel)
    import json
    print(json.dumps(summary, indent=2, default=str))

    # Print visualization data
    print("\nVisualization Data:")
    viz_data = get_funnel_visualization_data(funnel)
    print(json.dumps(viz_data, indent=2))
