"""
Group Health Scorer Module

Calculates health scores for each group based on activity levels, completion rates,
and overdue items. Assigns green/yellow/red status indicators.

Enhanced with DateRangeFilter support for custom date range health score calculations.

Usage:
    from group_health_scorer import (
        HealthStatus,
        GroupHealthScore,
        HealthScoreConfig,
        calculate_group_health_scores,
        calculate_health_scores_for_range,
        get_health_color,
        get_health_summary,
    )
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
from collections import defaultdict

from constants import HealthScoreDefaults

# Set up logging
logger = logging.getLogger(__name__)


class HealthStatus(Enum):
    """Health status indicator for a group."""
    GREEN = "green"      # Healthy (score >= 70)
    YELLOW = "yellow"    # Caution (score 40-69)
    RED = "red"          # Critical (score < 40)


@dataclass
class HealthScoreConfig:
    """Configuration for health score calculation.

    Attributes:
        activity_weight: Weight for activity level score (0-1)
        completion_weight: Weight for completion rate score (0-1)
        overdue_weight: Weight for overdue items score (0-1)
        green_threshold: Minimum score for GREEN status (default 70)
        yellow_threshold: Minimum score for YELLOW status (default 40)
        activity_lookback_days: Days to look back for activity calculation
        overdue_threshold_days: Days after which an item is considered overdue
    """
    activity_weight: float = HealthScoreDefaults.ACTIVITY_WEIGHT
    completion_weight: float = HealthScoreDefaults.COMPLETION_WEIGHT
    overdue_weight: float = HealthScoreDefaults.OVERDUE_WEIGHT
    green_threshold: int = HealthScoreDefaults.GREEN_THRESHOLD
    yellow_threshold: int = HealthScoreDefaults.YELLOW_THRESHOLD
    activity_lookback_days: int = HealthScoreDefaults.ACTIVITY_LOOKBACK_DAYS
    overdue_threshold_days: int = HealthScoreDefaults.OVERDUE_THRESHOLD_DAYS

    def __post_init__(self):
        """Validate weights sum to 1.0."""
        total = self.activity_weight + self.completion_weight + self.overdue_weight
        if abs(total - 1.0) > 0.01:
            logger.warning(f"Health score weights sum to {total}, normalizing to 1.0")
            self.activity_weight /= total
            self.completion_weight /= total
            self.overdue_weight /= total


@dataclass
class GroupHealthScore:
    """Health score details for a single group.

    Attributes:
        group: Group identifier (e.g., "NA", "NF", "BUNDLE_FAN")
        overall_score: Combined health score (0-100)
        status: Health status indicator (GREEN, YELLOW, RED)
        activity_score: Score based on recent activity level (0-100)
        completion_score: Score based on completion rate (0-100)
        overdue_score: Score based on overdue items (0-100, higher is better)
        activity_count: Number of changes in lookback period
        total_products: Total products in the group
        completed_products: Number of products that completed Phase 4
        overdue_count: Number of overdue items
        trend: Trend direction compared to previous period ("up", "down", "flat")
        last_activity_date: Date of most recent activity
    """
    group: str
    overall_score: float
    status: HealthStatus
    activity_score: float = 0.0
    completion_score: float = 0.0
    overdue_score: float = 0.0
    activity_count: int = 0
    total_products: int = 0
    completed_products: int = 0
    overdue_count: int = 0
    trend: str = "flat"
    last_activity_date: Optional[date] = None

    @property
    def status_label(self) -> str:
        """Get human-readable status label."""
        labels = {
            HealthStatus.GREEN: "Healthy",
            HealthStatus.YELLOW: "Needs Attention",
            HealthStatus.RED: "Critical",
        }
        return labels.get(self.status, "Unknown")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "group": self.group,
            "overall_score": round(self.overall_score, 1),
            "status": self.status.value,
            "status_label": self.status_label,
            "activity_score": round(self.activity_score, 1),
            "completion_score": round(self.completion_score, 1),
            "overdue_score": round(self.overdue_score, 1),
            "activity_count": self.activity_count,
            "total_products": self.total_products,
            "completed_products": self.completed_products,
            "overdue_count": self.overdue_count,
            "trend": self.trend,
            "last_activity_date": self.last_activity_date.isoformat() if self.last_activity_date else None,
        }


def get_health_status(score: float, config: HealthScoreConfig) -> HealthStatus:
    """Determine health status based on score and thresholds.

    Args:
        score: Health score (0-100)
        config: Health score configuration

    Returns:
        HealthStatus enum value
    """
    if score >= config.green_threshold:
        return HealthStatus.GREEN
    elif score >= config.yellow_threshold:
        return HealthStatus.YELLOW
    else:
        return HealthStatus.RED


def calculate_activity_score(
    activity_count: int,
    total_products: int,
    lookback_days: int
) -> float:
    """Calculate activity score based on changes relative to group size.

    A healthy group should have regular activity. The score is based on
    the ratio of changes to total products, normalized by time period.

    Args:
        activity_count: Number of changes in the period
        total_products: Total products in the group
        lookback_days: Number of days in the lookback period

    Returns:
        Activity score (0-100)
    """
    if total_products == 0:
        return 0.0

    # Calculate activity rate (changes per product per month)
    # Expecting at least some movement for healthy groups
    months = lookback_days / 30.0
    if months == 0:
        months = 1.0

    activity_rate = activity_count / (total_products * months)

    # Score calculation:
    # - 0% activity rate = 0 score
    # - 5% activity rate = 50 score (baseline expectation)
    # - 10%+ activity rate = 100 score (very active)

    if activity_rate >= 0.10:
        return 100.0
    elif activity_rate >= 0.05:
        # Linear interpolation from 50-100 for 5-10% rate
        return 50.0 + (activity_rate - 0.05) / 0.05 * 50.0
    elif activity_rate > 0:
        # Linear interpolation from 0-50 for 0-5% rate
        return activity_rate / 0.05 * 50.0
    else:
        return 0.0


def calculate_completion_score(
    completed_products: int,
    total_products: int
) -> float:
    """Calculate completion score based on products that completed Phase 4.

    Args:
        completed_products: Number of products that completed Phase 4
        total_products: Total products in the group

    Returns:
        Completion score (0-100)
    """
    if total_products == 0:
        return 0.0

    completion_rate = completed_products / total_products

    # Score is directly proportional to completion rate
    # with some bonus for high completion
    if completion_rate >= 0.90:
        return 100.0
    elif completion_rate >= 0.75:
        return 90.0 + (completion_rate - 0.75) / 0.15 * 10.0
    else:
        return completion_rate / 0.75 * 90.0


def calculate_overdue_score(
    overdue_count: int,
    total_products: int
) -> float:
    """Calculate overdue score (higher is better - fewer overdue items).

    Args:
        overdue_count: Number of overdue items
        total_products: Total products in the group

    Returns:
        Overdue score (0-100), where 100 means no overdue items
    """
    if total_products == 0:
        return 100.0  # No products = no overdue

    overdue_rate = overdue_count / total_products

    # Inverse scoring:
    # - 0% overdue = 100 score
    # - 5% overdue = 75 score
    # - 10% overdue = 50 score
    # - 20%+ overdue = 0 score

    if overdue_rate >= 0.20:
        return 0.0
    elif overdue_rate >= 0.10:
        return (0.20 - overdue_rate) / 0.10 * 50.0
    elif overdue_rate >= 0.05:
        return 50.0 + (0.10 - overdue_rate) / 0.05 * 25.0
    else:
        return 75.0 + (0.05 - overdue_rate) / 0.05 * 25.0


def determine_trend(
    current_activity: int,
    previous_activity: int
) -> str:
    """Determine trend direction based on activity comparison.

    Args:
        current_activity: Activity count in current period
        previous_activity: Activity count in previous period

    Returns:
        Trend direction ("up", "down", or "flat")
    """
    if previous_activity == 0:
        if current_activity > 0:
            return "up"
        return "flat"

    change_rate = (current_activity - previous_activity) / previous_activity

    if change_rate >= 0.10:  # 10% increase threshold
        return "up"
    elif change_rate <= -0.10:  # 10% decrease threshold
        return "down"
    else:
        return "flat"


def calculate_group_health_scores(
    changes: List[Dict[str, Any]],
    total_products: Dict[str, int],
    completed_by_group: Optional[Dict[str, int]] = None,
    overdue_by_group: Optional[Dict[str, int]] = None,
    config: Optional[HealthScoreConfig] = None,
    reference_date: Optional[date] = None
) -> Dict[str, GroupHealthScore]:
    """Calculate health scores for all groups.

    Args:
        changes: List of change records from historical data
        total_products: Dictionary mapping group to total product count
        completed_by_group: Dictionary mapping group to completed product count
        overdue_by_group: Dictionary mapping group to overdue item count
        config: Health score configuration (uses defaults if not provided)
        reference_date: Reference date for calculations (defaults to today)

    Returns:
        Dictionary mapping group names to GroupHealthScore objects
    """
    if config is None:
        config = HealthScoreConfig()

    if reference_date is None:
        reference_date = date.today()

    # Calculate date ranges
    lookback_start = reference_date - timedelta(days=config.activity_lookback_days)
    previous_start = lookback_start - timedelta(days=config.activity_lookback_days)
    previous_end = lookback_start - timedelta(days=1)

    # Aggregate activity by group for current and previous periods
    current_activity = defaultdict(int)
    previous_activity = defaultdict(int)
    last_activity_dates = {}
    processing_errors = 0

    for idx, change in enumerate(changes):
        # Row-level error isolation for change processing
        try:
            group = change.get("Group", "")
            if not group:
                continue

            # Get the parsed date or parse the timestamp - with error isolation
            change_date = None
            try:
                change_date = change.get("ParsedDate") or change.get("ParsedTimestamp")
                if change_date is None:
                    timestamp_str = change.get("Timestamp", "")
                    if timestamp_str:
                        try:
                            change_date = datetime.strptime(timestamp_str, "%Y-%m-%d %H:%M:%S").date()
                        except ValueError:
                            continue

                if hasattr(change_date, 'date'):
                    change_date = change_date.date()
            except Exception as date_error:
                logger.warning(
                    f"Health scorer: Change {idx} (Group: {group}): Error parsing date - "
                    f"{type(date_error).__name__}: {date_error}. Skipping this change."
                )
                processing_errors += 1
                continue

            if change_date is None:
                continue

            # Categorize by period - with error isolation
            try:
                if lookback_start <= change_date <= reference_date:
                    current_activity[group] += 1
                    # Track last activity date
                    if group not in last_activity_dates or change_date > last_activity_dates[group]:
                        last_activity_dates[group] = change_date
                elif previous_start <= change_date <= previous_end:
                    previous_activity[group] += 1
            except Exception as categorize_error:
                logger.warning(
                    f"Health scorer: Change {idx} (Group: {group}): Error categorizing by period - "
                    f"{type(categorize_error).__name__}: {categorize_error}. Skipping this change."
                )
                processing_errors += 1
                continue

        except Exception as change_error:
            # Catch any unexpected errors at the change level
            processing_errors += 1
            logger.error(
                f"Health scorer: Change {idx}: Unexpected error during processing - "
                f"{type(change_error).__name__}: {change_error}. "
                f"Skipping this change and continuing with remaining changes."
            )
            continue

    # Log processing errors summary if any occurred
    if processing_errors > 0:
        logger.warning(
            f"Health scorer: {processing_errors} changes encountered processing errors and were skipped."
        )

    # Calculate health scores for each group
    health_scores = {}
    all_groups = set(total_products.keys()) | set(current_activity.keys())

    for group in all_groups:
        group_total = total_products.get(group, 0)
        activity_count = current_activity.get(group, 0)
        prev_activity = previous_activity.get(group, 0)
        completed = (completed_by_group or {}).get(group, 0)
        overdue = (overdue_by_group or {}).get(group, 0)

        # Calculate individual scores
        activity_score = calculate_activity_score(
            activity_count,
            group_total,
            config.activity_lookback_days
        )

        completion_score = calculate_completion_score(completed, group_total)
        overdue_score = calculate_overdue_score(overdue, group_total)

        # Calculate weighted overall score
        overall_score = (
            config.activity_weight * activity_score +
            config.completion_weight * completion_score +
            config.overdue_weight * overdue_score
        )

        # Determine status and trend
        status = get_health_status(overall_score, config)
        trend = determine_trend(activity_count, prev_activity)

        health_scores[group] = GroupHealthScore(
            group=group,
            overall_score=overall_score,
            status=status,
            activity_score=activity_score,
            completion_score=completion_score,
            overdue_score=overdue_score,
            activity_count=activity_count,
            total_products=group_total,
            completed_products=completed,
            overdue_count=overdue,
            trend=trend,
            last_activity_date=last_activity_dates.get(group),
        )

    return health_scores


def calculate_health_scores_for_range(
    date_range_filter: Any,  # DateRangeFilter type
    total_products: Dict[str, int],
    completed_by_group: Optional[Dict[str, int]] = None,
    overdue_by_group: Optional[Dict[str, int]] = None,
    config: Optional[HealthScoreConfig] = None,
    changes: Optional[List[Dict[str, Any]]] = None
) -> Dict[str, GroupHealthScore]:
    """Calculate health scores for all groups within a custom date range.

    This function integrates with DateRangeFilter to ensure health score
    calculations respect the custom date range parameters.

    Args:
        date_range_filter: DateRangeFilter object specifying the date range
        total_products: Dictionary mapping group to total product count
        completed_by_group: Dictionary mapping group to completed product count
        overdue_by_group: Dictionary mapping group to overdue item count
        config: Health score configuration (uses defaults if not provided)
        changes: Optional pre-loaded changes (will load if not provided)

    Returns:
        Dictionary mapping group names to GroupHealthScore objects

    Example:
        >>> from date_range_filter import create_date_range
        >>> custom_range = create_date_range("2026-01-01", "2026-01-15")
        >>> scores = calculate_health_scores_for_range(
        ...     custom_range,
        ...     total_products={"NA": 1779, "NF": 1716}
        ... )
    """
    if config is None:
        config = HealthScoreConfig()

    # Use the date range filter's dates
    reference_date = date_range_filter.end_date

    # Adjust the activity lookback to match the date range
    # Create a modified config that uses the custom range's duration
    custom_config = HealthScoreConfig(
        activity_weight=config.activity_weight,
        completion_weight=config.completion_weight,
        overdue_weight=config.overdue_weight,
        green_threshold=config.green_threshold,
        yellow_threshold=config.yellow_threshold,
        activity_lookback_days=date_range_filter.days_in_range,
        overdue_threshold_days=config.overdue_threshold_days,
    )

    # Load changes for the date range if not provided
    if changes is None:
        from historical_data_loader import load_change_history
        # Load changes covering both current and previous periods
        previous_range = date_range_filter.get_previous_period()
        all_start = previous_range.start_date
        changes = load_change_history(start_date=all_start, end_date=reference_date)

    # Filter changes to the specified date range
    from historical_data_loader import filter_by_date_range_filter
    range_changes = filter_by_date_range_filter(changes, date_range_filter)

    # Calculate health scores using the filtered changes
    return calculate_group_health_scores(
        changes=range_changes,
        total_products=total_products,
        completed_by_group=completed_by_group,
        overdue_by_group=overdue_by_group,
        config=custom_config,
        reference_date=reference_date,
    )


def get_health_scores_comparison(
    date_range_filter: Any,  # DateRangeFilter type
    total_products: Dict[str, int],
    completed_by_group: Optional[Dict[str, int]] = None,
    overdue_by_group: Optional[Dict[str, int]] = None,
    config: Optional[HealthScoreConfig] = None
) -> Dict[str, Any]:
    """Get health scores with comparison to the previous period.

    Args:
        date_range_filter: DateRangeFilter object for the current period
        total_products: Dictionary mapping group to total product count
        completed_by_group: Dictionary mapping group to completed product count
        overdue_by_group: Dictionary mapping group to overdue item count
        config: Health score configuration

    Returns:
        Dictionary containing:
            - current_scores: Health scores for the current period
            - previous_scores: Health scores for the previous period
            - changes: Dictionary of score changes by group
            - summary: Summary of health score changes
    """
    # Calculate current period scores
    current_scores = calculate_health_scores_for_range(
        date_range_filter,
        total_products,
        completed_by_group,
        overdue_by_group,
        config
    )

    # Calculate previous period scores
    previous_range = date_range_filter.get_previous_period()
    previous_scores = calculate_health_scores_for_range(
        previous_range,
        total_products,
        completed_by_group,
        overdue_by_group,
        config
    )

    # Calculate changes
    changes = {}
    for group in set(current_scores.keys()) | set(previous_scores.keys()):
        current_score = current_scores.get(group)
        previous_score = previous_scores.get(group)

        if current_score and previous_score:
            score_change = current_score.overall_score - previous_score.overall_score
            status_change = "improved" if score_change > 5 else ("declined" if score_change < -5 else "stable")
        elif current_score:
            score_change = current_score.overall_score
            status_change = "new"
        else:
            score_change = 0
            status_change = "removed"

        changes[group] = {
            "current_score": current_score.overall_score if current_score else None,
            "previous_score": previous_score.overall_score if previous_score else None,
            "score_change": round(score_change, 1),
            "status_change": status_change,
        }

    # Generate summary
    improved = sum(1 for c in changes.values() if c["status_change"] == "improved")
    declined = sum(1 for c in changes.values() if c["status_change"] == "declined")
    stable = sum(1 for c in changes.values() if c["status_change"] == "stable")

    return {
        "current_scores": current_scores,
        "previous_scores": previous_scores,
        "changes": changes,
        "summary": {
            "improved_count": improved,
            "declined_count": declined,
            "stable_count": stable,
            "current_period": {
                "start": date_range_filter.start_date.isoformat(),
                "end": date_range_filter.end_date.isoformat(),
                "label": date_range_filter.label,
            },
            "previous_period": {
                "start": previous_range.start_date.isoformat(),
                "end": previous_range.end_date.isoformat(),
                "label": previous_range.label,
            },
        },
    }


def get_health_color(status: HealthStatus) -> str:
    """Get WCAG AA compliant hex color code for a health status.

    All colors meet minimum 3:1 contrast ratio against white background
    as required by WCAG 2.1 guidelines for graphical elements.

    Args:
        status: HealthStatus enum value

    Returns:
        Hex color code string
    """
    # WCAG AA compliant colors with enhanced contrast
    colors = {
        HealthStatus.GREEN: "#1B5E20",   # Green 900 - Contrast: 7.8:1
        HealthStatus.YELLOW: "#8B6914",  # Dark Gold - Contrast: 4.5:1
        HealthStatus.RED: "#B71C1C",     # Red 900 - Contrast: 6.9:1
    }
    return colors.get(status, "#546E7A")  # Blue Grey 600 as fallback - Contrast: 4.6:1


def get_health_summary(health_scores: Dict[str, GroupHealthScore]) -> Dict[str, Any]:
    """Get a summary of health scores across all groups.

    Args:
        health_scores: Dictionary of group health scores

    Returns:
        Summary dictionary with aggregate statistics
    """
    if not health_scores:
        return {
            "total_groups": 0,
            "green_count": 0,
            "yellow_count": 0,
            "red_count": 0,
            "average_score": 0.0,
            "healthiest_group": None,
            "most_critical_group": None,
            "groups_by_status": {
                "green": [],
                "yellow": [],
                "red": [],
            },
        }

    scores = list(health_scores.values())

    green_groups = [s for s in scores if s.status == HealthStatus.GREEN]
    yellow_groups = [s for s in scores if s.status == HealthStatus.YELLOW]
    red_groups = [s for s in scores if s.status == HealthStatus.RED]

    avg_score = sum(s.overall_score for s in scores) / len(scores)

    # Find healthiest and most critical
    sorted_scores = sorted(scores, key=lambda x: x.overall_score, reverse=True)
    healthiest = sorted_scores[0] if sorted_scores else None
    most_critical = sorted_scores[-1] if sorted_scores else None

    return {
        "total_groups": len(scores),
        "green_count": len(green_groups),
        "yellow_count": len(yellow_groups),
        "red_count": len(red_groups),
        "average_score": round(avg_score, 1),
        "healthiest_group": healthiest.group if healthiest else None,
        "healthiest_score": round(healthiest.overall_score, 1) if healthiest else None,
        "most_critical_group": most_critical.group if most_critical else None,
        "most_critical_score": round(most_critical.overall_score, 1) if most_critical else None,
        "groups_by_status": {
            "green": [s.group for s in green_groups],
            "yellow": [s.group for s in yellow_groups],
            "red": [s.group for s in red_groups],
        },
    }


def format_health_report(health_scores: Dict[str, GroupHealthScore]) -> str:
    """Format health scores as a text report.

    Args:
        health_scores: Dictionary of group health scores

    Returns:
        Formatted text report
    """
    if not health_scores:
        return "No health score data available."

    lines = []
    lines.append("=" * 60)
    lines.append("GROUP HEALTH SCORE REPORT")
    lines.append("=" * 60)

    # Summary
    summary = get_health_summary(health_scores)
    lines.append(f"\nOverall Summary:")
    lines.append(f"  Total Groups: {summary['total_groups']}")
    lines.append(f"  Average Score: {summary['average_score']}")
    lines.append(f"  Green (Healthy): {summary['green_count']}")
    lines.append(f"  Yellow (Caution): {summary['yellow_count']}")
    lines.append(f"  Red (Critical): {summary['red_count']}")

    # Individual groups sorted by score
    lines.append("\n" + "-" * 60)
    lines.append("Group Details (sorted by score):")
    lines.append("-" * 60)

    sorted_scores = sorted(
        health_scores.values(),
        key=lambda x: x.overall_score,
        reverse=True
    )

    status_icons = {
        HealthStatus.GREEN: "[OK]",
        HealthStatus.YELLOW: "[!!]",
        HealthStatus.RED: "[XX]",
    }

    for score in sorted_scores:
        icon = status_icons.get(score.status, "[??]")
        trend_icon = {"up": "+", "down": "-", "flat": "="}[score.trend]

        lines.append(f"\n{icon} {score.group}: {score.overall_score:.1f} ({score.status_label}) {trend_icon}")
        lines.append(f"    Activity Score: {score.activity_score:.1f} ({score.activity_count} changes)")
        lines.append(f"    Completion Score: {score.completion_score:.1f} ({score.completed_products}/{score.total_products})")
        lines.append(f"    Overdue Score: {score.overdue_score:.1f} ({score.overdue_count} overdue)")

    lines.append("\n" + "=" * 60)

    return "\n".join(lines)


if __name__ == "__main__":
    # Demo usage with sample data
    print("Group Health Scorer - Demo")
    print("=" * 60)

    # Sample data
    sample_total_products = {
        "NA": 1779,
        "NF": 1716,
        "NH": 893,
        "NM": 391,
        "NP": 394,
        "NT": 119,
        "NV": 0,
    }

    sample_completed = {
        "NA": 1500,
        "NF": 1200,
        "NH": 700,
        "NM": 300,
        "NP": 350,
        "NT": 100,
        "NV": 0,
    }

    sample_overdue = {
        "NA": 50,
        "NF": 100,
        "NH": 30,
        "NM": 20,
        "NP": 10,
        "NT": 5,
        "NV": 0,
    }

    # Create sample changes
    sample_changes = []
    today = date.today()
    for i in range(200):
        group = ["NA", "NF", "NH", "NM", "NP", "NT"][i % 6]
        days_ago = i % 30
        change_date = today - timedelta(days=days_ago)
        sample_changes.append({
            "Group": group,
            "Timestamp": change_date.strftime("%Y-%m-%d 10:00:00"),
            "ParsedDate": change_date,
        })

    # Calculate health scores
    health_scores = calculate_group_health_scores(
        changes=sample_changes,
        total_products=sample_total_products,
        completed_by_group=sample_completed,
        overdue_by_group=sample_overdue,
    )

    # Print report
    print(format_health_report(health_scores))

    # Print summary
    print("\nJSON Summary:")
    summary = get_health_summary(health_scores)
    import json
    print(json.dumps(summary, indent=2))
