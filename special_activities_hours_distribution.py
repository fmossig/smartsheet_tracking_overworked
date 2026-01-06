"""
Special Activities Hours Distribution Module

Calculate distribution of hours worked across special activity categories.
Identify time-intensive categories and efficiency patterns.

This module provides analytics for understanding how work hours are distributed
across different activity categories, identifying time-intensive areas, and
analyzing efficiency patterns for workload optimization.

Usage:
    from special_activities_hours_distribution import (
        CategoryIntensityLevel,
        EfficiencyLevel,
        HoursDistributionMetrics,
        CategoryHoursDistribution,
        EfficiencyPattern,
        HoursDistributionSummary,
        calculate_hours_distribution,
        identify_time_intensive_categories,
        analyze_efficiency_patterns,
        get_distribution_visualization_data,
        format_distribution_report,
    )
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Any, Tuple
from enum import Enum
from collections import defaultdict
import statistics

from period_comparison_calculator import (
    TrendDirection,
    calculate_percent_change,
    get_trend_direction,
    format_percent_change,
    get_trend_indicator,
)

# Set up logging
logger = logging.getLogger(__name__)


class CategoryIntensityLevel(Enum):
    """Classification of category time intensity based on hours consumption."""
    CRITICAL = "critical"      # Top 10% of hours
    HIGH = "high"              # Top 10-30% of hours
    MEDIUM = "medium"          # Middle 30-70% of hours
    LOW = "low"                # Bottom 30% of hours


class EfficiencyLevel(Enum):
    """Classification of category efficiency based on hours per item."""
    HIGHLY_EFFICIENT = "highly_efficient"    # Low hours per item
    EFFICIENT = "efficient"                   # Below average hours per item
    AVERAGE = "average"                       # Near average hours per item
    INEFFICIENT = "inefficient"               # Above average hours per item
    HIGHLY_INEFFICIENT = "highly_inefficient" # Very high hours per item


class WorkloadDistributionLevel(Enum):
    """Classification of workload distribution across users."""
    BALANCED = "balanced"          # Hours evenly distributed
    SLIGHTLY_UNEVEN = "slightly_uneven"  # Minor imbalance
    UNEVEN = "uneven"              # Noticeable imbalance
    HIGHLY_CONCENTRATED = "highly_concentrated"  # Most hours by few users


@dataclass
class HoursDistributionMetrics:
    """Core metrics for hours distribution analysis.

    Attributes:
        total_hours: Total hours across all categories
        total_items: Total number of activity items
        total_categories: Number of unique categories
        average_hours_per_category: Average hours per category
        average_hours_per_item: Average hours per activity item
        hours_std_deviation: Standard deviation of hours across categories
        gini_coefficient: Gini coefficient for distribution inequality (0-1)
    """
    total_hours: float = 0.0
    total_items: int = 0
    total_categories: int = 0
    average_hours_per_category: float = 0.0
    average_hours_per_item: float = 0.0
    hours_std_deviation: float = 0.0
    gini_coefficient: float = 0.0

    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        if self.total_categories > 0:
            self.average_hours_per_category = round(
                self.total_hours / self.total_categories, 2
            )
        if self.total_items > 0:
            self.average_hours_per_item = round(
                self.total_hours / self.total_items, 2
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "total_hours": round(self.total_hours, 2),
            "total_items": self.total_items,
            "total_categories": self.total_categories,
            "average_hours_per_category": self.average_hours_per_category,
            "average_hours_per_item": self.average_hours_per_item,
            "hours_std_deviation": round(self.hours_std_deviation, 2),
            "gini_coefficient": round(self.gini_coefficient, 3),
        }


@dataclass
class CategoryHoursDistribution:
    """Hours distribution metrics for a specific category.

    Attributes:
        category_name: Name of the activity category
        total_hours: Total hours spent in this category
        item_count: Number of activity items in this category
        percentage_of_total_hours: Percentage of total hours across all categories
        percentage_of_total_items: Percentage of total items across all categories
        hours_per_item: Average hours per item in this category
        rank_by_hours: Rank by total hours (1 = highest)
        rank_by_efficiency: Rank by hours per item (1 = most efficient/lowest)
        intensity_level: Classification of time intensity
        efficiency_level: Classification of efficiency
        by_user: Breakdown of hours by user for this category
        user_count: Number of users who worked in this category
        top_contributor: User with most hours in this category
        top_contributor_hours: Hours contributed by top contributor
        previous_period_hours: Hours from the previous period (if available)
        hours_percent_change: Percentage change in hours vs previous period
        trend: Trend direction based on hours change
    """
    category_name: str
    total_hours: float = 0.0
    item_count: int = 0
    percentage_of_total_hours: float = 0.0
    percentage_of_total_items: float = 0.0
    hours_per_item: float = 0.0
    rank_by_hours: int = 0
    rank_by_efficiency: int = 0
    intensity_level: CategoryIntensityLevel = CategoryIntensityLevel.LOW
    efficiency_level: EfficiencyLevel = EfficiencyLevel.AVERAGE
    by_user: Dict[str, float] = field(default_factory=dict)
    user_count: int = 0
    top_contributor: Optional[str] = None
    top_contributor_hours: float = 0.0
    previous_period_hours: Optional[float] = None
    hours_percent_change: Optional[float] = None
    trend: TrendDirection = TrendDirection.NO_DATA

    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        if self.item_count > 0:
            self.hours_per_item = round(self.total_hours / self.item_count, 2)
        if self.by_user:
            self.user_count = len(self.by_user)
            # Find top contributor
            sorted_users = sorted(
                self.by_user.items(),
                key=lambda x: x[1],
                reverse=True
            )
            if sorted_users:
                self.top_contributor = sorted_users[0][0]
                self.top_contributor_hours = sorted_users[0][1]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "category_name": self.category_name,
            "total_hours": round(self.total_hours, 2),
            "item_count": self.item_count,
            "percentage_of_total_hours": round(self.percentage_of_total_hours, 2),
            "percentage_of_total_items": round(self.percentage_of_total_items, 2),
            "hours_per_item": self.hours_per_item,
            "rank_by_hours": self.rank_by_hours,
            "rank_by_efficiency": self.rank_by_efficiency,
            "intensity_level": self.intensity_level.value,
            "efficiency_level": self.efficiency_level.value,
            "by_user": {k: round(v, 2) for k, v in self.by_user.items()},
            "user_count": self.user_count,
            "top_contributor": self.top_contributor,
            "top_contributor_hours": round(self.top_contributor_hours, 2) if self.top_contributor_hours else 0.0,
            "previous_period_hours": round(self.previous_period_hours, 2) if self.previous_period_hours is not None else None,
            "hours_percent_change": round(self.hours_percent_change, 2) if self.hours_percent_change is not None else None,
            "trend": self.trend.value,
        }


@dataclass
class EfficiencyPattern:
    """Identified efficiency pattern in the hours distribution.

    Attributes:
        pattern_type: Type of pattern identified
        description: Human-readable description of the pattern
        affected_categories: List of categories affected by this pattern
        severity: Severity level (info, warning, critical)
        recommendation: Suggested action based on the pattern
        metrics: Supporting metrics for this pattern
    """
    pattern_type: str
    description: str
    affected_categories: List[str] = field(default_factory=list)
    severity: str = "info"  # info, warning, critical
    recommendation: str = ""
    metrics: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pattern_type": self.pattern_type,
            "description": self.description,
            "affected_categories": self.affected_categories,
            "severity": self.severity,
            "recommendation": self.recommendation,
            "metrics": self.metrics,
        }


@dataclass
class HoursDistributionSummary:
    """Complete summary of hours distribution analysis.

    Attributes:
        start_date: Start of the analysis period
        end_date: End of the analysis period
        metrics: Core distribution metrics
        categories: List of CategoryHoursDistribution sorted by hours (descending)
        time_intensive_categories: Categories consuming most hours (top 20%)
        most_efficient_categories: Categories with lowest hours per item
        least_efficient_categories: Categories with highest hours per item
        efficiency_patterns: Identified efficiency patterns
        workload_distribution: Classification of workload balance
        has_trend_data: Whether trend data is available
    """
    start_date: date
    end_date: date
    metrics: HoursDistributionMetrics
    categories: List[CategoryHoursDistribution] = field(default_factory=list)
    time_intensive_categories: List[str] = field(default_factory=list)
    most_efficient_categories: List[str] = field(default_factory=list)
    least_efficient_categories: List[str] = field(default_factory=list)
    efficiency_patterns: List[EfficiencyPattern] = field(default_factory=list)
    workload_distribution: WorkloadDistributionLevel = WorkloadDistributionLevel.BALANCED
    has_trend_data: bool = False

    def __post_init__(self):
        """Calculate derived fields after initialization."""
        if self.categories:
            # Sort categories by total hours descending
            self.categories = sorted(
                self.categories,
                key=lambda x: x.total_hours,
                reverse=True
            )
            # Check for trend data
            self.has_trend_data = any(
                cat.previous_period_hours is not None for cat in self.categories
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "metrics": self.metrics.to_dict(),
            "categories": [cat.to_dict() for cat in self.categories],
            "time_intensive_categories": self.time_intensive_categories,
            "most_efficient_categories": self.most_efficient_categories,
            "least_efficient_categories": self.least_efficient_categories,
            "efficiency_patterns": [p.to_dict() for p in self.efficiency_patterns],
            "workload_distribution": self.workload_distribution.value,
            "has_trend_data": self.has_trend_data,
        }

    def get_top_categories(self, n: int = 5) -> List[CategoryHoursDistribution]:
        """Get top N categories by hours."""
        return self.categories[:n]

    def get_categories_by_intensity(
        self, level: CategoryIntensityLevel
    ) -> List[CategoryHoursDistribution]:
        """Get categories with a specific intensity level."""
        return [cat for cat in self.categories if cat.intensity_level == level]

    def get_categories_by_efficiency(
        self, level: EfficiencyLevel
    ) -> List[CategoryHoursDistribution]:
        """Get categories with a specific efficiency level."""
        return [cat for cat in self.categories if cat.efficiency_level == level]


def aggregate_hours_by_category(
    user_activity_data: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, Dict[str, float]]]:
    """Aggregate hours data by category from user activity data.

    Args:
        user_activity_data: Dictionary mapping usernames to their activity data.
            Each user entry contains 'count', 'hours', and 'categories' dict.

    Returns:
        Tuple of:
            - category_hours: Dict mapping category name to total hours
            - category_counts: Dict mapping category name to item count
            - category_by_user: Dict mapping category to user-hour breakdown
    """
    category_hours: Dict[str, float] = defaultdict(float)
    category_counts: Dict[str, int] = defaultdict(int)
    category_by_user: Dict[str, Dict[str, float]] = defaultdict(lambda: defaultdict(float))

    for user, data in user_activity_data.items():
        user_categories = data.get("categories", {})
        for category, hours in user_categories.items():
            category_hours[category] += hours
            category_counts[category] += 1
            category_by_user[category][user] += hours

    return dict(category_hours), dict(category_counts), dict(category_by_user)


def calculate_gini_coefficient(values: List[float]) -> float:
    """Calculate Gini coefficient for distribution inequality.

    Args:
        values: List of values to calculate inequality for

    Returns:
        Gini coefficient (0 = perfect equality, 1 = perfect inequality)
    """
    if not values or len(values) < 2:
        return 0.0

    # Sort values
    sorted_values = sorted(values)
    n = len(sorted_values)
    total = sum(sorted_values)

    if total == 0:
        return 0.0

    # Calculate Gini coefficient
    cumulative = 0.0
    for i, value in enumerate(sorted_values):
        cumulative += (2 * (i + 1) - n - 1) * value

    gini = cumulative / (n * total)
    return max(0.0, min(1.0, gini))  # Ensure it's between 0 and 1


def classify_intensity_level(
    percentage: float,
    cumulative_percentage: float
) -> CategoryIntensityLevel:
    """Classify category intensity based on percentage of total hours.

    Args:
        percentage: Category's percentage of total hours
        cumulative_percentage: Cumulative percentage up to this category

    Returns:
        CategoryIntensityLevel classification
    """
    # Categories in top 10% of cumulative hours are CRITICAL
    if cumulative_percentage <= 10:
        return CategoryIntensityLevel.CRITICAL
    # Categories in top 10-30% are HIGH
    elif cumulative_percentage <= 30:
        return CategoryIntensityLevel.HIGH
    # Categories in middle 30-70% are MEDIUM
    elif cumulative_percentage <= 70:
        return CategoryIntensityLevel.MEDIUM
    # Bottom 30% are LOW
    else:
        return CategoryIntensityLevel.LOW


def classify_efficiency_level(
    hours_per_item: float,
    average_hours_per_item: float
) -> EfficiencyLevel:
    """Classify category efficiency based on hours per item.

    Args:
        hours_per_item: Category's hours per item
        average_hours_per_item: Overall average hours per item

    Returns:
        EfficiencyLevel classification
    """
    if average_hours_per_item == 0:
        return EfficiencyLevel.AVERAGE

    ratio = hours_per_item / average_hours_per_item

    if ratio <= 0.5:
        return EfficiencyLevel.HIGHLY_EFFICIENT
    elif ratio <= 0.8:
        return EfficiencyLevel.EFFICIENT
    elif ratio <= 1.2:
        return EfficiencyLevel.AVERAGE
    elif ratio <= 1.5:
        return EfficiencyLevel.INEFFICIENT
    else:
        return EfficiencyLevel.HIGHLY_INEFFICIENT


def classify_workload_distribution(gini: float) -> WorkloadDistributionLevel:
    """Classify workload distribution based on Gini coefficient.

    Args:
        gini: Gini coefficient (0-1)

    Returns:
        WorkloadDistributionLevel classification
    """
    if gini <= 0.2:
        return WorkloadDistributionLevel.BALANCED
    elif gini <= 0.4:
        return WorkloadDistributionLevel.SLIGHTLY_UNEVEN
    elif gini <= 0.6:
        return WorkloadDistributionLevel.UNEVEN
    else:
        return WorkloadDistributionLevel.HIGHLY_CONCENTRATED


def calculate_hours_distribution(
    user_activity_data: Dict[str, Dict[str, Any]],
    start_date: date,
    end_date: date,
    previous_period_data: Optional[Dict[str, Dict[str, Any]]] = None
) -> HoursDistributionSummary:
    """Calculate complete hours distribution analysis.

    Args:
        user_activity_data: Dictionary mapping usernames to their activity data
        start_date: Start of the analysis period
        end_date: End of the analysis period
        previous_period_data: Optional previous period data for trend calculation

    Returns:
        HoursDistributionSummary with all distribution metrics and patterns

    Example:
        >>> user_data = {
        ...     "UserA": {"count": 3, "hours": 10.5, "categories": {"Meeting": 5.0, "Research": 5.5}},
        ...     "UserB": {"count": 2, "hours": 8.0, "categories": {"Meeting": 8.0}}
        ... }
        >>> summary = calculate_hours_distribution(user_data, date(2026, 1, 1), date(2026, 1, 31))
        >>> print(f"Total hours: {summary.metrics.total_hours}")
    """
    logger.info(f"Calculating hours distribution for {start_date} to {end_date}")

    # Aggregate current period data
    category_hours, category_counts, category_by_user = aggregate_hours_by_category(
        user_activity_data
    )

    # Calculate totals
    total_hours = sum(category_hours.values())
    total_items = sum(category_counts.values())
    total_categories = len(category_hours)

    logger.info(
        f"Found {total_categories} categories with {total_hours:.1f} total hours "
        f"and {total_items} items"
    )

    # Aggregate previous period data if available
    prev_category_hours: Dict[str, float] = {}
    if previous_period_data:
        prev_category_hours, _, _ = aggregate_hours_by_category(previous_period_data)
        logger.info(f"Previous period has {len(prev_category_hours)} categories")

    # Calculate standard deviation of hours
    hours_values = list(category_hours.values())
    hours_std = statistics.stdev(hours_values) if len(hours_values) > 1 else 0.0

    # Calculate Gini coefficient for distribution inequality
    gini = calculate_gini_coefficient(hours_values)

    # Calculate average hours per item (overall)
    avg_hours_per_item = total_hours / total_items if total_items > 0 else 0.0

    # Build metrics
    metrics = HoursDistributionMetrics(
        total_hours=total_hours,
        total_items=total_items,
        total_categories=total_categories,
        hours_std_deviation=hours_std,
        gini_coefficient=gini,
    )

    # Sort categories by hours to calculate cumulative percentages
    sorted_categories = sorted(
        category_hours.items(),
        key=lambda x: x[1],
        reverse=True
    )

    # Build category distribution list with rankings
    categories: List[CategoryHoursDistribution] = []
    cumulative_pct = 0.0

    for rank, (category_name, hours) in enumerate(sorted_categories, 1):
        item_count = category_counts.get(category_name, 0)
        user_breakdown = dict(category_by_user.get(category_name, {}))
        hours_per_item = hours / item_count if item_count > 0 else 0.0

        # Calculate percentages
        pct_hours = (hours / total_hours * 100) if total_hours > 0 else 0.0
        pct_items = (item_count / total_items * 100) if total_items > 0 else 0.0
        cumulative_pct += pct_hours

        # Classify intensity and efficiency
        intensity = classify_intensity_level(pct_hours, cumulative_pct)
        efficiency = classify_efficiency_level(hours_per_item, avg_hours_per_item)

        # Calculate trend data
        prev_hours = prev_category_hours.get(category_name)
        hours_pct_change: Optional[float] = None
        trend = TrendDirection.NO_DATA

        if prev_hours is not None:
            hours_pct_change = calculate_percent_change(
                int(hours * 100),
                int(prev_hours * 100)
            )
            trend = get_trend_direction(hours_pct_change)

        distribution = CategoryHoursDistribution(
            category_name=category_name,
            total_hours=hours,
            item_count=item_count,
            percentage_of_total_hours=pct_hours,
            percentage_of_total_items=pct_items,
            hours_per_item=hours_per_item,
            rank_by_hours=rank,
            intensity_level=intensity,
            efficiency_level=efficiency,
            by_user=user_breakdown,
            previous_period_hours=prev_hours,
            hours_percent_change=hours_pct_change,
            trend=trend,
        )
        categories.append(distribution)

    # Assign efficiency rankings
    efficiency_sorted = sorted(
        enumerate(categories),
        key=lambda x: x[1].hours_per_item
    )
    for efficiency_rank, (original_idx, _) in enumerate(efficiency_sorted, 1):
        categories[original_idx].rank_by_efficiency = efficiency_rank

    # Identify time-intensive categories (top 20% by hours)
    time_intensive = [
        cat.category_name
        for cat in categories
        if cat.intensity_level in [CategoryIntensityLevel.CRITICAL, CategoryIntensityLevel.HIGH]
    ]

    # Identify most/least efficient categories
    most_efficient = [
        cat.category_name
        for cat in categories
        if cat.efficiency_level in [EfficiencyLevel.HIGHLY_EFFICIENT, EfficiencyLevel.EFFICIENT]
    ][:5]

    least_efficient = [
        cat.category_name
        for cat in categories
        if cat.efficiency_level in [EfficiencyLevel.HIGHLY_INEFFICIENT, EfficiencyLevel.INEFFICIENT]
    ][:5]

    # Classify workload distribution
    workload_dist = classify_workload_distribution(gini)

    # Analyze efficiency patterns
    patterns = _identify_efficiency_patterns(
        categories,
        metrics,
        workload_dist
    )

    summary = HoursDistributionSummary(
        start_date=start_date,
        end_date=end_date,
        metrics=metrics,
        categories=categories,
        time_intensive_categories=time_intensive,
        most_efficient_categories=most_efficient,
        least_efficient_categories=least_efficient,
        efficiency_patterns=patterns,
        workload_distribution=workload_dist,
    )

    return summary


def _identify_efficiency_patterns(
    categories: List[CategoryHoursDistribution],
    metrics: HoursDistributionMetrics,
    workload_dist: WorkloadDistributionLevel
) -> List[EfficiencyPattern]:
    """Identify efficiency patterns in the hours distribution.

    Args:
        categories: List of category distributions
        metrics: Overall distribution metrics
        workload_dist: Workload distribution classification

    Returns:
        List of identified EfficiencyPattern objects
    """
    patterns: List[EfficiencyPattern] = []

    # Pattern 1: High concentration (single category dominance)
    if categories and categories[0].percentage_of_total_hours > 40:
        top_cat = categories[0]
        patterns.append(EfficiencyPattern(
            pattern_type="high_concentration",
            description=f"Category '{top_cat.category_name}' consumes {top_cat.percentage_of_total_hours:.1f}% of total hours",
            affected_categories=[top_cat.category_name],
            severity="warning",
            recommendation="Consider if this concentration is expected or if workload should be redistributed",
            metrics={
                "dominant_category": top_cat.category_name,
                "percentage": top_cat.percentage_of_total_hours,
            }
        ))

    # Pattern 2: Efficiency outliers (very high hours per item)
    inefficient_cats = [
        cat for cat in categories
        if cat.efficiency_level == EfficiencyLevel.HIGHLY_INEFFICIENT
    ]
    if inefficient_cats:
        patterns.append(EfficiencyPattern(
            pattern_type="low_efficiency_categories",
            description=f"{len(inefficient_cats)} categories have significantly high hours per item",
            affected_categories=[cat.category_name for cat in inefficient_cats],
            severity="warning" if len(inefficient_cats) > 2 else "info",
            recommendation="Review these categories for process improvement opportunities",
            metrics={
                "categories": [
                    {"name": cat.category_name, "hours_per_item": cat.hours_per_item}
                    for cat in inefficient_cats
                ],
                "average_hours_per_item": metrics.average_hours_per_item,
            }
        ))

    # Pattern 3: User concentration within categories
    concentrated_categories = []
    for cat in categories:
        if cat.user_count > 0 and cat.top_contributor_hours > 0:
            concentration = cat.top_contributor_hours / cat.total_hours
            if concentration > 0.7 and cat.user_count > 1:
                concentrated_categories.append({
                    "category": cat.category_name,
                    "top_contributor": cat.top_contributor,
                    "concentration": concentration,
                })

    if concentrated_categories:
        patterns.append(EfficiencyPattern(
            pattern_type="user_concentration",
            description=f"{len(concentrated_categories)} categories have work concentrated with single users",
            affected_categories=[c["category"] for c in concentrated_categories],
            severity="info",
            recommendation="Consider cross-training to distribute expertise",
            metrics={"concentrated_categories": concentrated_categories}
        ))

    # Pattern 4: Overall workload imbalance
    if workload_dist in [WorkloadDistributionLevel.UNEVEN, WorkloadDistributionLevel.HIGHLY_CONCENTRATED]:
        patterns.append(EfficiencyPattern(
            pattern_type="workload_imbalance",
            description=f"Workload distribution is {workload_dist.value}",
            affected_categories=[cat.category_name for cat in categories[:3]],
            severity="warning" if workload_dist == WorkloadDistributionLevel.HIGHLY_CONCENTRATED else "info",
            recommendation="Review resource allocation across categories",
            metrics={
                "gini_coefficient": metrics.gini_coefficient,
                "distribution_level": workload_dist.value,
            }
        ))

    # Pattern 5: Trending categories (significant changes)
    trending_up = [
        cat for cat in categories
        if cat.trend == TrendDirection.UP and cat.hours_percent_change and cat.hours_percent_change > 25
    ]
    trending_down = [
        cat for cat in categories
        if cat.trend == TrendDirection.DOWN and cat.hours_percent_change and cat.hours_percent_change < -25
    ]

    if trending_up:
        patterns.append(EfficiencyPattern(
            pattern_type="increasing_demand",
            description=f"{len(trending_up)} categories show significant hour increases (>25%)",
            affected_categories=[cat.category_name for cat in trending_up],
            severity="info",
            recommendation="Monitor capacity for these growing categories",
            metrics={
                "trending_categories": [
                    {"name": cat.category_name, "change": cat.hours_percent_change}
                    for cat in trending_up
                ]
            }
        ))

    if trending_down:
        patterns.append(EfficiencyPattern(
            pattern_type="decreasing_demand",
            description=f"{len(trending_down)} categories show significant hour decreases (>25%)",
            affected_categories=[cat.category_name for cat in trending_down],
            severity="info",
            recommendation="Evaluate if reduced hours indicate efficiency gains or reduced activity",
            metrics={
                "trending_categories": [
                    {"name": cat.category_name, "change": cat.hours_percent_change}
                    for cat in trending_down
                ]
            }
        ))

    return patterns


def identify_time_intensive_categories(
    summary: HoursDistributionSummary,
    threshold_percentile: float = 80.0
) -> List[Dict[str, Any]]:
    """Identify time-intensive categories (categories consuming most hours).

    Args:
        summary: HoursDistributionSummary from calculate_hours_distribution
        threshold_percentile: Percentile threshold for "time-intensive" (default 80%)

    Returns:
        List of dictionaries with category details for time-intensive categories

    Example:
        >>> time_intensive = identify_time_intensive_categories(summary)
        >>> for cat in time_intensive:
        ...     print(f"{cat['name']}: {cat['hours']}h ({cat['percentage']}%)")
    """
    # Calculate threshold based on percentile
    if not summary.categories:
        return []

    hours_values = sorted([cat.total_hours for cat in summary.categories], reverse=True)
    threshold_index = int(len(hours_values) * (100 - threshold_percentile) / 100)
    threshold_hours = hours_values[threshold_index] if threshold_index < len(hours_values) else 0

    result = []
    cumulative_hours = 0.0

    for cat in summary.categories:
        if cat.total_hours >= threshold_hours:
            cumulative_hours += cat.total_hours
            result.append({
                "name": cat.category_name,
                "hours": round(cat.total_hours, 2),
                "percentage": round(cat.percentage_of_total_hours, 1),
                "items": cat.item_count,
                "hours_per_item": cat.hours_per_item,
                "intensity_level": cat.intensity_level.value,
                "cumulative_hours": round(cumulative_hours, 2),
                "cumulative_percentage": round(
                    cumulative_hours / summary.metrics.total_hours * 100, 1
                ) if summary.metrics.total_hours > 0 else 0.0,
                "trend": cat.trend.value if cat.trend != TrendDirection.NO_DATA else None,
            })

    return result


def analyze_efficiency_patterns(
    summary: HoursDistributionSummary
) -> Dict[str, Any]:
    """Analyze efficiency patterns in the hours distribution.

    Args:
        summary: HoursDistributionSummary from calculate_hours_distribution

    Returns:
        Dictionary with comprehensive efficiency analysis

    Example:
        >>> analysis = analyze_efficiency_patterns(summary)
        >>> print(f"Overall efficiency: {analysis['overall_efficiency']}")
        >>> for pattern in analysis['patterns']:
        ...     print(f"- {pattern['description']}")
    """
    # Calculate overall efficiency metrics
    efficient_categories = len([
        cat for cat in summary.categories
        if cat.efficiency_level in [EfficiencyLevel.HIGHLY_EFFICIENT, EfficiencyLevel.EFFICIENT]
    ])
    total_categories = len(summary.categories)
    efficiency_ratio = efficient_categories / total_categories if total_categories > 0 else 0

    # Determine overall efficiency classification
    if efficiency_ratio >= 0.6:
        overall_efficiency = "excellent"
    elif efficiency_ratio >= 0.4:
        overall_efficiency = "good"
    elif efficiency_ratio >= 0.2:
        overall_efficiency = "average"
    else:
        overall_efficiency = "needs_improvement"

    # Find optimization opportunities
    optimization_opportunities = []
    for cat in summary.categories:
        if cat.efficiency_level in [EfficiencyLevel.INEFFICIENT, EfficiencyLevel.HIGHLY_INEFFICIENT]:
            potential_savings = (
                cat.hours_per_item - summary.metrics.average_hours_per_item
            ) * cat.item_count
            if potential_savings > 0:
                optimization_opportunities.append({
                    "category": cat.category_name,
                    "current_hours_per_item": cat.hours_per_item,
                    "target_hours_per_item": summary.metrics.average_hours_per_item,
                    "potential_hours_saved": round(potential_savings, 1),
                    "potential_percentage_saved": round(
                        potential_savings / summary.metrics.total_hours * 100, 1
                    ) if summary.metrics.total_hours > 0 else 0.0,
                })

    # Sort opportunities by potential savings
    optimization_opportunities.sort(key=lambda x: x["potential_hours_saved"], reverse=True)

    return {
        "overall_efficiency": overall_efficiency,
        "efficiency_ratio": round(efficiency_ratio * 100, 1),
        "efficient_categories_count": efficient_categories,
        "total_categories_count": total_categories,
        "patterns": [p.to_dict() for p in summary.efficiency_patterns],
        "optimization_opportunities": optimization_opportunities[:5],  # Top 5
        "workload_distribution": summary.workload_distribution.value,
        "distribution_equality": {
            "gini_coefficient": round(summary.metrics.gini_coefficient, 3),
            "interpretation": _interpret_gini(summary.metrics.gini_coefficient),
        },
        "category_efficiency_breakdown": {
            "highly_efficient": len([c for c in summary.categories if c.efficiency_level == EfficiencyLevel.HIGHLY_EFFICIENT]),
            "efficient": len([c for c in summary.categories if c.efficiency_level == EfficiencyLevel.EFFICIENT]),
            "average": len([c for c in summary.categories if c.efficiency_level == EfficiencyLevel.AVERAGE]),
            "inefficient": len([c for c in summary.categories if c.efficiency_level == EfficiencyLevel.INEFFICIENT]),
            "highly_inefficient": len([c for c in summary.categories if c.efficiency_level == EfficiencyLevel.HIGHLY_INEFFICIENT]),
        }
    }


def _interpret_gini(gini: float) -> str:
    """Provide human-readable interpretation of Gini coefficient."""
    if gini <= 0.2:
        return "Hours are very evenly distributed across categories"
    elif gini <= 0.4:
        return "Hours distribution shows moderate inequality"
    elif gini <= 0.6:
        return "Hours are unevenly distributed with some concentration"
    else:
        return "Hours are highly concentrated in few categories"


def get_distribution_visualization_data(
    summary: HoursDistributionSummary
) -> Dict[str, Any]:
    """Get data formatted for visualization (charts, graphs).

    Args:
        summary: HoursDistributionSummary from calculate_hours_distribution

    Returns:
        Dictionary with data formatted for various chart types

    Example:
        >>> viz_data = get_distribution_visualization_data(summary)
        >>> pie_data = viz_data["pie_chart"]
        >>> bar_data = viz_data["bar_chart"]
    """
    # Pie chart data (category distribution)
    pie_data = [
        {
            "name": cat.category_name,
            "value": round(cat.total_hours, 2),
            "percentage": round(cat.percentage_of_total_hours, 1),
        }
        for cat in summary.categories
    ]

    # Bar chart data (hours per item comparison)
    bar_data = [
        {
            "category": cat.category_name,
            "hours_per_item": cat.hours_per_item,
            "average": summary.metrics.average_hours_per_item,
        }
        for cat in summary.categories
    ]

    # Stacked bar data (by user within category)
    stacked_data = []
    all_users = set()
    for cat in summary.categories:
        all_users.update(cat.by_user.keys())
    all_users = sorted(all_users)

    for cat in summary.categories:
        entry = {"category": cat.category_name}
        for user in all_users:
            entry[user] = round(cat.by_user.get(user, 0), 2)
        stacked_data.append(entry)

    # Trend data (if available)
    trend_data = []
    if summary.has_trend_data:
        for cat in summary.categories:
            if cat.previous_period_hours is not None:
                trend_data.append({
                    "category": cat.category_name,
                    "current": round(cat.total_hours, 2),
                    "previous": round(cat.previous_period_hours, 2),
                    "change_percent": cat.hours_percent_change,
                    "trend": cat.trend.value,
                })

    # Intensity distribution (for donut/ring chart)
    intensity_data = {
        "critical": len([c for c in summary.categories if c.intensity_level == CategoryIntensityLevel.CRITICAL]),
        "high": len([c for c in summary.categories if c.intensity_level == CategoryIntensityLevel.HIGH]),
        "medium": len([c for c in summary.categories if c.intensity_level == CategoryIntensityLevel.MEDIUM]),
        "low": len([c for c in summary.categories if c.intensity_level == CategoryIntensityLevel.LOW]),
    }

    return {
        "pie_chart": pie_data,
        "bar_chart": bar_data,
        "stacked_bar": stacked_data,
        "users": list(all_users),
        "trend_data": trend_data,
        "intensity_distribution": intensity_data,
        "summary_metrics": {
            "total_hours": round(summary.metrics.total_hours, 2),
            "total_items": summary.metrics.total_items,
            "categories": summary.metrics.total_categories,
            "avg_hours_per_item": summary.metrics.average_hours_per_item,
        }
    }


def format_distribution_report(
    summary: HoursDistributionSummary
) -> Dict[str, Any]:
    """Format distribution analysis as a report-ready structure.

    Args:
        summary: HoursDistributionSummary from calculate_hours_distribution

    Returns:
        Dictionary with formatted report sections

    Example:
        >>> report = format_distribution_report(summary)
        >>> print(report["executive_summary"])
    """
    # Executive summary
    top_cats = summary.get_top_categories(3)
    top_cats_text = ", ".join([
        f"{cat.category_name} ({cat.percentage_of_total_hours:.1f}%)"
        for cat in top_cats
    ])

    efficiency_analysis = analyze_efficiency_patterns(summary)

    executive_summary = (
        f"Analysis of {summary.metrics.total_hours:.1f} total hours across "
        f"{summary.metrics.total_categories} categories from {summary.start_date} to {summary.end_date}. "
        f"Top categories by hours: {top_cats_text}. "
        f"Overall efficiency rating: {efficiency_analysis['overall_efficiency']}. "
        f"Workload distribution: {summary.workload_distribution.value}."
    )

    # Key findings
    key_findings = []

    # Time-intensive finding
    if summary.time_intensive_categories:
        key_findings.append({
            "type": "time_intensive",
            "title": "Time-Intensive Categories",
            "finding": f"{len(summary.time_intensive_categories)} categories consume the majority of hours",
            "categories": summary.time_intensive_categories[:5],
        })

    # Efficiency finding
    if efficiency_analysis["optimization_opportunities"]:
        potential_savings = sum(
            opp["potential_hours_saved"]
            for opp in efficiency_analysis["optimization_opportunities"]
        )
        key_findings.append({
            "type": "efficiency",
            "title": "Efficiency Opportunities",
            "finding": f"Potential to save {potential_savings:.1f} hours by improving efficiency in underperforming categories",
            "opportunities": efficiency_analysis["optimization_opportunities"][:3],
        })

    # Trend finding
    if summary.has_trend_data:
        trending_up = [
            cat for cat in summary.categories
            if cat.trend == TrendDirection.UP
        ]
        trending_down = [
            cat for cat in summary.categories
            if cat.trend == TrendDirection.DOWN
        ]
        if trending_up or trending_down:
            key_findings.append({
                "type": "trends",
                "title": "Category Trends",
                "finding": f"{len(trending_up)} categories increasing, {len(trending_down)} decreasing vs previous period",
                "increasing": [cat.category_name for cat in trending_up[:3]],
                "decreasing": [cat.category_name for cat in trending_down[:3]],
            })

    # Category table
    category_table = [
        {
            "rank": cat.rank_by_hours,
            "category": cat.category_name,
            "hours": f"{cat.total_hours:.1f}",
            "percentage": f"{cat.percentage_of_total_hours:.1f}%",
            "items": cat.item_count,
            "hours_per_item": f"{cat.hours_per_item:.2f}",
            "intensity": cat.intensity_level.value,
            "efficiency": cat.efficiency_level.value,
            "trend": get_trend_indicator(cat.trend) if cat.trend != TrendDirection.NO_DATA else "-",
        }
        for cat in summary.categories
    ]

    return {
        "period": {
            "start": summary.start_date.isoformat(),
            "end": summary.end_date.isoformat(),
        },
        "executive_summary": executive_summary,
        "key_metrics": {
            "total_hours": round(summary.metrics.total_hours, 2),
            "total_items": summary.metrics.total_items,
            "total_categories": summary.metrics.total_categories,
            "avg_hours_per_item": summary.metrics.average_hours_per_item,
            "avg_hours_per_category": summary.metrics.average_hours_per_category,
        },
        "key_findings": key_findings,
        "efficiency_analysis": efficiency_analysis,
        "category_table": category_table,
        "patterns": [p.to_dict() for p in summary.efficiency_patterns],
        "recommendations": _generate_recommendations(summary, efficiency_analysis),
    }


def _generate_recommendations(
    summary: HoursDistributionSummary,
    efficiency_analysis: Dict[str, Any]
) -> List[Dict[str, str]]:
    """Generate actionable recommendations based on analysis."""
    recommendations = []

    # Recommendation based on overall efficiency
    if efficiency_analysis["overall_efficiency"] == "needs_improvement":
        recommendations.append({
            "priority": "high",
            "area": "Process Efficiency",
            "recommendation": "Review high-hour categories for process optimization opportunities",
            "expected_impact": "Could reduce total hours by 10-20%",
        })
    elif efficiency_analysis["overall_efficiency"] == "average":
        recommendations.append({
            "priority": "medium",
            "area": "Process Efficiency",
            "recommendation": "Focus on improving efficiency in bottom-performing categories",
            "expected_impact": "Could improve overall efficiency rating to 'good'",
        })

    # Recommendation based on workload distribution
    if summary.workload_distribution == WorkloadDistributionLevel.HIGHLY_CONCENTRATED:
        recommendations.append({
            "priority": "high",
            "area": "Workload Balance",
            "recommendation": "Redistribute work across more categories or add resources to high-volume areas",
            "expected_impact": "Better resource utilization and reduced bottlenecks",
        })
    elif summary.workload_distribution == WorkloadDistributionLevel.UNEVEN:
        recommendations.append({
            "priority": "medium",
            "area": "Workload Balance",
            "recommendation": "Monitor workload trends to prevent further concentration",
            "expected_impact": "Maintain sustainable workload distribution",
        })

    # Recommendation based on optimization opportunities
    if efficiency_analysis["optimization_opportunities"]:
        top_opp = efficiency_analysis["optimization_opportunities"][0]
        recommendations.append({
            "priority": "medium",
            "area": "Category Optimization",
            "recommendation": f"Investigate efficiency in '{top_opp['category']}' category",
            "expected_impact": f"Could save {top_opp['potential_hours_saved']:.1f} hours",
        })

    # Recommendation based on trends (if available)
    trending_patterns = [
        p for p in summary.efficiency_patterns
        if p.pattern_type in ["increasing_demand", "decreasing_demand"]
    ]
    for pattern in trending_patterns[:1]:
        recommendations.append({
            "priority": "low",
            "area": "Capacity Planning",
            "recommendation": pattern.recommendation,
            "expected_impact": "Better resource planning for future periods",
        })

    return recommendations


if __name__ == "__main__":
    # Demo usage with sample data
    print("Special Activities Hours Distribution - Demo")
    print("=" * 60)

    # Sample user activity data (mimicking get_special_activities output)
    sample_data = {
        "UserA": {
            "count": 5,
            "hours": 15.5,
            "categories": {
                "Meetings": 8.0,
                "Compliance": 4.5,
                "Research": 3.0,
            }
        },
        "UserB": {
            "count": 4,
            "hours": 12.0,
            "categories": {
                "Meetings": 6.0,
                "Organisatorische Aufgaben": 4.0,
                "Research": 2.0,
            }
        },
        "UserC": {
            "count": 3,
            "hours": 8.5,
            "categories": {
                "Compliance": 5.0,
                "Meetings": 3.5,
            }
        },
    }

    # Previous period data for trend comparison
    previous_data = {
        "UserA": {
            "count": 4,
            "hours": 12.0,
            "categories": {
                "Meetings": 6.0,
                "Compliance": 6.0,
            }
        },
        "UserB": {
            "count": 3,
            "hours": 10.0,
            "categories": {
                "Meetings": 7.0,
                "Research": 3.0,
            }
        },
    }

    # Calculate hours distribution
    summary = calculate_hours_distribution(
        user_activity_data=sample_data,
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 31),
        previous_period_data=previous_data,
    )

    print(f"\nPeriod: {summary.start_date} to {summary.end_date}")
    print(f"Total Hours: {summary.metrics.total_hours:.1f}")
    print(f"Total Items: {summary.metrics.total_items}")
    print(f"Categories: {summary.metrics.total_categories}")
    print(f"Avg Hours/Item: {summary.metrics.average_hours_per_item:.2f}")
    print(f"Gini Coefficient: {summary.metrics.gini_coefficient:.3f}")
    print(f"Workload Distribution: {summary.workload_distribution.value}")

    print("\n" + "-" * 60)
    print("Hours Distribution by Category:")
    print("-" * 60)

    for cat in summary.categories:
        trend_info = ""
        if cat.hours_percent_change is not None:
            indicator = get_trend_indicator(cat.trend)
            trend_info = f" [{indicator} {format_percent_change(cat.hours_percent_change)}]"

        print(f"  {cat.rank_by_hours}. {cat.category_name}: {cat.total_hours:.1f}h "
              f"({cat.percentage_of_total_hours:.1f}%) - {cat.item_count} items "
              f"[{cat.intensity_level.value.upper()}]{trend_info}")
        print(f"     Hours/Item: {cat.hours_per_item:.2f} ({cat.efficiency_level.value})")

    # Time-intensive categories
    print("\n" + "-" * 60)
    print("Time-Intensive Categories:")
    print("-" * 60)
    time_intensive = identify_time_intensive_categories(summary)
    for cat in time_intensive:
        print(f"  - {cat['name']}: {cat['hours']}h ({cat['percentage']}%) "
              f"[{cat['intensity_level'].upper()}]")

    # Efficiency analysis
    print("\n" + "-" * 60)
    print("Efficiency Analysis:")
    print("-" * 60)
    efficiency = analyze_efficiency_patterns(summary)
    print(f"  Overall Efficiency: {efficiency['overall_efficiency']}")
    print(f"  Efficient Categories: {efficiency['efficient_categories_count']}/{efficiency['total_categories_count']}")
    print(f"  Distribution: {efficiency['distribution_equality']['interpretation']}")

    if efficiency["optimization_opportunities"]:
        print("\n  Optimization Opportunities:")
        for opp in efficiency["optimization_opportunities"][:3]:
            print(f"    - {opp['category']}: Save {opp['potential_hours_saved']:.1f}h "
                  f"({opp['potential_percentage_saved']:.1f}%)")

    # Efficiency patterns
    print("\n" + "-" * 60)
    print("Identified Patterns:")
    print("-" * 60)
    for pattern in summary.efficiency_patterns:
        print(f"  [{pattern.severity.upper()}] {pattern.pattern_type}:")
        print(f"    {pattern.description}")
        print(f"    Recommendation: {pattern.recommendation}")

    # Format report
    print("\n" + "-" * 60)
    print("Report Summary:")
    print("-" * 60)
    report = format_distribution_report(summary)
    print(f"  {report['executive_summary']}")

    print("\n  Key Findings:")
    for finding in report["key_findings"]:
        print(f"    - {finding['title']}: {finding['finding']}")

    print("\n  Recommendations:")
    for rec in report["recommendations"]:
        print(f"    [{rec['priority'].upper()}] {rec['area']}: {rec['recommendation']}")
