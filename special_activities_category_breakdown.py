"""
Special Activities Category Breakdown Module

Aggregates special activities data by category with total hours and item counts.
Calculates category percentages and trends for period comparisons.

This module provides analytics for special activities data, allowing users to
understand the distribution of activities across categories and track changes
over time. It includes period-over-period comparison functionality to analyze
changes in category distribution and total hours.

Usage:
    from special_activities_category_breakdown import (
        CategoryBreakdown,
        CategoryBreakdownSummary,
        SpecialActivitiesPeriodComparison,
        calculate_category_breakdown,
        get_category_breakdown_summary,
        calculate_category_trends,
        compare_special_activities_periods,
        format_period_comparison_table,
        get_period_comparison_summary,
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
    format_percent_change,
    get_trend_indicator,
)

# Set up logging
logger = logging.getLogger(__name__)


@dataclass
class SpecialActivitiesPeriodComparison:
    """Comparison of special activities between current and previous periods.

    This dataclass provides a comprehensive comparison of special activities
    across two time periods, including total hours, category distribution changes,
    and trend analysis.

    Attributes:
        current_period_start: Start date of the current period
        current_period_end: End date of the current period
        previous_period_start: Start date of the previous period
        previous_period_end: End date of the previous period
        current_total_hours: Total hours in current period
        previous_total_hours: Total hours in previous period
        hours_change: Absolute change in total hours
        hours_percent_change: Percentage change in total hours
        hours_trend: Trend direction for total hours
        current_total_items: Total activity items in current period
        previous_total_items: Total activity items in previous period
        items_change: Absolute change in total items
        items_percent_change: Percentage change in total items
        items_trend: Trend direction for total items
        current_category_count: Number of categories in current period
        previous_category_count: Number of categories in previous period
        category_comparisons: Detailed comparison for each category
        new_categories: Categories appearing only in current period
        dropped_categories: Categories appearing only in previous period
        top_increasing_categories: Categories with largest hours increase
        top_decreasing_categories: Categories with largest hours decrease
    """
    current_period_start: date
    current_period_end: date
    previous_period_start: date
    previous_period_end: date
    current_total_hours: float
    previous_total_hours: float
    hours_change: float = 0.0
    hours_percent_change: float = 0.0
    hours_trend: TrendDirection = TrendDirection.NO_DATA
    current_total_items: int = 0
    previous_total_items: int = 0
    items_change: int = 0
    items_percent_change: float = 0.0
    items_trend: TrendDirection = TrendDirection.NO_DATA
    current_category_count: int = 0
    previous_category_count: int = 0
    category_comparisons: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    new_categories: List[str] = field(default_factory=list)
    dropped_categories: List[str] = field(default_factory=list)
    top_increasing_categories: List[Dict[str, Any]] = field(default_factory=list)
    top_decreasing_categories: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        # Calculate hours change
        self.hours_change = self.current_total_hours - self.previous_total_hours
        self.hours_percent_change = calculate_percent_change(
            int(self.current_total_hours * 100),
            int(self.previous_total_hours * 100)
        )
        self.hours_trend = get_trend_direction(self.hours_percent_change)

        # Calculate items change
        self.items_change = self.current_total_items - self.previous_total_items
        self.items_percent_change = calculate_percent_change(
            self.current_total_items,
            self.previous_total_items
        )
        self.items_trend = get_trend_direction(self.items_percent_change)

        # Calculate top increasing/decreasing categories
        self._calculate_top_movers()

    def _calculate_top_movers(self):
        """Calculate top increasing and decreasing categories."""
        # Sort categories by hours change
        sorted_categories = sorted(
            self.category_comparisons.items(),
            key=lambda x: x[1].get("hours_change", 0),
            reverse=True
        )

        # Top increasing (positive change)
        self.top_increasing_categories = [
            {"category": cat, **data}
            for cat, data in sorted_categories
            if data.get("hours_change", 0) > 0
        ][:3]

        # Top decreasing (negative change)
        self.top_decreasing_categories = [
            {"category": cat, **data}
            for cat, data in sorted_categories[::-1]
            if data.get("hours_change", 0) < 0
        ][:3]

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "current_period": {
                "start": self.current_period_start.isoformat(),
                "end": self.current_period_end.isoformat(),
                "total_hours": round(self.current_total_hours, 2),
                "total_items": self.current_total_items,
                "category_count": self.current_category_count,
            },
            "previous_period": {
                "start": self.previous_period_start.isoformat(),
                "end": self.previous_period_end.isoformat(),
                "total_hours": round(self.previous_total_hours, 2),
                "total_items": self.previous_total_items,
                "category_count": self.previous_category_count,
            },
            "comparison": {
                "hours_change": round(self.hours_change, 2),
                "hours_percent_change": round(self.hours_percent_change, 2),
                "hours_trend": self.hours_trend.value,
                "items_change": self.items_change,
                "items_percent_change": round(self.items_percent_change, 2),
                "items_trend": self.items_trend.value,
            },
            "category_comparisons": self.category_comparisons,
            "new_categories": self.new_categories,
            "dropped_categories": self.dropped_categories,
            "top_increasing_categories": self.top_increasing_categories,
            "top_decreasing_categories": self.top_decreasing_categories,
        }

    def get_summary_text(self) -> str:
        """Generate a human-readable summary of the comparison."""
        hours_indicator = get_trend_indicator(self.hours_trend)
        items_indicator = get_trend_indicator(self.items_trend)

        summary_parts = [
            f"Period: {self.current_period_start.strftime('%b %d')} - {self.current_period_end.strftime('%b %d')} vs "
            f"{self.previous_period_start.strftime('%b %d')} - {self.previous_period_end.strftime('%b %d')}",
            f"Total Hours: {self.current_total_hours:.1f}h {hours_indicator} {format_percent_change(self.hours_percent_change)} "
            f"(was {self.previous_total_hours:.1f}h)",
            f"Activities: {self.current_total_items} {items_indicator} {format_percent_change(self.items_percent_change)} "
            f"(was {self.previous_total_items})",
        ]

        if self.new_categories:
            summary_parts.append(f"New categories: {', '.join(self.new_categories)}")

        if self.dropped_categories:
            summary_parts.append(f"Dropped categories: {', '.join(self.dropped_categories)}")

        return " | ".join(summary_parts)


@dataclass
class CategoryBreakdown:
    """Statistics for a specific special activity category.

    Attributes:
        category_name: The name of the activity category
        total_hours: Total hours spent on activities in this category
        item_count: Number of activities in this category
        percentage_of_total_hours: Percentage of total hours across all categories
        percentage_of_total_items: Percentage of total items across all categories
        by_user: Breakdown of hours by user for this category
        previous_period_hours: Hours from the previous period (if available)
        previous_period_count: Item count from previous period (if available)
        hours_percent_change: Percentage change in hours vs previous period
        count_percent_change: Percentage change in count vs previous period
        trend: Trend direction based on hours change (UP, DOWN, FLAT)
    """
    category_name: str
    total_hours: float
    item_count: int
    percentage_of_total_hours: float = 0.0
    percentage_of_total_items: float = 0.0
    by_user: Dict[str, float] = field(default_factory=dict)
    previous_period_hours: Optional[float] = None
    previous_period_count: Optional[int] = None
    hours_percent_change: Optional[float] = None
    count_percent_change: Optional[float] = None
    trend: TrendDirection = TrendDirection.NO_DATA

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "category_name": self.category_name,
            "total_hours": round(self.total_hours, 2),
            "item_count": self.item_count,
            "percentage_of_total_hours": round(self.percentage_of_total_hours, 2),
            "percentage_of_total_items": round(self.percentage_of_total_items, 2),
            "by_user": {k: round(v, 2) for k, v in self.by_user.items()},
            "previous_period_hours": round(self.previous_period_hours, 2) if self.previous_period_hours is not None else None,
            "previous_period_count": self.previous_period_count,
            "hours_percent_change": round(self.hours_percent_change, 2) if self.hours_percent_change is not None else None,
            "count_percent_change": round(self.count_percent_change, 2) if self.count_percent_change is not None else None,
            "trend": self.trend.value,
        }


@dataclass
class CategoryBreakdownSummary:
    """Summary of all category breakdowns for a period.

    Attributes:
        start_date: Start of the analysis period
        end_date: End of the analysis period
        total_hours: Total hours across all categories
        total_items: Total number of activity items
        category_count: Number of unique categories
        categories: List of CategoryBreakdown objects sorted by hours (descending)
        top_category: The category with the most hours
        has_trend_data: Whether trend data is available
    """
    start_date: date
    end_date: date
    total_hours: float
    total_items: int
    category_count: int
    categories: List[CategoryBreakdown] = field(default_factory=list)
    top_category: Optional[str] = None
    has_trend_data: bool = False

    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        if self.categories:
            # Sort categories by total hours descending
            self.categories = sorted(
                self.categories,
                key=lambda x: x.total_hours,
                reverse=True
            )
            self.top_category = self.categories[0].category_name if self.categories else None
            self.has_trend_data = any(
                cat.previous_period_hours is not None for cat in self.categories
            )

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "start_date": self.start_date.isoformat(),
            "end_date": self.end_date.isoformat(),
            "total_hours": round(self.total_hours, 2),
            "total_items": self.total_items,
            "category_count": self.category_count,
            "categories": [cat.to_dict() for cat in self.categories],
            "top_category": self.top_category,
            "has_trend_data": self.has_trend_data,
        }

    def get_top_categories(self, n: int = 5) -> List[CategoryBreakdown]:
        """Get the top N categories by hours.

        Args:
            n: Number of categories to return (default 5)

        Returns:
            List of CategoryBreakdown objects for top N categories
        """
        return self.categories[:n]

    def get_trending_up(self) -> List[CategoryBreakdown]:
        """Get categories with increasing hours (UP trend).

        Returns:
            List of CategoryBreakdown objects with UP trend
        """
        return [cat for cat in self.categories if cat.trend == TrendDirection.UP]

    def get_trending_down(self) -> List[CategoryBreakdown]:
        """Get categories with decreasing hours (DOWN trend).

        Returns:
            List of CategoryBreakdown objects with DOWN trend
        """
        return [cat for cat in self.categories if cat.trend == TrendDirection.DOWN]


def aggregate_by_category(
    user_activity_data: Dict[str, Dict[str, Any]]
) -> Tuple[Dict[str, float], Dict[str, int], Dict[str, Dict[str, float]]]:
    """Aggregate special activities data by category.

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
            category_counts[category] += 1  # Each entry is one activity
            category_by_user[category][user] += hours

    return dict(category_hours), dict(category_counts), dict(category_by_user)


def calculate_category_breakdown(
    user_activity_data: Dict[str, Dict[str, Any]],
    start_date: date,
    end_date: date,
    previous_period_data: Optional[Dict[str, Dict[str, Any]]] = None
) -> CategoryBreakdownSummary:
    """Calculate category breakdown from special activities data.

    Args:
        user_activity_data: Dictionary mapping usernames to their activity data
        start_date: Start of the analysis period
        end_date: End of the analysis period
        previous_period_data: Optional previous period data for trend calculation

    Returns:
        CategoryBreakdownSummary with all category statistics

    Example:
        >>> user_data = {
        ...     "UserA": {"count": 3, "hours": 10.5, "categories": {"Meeting": 5.0, "Research": 5.5}},
        ...     "UserB": {"count": 2, "hours": 8.0, "categories": {"Meeting": 8.0}}
        ... }
        >>> summary = calculate_category_breakdown(user_data, date(2026, 1, 1), date(2026, 1, 31))
        >>> summary.total_hours
        18.5
    """
    logger.info(f"Calculating category breakdown for {start_date} to {end_date}")

    # Aggregate current period data
    category_hours, category_counts, category_by_user = aggregate_by_category(user_activity_data)

    # Calculate totals
    total_hours = sum(category_hours.values())
    total_items = sum(category_counts.values())
    category_count = len(category_hours)

    logger.info(f"Found {category_count} categories with {total_hours:.1f} total hours and {total_items} items")

    # Aggregate previous period data if available
    prev_category_hours: Dict[str, float] = {}
    prev_category_counts: Dict[str, int] = {}
    if previous_period_data:
        prev_category_hours, prev_category_counts, _ = aggregate_by_category(previous_period_data)
        logger.info(f"Previous period has {len(prev_category_hours)} categories")

    # Build category breakdown list
    categories: List[CategoryBreakdown] = []

    for category_name, hours in category_hours.items():
        item_count = category_counts.get(category_name, 0)
        user_breakdown = dict(category_by_user.get(category_name, {}))

        # Calculate percentages
        pct_hours = (hours / total_hours * 100) if total_hours > 0 else 0.0
        pct_items = (item_count / total_items * 100) if total_items > 0 else 0.0

        # Calculate trend data
        prev_hours = prev_category_hours.get(category_name)
        prev_count = prev_category_counts.get(category_name)
        hours_pct_change: Optional[float] = None
        count_pct_change: Optional[float] = None
        trend = TrendDirection.NO_DATA

        if prev_hours is not None:
            hours_pct_change = calculate_percent_change(int(hours * 100), int(prev_hours * 100))
            trend = get_trend_direction(hours_pct_change)

        if prev_count is not None:
            count_pct_change = calculate_percent_change(item_count, prev_count)

        breakdown = CategoryBreakdown(
            category_name=category_name,
            total_hours=hours,
            item_count=item_count,
            percentage_of_total_hours=pct_hours,
            percentage_of_total_items=pct_items,
            by_user=user_breakdown,
            previous_period_hours=prev_hours,
            previous_period_count=prev_count,
            hours_percent_change=hours_pct_change,
            count_percent_change=count_pct_change,
            trend=trend,
        )
        categories.append(breakdown)

    summary = CategoryBreakdownSummary(
        start_date=start_date,
        end_date=end_date,
        total_hours=total_hours,
        total_items=total_items,
        category_count=category_count,
        categories=categories,
    )

    return summary


def get_category_breakdown_summary(summary: CategoryBreakdownSummary) -> Dict[str, Any]:
    """Get a formatted summary of category breakdown for display.

    Args:
        summary: CategoryBreakdownSummary object

    Returns:
        Dictionary with formatted summary data

    Example:
        >>> summary = calculate_category_breakdown(data, start, end)
        >>> formatted = get_category_breakdown_summary(summary)
        >>> print(formatted["top_categories"])
    """
    # Get top 5 categories
    top_categories = [
        {
            "name": cat.category_name,
            "hours": round(cat.total_hours, 1),
            "percentage": f"{cat.percentage_of_total_hours:.1f}%",
            "items": cat.item_count,
            "trend": cat.trend.value if cat.trend != TrendDirection.NO_DATA else None,
            "change": format_percent_change(cat.hours_percent_change) if cat.hours_percent_change is not None else None,
        }
        for cat in summary.get_top_categories(5)
    ]

    # Get trending categories
    trending_up = [
        {"name": cat.category_name, "change": format_percent_change(cat.hours_percent_change)}
        for cat in summary.get_trending_up()[:3]
        if cat.hours_percent_change is not None
    ]

    trending_down = [
        {"name": cat.category_name, "change": format_percent_change(cat.hours_percent_change)}
        for cat in summary.get_trending_down()[:3]
        if cat.hours_percent_change is not None
    ]

    return {
        "period": {
            "start": summary.start_date.isoformat(),
            "end": summary.end_date.isoformat(),
        },
        "totals": {
            "hours": round(summary.total_hours, 1),
            "items": summary.total_items,
            "categories": summary.category_count,
        },
        "top_category": summary.top_category,
        "top_categories": top_categories,
        "trending_up": trending_up,
        "trending_down": trending_down,
        "has_trend_data": summary.has_trend_data,
    }


def calculate_category_trends(
    current_summary: CategoryBreakdownSummary,
    previous_summary: Optional[CategoryBreakdownSummary] = None
) -> Dict[str, Dict[str, Any]]:
    """Calculate trends for each category between two periods.

    Args:
        current_summary: CategoryBreakdownSummary for current period
        previous_summary: Optional CategoryBreakdownSummary for previous period

    Returns:
        Dictionary mapping category name to trend details

    Example:
        >>> current = calculate_category_breakdown(current_data, start, end)
        >>> previous = calculate_category_breakdown(prev_data, prev_start, prev_end)
        >>> trends = calculate_category_trends(current, previous)
    """
    trends: Dict[str, Dict[str, Any]] = {}

    # Build previous period lookup
    prev_lookup: Dict[str, CategoryBreakdown] = {}
    if previous_summary:
        prev_lookup = {cat.category_name: cat for cat in previous_summary.categories}

    for cat in current_summary.categories:
        prev_cat = prev_lookup.get(cat.category_name)

        if prev_cat:
            hours_change = cat.total_hours - prev_cat.total_hours
            count_change = cat.item_count - prev_cat.item_count
            hours_pct_change = calculate_percent_change(
                int(cat.total_hours * 100),
                int(prev_cat.total_hours * 100)
            )
            count_pct_change = calculate_percent_change(cat.item_count, prev_cat.item_count)
            trend = get_trend_direction(hours_pct_change)
        else:
            # New category (not in previous period)
            hours_change = cat.total_hours
            count_change = cat.item_count
            hours_pct_change = 100.0 if cat.total_hours > 0 else 0.0
            count_pct_change = 100.0 if cat.item_count > 0 else 0.0
            trend = TrendDirection.UP if cat.total_hours > 0 else TrendDirection.FLAT

        trends[cat.category_name] = {
            "current_hours": cat.total_hours,
            "previous_hours": prev_cat.total_hours if prev_cat else 0.0,
            "hours_change": round(hours_change, 2),
            "hours_percent_change": hours_pct_change,
            "current_count": cat.item_count,
            "previous_count": prev_cat.item_count if prev_cat else 0,
            "count_change": count_change,
            "count_percent_change": count_pct_change,
            "trend": trend.value,
            "trend_indicator": get_trend_indicator(trend),
        }

    # Check for categories that were in previous but not in current (dropped)
    if previous_summary:
        current_names = {cat.category_name for cat in current_summary.categories}
        for prev_cat in previous_summary.categories:
            if prev_cat.category_name not in current_names:
                trends[prev_cat.category_name] = {
                    "current_hours": 0.0,
                    "previous_hours": prev_cat.total_hours,
                    "hours_change": -prev_cat.total_hours,
                    "hours_percent_change": -100.0,
                    "current_count": 0,
                    "previous_count": prev_cat.item_count,
                    "count_change": -prev_cat.item_count,
                    "count_percent_change": -100.0,
                    "trend": TrendDirection.DOWN.value,
                    "trend_indicator": get_trend_indicator(TrendDirection.DOWN),
                }

    return trends


def format_category_table(summary: CategoryBreakdownSummary) -> List[List[str]]:
    """Format category breakdown as table data for display.

    Args:
        summary: CategoryBreakdownSummary object

    Returns:
        List of lists representing table rows with headers

    Example:
        >>> table_data = format_category_table(summary)
        >>> # Returns [["Category", "Hours", "Items", "% Hours", "Trend"], ...]
    """
    headers = ["Category", "Hours", "Items", "% of Total", "Trend"]
    rows: List[List[str]] = [headers]

    for cat in summary.categories:
        trend_str = ""
        if cat.trend != TrendDirection.NO_DATA and cat.hours_percent_change is not None:
            indicator = get_trend_indicator(cat.trend)
            trend_str = f"{indicator} {format_percent_change(cat.hours_percent_change)}"

        rows.append([
            cat.category_name,
            f"{cat.total_hours:.1f}",
            str(cat.item_count),
            f"{cat.percentage_of_total_hours:.1f}%",
            trend_str,
        ])

    # Add totals row
    rows.append([
        "Total",
        f"{summary.total_hours:.1f}",
        str(summary.total_items),
        "100.0%",
        "",
    ])

    return rows


def compare_special_activities_periods(
    current_data: Dict[str, Dict[str, Any]],
    previous_data: Dict[str, Dict[str, Any]],
    current_start: date,
    current_end: date,
    previous_start: date,
    previous_end: date,
) -> SpecialActivitiesPeriodComparison:
    """Compare special activities between current and previous periods.

    This function provides a comprehensive comparison of special activities data
    between two time periods, calculating changes in total hours, category
    distribution, and identifying trends.

    Args:
        current_data: User activity data for the current period
            (format: {username: {count, hours, categories: {cat: hours}}})
        previous_data: User activity data for the previous period
        current_start: Start date of the current period
        current_end: End date of the current period
        previous_start: Start date of the previous period
        previous_end: End date of the previous period

    Returns:
        SpecialActivitiesPeriodComparison with all comparison metrics

    Example:
        >>> current = get_special_activities(start_date, end_date)
        >>> previous = get_special_activities(prev_start, prev_end)
        >>> comparison = compare_special_activities_periods(
        ...     current_data=current[0],
        ...     previous_data=previous[0],
        ...     current_start=start_date,
        ...     current_end=end_date,
        ...     previous_start=prev_start,
        ...     previous_end=prev_end
        ... )
        >>> print(comparison.get_summary_text())
    """
    logger.info(
        f"Comparing special activities: {current_start} to {current_end} vs "
        f"{previous_start} to {previous_end}"
    )

    # Aggregate data by category for both periods
    current_hours, current_counts, current_by_user = aggregate_by_category(current_data)
    previous_hours, previous_counts, previous_by_user = aggregate_by_category(previous_data)

    # Calculate totals
    current_total_hours = sum(current_hours.values())
    previous_total_hours = sum(previous_hours.values())
    current_total_items = sum(current_counts.values())
    previous_total_items = sum(previous_counts.values())

    # Get category sets
    current_categories = set(current_hours.keys())
    previous_categories = set(previous_hours.keys())
    all_categories = current_categories | previous_categories

    # Identify new and dropped categories
    new_categories = list(current_categories - previous_categories)
    dropped_categories = list(previous_categories - current_categories)

    # Build category comparisons
    category_comparisons: Dict[str, Dict[str, Any]] = {}

    for category in all_categories:
        curr_hours = current_hours.get(category, 0.0)
        prev_hours = previous_hours.get(category, 0.0)
        curr_count = current_counts.get(category, 0)
        prev_count = previous_counts.get(category, 0)

        hours_change = curr_hours - prev_hours
        hours_pct_change = calculate_percent_change(
            int(curr_hours * 100),
            int(prev_hours * 100)
        )
        count_change = curr_count - prev_count
        count_pct_change = calculate_percent_change(curr_count, prev_count)

        trend = get_trend_direction(hours_pct_change)

        # Calculate distribution percentage changes
        curr_dist_pct = (curr_hours / current_total_hours * 100) if current_total_hours > 0 else 0.0
        prev_dist_pct = (prev_hours / previous_total_hours * 100) if previous_total_hours > 0 else 0.0
        dist_pct_change = curr_dist_pct - prev_dist_pct

        category_comparisons[category] = {
            "current_hours": round(curr_hours, 2),
            "previous_hours": round(prev_hours, 2),
            "hours_change": round(hours_change, 2),
            "hours_percent_change": round(hours_pct_change, 2),
            "current_count": curr_count,
            "previous_count": prev_count,
            "count_change": count_change,
            "count_percent_change": round(count_pct_change, 2),
            "current_distribution_pct": round(curr_dist_pct, 2),
            "previous_distribution_pct": round(prev_dist_pct, 2),
            "distribution_change": round(dist_pct_change, 2),
            "trend": trend.value,
            "trend_indicator": get_trend_indicator(trend),
            "is_new": category in new_categories,
            "is_dropped": category in dropped_categories,
        }

    comparison = SpecialActivitiesPeriodComparison(
        current_period_start=current_start,
        current_period_end=current_end,
        previous_period_start=previous_start,
        previous_period_end=previous_end,
        current_total_hours=current_total_hours,
        previous_total_hours=previous_total_hours,
        current_total_items=current_total_items,
        previous_total_items=previous_total_items,
        current_category_count=len(current_categories),
        previous_category_count=len(previous_categories),
        category_comparisons=category_comparisons,
        new_categories=new_categories,
        dropped_categories=dropped_categories,
    )

    logger.info(
        f"Comparison complete: {comparison.hours_change:+.1f}h ({format_percent_change(comparison.hours_percent_change)}), "
        f"{len(new_categories)} new categories, {len(dropped_categories)} dropped"
    )

    return comparison


def format_period_comparison_table(
    comparison: SpecialActivitiesPeriodComparison,
) -> List[List[str]]:
    """Format period comparison as table data for display.

    Args:
        comparison: SpecialActivitiesPeriodComparison object

    Returns:
        List of lists representing table rows with headers
    """
    headers = ["Category", "Current", "Previous", "Change", "% Change", "Distribution Δ"]
    rows: List[List[str]] = [headers]

    # Sort categories by current hours (descending)
    sorted_categories = sorted(
        comparison.category_comparisons.items(),
        key=lambda x: x[1]["current_hours"],
        reverse=True
    )

    for category, data in sorted_categories:
        indicator = data["trend_indicator"]
        change_str = f"{indicator} {data['hours_change']:+.1f}h"
        pct_change_str = format_percent_change(data["hours_percent_change"])
        dist_change_str = f"{data['distribution_change']:+.1f}%"

        # Add markers for new/dropped categories
        category_name = category
        if data["is_new"]:
            category_name += " (NEW)"
        elif data["is_dropped"]:
            category_name += " (DROPPED)"

        rows.append([
            category_name,
            f"{data['current_hours']:.1f}h",
            f"{data['previous_hours']:.1f}h",
            change_str,
            pct_change_str,
            dist_change_str,
        ])

    # Add totals row
    hours_indicator = get_trend_indicator(comparison.hours_trend)
    rows.append([
        "TOTAL",
        f"{comparison.current_total_hours:.1f}h",
        f"{comparison.previous_total_hours:.1f}h",
        f"{hours_indicator} {comparison.hours_change:+.1f}h",
        format_percent_change(comparison.hours_percent_change),
        "",
    ])

    return rows


def get_period_comparison_summary(
    comparison: SpecialActivitiesPeriodComparison,
) -> Dict[str, Any]:
    """Get a formatted summary of the period comparison for display.

    Args:
        comparison: SpecialActivitiesPeriodComparison object

    Returns:
        Dictionary with formatted comparison summary data
    """
    return {
        "period": {
            "current": {
                "start": comparison.current_period_start.isoformat(),
                "end": comparison.current_period_end.isoformat(),
                "label": f"{comparison.current_period_start.strftime('%b %d')} - {comparison.current_period_end.strftime('%b %d')}",
            },
            "previous": {
                "start": comparison.previous_period_start.isoformat(),
                "end": comparison.previous_period_end.isoformat(),
                "label": f"{comparison.previous_period_start.strftime('%b %d')} - {comparison.previous_period_end.strftime('%b %d')}",
            },
        },
        "totals": {
            "current_hours": round(comparison.current_total_hours, 1),
            "previous_hours": round(comparison.previous_total_hours, 1),
            "hours_change": round(comparison.hours_change, 1),
            "hours_percent_change": format_percent_change(comparison.hours_percent_change),
            "hours_trend": comparison.hours_trend.value,
            "hours_trend_indicator": get_trend_indicator(comparison.hours_trend),
            "current_items": comparison.current_total_items,
            "previous_items": comparison.previous_total_items,
            "items_change": comparison.items_change,
            "items_percent_change": format_percent_change(comparison.items_percent_change),
            "items_trend": comparison.items_trend.value,
        },
        "category_stats": {
            "current_count": comparison.current_category_count,
            "previous_count": comparison.previous_category_count,
            "new_categories": comparison.new_categories,
            "dropped_categories": comparison.dropped_categories,
        },
        "top_movers": {
            "increasing": [
                {
                    "category": cat["category"],
                    "change": f"+{cat['hours_change']:.1f}h",
                    "percent_change": format_percent_change(cat["hours_percent_change"]),
                }
                for cat in comparison.top_increasing_categories
            ],
            "decreasing": [
                {
                    "category": cat["category"],
                    "change": f"{cat['hours_change']:.1f}h",
                    "percent_change": format_percent_change(cat["hours_percent_change"]),
                }
                for cat in comparison.top_decreasing_categories
            ],
        },
        "summary_text": comparison.get_summary_text(),
    }


if __name__ == "__main__":
    # Demo usage with sample data
    print("Special Activities Category Breakdown - Demo")
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

    # Calculate breakdown for current period
    summary = calculate_category_breakdown(
        user_activity_data=sample_data,
        start_date=date(2026, 1, 1),
        end_date=date(2026, 1, 31),
        previous_period_data=previous_data,
    )

    print(f"\nPeriod: {summary.start_date} to {summary.end_date}")
    print(f"Total Hours: {summary.total_hours:.1f}")
    print(f"Total Items: {summary.total_items}")
    print(f"Categories: {summary.category_count}")
    print(f"Top Category: {summary.top_category}")

    print("\n" + "-" * 60)
    print("Category Breakdown:")
    print("-" * 60)

    for cat in summary.categories:
        trend_info = ""
        if cat.hours_percent_change is not None:
            indicator = get_trend_indicator(cat.trend)
            trend_info = f" [{indicator} {format_percent_change(cat.hours_percent_change)}]"

        print(f"  {cat.category_name}: {cat.total_hours:.1f}h ({cat.percentage_of_total_hours:.1f}%)"
              f" - {cat.item_count} items{trend_info}")

    # Get formatted summary
    print("\n" + "-" * 60)
    print("Formatted Summary:")
    print("-" * 60)
    formatted = get_category_breakdown_summary(summary)
    print(f"Top Categories: {formatted['top_categories']}")
    print(f"Trending Up: {formatted['trending_up']}")
    print(f"Trending Down: {formatted['trending_down']}")

    # Calculate trends
    print("\n" + "-" * 60)
    print("Category Trends:")
    print("-" * 60)

    # Create previous summary for trend calculation
    prev_summary = calculate_category_breakdown(
        user_activity_data=previous_data,
        start_date=date(2025, 12, 1),
        end_date=date(2025, 12, 31),
    )

    trends = calculate_category_trends(summary, prev_summary)
    for cat_name, trend_data in trends.items():
        print(f"  {cat_name}: {trend_data['current_hours']:.1f}h vs {trend_data['previous_hours']:.1f}h "
              f"({trend_data['trend_indicator']} {trend_data['hours_percent_change']:.1f}%)")

    # Display as table
    print("\n" + "-" * 60)
    print("Table Format:")
    print("-" * 60)
    table = format_category_table(summary)
    for row in table:
        print("  " + " | ".join(f"{cell:15}" for cell in row))
