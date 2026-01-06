"""
User Performance Trends Calculator Module

Tracks individual user performance trends over multiple weeks or months.
Calculates moving averages and identifies improving or declining productivity.

This module provides:
- Weekly and monthly performance aggregation per user
- Moving average calculations (simple, weighted, exponential)
- Trend analysis with improvement/decline detection
- Multi-period comparison and momentum tracking
- Performance visualization data for reports

Usage:
    from user_performance_trends_calculator import (
        UserPerformanceTrend,
        PeriodPerformance,
        TrendStatus,
        PerformanceMomentum,
        calculate_user_performance_trend,
        calculate_all_users_performance_trends,
        calculate_performance_trends_for_range,
        get_user_trend_comparison,
        get_improving_users,
        get_declining_users,
        get_performance_trends_summary,
        get_performance_trends_visualization_data,
        format_performance_trends_report,
    )
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Any, Union, Tuple
from enum import Enum
from collections import defaultdict

# Import from trend_line_calculator for trend math
from trend_line_calculator import (
    TimeSeriesDataPoint,
    TrendLineResult,
    TrendDirection,
    MovingAverageResult,
    MovingAverageType,
    calculate_linear_regression,
    calculate_simple_moving_average,
    calculate_exponential_moving_average,
)

# Set up logging
logger = logging.getLogger(__name__)


class TrendStatus(Enum):
    """Status of user performance trend."""
    SIGNIFICANTLY_IMPROVING = "significantly_improving"  # Strong upward trend
    IMPROVING = "improving"  # Moderate upward trend
    STABLE = "stable"  # No significant change
    DECLINING = "declining"  # Moderate downward trend
    SIGNIFICANTLY_DECLINING = "significantly_declining"  # Strong downward trend
    INSUFFICIENT_DATA = "insufficient_data"  # Not enough periods for analysis


class PerformanceMomentum(Enum):
    """Momentum of performance change (acceleration/deceleration)."""
    ACCELERATING = "accelerating"  # Improvement rate is increasing
    STEADY = "steady"  # Change rate is consistent
    DECELERATING = "decelerating"  # Improvement rate is decreasing
    INSUFFICIENT_DATA = "insufficient_data"


@dataclass
class PeriodPerformance:
    """Performance metrics for a specific period (week or month).

    Attributes:
        period_number: Sequential period number (1-based)
        period_label: Human-readable label (e.g., "Week 1 (01/06)" or "January 2026")
        period_start: Start date of the period
        period_end: End date of the period
        total_items: Total items processed in this period
        items_per_day: Average items per day for this period
        active_days: Number of days with activity
        percent_change_from_previous: Percentage change from previous period
    """
    period_number: int
    period_label: str
    period_start: date
    period_end: date
    total_items: int = 0
    items_per_day: float = 0.0
    active_days: int = 0
    percent_change_from_previous: Optional[float] = None

    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        days_in_period = (self.period_end - self.period_start).days + 1
        if self.total_items > 0 and days_in_period > 0:
            self.items_per_day = self.total_items / days_in_period

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "period_number": self.period_number,
            "period_label": self.period_label,
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "total_items": self.total_items,
            "items_per_day": round(self.items_per_day, 2),
            "active_days": self.active_days,
            "percent_change_from_previous": round(self.percent_change_from_previous, 2) if self.percent_change_from_previous is not None else None,
        }


@dataclass
class MovingAverageMetrics:
    """Moving average metrics for trend smoothing.

    Attributes:
        ma_type: Type of moving average
        window_size: Window size for the MA calculation
        current_value: Most recent MA value
        previous_value: Previous MA value (for trend comparison)
        values: List of all MA values aligned with periods
        trend_direction: Direction based on MA comparison
    """
    ma_type: MovingAverageType
    window_size: int
    current_value: Optional[float] = None
    previous_value: Optional[float] = None
    values: List[Optional[float]] = field(default_factory=list)
    trend_direction: TrendDirection = TrendDirection.INSUFFICIENT_DATA

    def __post_init__(self):
        """Calculate trend direction from MA values."""
        if self.current_value is not None and self.previous_value is not None:
            if self.previous_value > 0:
                change_ratio = (self.current_value - self.previous_value) / self.previous_value
                if change_ratio >= 0.10:
                    self.trend_direction = TrendDirection.STRONG_UP
                elif change_ratio >= 0.03:
                    self.trend_direction = TrendDirection.UP
                elif change_ratio <= -0.10:
                    self.trend_direction = TrendDirection.STRONG_DOWN
                elif change_ratio <= -0.03:
                    self.trend_direction = TrendDirection.DOWN
                else:
                    self.trend_direction = TrendDirection.FLAT

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "ma_type": self.ma_type.value,
            "window_size": self.window_size,
            "current_value": round(self.current_value, 2) if self.current_value is not None else None,
            "previous_value": round(self.previous_value, 2) if self.previous_value is not None else None,
            "values": [round(v, 2) if v is not None else None for v in self.values],
            "trend_direction": self.trend_direction.value,
        }


@dataclass
class UserPerformanceTrend:
    """Complete performance trend analysis for a user.

    Attributes:
        user: User identifier (initials like "JHU", "DM")
        period_type: Type of periods analyzed ("weekly" or "monthly")
        periods: List of PeriodPerformance objects for each period
        trend_status: Overall trend classification
        momentum: Trend momentum (acceleration/deceleration)
        overall_items: Total items across all periods
        overall_items_per_day: Average items per day across all periods
        baseline_items_per_day: Average items/day in first half of periods (baseline)
        recent_items_per_day: Average items/day in second half of periods (recent)
        improvement_percentage: Percentage improvement from baseline to recent
        trend_line: Linear regression result for items per day
        simple_ma: Simple moving average results
        exponential_ma: Exponential moving average results
        date_range_start: Start date of analysis
        date_range_end: End date of analysis
        periods_analyzed: Number of periods analyzed
        confidence_score: Confidence in the trend assessment (0-1)
    """
    user: str
    period_type: str = "weekly"
    periods: List[PeriodPerformance] = field(default_factory=list)
    trend_status: TrendStatus = TrendStatus.INSUFFICIENT_DATA
    momentum: PerformanceMomentum = PerformanceMomentum.INSUFFICIENT_DATA
    overall_items: int = 0
    overall_items_per_day: float = 0.0
    baseline_items_per_day: float = 0.0
    recent_items_per_day: float = 0.0
    improvement_percentage: float = 0.0
    trend_line: Optional[TrendLineResult] = None
    simple_ma: Optional[MovingAverageMetrics] = None
    exponential_ma: Optional[MovingAverageMetrics] = None
    date_range_start: Optional[date] = None
    date_range_end: Optional[date] = None
    periods_analyzed: int = 0
    confidence_score: float = 0.0

    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        self.periods_analyzed = len(self.periods)
        if self.periods:
            self.date_range_start = min(p.period_start for p in self.periods)
            self.date_range_end = max(p.period_end for p in self.periods)
            self.overall_items = sum(p.total_items for p in self.periods)

            # Calculate overall items per day
            total_days = (self.date_range_end - self.date_range_start).days + 1
            if total_days > 0:
                self.overall_items_per_day = self.overall_items / total_days

    @property
    def is_improving(self) -> bool:
        """Check if the user's performance is improving."""
        return self.trend_status in (TrendStatus.IMPROVING, TrendStatus.SIGNIFICANTLY_IMPROVING)

    @property
    def is_declining(self) -> bool:
        """Check if the user's performance is declining."""
        return self.trend_status in (TrendStatus.DECLINING, TrendStatus.SIGNIFICANTLY_DECLINING)

    @property
    def is_stable(self) -> bool:
        """Check if the user's performance is stable."""
        return self.trend_status == TrendStatus.STABLE

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "user": self.user,
            "period_type": self.period_type,
            "periods": [p.to_dict() for p in self.periods],
            "trend_status": self.trend_status.value,
            "momentum": self.momentum.value,
            "overall_items": self.overall_items,
            "overall_items_per_day": round(self.overall_items_per_day, 2),
            "baseline_items_per_day": round(self.baseline_items_per_day, 2),
            "recent_items_per_day": round(self.recent_items_per_day, 2),
            "improvement_percentage": round(self.improvement_percentage, 2),
            "trend_line": self.trend_line.to_dict() if self.trend_line else None,
            "simple_ma": self.simple_ma.to_dict() if self.simple_ma else None,
            "exponential_ma": self.exponential_ma.to_dict() if self.exponential_ma else None,
            "date_range_start": self.date_range_start.isoformat() if self.date_range_start else None,
            "date_range_end": self.date_range_end.isoformat() if self.date_range_end else None,
            "periods_analyzed": self.periods_analyzed,
            "confidence_score": round(self.confidence_score, 2),
            "is_improving": self.is_improving,
            "is_declining": self.is_declining,
            "is_stable": self.is_stable,
        }


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


def _generate_month_periods(
    start_date: date,
    end_date: date
) -> List[Tuple[date, date, str, int]]:
    """Generate month periods between start and end dates.

    Args:
        start_date: Start of the date range
        end_date: End of the date range

    Returns:
        List of tuples (month_start, month_end, label, month_number)
    """
    periods = []
    current = start_date.replace(day=1)
    month_num = 1

    while current <= end_date:
        # Calculate end of month
        if current.month == 12:
            next_month = current.replace(year=current.year + 1, month=1, day=1)
        else:
            next_month = current.replace(month=current.month + 1, day=1)
        month_end = next_month - timedelta(days=1)

        # Clamp to actual range
        actual_start = max(current, start_date)
        actual_end = min(month_end, end_date)

        label = f"{current.strftime('%B %Y')}"
        periods.append((actual_start, actual_end, label, month_num))

        current = next_month
        month_num += 1

    return periods


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


def _calculate_period_performances(
    changes: List[Dict[str, Any]],
    user: str,
    period_type: str = "weekly"
) -> List[PeriodPerformance]:
    """Calculate performance metrics for each period.

    Args:
        changes: List of change records (already filtered by user)
        user: User identifier
        period_type: "weekly" or "monthly"

    Returns:
        List of PeriodPerformance objects
    """
    if not changes:
        return []

    # Get date range
    start_date, end_date = _get_date_range(changes)
    if not start_date or not end_date:
        return []

    # Generate periods
    if period_type == "monthly":
        periods = _generate_month_periods(start_date, end_date)
    else:
        periods = _generate_week_periods(start_date, end_date)

    # Count items per period
    period_items: Dict[int, int] = defaultdict(int)
    period_active_days: Dict[int, set] = defaultdict(set)

    for change in changes:
        ts = change.get('ParsedTimestamp') or change.get('ParsedDate')
        if not ts:
            continue
        change_date = ts.date() if hasattr(ts, 'date') else ts

        # Find which period this change belongs to
        for period_start, period_end, label, period_num in periods:
            if period_start <= change_date <= period_end:
                period_items[period_num] += 1
                period_active_days[period_num].add(change_date)
                break

    # Build PeriodPerformance objects
    result = []
    previous_items_per_day = None

    for period_start, period_end, label, period_num in periods:
        total_items = period_items.get(period_num, 0)
        days_in_period = (period_end - period_start).days + 1
        items_per_day = total_items / max(days_in_period, 1)
        active_days = len(period_active_days.get(period_num, set()))

        # Calculate percent change from previous period
        percent_change = None
        if previous_items_per_day is not None and previous_items_per_day > 0:
            percent_change = ((items_per_day - previous_items_per_day) / previous_items_per_day) * 100
        elif previous_items_per_day == 0 and items_per_day > 0:
            percent_change = 100.0
        elif previous_items_per_day == 0 and items_per_day == 0:
            percent_change = 0.0

        performance = PeriodPerformance(
            period_number=period_num,
            period_label=label,
            period_start=period_start,
            period_end=period_end,
            total_items=total_items,
            items_per_day=items_per_day,
            active_days=active_days,
            percent_change_from_previous=percent_change,
        )
        result.append(performance)
        previous_items_per_day = items_per_day

    return result


def _calculate_trend_status(
    trend_line: TrendLineResult,
    improvement_percentage: float,
    periods_count: int,
    ma_trend: Optional[TrendDirection] = None
) -> TrendStatus:
    """Determine the overall trend status.

    Args:
        trend_line: Linear regression result
        improvement_percentage: Percentage improvement from baseline to recent
        periods_count: Number of periods analyzed
        ma_trend: Moving average trend direction for additional confirmation

    Returns:
        TrendStatus classification
    """
    if periods_count < 3:
        return TrendStatus.INSUFFICIENT_DATA

    # Use combination of regression direction and improvement percentage
    direction = trend_line.direction

    if direction == TrendDirection.STRONG_UP or improvement_percentage >= 20:
        return TrendStatus.SIGNIFICANTLY_IMPROVING
    elif direction == TrendDirection.UP or improvement_percentage >= 5:
        return TrendStatus.IMPROVING
    elif direction == TrendDirection.STRONG_DOWN or improvement_percentage <= -20:
        return TrendStatus.SIGNIFICANTLY_DECLINING
    elif direction == TrendDirection.DOWN or improvement_percentage <= -5:
        return TrendStatus.DECLINING
    else:
        return TrendStatus.STABLE


def _calculate_momentum(
    periods: List[PeriodPerformance]
) -> PerformanceMomentum:
    """Calculate the momentum of performance change.

    Compares the rate of change in recent periods vs earlier periods.

    Args:
        periods: List of PeriodPerformance objects

    Returns:
        PerformanceMomentum classification
    """
    if len(periods) < 4:
        return PerformanceMomentum.INSUFFICIENT_DATA

    # Get percent changes (skip first period which has no previous)
    changes = [p.percent_change_from_previous for p in periods[1:]
               if p.percent_change_from_previous is not None]

    if len(changes) < 3:
        return PerformanceMomentum.INSUFFICIENT_DATA

    # Compare first half vs second half of changes
    mid = len(changes) // 2
    first_half_avg = sum(changes[:mid]) / mid if mid > 0 else 0
    second_half_avg = sum(changes[mid:]) / (len(changes) - mid) if len(changes) > mid else 0

    # Determine momentum
    diff = second_half_avg - first_half_avg
    if diff > 5:  # 5 percentage points faster improvement
        return PerformanceMomentum.ACCELERATING
    elif diff < -5:  # 5 percentage points slower improvement
        return PerformanceMomentum.DECELERATING
    else:
        return PerformanceMomentum.STEADY


def _calculate_moving_averages(
    periods: List[PeriodPerformance],
    window_size: int = 3
) -> Tuple[Optional[MovingAverageMetrics], Optional[MovingAverageMetrics]]:
    """Calculate simple and exponential moving averages.

    Args:
        periods: List of PeriodPerformance objects
        window_size: Window size for MA calculations

    Returns:
        Tuple of (simple_ma_metrics, exponential_ma_metrics)
    """
    if len(periods) < window_size:
        return None, None

    # Create data points from periods
    data_points = [
        TimeSeriesDataPoint(
            date=p.period_start,
            value=p.items_per_day,
            label=p.period_label
        )
        for p in periods
    ]

    # Calculate simple moving average
    sma_result = calculate_simple_moving_average(data_points, window_size)
    sma_values = sma_result.ma_values

    # Get current and previous SMA values
    sma_current = None
    sma_previous = None
    for i in range(len(sma_values) - 1, -1, -1):
        if sma_current is None and sma_values[i] is not None:
            sma_current = sma_values[i]
        elif sma_current is not None and sma_previous is None and sma_values[i] is not None:
            sma_previous = sma_values[i]
            break

    simple_ma = MovingAverageMetrics(
        ma_type=MovingAverageType.SIMPLE,
        window_size=window_size,
        current_value=sma_current,
        previous_value=sma_previous,
        values=sma_values,
    )

    # Calculate exponential moving average
    ema_result = calculate_exponential_moving_average(data_points, window_size)
    ema_values = ema_result.ma_values

    # Get current and previous EMA values
    ema_current = None
    ema_previous = None
    for i in range(len(ema_values) - 1, -1, -1):
        if ema_current is None and ema_values[i] is not None:
            ema_current = ema_values[i]
        elif ema_current is not None and ema_previous is None and ema_values[i] is not None:
            ema_previous = ema_values[i]
            break

    exponential_ma = MovingAverageMetrics(
        ma_type=MovingAverageType.EXPONENTIAL,
        window_size=window_size,
        current_value=ema_current,
        previous_value=ema_previous,
        values=ema_values,
    )

    return simple_ma, exponential_ma


def calculate_user_performance_trend(
    changes: List[Dict[str, Any]],
    user: str,
    period_type: str = "weekly",
    ma_window_size: int = 3
) -> UserPerformanceTrend:
    """Calculate performance trend for a specific user.

    Args:
        changes: List of change records from historical data
        user: User identifier to calculate trends for
        period_type: "weekly" or "monthly" period aggregation
        ma_window_size: Window size for moving average calculations

    Returns:
        UserPerformanceTrend object with comprehensive trend analysis

    Example:
        >>> from historical_data_loader import load_change_history
        >>> changes = load_change_history()
        >>> trend = calculate_user_performance_trend(changes, "JHU")
        >>> print(f"Trend: {trend.trend_status.value}")
        >>> print(f"Improvement: {trend.improvement_percentage:.1f}%")
    """
    if not changes:
        return UserPerformanceTrend(user=user, period_type=period_type)

    # Filter changes for this user
    user_changes = [c for c in changes if c.get("User") == user]

    if not user_changes:
        return UserPerformanceTrend(user=user, period_type=period_type)

    # Calculate period performances
    periods = _calculate_period_performances(user_changes, user, period_type)

    if len(periods) < 2:
        return UserPerformanceTrend(
            user=user,
            period_type=period_type,
            periods=periods,
            trend_status=TrendStatus.INSUFFICIENT_DATA,
        )

    # Calculate baseline vs recent performance
    mid_point = len(periods) // 2
    baseline_periods = periods[:mid_point] if mid_point > 0 else periods[:1]
    recent_periods = periods[mid_point:] if mid_point < len(periods) else periods[-1:]

    baseline_items_per_day = sum(p.items_per_day for p in baseline_periods) / len(baseline_periods) if baseline_periods else 0
    recent_items_per_day = sum(p.items_per_day for p in recent_periods) / len(recent_periods) if recent_periods else 0

    # Calculate improvement percentage
    if baseline_items_per_day > 0:
        improvement_percentage = ((recent_items_per_day - baseline_items_per_day) / baseline_items_per_day) * 100
    elif baseline_items_per_day == 0 and recent_items_per_day > 0:
        improvement_percentage = 100.0
    else:
        improvement_percentage = 0.0

    # Create data points for trend analysis
    data_points = [
        TimeSeriesDataPoint(
            date=p.period_start,
            value=p.items_per_day,
            label=p.period_label
        )
        for p in periods
    ]

    # Calculate linear regression trend
    trend_line = calculate_linear_regression(data_points)

    # Calculate moving averages
    simple_ma, exponential_ma = _calculate_moving_averages(periods, ma_window_size)

    # Determine trend status
    trend_status = _calculate_trend_status(
        trend_line,
        improvement_percentage,
        len(periods),
        exponential_ma.trend_direction if exponential_ma else None
    )

    # Calculate momentum
    momentum = _calculate_momentum(periods)

    # Calculate confidence score (based on R-squared and period count)
    base_confidence = trend_line.r_squared
    period_factor = min(len(periods) / 8, 1.0)  # Full confidence at 8+ periods
    confidence_score = base_confidence * period_factor

    return UserPerformanceTrend(
        user=user,
        period_type=period_type,
        periods=periods,
        trend_status=trend_status,
        momentum=momentum,
        baseline_items_per_day=baseline_items_per_day,
        recent_items_per_day=recent_items_per_day,
        improvement_percentage=improvement_percentage,
        trend_line=trend_line,
        simple_ma=simple_ma,
        exponential_ma=exponential_ma,
        confidence_score=confidence_score,
    )


def calculate_all_users_performance_trends(
    changes: List[Dict[str, Any]],
    period_type: str = "weekly",
    ma_window_size: int = 3
) -> Dict[str, UserPerformanceTrend]:
    """Calculate performance trends for all users in the data.

    Args:
        changes: List of change records from historical data
        period_type: "weekly" or "monthly" period aggregation
        ma_window_size: Window size for moving average calculations

    Returns:
        Dictionary mapping user identifiers to UserPerformanceTrend objects

    Example:
        >>> from historical_data_loader import load_change_history
        >>> changes = load_change_history()
        >>> all_trends = calculate_all_users_performance_trends(changes)
        >>> for user, trend in all_trends.items():
        ...     print(f"{user}: {trend.trend_status.value}")
    """
    if not changes:
        return {}

    # Get all unique users
    users = set(c.get("User") for c in changes if c.get("User"))

    result = {}
    for user in users:
        result[user] = calculate_user_performance_trend(
            changes, user, period_type, ma_window_size
        )

    return result


def calculate_performance_trends_for_range(
    date_range_filter: Any,  # DateRangeFilter type
    user: Optional[str] = None,
    period_type: str = "weekly",
    changes: Optional[List[Dict[str, Any]]] = None
) -> Union[UserPerformanceTrend, Dict[str, UserPerformanceTrend]]:
    """Calculate performance trends for a specific date range.

    Args:
        date_range_filter: DateRangeFilter object specifying the date range
        user: Optional specific user to analyze. If None, analyzes all users
        period_type: "weekly" or "monthly" period aggregation
        changes: Optional pre-loaded changes (will load if not provided)

    Returns:
        If user specified: UserPerformanceTrend for that user
        If user is None: Dict mapping users to their trends

    Example:
        >>> from date_range_filter import get_preset_range, DateRangePreset
        >>> last_month = get_preset_range(DateRangePreset.LAST_30_DAYS)
        >>> trends = calculate_performance_trends_for_range(last_month)
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
        return calculate_user_performance_trend(changes, user, period_type)
    else:
        return calculate_all_users_performance_trends(changes, period_type)


def get_user_trend_comparison(
    changes: List[Dict[str, Any]],
    period_type: str = "weekly"
) -> List[Dict[str, Any]]:
    """Get a comparison of all users' trends sorted by improvement.

    Args:
        changes: List of change records
        period_type: "weekly" or "monthly" period aggregation

    Returns:
        List of dictionaries with user trend info, sorted by improvement percentage

    Example:
        >>> comparison = get_user_trend_comparison(changes)
        >>> for item in comparison:
        ...     print(f"{item['user']}: {item['improvement_percentage']:.1f}%")
    """
    all_trends = calculate_all_users_performance_trends(changes, period_type)

    comparison = []
    for user, trend in all_trends.items():
        comparison.append({
            "user": user,
            "trend_status": trend.trend_status.value,
            "momentum": trend.momentum.value,
            "improvement_percentage": trend.improvement_percentage,
            "baseline_items_per_day": trend.baseline_items_per_day,
            "recent_items_per_day": trend.recent_items_per_day,
            "overall_items_per_day": trend.overall_items_per_day,
            "periods_analyzed": trend.periods_analyzed,
            "confidence_score": trend.confidence_score,
            "is_improving": trend.is_improving,
            "is_declining": trend.is_declining,
        })

    # Sort by improvement percentage (descending)
    comparison.sort(key=lambda x: x["improvement_percentage"], reverse=True)

    return comparison


def get_improving_users(
    changes: List[Dict[str, Any]],
    period_type: str = "weekly",
    min_improvement: float = 5.0
) -> List[UserPerformanceTrend]:
    """Get list of users with improving performance trends.

    Args:
        changes: List of change records
        period_type: "weekly" or "monthly" period aggregation
        min_improvement: Minimum improvement percentage to qualify

    Returns:
        List of UserPerformanceTrend objects for improving users

    Example:
        >>> improving = get_improving_users(changes, min_improvement=10)
        >>> print(f"{len(improving)} users are improving by >10%")
    """
    all_trends = calculate_all_users_performance_trends(changes, period_type)

    improving = [
        trend for trend in all_trends.values()
        if trend.is_improving and trend.improvement_percentage >= min_improvement
    ]

    # Sort by improvement percentage (descending)
    improving.sort(key=lambda x: x.improvement_percentage, reverse=True)

    return improving


def get_declining_users(
    changes: List[Dict[str, Any]],
    period_type: str = "weekly",
    min_decline: float = 5.0
) -> List[UserPerformanceTrend]:
    """Get list of users with declining performance trends.

    Args:
        changes: List of change records
        period_type: "weekly" or "monthly" period aggregation
        min_decline: Minimum decline percentage (as positive number) to qualify

    Returns:
        List of UserPerformanceTrend objects for declining users

    Example:
        >>> declining = get_declining_users(changes, min_decline=10)
        >>> print(f"{len(declining)} users are declining by >10%")
    """
    all_trends = calculate_all_users_performance_trends(changes, period_type)

    declining = [
        trend for trend in all_trends.values()
        if trend.is_declining and trend.improvement_percentage <= -min_decline
    ]

    # Sort by improvement percentage (ascending - most decline first)
    declining.sort(key=lambda x: x.improvement_percentage)

    return declining


def get_performance_trends_summary(
    changes: List[Dict[str, Any]],
    period_type: str = "weekly"
) -> Dict[str, Any]:
    """Get a comprehensive summary of all users' performance trends.

    Args:
        changes: List of change records
        period_type: "weekly" or "monthly" period aggregation

    Returns:
        Dictionary with comprehensive trend summary

    Example:
        >>> summary = get_performance_trends_summary(changes)
        >>> print(f"Improving: {summary['improving_count']}")
        >>> print(f"Declining: {summary['declining_count']}")
    """
    all_trends = calculate_all_users_performance_trends(changes, period_type)

    if not all_trends:
        return {
            "total_users": 0,
            "improving_count": 0,
            "declining_count": 0,
            "stable_count": 0,
            "insufficient_data_count": 0,
            "users": {},
            "comparison": [],
            "top_improvers": [],
            "biggest_decliners": [],
            "average_improvement": 0,
        }

    # Count by status
    improving_count = sum(1 for t in all_trends.values() if t.is_improving)
    declining_count = sum(1 for t in all_trends.values() if t.is_declining)
    stable_count = sum(1 for t in all_trends.values() if t.is_stable)
    insufficient_count = sum(1 for t in all_trends.values()
                            if t.trend_status == TrendStatus.INSUFFICIENT_DATA)

    # Calculate average improvement
    valid_trends = [t for t in all_trends.values()
                   if t.trend_status != TrendStatus.INSUFFICIENT_DATA]
    avg_improvement = sum(t.improvement_percentage for t in valid_trends) / len(valid_trends) if valid_trends else 0

    # Get comparison
    comparison = get_user_trend_comparison(changes, period_type)

    # Get top improvers and biggest decliners
    top_improvers = [
        {"user": t.user, "improvement": t.improvement_percentage}
        for t in sorted(valid_trends, key=lambda x: x.improvement_percentage, reverse=True)[:5]
        if t.improvement_percentage > 0
    ]

    biggest_decliners = [
        {"user": t.user, "decline": abs(t.improvement_percentage)}
        for t in sorted(valid_trends, key=lambda x: x.improvement_percentage)[:5]
        if t.improvement_percentage < 0
    ]

    # Get date range
    start_date, end_date = _get_date_range(changes)

    return {
        "total_users": len(all_trends),
        "improving_count": improving_count,
        "declining_count": declining_count,
        "stable_count": stable_count,
        "insufficient_data_count": insufficient_count,
        "average_improvement": round(avg_improvement, 2),
        "period_type": period_type,
        "date_range_start": start_date.isoformat() if start_date else None,
        "date_range_end": end_date.isoformat() if end_date else None,
        "users": {user: trend.to_dict() for user, trend in all_trends.items()},
        "comparison": comparison,
        "top_improvers": top_improvers,
        "biggest_decliners": biggest_decliners,
    }


def get_performance_trends_visualization_data(
    changes: List[Dict[str, Any]],
    period_type: str = "weekly"
) -> Dict[str, Any]:
    """Get data formatted for performance trends visualization.

    This function prepares data in a format suitable for rendering
    charts in PDF reports or other visualizations.

    Args:
        changes: List of change records
        period_type: "weekly" or "monthly" period aggregation

    Returns:
        Dictionary with visualization-ready data

    Example:
        >>> viz_data = get_performance_trends_visualization_data(changes)
        >>> # Use viz_data to render trend charts
    """
    all_trends = calculate_all_users_performance_trends(changes, period_type)

    # Prepare trend comparison bar chart data
    comparison_chart_data = []
    for user, trend in sorted(all_trends.items(), key=lambda x: x[1].improvement_percentage, reverse=True):
        # Color based on trend status
        color_map = {
            TrendStatus.SIGNIFICANTLY_IMPROVING: "#1B5E20",  # Dark Green
            TrendStatus.IMPROVING: "#388E3C",  # Green
            TrendStatus.STABLE: "#8B6914",  # Gold
            TrendStatus.DECLINING: "#E65100",  # Orange
            TrendStatus.SIGNIFICANTLY_DECLINING: "#B71C1C",  # Red
            TrendStatus.INSUFFICIENT_DATA: "#757575",  # Gray
        }

        comparison_chart_data.append({
            "user": user,
            "improvement_percentage": trend.improvement_percentage,
            "trend_status": trend.trend_status.value,
            "color": color_map.get(trend.trend_status, "#757575"),
            "baseline": trend.baseline_items_per_day,
            "recent": trend.recent_items_per_day,
        })

    # Prepare time series data for each user
    user_time_series = {}
    for user, trend in all_trends.items():
        if trend.periods:
            user_time_series[user] = {
                "labels": [p.period_label for p in trend.periods],
                "items_per_day": [p.items_per_day for p in trend.periods],
                "trend_line": trend.trend_line.trend_line_points if trend.trend_line else [],
                "simple_ma": trend.simple_ma.values if trend.simple_ma else [],
                "exponential_ma": trend.exponential_ma.values if trend.exponential_ma else [],
            }

    # Summary statistics
    summary = get_performance_trends_summary(changes, period_type)

    return {
        "comparison_chart_data": comparison_chart_data,
        "user_time_series": user_time_series,
        "summary": {
            "total_users": summary["total_users"],
            "improving_count": summary["improving_count"],
            "declining_count": summary["declining_count"],
            "stable_count": summary["stable_count"],
            "average_improvement": summary["average_improvement"],
        },
        "top_improvers": summary["top_improvers"],
        "biggest_decliners": summary["biggest_decliners"],
    }


def format_performance_trends_report(
    changes: List[Dict[str, Any]],
    period_type: str = "weekly",
    include_period_details: bool = True
) -> str:
    """Format performance trends as a text report.

    Args:
        changes: List of change records from historical data
        period_type: "weekly" or "monthly" period aggregation
        include_period_details: Whether to include per-period breakdown

    Returns:
        Formatted text report string
    """
    summary = get_performance_trends_summary(changes, period_type)
    all_trends = calculate_all_users_performance_trends(changes, period_type)

    lines = []
    lines.append("=" * 80)
    lines.append("USER PERFORMANCE TRENDS REPORT")
    lines.append("=" * 80)

    if summary["total_users"] == 0:
        lines.append("\nNo user data available.")
        return "\n".join(lines)

    # Date range info
    if summary.get("date_range_start") and summary.get("date_range_end"):
        lines.append(f"\nPeriod: {summary['date_range_start']} to {summary['date_range_end']}")
        lines.append(f"Analysis Type: {period_type.capitalize()}")

    # Summary
    lines.append("\n" + "-" * 80)
    lines.append("OVERVIEW")
    lines.append("-" * 80)
    lines.append(f"  Total Users Analyzed: {summary['total_users']}")
    lines.append(f"  Improving: {summary['improving_count']} users")
    lines.append(f"  Stable: {summary['stable_count']} users")
    lines.append(f"  Declining: {summary['declining_count']} users")
    lines.append(f"  Insufficient Data: {summary['insufficient_data_count']} users")
    lines.append(f"  Average Improvement: {summary['average_improvement']:+.1f}%")

    # Top Improvers
    if summary["top_improvers"]:
        lines.append("\n" + "-" * 80)
        lines.append("TOP IMPROVERS")
        lines.append("-" * 80)
        for item in summary["top_improvers"]:
            lines.append(f"  {item['user']}: +{item['improvement']:.1f}%")

    # Biggest Decliners
    if summary["biggest_decliners"]:
        lines.append("\n" + "-" * 80)
        lines.append("NEEDS ATTENTION (Declining)")
        lines.append("-" * 80)
        for item in summary["biggest_decliners"]:
            lines.append(f"  {item['user']}: -{item['decline']:.1f}%")

    # User Rankings by Trend
    lines.append("\n" + "-" * 80)
    lines.append("USER TREND RANKINGS")
    lines.append("-" * 80)
    lines.append(f"{'User':<10} {'Status':<25} {'Improvement':>12} {'Baseline':>12} {'Recent':>12} {'Momentum':<15}")
    lines.append("-" * 80)

    comparison = summary["comparison"]
    for item in comparison:
        lines.append(
            f"{item['user']:<10} "
            f"{item['trend_status']:<25} "
            f"{item['improvement_percentage']:>+11.1f}% "
            f"{item['baseline_items_per_day']:>12.2f} "
            f"{item['recent_items_per_day']:>12.2f} "
            f"{item['momentum']:<15}"
        )

    # Per-user period details
    if include_period_details:
        lines.append("\n" + "-" * 80)
        lines.append("PERIOD DETAILS BY USER")
        lines.append("-" * 80)

        for user in sorted(all_trends.keys()):
            trend = all_trends[user]
            if trend.periods:
                lines.append(f"\n  {user} ({trend.trend_status.value}):")
                for period in trend.periods:
                    change_str = f"{period.percent_change_from_previous:+.1f}%" if period.percent_change_from_previous is not None else "N/A"
                    lines.append(
                        f"    {period.period_label}: "
                        f"{period.total_items} items, "
                        f"{period.items_per_day:.2f}/day, "
                        f"Change: {change_str}"
                    )

    lines.append("\n" + "=" * 80)

    return "\n".join(lines)


if __name__ == "__main__":
    # Demo usage
    print("User Performance Trends Calculator - Demo")
    print("=" * 80)

    # Create sample data spanning multiple weeks
    import random
    sample_changes = []
    groups = ["NA", "NF", "NH", "NM", "NP", "NT", "NV"]
    users = ["DM", "JHU", "HI", "MK", "AS"]

    # Create data with different trends for each user
    user_trends = {
        "DM": 1.05,    # Improving
        "JHU": 0.95,   # Declining
        "HI": 1.0,     # Stable
        "MK": 1.10,    # Significantly improving
        "AS": 0.90,    # Significantly declining
    }

    base_date = datetime.now() - timedelta(days=60)

    for day_offset in range(60):
        current_date = base_date + timedelta(days=day_offset)
        week_number = day_offset // 7

        for user in users:
            # Calculate base items for this user with trend
            trend_factor = user_trends[user] ** week_number
            base_items = int(10 * trend_factor + random.randint(-3, 3))
            base_items = max(1, base_items)  # Ensure at least 1 item

            for _ in range(base_items):
                group = random.choice(groups)
                phase = random.randint(1, 5)
                row_id = f"row_{random.randint(1, 1000)}"

                sample_changes.append({
                    "Group": group,
                    "RowID": row_id,
                    "Phase": str(phase),
                    "User": user,
                    "ParsedTimestamp": current_date,
                })

    print(f"\nGenerated {len(sample_changes)} sample changes over 60 days")

    # Calculate and print report
    print(format_performance_trends_report(sample_changes, period_type="weekly"))

    # Print summary
    print("\nJSON Summary (excerpt):")
    summary = get_performance_trends_summary(sample_changes)
    import json
    print(json.dumps({
        "total_users": summary["total_users"],
        "improving_count": summary["improving_count"],
        "declining_count": summary["declining_count"],
        "stable_count": summary["stable_count"],
        "average_improvement": summary["average_improvement"],
        "top_improvers": summary["top_improvers"],
        "biggest_decliners": summary["biggest_decliners"],
    }, indent=2, default=str))

    # Print visualization data
    print("\nVisualization Data (summary):")
    viz_data = get_performance_trends_visualization_data(sample_changes)
    print(json.dumps(viz_data["summary"], indent=2))
