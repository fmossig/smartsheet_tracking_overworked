"""
Trend Line Calculator Module

Provides linear regression and moving average calculations for time-series data.
Prepares data structures for trend line visualization in reports and charts.

This module supports:
- Simple linear regression for trend analysis
- Simple, weighted, and exponential moving averages
- Trend line visualization data preparation
- Forecast calculations based on trend data

Usage:
    from trend_line_calculator import (
        TimeSeriesDataPoint,
        TrendLineResult,
        MovingAverageResult,
        TrendDirection,
        calculate_linear_regression,
        calculate_simple_moving_average,
        calculate_weighted_moving_average,
        calculate_exponential_moving_average,
        get_trend_visualization_data,
        forecast_value,
        analyze_trend,
    )
"""

import logging
import math
from dataclasses import dataclass, field
from datetime import datetime, timedelta, date
from typing import List, Dict, Optional, Any, Union, Tuple
from enum import Enum

# Set up logging
logger = logging.getLogger(__name__)


class TrendDirection(Enum):
    """Direction of the trend based on regression analysis."""
    STRONG_UP = "strong_up"
    UP = "up"
    FLAT = "flat"
    DOWN = "down"
    STRONG_DOWN = "strong_down"
    INSUFFICIENT_DATA = "insufficient_data"


class MovingAverageType(Enum):
    """Types of moving average calculations."""
    SIMPLE = "simple"
    WEIGHTED = "weighted"
    EXPONENTIAL = "exponential"


@dataclass
class TimeSeriesDataPoint:
    """A single data point in a time series.

    Attributes:
        date: The date of the data point
        value: The numeric value for this date
        label: Optional label for the data point
        metadata: Optional additional data for this point
    """
    date: date
    value: float
    label: Optional[str] = None
    metadata: Optional[Dict[str, Any]] = None

    def __post_init__(self):
        """Validate and normalize data after initialization."""
        # Convert string date if provided
        if isinstance(self.date, str):
            self.date = datetime.strptime(self.date, "%Y-%m-%d").date()
        elif isinstance(self.date, datetime):
            self.date = self.date.date()

        # Ensure value is a float
        self.value = float(self.value)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a serializable dictionary."""
        return {
            "date": self.date.isoformat(),
            "value": self.value,
            "label": self.label,
            "metadata": self.metadata,
        }


@dataclass
class TrendLineResult:
    """Result of a linear regression trend line calculation.

    Attributes:
        slope: The slope of the trend line (change per day)
        intercept: The y-intercept of the trend line
        r_squared: Coefficient of determination (0-1, how well the line fits)
        direction: The direction of the trend
        data_points: Original data points used for calculation
        trend_line_points: Calculated trend line values for each data point
        start_date: First date in the series
        end_date: Last date in the series
        average_value: Mean of all values
        std_deviation: Standard deviation of values
        percent_change: Percent change from first to last value
        daily_change_rate: Average daily change based on slope
    """
    slope: float
    intercept: float
    r_squared: float
    direction: TrendDirection
    data_points: List[TimeSeriesDataPoint]
    trend_line_points: List[float] = field(default_factory=list)
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    average_value: float = 0.0
    std_deviation: float = 0.0
    percent_change: float = 0.0
    daily_change_rate: float = 0.0

    def __post_init__(self):
        """Calculate derived metrics after initialization."""
        if self.data_points:
            self.start_date = min(dp.date for dp in self.data_points)
            self.end_date = max(dp.date for dp in self.data_points)

            values = [dp.value for dp in self.data_points]
            self.average_value = sum(values) / len(values) if values else 0.0

            if len(values) > 1:
                mean = self.average_value
                variance = sum((v - mean) ** 2 for v in values) / len(values)
                self.std_deviation = math.sqrt(variance)

                # Calculate percent change
                first_val = values[0]
                last_val = values[-1]
                if first_val != 0:
                    self.percent_change = ((last_val - first_val) / abs(first_val)) * 100
                else:
                    self.percent_change = 100.0 if last_val > 0 else (-100.0 if last_val < 0 else 0.0)

            self.daily_change_rate = self.slope

    def forecast(self, days_ahead: int) -> float:
        """Forecast a value N days after the end date.

        Args:
            days_ahead: Number of days to forecast ahead

        Returns:
            Forecasted value based on the trend line
        """
        if not self.data_points:
            return 0.0

        # X value is days from start date
        days_from_start = (self.end_date - self.start_date).days + days_ahead
        return self.intercept + self.slope * days_from_start

    def get_trend_line_value(self, target_date: date) -> float:
        """Get the trend line value for a specific date.

        Args:
            target_date: The date to get the trend value for

        Returns:
            The trend line value at the specified date
        """
        if not self.data_points or not self.start_date:
            return 0.0

        days_from_start = (target_date - self.start_date).days
        return self.intercept + self.slope * days_from_start

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a serializable dictionary."""
        return {
            "slope": round(self.slope, 6),
            "intercept": round(self.intercept, 4),
            "r_squared": round(self.r_squared, 4),
            "direction": self.direction.value,
            "start_date": self.start_date.isoformat() if self.start_date else None,
            "end_date": self.end_date.isoformat() if self.end_date else None,
            "average_value": round(self.average_value, 4),
            "std_deviation": round(self.std_deviation, 4),
            "percent_change": round(self.percent_change, 2),
            "daily_change_rate": round(self.daily_change_rate, 6),
            "trend_line_points": [round(v, 4) for v in self.trend_line_points],
            "data_point_count": len(self.data_points),
        }


@dataclass
class MovingAverageResult:
    """Result of a moving average calculation.

    Attributes:
        ma_type: Type of moving average (simple, weighted, exponential)
        window_size: Size of the moving average window
        data_points: Original data points
        ma_values: Calculated moving average values (aligned with data points)
        smoothing_factor: Alpha value for exponential MA (None for other types)
    """
    ma_type: MovingAverageType
    window_size: int
    data_points: List[TimeSeriesDataPoint]
    ma_values: List[Optional[float]] = field(default_factory=list)
    smoothing_factor: Optional[float] = None

    def __post_init__(self):
        """Validate parameters after initialization."""
        if self.window_size < 1:
            raise ValueError("Window size must be at least 1")

        if self.ma_type == MovingAverageType.EXPONENTIAL:
            if self.smoothing_factor is None:
                # Default smoothing factor: 2 / (window_size + 1)
                self.smoothing_factor = 2.0 / (self.window_size + 1)

    def get_latest_ma(self) -> Optional[float]:
        """Get the most recent moving average value."""
        for val in reversed(self.ma_values):
            if val is not None:
                return val
        return None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to a serializable dictionary."""
        return {
            "ma_type": self.ma_type.value,
            "window_size": self.window_size,
            "smoothing_factor": round(self.smoothing_factor, 4) if self.smoothing_factor else None,
            "ma_values": [round(v, 4) if v is not None else None for v in self.ma_values],
            "latest_ma": round(self.get_latest_ma(), 4) if self.get_latest_ma() is not None else None,
            "data_point_count": len(self.data_points),
        }


def calculate_linear_regression(
    data_points: List[TimeSeriesDataPoint],
    slope_threshold: float = 0.01,
    strong_slope_threshold: float = 0.05
) -> TrendLineResult:
    """Calculate a linear regression trend line for time series data.

    Uses the least squares method to find the best-fit line through the data points.

    Args:
        data_points: List of TimeSeriesDataPoint objects (must have at least 2 points)
        slope_threshold: Threshold for UP/DOWN vs FLAT classification (per day)
        strong_slope_threshold: Threshold for STRONG_UP/STRONG_DOWN classification

    Returns:
        TrendLineResult with slope, intercept, R-squared, and trend direction

    Example:
        >>> from datetime import date
        >>> points = [
        ...     TimeSeriesDataPoint(date(2026, 1, 1), 100),
        ...     TimeSeriesDataPoint(date(2026, 1, 2), 110),
        ...     TimeSeriesDataPoint(date(2026, 1, 3), 120),
        ... ]
        >>> result = calculate_linear_regression(points)
        >>> print(f"Slope: {result.slope}, Direction: {result.direction.value}")
    """
    if len(data_points) < 2:
        return TrendLineResult(
            slope=0.0,
            intercept=data_points[0].value if data_points else 0.0,
            r_squared=0.0,
            direction=TrendDirection.INSUFFICIENT_DATA,
            data_points=data_points,
            trend_line_points=[dp.value for dp in data_points] if data_points else [],
        )

    # Sort by date
    sorted_points = sorted(data_points, key=lambda dp: dp.date)

    # Convert dates to numeric values (days from first date)
    start_date = sorted_points[0].date
    x_values = [(dp.date - start_date).days for dp in sorted_points]
    y_values = [dp.value for dp in sorted_points]

    n = len(x_values)

    # Calculate means
    x_mean = sum(x_values) / n
    y_mean = sum(y_values) / n

    # Calculate slope and intercept using least squares
    numerator = sum((x - x_mean) * (y - y_mean) for x, y in zip(x_values, y_values))
    denominator = sum((x - x_mean) ** 2 for x in x_values)

    if denominator == 0:
        # All x values are the same (same date)
        slope = 0.0
        intercept = y_mean
    else:
        slope = numerator / denominator
        intercept = y_mean - slope * x_mean

    # Calculate R-squared
    y_pred = [intercept + slope * x for x in x_values]
    ss_res = sum((y - yp) ** 2 for y, yp in zip(y_values, y_pred))
    ss_tot = sum((y - y_mean) ** 2 for y in y_values)

    r_squared = 1 - (ss_res / ss_tot) if ss_tot != 0 else 0.0
    r_squared = max(0.0, min(1.0, r_squared))  # Clamp to [0, 1]

    # Determine trend direction based on normalized slope
    # Normalize slope relative to the mean value
    normalized_slope = slope / y_mean if y_mean != 0 else slope

    if abs(normalized_slope) < slope_threshold:
        direction = TrendDirection.FLAT
    elif normalized_slope >= strong_slope_threshold:
        direction = TrendDirection.STRONG_UP
    elif normalized_slope > slope_threshold:
        direction = TrendDirection.UP
    elif normalized_slope <= -strong_slope_threshold:
        direction = TrendDirection.STRONG_DOWN
    else:
        direction = TrendDirection.DOWN

    return TrendLineResult(
        slope=slope,
        intercept=intercept,
        r_squared=r_squared,
        direction=direction,
        data_points=sorted_points,
        trend_line_points=y_pred,
    )


def calculate_simple_moving_average(
    data_points: List[TimeSeriesDataPoint],
    window_size: int = 7
) -> MovingAverageResult:
    """Calculate a simple moving average for time series data.

    Simple Moving Average (SMA) is the unweighted mean of the previous n data points.

    Args:
        data_points: List of TimeSeriesDataPoint objects
        window_size: Number of periods for the moving average (default 7)

    Returns:
        MovingAverageResult with calculated MA values

    Example:
        >>> from datetime import date
        >>> points = [TimeSeriesDataPoint(date(2026, 1, i), i * 10) for i in range(1, 15)]
        >>> result = calculate_simple_moving_average(points, window_size=3)
        >>> print(f"Latest MA: {result.get_latest_ma()}")
    """
    if not data_points:
        return MovingAverageResult(
            ma_type=MovingAverageType.SIMPLE,
            window_size=window_size,
            data_points=[],
            ma_values=[],
        )

    # Sort by date
    sorted_points = sorted(data_points, key=lambda dp: dp.date)
    values = [dp.value for dp in sorted_points]
    n = len(values)

    ma_values: List[Optional[float]] = []

    for i in range(n):
        if i < window_size - 1:
            # Not enough data points yet
            ma_values.append(None)
        else:
            # Calculate average of last window_size values
            window = values[i - window_size + 1:i + 1]
            ma_values.append(sum(window) / window_size)

    return MovingAverageResult(
        ma_type=MovingAverageType.SIMPLE,
        window_size=window_size,
        data_points=sorted_points,
        ma_values=ma_values,
    )


def calculate_weighted_moving_average(
    data_points: List[TimeSeriesDataPoint],
    window_size: int = 7,
    weights: Optional[List[float]] = None
) -> MovingAverageResult:
    """Calculate a weighted moving average for time series data.

    Weighted Moving Average (WMA) gives more weight to recent data points.

    Args:
        data_points: List of TimeSeriesDataPoint objects
        window_size: Number of periods for the moving average (default 7)
        weights: Optional custom weights. If None, uses linearly increasing weights.

    Returns:
        MovingAverageResult with calculated WMA values

    Example:
        >>> from datetime import date
        >>> points = [TimeSeriesDataPoint(date(2026, 1, i), i * 10) for i in range(1, 15)]
        >>> result = calculate_weighted_moving_average(points, window_size=3)
        >>> print(f"Latest WMA: {result.get_latest_ma()}")
    """
    if not data_points:
        return MovingAverageResult(
            ma_type=MovingAverageType.WEIGHTED,
            window_size=window_size,
            data_points=[],
            ma_values=[],
        )

    # Sort by date
    sorted_points = sorted(data_points, key=lambda dp: dp.date)
    values = [dp.value for dp in sorted_points]
    n = len(values)

    # Generate default weights if not provided (linearly increasing)
    if weights is None:
        weights = list(range(1, window_size + 1))

    if len(weights) != window_size:
        raise ValueError(f"Weights length ({len(weights)}) must match window size ({window_size})")

    weight_sum = sum(weights)

    ma_values: List[Optional[float]] = []

    for i in range(n):
        if i < window_size - 1:
            # Not enough data points yet
            ma_values.append(None)
        else:
            # Calculate weighted average
            window = values[i - window_size + 1:i + 1]
            weighted_sum = sum(v * w for v, w in zip(window, weights))
            ma_values.append(weighted_sum / weight_sum)

    return MovingAverageResult(
        ma_type=MovingAverageType.WEIGHTED,
        window_size=window_size,
        data_points=sorted_points,
        ma_values=ma_values,
    )


def calculate_exponential_moving_average(
    data_points: List[TimeSeriesDataPoint],
    window_size: int = 7,
    smoothing_factor: Optional[float] = None
) -> MovingAverageResult:
    """Calculate an exponential moving average for time series data.

    Exponential Moving Average (EMA) applies exponentially decreasing weights
    to older data points, giving more importance to recent values.

    Args:
        data_points: List of TimeSeriesDataPoint objects
        window_size: Number of periods (used to calculate default smoothing factor)
        smoothing_factor: Alpha value (0-1). If None, uses 2/(window_size+1)

    Returns:
        MovingAverageResult with calculated EMA values

    Example:
        >>> from datetime import date
        >>> points = [TimeSeriesDataPoint(date(2026, 1, i), i * 10) for i in range(1, 15)]
        >>> result = calculate_exponential_moving_average(points, window_size=5)
        >>> print(f"Latest EMA: {result.get_latest_ma()}")
    """
    if not data_points:
        return MovingAverageResult(
            ma_type=MovingAverageType.EXPONENTIAL,
            window_size=window_size,
            data_points=[],
            ma_values=[],
            smoothing_factor=smoothing_factor,
        )

    # Sort by date
    sorted_points = sorted(data_points, key=lambda dp: dp.date)
    values = [dp.value for dp in sorted_points]
    n = len(values)

    # Calculate smoothing factor if not provided
    if smoothing_factor is None:
        smoothing_factor = 2.0 / (window_size + 1)

    if not (0 < smoothing_factor <= 1):
        raise ValueError("Smoothing factor must be between 0 and 1 (exclusive of 0)")

    ma_values: List[Optional[float]] = []

    # First EMA value is the simple average of first window_size values
    # or the first value if not enough data
    if n >= window_size:
        first_ema = sum(values[:window_size]) / window_size
        start_idx = window_size - 1

        # Fill None for initial values
        for _ in range(start_idx):
            ma_values.append(None)

        ma_values.append(first_ema)

        # Calculate subsequent EMA values
        ema = first_ema
        for i in range(start_idx + 1, n):
            ema = values[i] * smoothing_factor + ema * (1 - smoothing_factor)
            ma_values.append(ema)
    else:
        # Not enough data for full EMA, use progressive approach
        if n > 0:
            ema = values[0]
            ma_values.append(ema)

            for i in range(1, n):
                ema = values[i] * smoothing_factor + ema * (1 - smoothing_factor)
                ma_values.append(ema)

    return MovingAverageResult(
        ma_type=MovingAverageType.EXPONENTIAL,
        window_size=window_size,
        data_points=sorted_points,
        ma_values=ma_values,
        smoothing_factor=smoothing_factor,
    )


def forecast_value(
    trend_result: TrendLineResult,
    target_date: date
) -> Dict[str, Any]:
    """Forecast a value for a target date based on trend analysis.

    Args:
        trend_result: TrendLineResult from calculate_linear_regression
        target_date: The date to forecast for

    Returns:
        Dictionary with forecasted value and confidence metrics

    Example:
        >>> result = calculate_linear_regression(data_points)
        >>> forecast = forecast_value(result, date(2026, 2, 1))
        >>> print(f"Forecasted value: {forecast['value']}")
    """
    if not trend_result.start_date:
        return {
            "value": None,
            "confidence": 0.0,
            "error_margin": None,
            "days_from_last_data": None,
            "is_extrapolation": True,
        }

    forecasted_value = trend_result.get_trend_line_value(target_date)

    # Calculate days from last data point
    days_from_last = (target_date - trend_result.end_date).days if trend_result.end_date else 0
    is_extrapolation = target_date > trend_result.end_date if trend_result.end_date else True

    # Confidence decreases for extrapolation
    base_confidence = trend_result.r_squared
    if is_extrapolation and days_from_last > 0:
        # Reduce confidence by 5% per day of extrapolation
        decay = max(0, 1 - (days_from_last * 0.05))
        confidence = base_confidence * decay
    else:
        confidence = base_confidence

    # Error margin based on standard deviation and extrapolation distance
    error_margin = trend_result.std_deviation
    if is_extrapolation and days_from_last > 0:
        error_margin *= (1 + days_from_last * 0.1)

    return {
        "date": target_date.isoformat(),
        "value": round(forecasted_value, 4),
        "confidence": round(confidence, 4),
        "error_margin": round(error_margin, 4),
        "days_from_last_data": days_from_last,
        "is_extrapolation": is_extrapolation,
    }


def analyze_trend(
    data_points: List[TimeSeriesDataPoint],
    ma_window_size: int = 7
) -> Dict[str, Any]:
    """Perform comprehensive trend analysis on time series data.

    Combines linear regression and moving average analysis for a complete
    trend assessment.

    Args:
        data_points: List of TimeSeriesDataPoint objects
        ma_window_size: Window size for moving average calculations

    Returns:
        Dictionary with complete trend analysis results

    Example:
        >>> from datetime import date
        >>> points = [TimeSeriesDataPoint(date(2026, 1, i), 100 + i * 5) for i in range(1, 31)]
        >>> analysis = analyze_trend(points)
        >>> print(f"Trend: {analysis['linear_regression']['direction']}")
    """
    if not data_points:
        return {
            "data_points": 0,
            "linear_regression": None,
            "simple_ma": None,
            "exponential_ma": None,
            "summary": {
                "trend_direction": TrendDirection.INSUFFICIENT_DATA.value,
                "confidence": 0.0,
                "recommendation": "Insufficient data for trend analysis",
            },
        }

    # Calculate linear regression
    lr_result = calculate_linear_regression(data_points)

    # Calculate moving averages
    sma_result = calculate_simple_moving_average(data_points, ma_window_size)
    ema_result = calculate_exponential_moving_average(data_points, ma_window_size)

    # Generate summary
    latest_sma = sma_result.get_latest_ma()
    latest_ema = ema_result.get_latest_ma()
    latest_value = data_points[-1].value if data_points else 0

    # Determine if value is above or below moving averages
    above_sma = latest_value > latest_sma if latest_sma is not None else None
    above_ema = latest_value > latest_ema if latest_ema is not None else None

    # Generate recommendation based on trend
    if lr_result.direction == TrendDirection.STRONG_UP:
        recommendation = "Strong upward trend detected. Values are increasing significantly."
    elif lr_result.direction == TrendDirection.UP:
        recommendation = "Moderate upward trend detected. Values are generally increasing."
    elif lr_result.direction == TrendDirection.STRONG_DOWN:
        recommendation = "Strong downward trend detected. Values are decreasing significantly."
    elif lr_result.direction == TrendDirection.DOWN:
        recommendation = "Moderate downward trend detected. Values are generally decreasing."
    elif lr_result.direction == TrendDirection.FLAT:
        recommendation = "Values are relatively stable with no significant trend."
    else:
        recommendation = "Insufficient data to determine a reliable trend."

    return {
        "data_points": len(data_points),
        "date_range": {
            "start": lr_result.start_date.isoformat() if lr_result.start_date else None,
            "end": lr_result.end_date.isoformat() if lr_result.end_date else None,
        },
        "linear_regression": lr_result.to_dict(),
        "simple_ma": sma_result.to_dict(),
        "exponential_ma": ema_result.to_dict(),
        "current_vs_ma": {
            "latest_value": round(latest_value, 4),
            "latest_sma": round(latest_sma, 4) if latest_sma is not None else None,
            "latest_ema": round(latest_ema, 4) if latest_ema is not None else None,
            "above_sma": above_sma,
            "above_ema": above_ema,
        },
        "summary": {
            "trend_direction": lr_result.direction.value,
            "confidence": round(lr_result.r_squared, 4),
            "percent_change": round(lr_result.percent_change, 2),
            "daily_change_rate": round(lr_result.daily_change_rate, 6),
            "recommendation": recommendation,
        },
    }


def get_trend_visualization_data(
    data_points: List[TimeSeriesDataPoint],
    include_regression: bool = True,
    include_sma: bool = True,
    include_ema: bool = True,
    ma_window_size: int = 7
) -> Dict[str, Any]:
    """Prepare data structures for trend line visualization in charts.

    Generates data suitable for ReportLab or other charting libraries.

    Args:
        data_points: List of TimeSeriesDataPoint objects
        include_regression: Include linear regression trend line
        include_sma: Include simple moving average line
        include_ema: Include exponential moving average line
        ma_window_size: Window size for moving average calculations

    Returns:
        Dictionary with visualization-ready data series

    Example:
        >>> viz_data = get_trend_visualization_data(data_points)
        >>> print(viz_data['chart_data']['labels'])  # Date labels
        >>> print(viz_data['chart_data']['actual_values'])  # Original values
        >>> print(viz_data['chart_data']['trend_line'])  # Regression line
    """
    if not data_points:
        return {
            "chart_data": {
                "labels": [],
                "actual_values": [],
                "trend_line": [],
                "sma": [],
                "ema": [],
            },
            "metrics": {
                "trend_direction": TrendDirection.INSUFFICIENT_DATA.value,
                "r_squared": 0.0,
                "slope": 0.0,
            },
            "annotations": [],
        }

    # Sort data points by date
    sorted_points = sorted(data_points, key=lambda dp: dp.date)

    # Prepare base data
    labels = [dp.date.strftime("%Y-%m-%d") for dp in sorted_points]
    actual_values = [dp.value for dp in sorted_points]

    # Calculate trend lines if requested
    trend_line = []
    sma = []
    ema = []
    metrics = {}

    if include_regression:
        lr_result = calculate_linear_regression(sorted_points)
        trend_line = lr_result.trend_line_points
        metrics = {
            "trend_direction": lr_result.direction.value,
            "r_squared": round(lr_result.r_squared, 4),
            "slope": round(lr_result.slope, 6),
            "intercept": round(lr_result.intercept, 4),
            "percent_change": round(lr_result.percent_change, 2),
            "average_value": round(lr_result.average_value, 4),
        }

    if include_sma:
        sma_result = calculate_simple_moving_average(sorted_points, ma_window_size)
        sma = sma_result.ma_values

    if include_ema:
        ema_result = calculate_exponential_moving_average(sorted_points, ma_window_size)
        ema = ema_result.ma_values

    # Generate annotations for significant points
    annotations = []
    if len(actual_values) >= 2:
        # Mark highest and lowest points
        max_idx = actual_values.index(max(actual_values))
        min_idx = actual_values.index(min(actual_values))

        annotations.append({
            "type": "max",
            "index": max_idx,
            "date": labels[max_idx],
            "value": actual_values[max_idx],
            "label": f"Max: {actual_values[max_idx]:.2f}",
        })
        annotations.append({
            "type": "min",
            "index": min_idx,
            "date": labels[min_idx],
            "value": actual_values[min_idx],
            "label": f"Min: {actual_values[min_idx]:.2f}",
        })

    return {
        "chart_data": {
            "labels": labels,
            "actual_values": [round(v, 4) for v in actual_values],
            "trend_line": [round(v, 4) for v in trend_line] if trend_line else [],
            "sma": [round(v, 4) if v is not None else None for v in sma],
            "ema": [round(v, 4) if v is not None else None for v in ema],
        },
        "metrics": metrics,
        "annotations": annotations,
        "ma_window_size": ma_window_size,
        "series_count": sum([
            1,  # Actual values always included
            1 if include_regression else 0,
            1 if include_sma else 0,
            1 if include_ema else 0,
        ]),
    }


def create_data_points_from_daily_counts(
    daily_counts: Dict[str, int]
) -> List[TimeSeriesDataPoint]:
    """Create TimeSeriesDataPoint list from a dictionary of daily counts.

    Helper function to convert data from historical_data_loader's
    get_daily_change_counts() format.

    Args:
        daily_counts: Dictionary mapping date strings (YYYY-MM-DD) to counts

    Returns:
        List of TimeSeriesDataPoint objects sorted by date

    Example:
        >>> from historical_data_loader import get_daily_change_counts
        >>> counts = get_daily_change_counts(changes)
        >>> data_points = create_data_points_from_daily_counts(counts)
        >>> trend = calculate_linear_regression(data_points)
    """
    data_points = []
    for date_str, count in daily_counts.items():
        try:
            d = datetime.strptime(date_str, "%Y-%m-%d").date()
            data_points.append(TimeSeriesDataPoint(date=d, value=float(count)))
        except (ValueError, TypeError) as e:
            logger.warning(f"Skipping invalid date entry: {date_str} - {e}")
            continue

    return sorted(data_points, key=lambda dp: dp.date)


def format_trend_report(
    analysis: Dict[str, Any]
) -> Dict[str, Any]:
    """Format trend analysis results for human-readable reports.

    Args:
        analysis: Result from analyze_trend()

    Returns:
        Dictionary with formatted report sections
    """
    if not analysis or analysis.get("data_points", 0) == 0:
        return {
            "executive_summary": "Insufficient data for trend analysis.",
            "key_metrics": {},
            "findings": [],
            "recommendations": ["Collect more data points for meaningful trend analysis."],
        }

    summary = analysis.get("summary", {})
    lr = analysis.get("linear_regression", {})
    current_vs_ma = analysis.get("current_vs_ma", {})

    direction = summary.get("trend_direction", "unknown")
    confidence = summary.get("confidence", 0)
    percent_change = summary.get("percent_change", 0)

    # Generate executive summary
    direction_text = {
        "strong_up": "a strong upward",
        "up": "a moderate upward",
        "flat": "a stable/flat",
        "down": "a moderate downward",
        "strong_down": "a strong downward",
        "insufficient_data": "an unclear",
    }.get(direction, "an unknown")

    confidence_text = "high" if confidence > 0.7 else ("moderate" if confidence > 0.4 else "low")

    executive_summary = (
        f"The data shows {direction_text} trend with {confidence_text} confidence "
        f"(R-squared: {confidence:.2%}). "
        f"Overall change: {percent_change:+.1f}% over the analysis period."
    )

    # Key metrics
    key_metrics = {
        "data_points": analysis.get("data_points", 0),
        "date_range": analysis.get("date_range", {}),
        "trend_direction": direction,
        "confidence_score": f"{confidence:.2%}",
        "total_change": f"{percent_change:+.1f}%",
        "daily_change": f"{summary.get('daily_change_rate', 0):+.4f}",
        "latest_value": current_vs_ma.get("latest_value"),
    }

    # Key findings
    findings = []

    if direction in ["strong_up", "up"]:
        findings.append("Values are showing a positive growth pattern.")
    elif direction in ["strong_down", "down"]:
        findings.append("Values are showing a declining pattern.")
    else:
        findings.append("Values are relatively stable over the analysis period.")

    if current_vs_ma.get("above_sma") is True:
        findings.append("Current value is above the simple moving average, indicating recent strength.")
    elif current_vs_ma.get("above_sma") is False:
        findings.append("Current value is below the simple moving average, indicating recent weakness.")

    if confidence > 0.7:
        findings.append("The trend line fits the data well, suggesting a consistent pattern.")
    elif confidence < 0.3:
        findings.append("High variability in data - trend predictions should be used with caution.")

    # Recommendations
    recommendations = []
    if summary.get("recommendation"):
        recommendations.append(summary["recommendation"])

    if confidence < 0.4:
        recommendations.append("Consider collecting more data or investigating sources of variability.")

    return {
        "executive_summary": executive_summary,
        "key_metrics": key_metrics,
        "findings": findings,
        "recommendations": recommendations,
    }


if __name__ == "__main__":
    # Demo usage
    print("Trend Line Calculator - Demo")
    print("=" * 60)

    # Create sample data points
    from datetime import date, timedelta

    # Simulated upward trending data with some noise
    base_date = date(2026, 1, 1)
    sample_data = []
    for i in range(30):
        d = base_date + timedelta(days=i)
        # Upward trend with some noise
        value = 100 + i * 2 + (i % 5 - 2) * 3
        sample_data.append(TimeSeriesDataPoint(date=d, value=value))

    print(f"\nGenerated {len(sample_data)} sample data points")
    print(f"Date range: {sample_data[0].date} to {sample_data[-1].date}")
    print(f"Value range: {min(dp.value for dp in sample_data):.1f} to {max(dp.value for dp in sample_data):.1f}")

    # Linear Regression
    print("\n" + "-" * 60)
    print("Linear Regression Analysis")
    print("-" * 60)

    lr_result = calculate_linear_regression(sample_data)
    print(f"Slope: {lr_result.slope:.4f} (change per day)")
    print(f"Intercept: {lr_result.intercept:.4f}")
    print(f"R-squared: {lr_result.r_squared:.4f}")
    print(f"Direction: {lr_result.direction.value}")
    print(f"Percent change: {lr_result.percent_change:.2f}%")

    # Moving Averages
    print("\n" + "-" * 60)
    print("Moving Average Analysis (7-day window)")
    print("-" * 60)

    sma_result = calculate_simple_moving_average(sample_data, 7)
    print(f"Simple MA (latest): {sma_result.get_latest_ma():.2f}")

    wma_result = calculate_weighted_moving_average(sample_data, 7)
    print(f"Weighted MA (latest): {wma_result.get_latest_ma():.2f}")

    ema_result = calculate_exponential_moving_average(sample_data, 7)
    print(f"Exponential MA (latest): {ema_result.get_latest_ma():.2f}")

    # Forecast
    print("\n" + "-" * 60)
    print("Forecast (7 days ahead)")
    print("-" * 60)

    forecast_date = base_date + timedelta(days=37)
    forecast = forecast_value(lr_result, forecast_date)
    print(f"Forecast for {forecast['date']}: {forecast['value']:.2f}")
    print(f"Confidence: {forecast['confidence']:.2%}")
    print(f"Error margin: +/- {forecast['error_margin']:.2f}")

    # Comprehensive Analysis
    print("\n" + "-" * 60)
    print("Comprehensive Trend Analysis")
    print("-" * 60)

    analysis = analyze_trend(sample_data)
    print(f"Summary: {analysis['summary']['recommendation']}")
    print(f"Confidence: {analysis['summary']['confidence']:.2%}")

    # Visualization Data
    print("\n" + "-" * 60)
    print("Visualization Data Sample")
    print("-" * 60)

    viz_data = get_trend_visualization_data(sample_data)
    print(f"Series count: {viz_data['series_count']}")
    print(f"Data points: {len(viz_data['chart_data']['labels'])}")
    print(f"Annotations: {len(viz_data['annotations'])}")
    for anno in viz_data['annotations']:
        print(f"  - {anno['label']} on {anno['date']}")

    # Formatted Report
    print("\n" + "-" * 60)
    print("Trend Report")
    print("-" * 60)

    report = format_trend_report(analysis)
    print(f"\nExecutive Summary:")
    print(f"  {report['executive_summary']}")
    print(f"\nKey Findings:")
    for finding in report['findings']:
        print(f"  - {finding}")
    print(f"\nRecommendations:")
    for rec in report['recommendations']:
        print(f"  - {rec}")
