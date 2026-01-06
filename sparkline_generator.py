"""
Sparkline Generator Module

Provides compact sparkline charts for showing mini trends inline with text or tables.
Implements simple line-based visualization for historical patterns using ReportLab.

Sparklines are designed to be small enough to fit inline with text or within table cells,
typically 40-120 points wide and 15-30 points high.

Key features:
- Multiple sparkline styles (line, area, bars, win/loss)
- WCAG AA compliant color options
- Min/max point highlighting
- Reference line support
- Configurable dimensions for inline or table use

Usage:
    from sparkline_generator import (
        create_sparkline,
        SparklineStyle,
        SparklineOptions,
        SPARKLINE_COLORS,
    )

    # Simple line sparkline
    sparkline = create_sparkline([10, 15, 12, 18, 14, 20])
    story.append(sparkline)

    # Area sparkline with highlighting
    sparkline = create_sparkline(
        [10, 15, 12, 18, 14, 20],
        style=SparklineStyle.AREA,
        highlight_min=True,
        highlight_max=True,
    )
"""

import logging
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Any, Tuple, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from reportlab.lib.colors import Color
    from reportlab.graphics.shapes import Drawing

from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing, Line, String, Polygon, Rect, Circle, Group

logger = logging.getLogger(__name__)


# =============================================================================
# Sparkline Constants
# =============================================================================

# Default dimensions for sparklines
DEFAULT_SPARKLINE_WIDTH = 60   # Points - compact for inline use
DEFAULT_SPARKLINE_HEIGHT = 20  # Points - fits in table cells

# Dimensions for different contexts
INLINE_SPARKLINE_WIDTH = 50    # For inline with text
INLINE_SPARKLINE_HEIGHT = 15

TABLE_SPARKLINE_WIDTH = 80     # For table cells
TABLE_SPARKLINE_HEIGHT = 25

DASHBOARD_SPARKLINE_WIDTH = 120  # For dashboard widgets
DASHBOARD_SPARKLINE_HEIGHT = 40

# WCAG AA compliant sparkline colors
SPARKLINE_COLORS = {
    "primary": colors.HexColor("#0077B6"),       # Blue - Contrast: 4.5:1
    "positive": colors.HexColor("#1B5E20"),      # Green - Contrast: 7.8:1
    "negative": colors.HexColor("#B71C1C"),      # Red - Contrast: 6.9:1
    "neutral": colors.HexColor("#546E7A"),       # Blue Grey - Contrast: 4.6:1
    "accent": colors.HexColor("#C65102"),        # Orange - Contrast: 4.8:1
    "min_point": colors.HexColor("#B71C1C"),     # Red for minimum point
    "max_point": colors.HexColor("#1B5E20"),     # Green for maximum point
    "reference": colors.HexColor("#757575"),     # Grey for reference line
    "fill": colors.HexColor("#E3F2FD"),          # Light blue fill for area
    "bar_positive": colors.HexColor("#1B5E20"),
    "bar_negative": colors.HexColor("#B71C1C"),
}

# Fill colors with transparency effect (lighter versions)
SPARKLINE_FILL_COLORS = {
    "primary": colors.HexColor("#B3E5FC"),       # Light Blue
    "positive": colors.HexColor("#C8E6C9"),      # Light Green
    "negative": colors.HexColor("#FFCDD2"),      # Light Red
    "neutral": colors.HexColor("#CFD8DC"),       # Light Grey
}


# =============================================================================
# Sparkline Style Definitions
# =============================================================================

class SparklineStyle(Enum):
    """Sparkline visualization style options."""
    LINE = "line"         # Simple connected line
    AREA = "area"         # Filled area under the line
    BARS = "bars"         # Vertical bars for each data point
    WIN_LOSS = "win_loss" # Binary win/loss bars (above/below zero)
    DOTS = "dots"         # Individual dots (useful for sparse data)


class SparklineTrend(Enum):
    """Overall trend direction for styling."""
    UP = "up"
    DOWN = "down"
    FLAT = "flat"
    VOLATILE = "volatile"


# =============================================================================
# Sparkline Options Dataclass
# =============================================================================

@dataclass
class SparklineOptions:
    """Configuration options for sparkline generation.

    Attributes:
        width: Width in points. Defaults to 60.
        height: Height in points. Defaults to 20.
        style: Visualization style. Defaults to LINE.
        line_color: Color for the line/bars. Auto-selected if None.
        fill_color: Fill color for area style. Auto-selected if None.
        stroke_width: Line thickness in points. Defaults to 1.5.
        highlight_min: If True, highlight the minimum point. Defaults to False.
        highlight_max: If True, highlight the maximum point. Defaults to False.
        show_reference_line: If True, show a reference line at zero or mean.
        reference_value: Value for reference line. Uses mean if None.
        show_endpoints: If True, show dots at start and end points.
        padding: Internal padding in points. Defaults to 2.
        bar_spacing: Spacing between bars (0-1). Defaults to 0.2.
        auto_color_trend: If True, color based on trend direction.
    """
    width: int = DEFAULT_SPARKLINE_WIDTH
    height: int = DEFAULT_SPARKLINE_HEIGHT
    style: SparklineStyle = SparklineStyle.LINE
    line_color: Optional["Color"] = None
    fill_color: Optional["Color"] = None
    stroke_width: float = 1.5
    highlight_min: bool = False
    highlight_max: bool = False
    show_reference_line: bool = False
    reference_value: Optional[float] = None
    show_endpoints: bool = False
    padding: float = 2
    bar_spacing: float = 0.2
    auto_color_trend: bool = True


# =============================================================================
# Sparkline Metrics Dataclass
# =============================================================================

@dataclass
class SparklineMetrics:
    """Metrics extracted from sparkline data.

    Provides summary statistics that can be displayed alongside the sparkline.
    """
    min_value: float
    max_value: float
    first_value: float
    last_value: float
    mean_value: float
    trend: SparklineTrend
    change_percent: float
    data_points: int

    def to_dict(self) -> Dict[str, Any]:
        """Convert metrics to a dictionary for JSON serialization."""
        return {
            "min_value": self.min_value,
            "max_value": self.max_value,
            "first_value": self.first_value,
            "last_value": self.last_value,
            "mean_value": round(self.mean_value, 2),
            "trend": self.trend.value,
            "change_percent": round(self.change_percent, 2),
            "data_points": self.data_points,
        }


# =============================================================================
# Helper Functions
# =============================================================================

def calculate_sparkline_metrics(data: List[float]) -> SparklineMetrics:
    """Calculate metrics from sparkline data.

    Args:
        data: List of numeric values.

    Returns:
        SparklineMetrics object with summary statistics.
    """
    if not data:
        return SparklineMetrics(
            min_value=0, max_value=0, first_value=0, last_value=0,
            mean_value=0, trend=SparklineTrend.FLAT, change_percent=0, data_points=0
        )

    min_val = min(data)
    max_val = max(data)
    first_val = data[0]
    last_val = data[-1]
    mean_val = sum(data) / len(data)

    # Calculate trend and change
    if len(data) >= 2:
        change = last_val - first_val
        if first_val != 0:
            change_pct = (change / abs(first_val)) * 100
        else:
            change_pct = 100.0 if change > 0 else (-100.0 if change < 0 else 0.0)

        # Determine trend based on change percentage and consistency
        # Check if trend is monotonic (consistently increasing or decreasing)
        is_monotonic = True
        is_increasing = data[1] >= data[0]
        for i in range(1, len(data)):
            if is_increasing and data[i] < data[i-1]:
                is_monotonic = False
                break
            elif not is_increasing and data[i] > data[i-1]:
                is_monotonic = False
                break

        # Calculate volatility (coefficient of variation)
        volatility = (max_val - min_val) / mean_val if mean_val != 0 else 0

        if abs(change_pct) < 5:
            trend = SparklineTrend.FLAT
        elif is_monotonic or abs(change_pct) > 20:
            # Clear directional trend if monotonic or large change
            trend = SparklineTrend.UP if change > 0 else SparklineTrend.DOWN
        elif volatility > 1.0:
            # Only mark as volatile if high volatility AND no clear direction
            trend = SparklineTrend.VOLATILE
        elif change > 0:
            trend = SparklineTrend.UP
        else:
            trend = SparklineTrend.DOWN
    else:
        change_pct = 0
        trend = SparklineTrend.FLAT

    return SparklineMetrics(
        min_value=min_val,
        max_value=max_val,
        first_value=first_val,
        last_value=last_val,
        mean_value=mean_val,
        trend=trend,
        change_percent=change_pct,
        data_points=len(data),
    )


def get_trend_color(trend: SparklineTrend) -> "Color":
    """Get the appropriate color for a trend direction.

    Args:
        trend: The trend direction.

    Returns:
        ReportLab color object.
    """
    if trend == SparklineTrend.UP:
        return SPARKLINE_COLORS["positive"]
    elif trend == SparklineTrend.DOWN:
        return SPARKLINE_COLORS["negative"]
    elif trend == SparklineTrend.VOLATILE:
        return SPARKLINE_COLORS["accent"]
    else:
        return SPARKLINE_COLORS["neutral"]


def normalize_data(
    data: List[float],
    height: float,
    padding: float,
) -> Tuple[List[float], float, float]:
    """Normalize data values to fit within the sparkline height.

    Args:
        data: List of numeric values.
        height: Total height of the sparkline.
        padding: Padding from top and bottom.

    Returns:
        Tuple of (normalized_values, min_value, max_value).
    """
    if not data:
        return [], 0, 0

    min_val = min(data)
    max_val = max(data)

    # Handle case where all values are the same
    value_range = max_val - min_val
    if value_range == 0:
        # Place all points at the middle
        normalized = [(height - 2 * padding) / 2 + padding] * len(data)
        return normalized, min_val, max_val

    available_height = height - 2 * padding
    normalized = [
        padding + ((v - min_val) / value_range) * available_height
        for v in data
    ]

    return normalized, min_val, max_val


# =============================================================================
# Sparkline Drawing Functions
# =============================================================================

def _draw_line_sparkline(
    drawing: "Drawing",
    data: List[float],
    options: SparklineOptions,
    metrics: SparklineMetrics,
) -> None:
    """Draw a line-style sparkline."""
    if len(data) < 2:
        return

    normalized, min_val, max_val = normalize_data(data, options.height, options.padding)

    # Calculate x positions
    n_points = len(data)
    available_width = options.width - 2 * options.padding
    x_step = available_width / (n_points - 1) if n_points > 1 else 0

    # Determine line color
    if options.line_color:
        line_color = options.line_color
    elif options.auto_color_trend:
        line_color = get_trend_color(metrics.trend)
    else:
        line_color = SPARKLINE_COLORS["primary"]

    # Draw connecting lines
    for i in range(len(data) - 1):
        x1 = options.padding + i * x_step
        y1 = normalized[i]
        x2 = options.padding + (i + 1) * x_step
        y2 = normalized[i + 1]

        line = Line(x1, y1, x2, y2)
        line.strokeColor = line_color
        line.strokeWidth = options.stroke_width
        drawing.add(line)

    # Highlight min/max points
    _draw_highlight_points(drawing, data, normalized, options, x_step, min_val, max_val)

    # Draw endpoints if requested
    if options.show_endpoints:
        _draw_endpoints(drawing, data, normalized, options, x_step, line_color)


def _draw_area_sparkline(
    drawing: "Drawing",
    data: List[float],
    options: SparklineOptions,
    metrics: SparklineMetrics,
) -> None:
    """Draw an area-style sparkline (filled under the line)."""
    if len(data) < 2:
        return

    normalized, min_val, max_val = normalize_data(data, options.height, options.padding)

    n_points = len(data)
    available_width = options.width - 2 * options.padding
    x_step = available_width / (n_points - 1) if n_points > 1 else 0

    # Determine colors
    if options.line_color:
        line_color = options.line_color
    elif options.auto_color_trend:
        line_color = get_trend_color(metrics.trend)
    else:
        line_color = SPARKLINE_COLORS["primary"]

    if options.fill_color:
        fill_color = options.fill_color
    else:
        fill_color = SPARKLINE_FILL_COLORS.get("primary", colors.lightblue)

    # Build polygon points for the fill area
    points = []

    # Start at bottom left
    points.append(options.padding)
    points.append(options.padding)

    # Add all data points
    for i, y_val in enumerate(normalized):
        x = options.padding + i * x_step
        points.append(x)
        points.append(y_val)

    # Close at bottom right
    points.append(options.padding + (n_points - 1) * x_step)
    points.append(options.padding)

    # Draw filled polygon
    polygon = Polygon(points)
    polygon.fillColor = fill_color
    polygon.strokeColor = None
    drawing.add(polygon)

    # Draw the line on top
    for i in range(len(data) - 1):
        x1 = options.padding + i * x_step
        y1 = normalized[i]
        x2 = options.padding + (i + 1) * x_step
        y2 = normalized[i + 1]

        line = Line(x1, y1, x2, y2)
        line.strokeColor = line_color
        line.strokeWidth = options.stroke_width
        drawing.add(line)

    # Highlight min/max points
    _draw_highlight_points(drawing, data, normalized, options, x_step, min_val, max_val)


def _draw_bar_sparkline(
    drawing: "Drawing",
    data: List[float],
    options: SparklineOptions,
    metrics: SparklineMetrics,
) -> None:
    """Draw a bar-style sparkline."""
    if not data:
        return

    normalized, min_val, max_val = normalize_data(data, options.height, options.padding)

    n_bars = len(data)
    available_width = options.width - 2 * options.padding
    bar_width = available_width / n_bars
    spacing = bar_width * options.bar_spacing
    actual_bar_width = bar_width - spacing

    # Determine bar color
    if options.line_color:
        bar_color = options.line_color
    elif options.auto_color_trend:
        bar_color = get_trend_color(metrics.trend)
    else:
        bar_color = SPARKLINE_COLORS["primary"]

    # Calculate baseline (bottom of chart or zero line if data spans positive/negative)
    baseline_y = options.padding

    # Draw bars
    for i, y_val in enumerate(normalized):
        x = options.padding + i * bar_width + spacing / 2
        bar_height = y_val - baseline_y

        rect = Rect(x, baseline_y, actual_bar_width, bar_height)
        rect.fillColor = bar_color
        rect.strokeColor = None
        drawing.add(rect)


def _draw_win_loss_sparkline(
    drawing: "Drawing",
    data: List[float],
    options: SparklineOptions,
    metrics: SparklineMetrics,
) -> None:
    """Draw a win/loss sparkline (binary bars above/below center line)."""
    if not data:
        return

    n_bars = len(data)
    available_width = options.width - 2 * options.padding
    bar_width = available_width / n_bars
    spacing = bar_width * options.bar_spacing
    actual_bar_width = bar_width - spacing

    center_y = options.height / 2
    half_height = (options.height - 2 * options.padding) / 2 - 1

    positive_color = SPARKLINE_COLORS["bar_positive"]
    negative_color = SPARKLINE_COLORS["bar_negative"]

    for i, value in enumerate(data):
        x = options.padding + i * bar_width + spacing / 2

        if value >= 0:
            rect = Rect(x, center_y, actual_bar_width, half_height)
            rect.fillColor = positive_color
        else:
            rect = Rect(x, center_y - half_height, actual_bar_width, half_height)
            rect.fillColor = negative_color

        rect.strokeColor = None
        drawing.add(rect)

    # Draw center line
    center_line = Line(
        options.padding,
        center_y,
        options.width - options.padding,
        center_y
    )
    center_line.strokeColor = SPARKLINE_COLORS["reference"]
    center_line.strokeWidth = 0.5
    drawing.add(center_line)


def _draw_dots_sparkline(
    drawing: "Drawing",
    data: List[float],
    options: SparklineOptions,
    metrics: SparklineMetrics,
) -> None:
    """Draw a dots-style sparkline (individual points)."""
    if not data:
        return

    normalized, min_val, max_val = normalize_data(data, options.height, options.padding)

    n_points = len(data)
    available_width = options.width - 2 * options.padding
    x_step = available_width / (n_points - 1) if n_points > 1 else options.width / 2

    # Determine dot color
    if options.line_color:
        dot_color = options.line_color
    elif options.auto_color_trend:
        dot_color = get_trend_color(metrics.trend)
    else:
        dot_color = SPARKLINE_COLORS["primary"]

    dot_radius = max(1.5, options.stroke_width)

    for i, y_val in enumerate(normalized):
        if n_points > 1:
            x = options.padding + i * x_step
        else:
            x = options.width / 2

        circle = Circle(x, y_val, dot_radius)
        circle.fillColor = dot_color
        circle.strokeColor = None
        drawing.add(circle)


def _draw_highlight_points(
    drawing: "Drawing",
    data: List[float],
    normalized: List[float],
    options: SparklineOptions,
    x_step: float,
    min_val: float,
    max_val: float,
) -> None:
    """Draw highlight dots for min/max points."""
    if not (options.highlight_min or options.highlight_max):
        return

    dot_radius = max(2.0, options.stroke_width + 0.5)

    for i, value in enumerate(data):
        x = options.padding + i * x_step
        y = normalized[i]

        if options.highlight_max and value == max_val:
            circle = Circle(x, y, dot_radius)
            circle.fillColor = SPARKLINE_COLORS["max_point"]
            circle.strokeColor = colors.white
            circle.strokeWidth = 0.5
            drawing.add(circle)

        if options.highlight_min and value == min_val:
            circle = Circle(x, y, dot_radius)
            circle.fillColor = SPARKLINE_COLORS["min_point"]
            circle.strokeColor = colors.white
            circle.strokeWidth = 0.5
            drawing.add(circle)


def _draw_endpoints(
    drawing: "Drawing",
    data: List[float],
    normalized: List[float],
    options: SparklineOptions,
    x_step: float,
    line_color: "Color",
) -> None:
    """Draw dots at the start and end points."""
    if len(data) < 1:
        return

    dot_radius = max(1.5, options.stroke_width)

    # Start point
    start_x = options.padding
    start_y = normalized[0]
    start_circle = Circle(start_x, start_y, dot_radius)
    start_circle.fillColor = line_color
    start_circle.strokeColor = colors.white
    start_circle.strokeWidth = 0.5
    drawing.add(start_circle)

    if len(data) > 1:
        # End point
        end_x = options.padding + (len(data) - 1) * x_step
        end_y = normalized[-1]
        end_circle = Circle(end_x, end_y, dot_radius)
        end_circle.fillColor = line_color
        end_circle.strokeColor = colors.white
        end_circle.strokeWidth = 0.5
        drawing.add(end_circle)


def _draw_reference_line(
    drawing: "Drawing",
    options: SparklineOptions,
    metrics: SparklineMetrics,
) -> None:
    """Draw a reference line (e.g., at zero or mean)."""
    if not options.show_reference_line:
        return

    ref_value = options.reference_value if options.reference_value is not None else metrics.mean_value

    # Normalize the reference value
    if metrics.max_value == metrics.min_value:
        ref_y = options.height / 2
    else:
        value_range = metrics.max_value - metrics.min_value
        available_height = options.height - 2 * options.padding
        ref_y = options.padding + ((ref_value - metrics.min_value) / value_range) * available_height

    # Clamp to visible area
    ref_y = max(options.padding, min(options.height - options.padding, ref_y))

    line = Line(
        options.padding,
        ref_y,
        options.width - options.padding,
        ref_y
    )
    line.strokeColor = SPARKLINE_COLORS["reference"]
    line.strokeWidth = 0.5
    line.strokeDashArray = [2, 2]
    drawing.add(line)


# =============================================================================
# Main Sparkline Creation Function
# =============================================================================

def create_sparkline(
    data: List[float],
    width: int = DEFAULT_SPARKLINE_WIDTH,
    height: int = DEFAULT_SPARKLINE_HEIGHT,
    style: Union[SparklineStyle, str] = SparklineStyle.LINE,
    line_color: Optional["Color"] = None,
    fill_color: Optional["Color"] = None,
    highlight_min: bool = False,
    highlight_max: bool = False,
    show_reference_line: bool = False,
    show_endpoints: bool = False,
    auto_color_trend: bool = True,
    **kwargs,
) -> "Drawing":
    """Create a compact sparkline chart.

    Generates a ReportLab Drawing containing a sparkline visualization of the
    provided data. Sparklines are designed to be compact enough to fit inline
    with text or within table cells.

    Args:
        data: List of numeric values to visualize.
        width: Width in points. Defaults to 60.
        height: Height in points. Defaults to 20.
        style: Visualization style ('line', 'area', 'bars', 'win_loss', 'dots')
            or SparklineStyle enum. Defaults to LINE.
        line_color: ReportLab color for the line/bars. Auto-selected based on
            trend if None and auto_color_trend is True.
        fill_color: Fill color for area style. Auto-selected if None.
        highlight_min: If True, highlight the minimum value with a red dot.
        highlight_max: If True, highlight the maximum value with a green dot.
        show_reference_line: If True, show a dashed reference line at the mean.
        show_endpoints: If True, show dots at the first and last points.
        auto_color_trend: If True, automatically color based on trend direction.
            Positive trends are green, negative are red. Defaults to True.
        **kwargs: Additional options passed to SparklineOptions.

    Returns:
        reportlab.graphics.shapes.Drawing: A Drawing object containing the
            sparkline chart. Can be added to a ReportLab story or rendered.

    Example:
        >>> # Simple line sparkline
        >>> sparkline = create_sparkline([10, 15, 12, 18, 14, 20])
        >>> story.append(sparkline)

        >>> # Area sparkline with highlighting
        >>> sparkline = create_sparkline(
        ...     [10, 15, 12, 18, 14, 20],
        ...     style='area',
        ...     highlight_min=True,
        ...     highlight_max=True,
        ... )

        >>> # Bar sparkline
        >>> sparkline = create_sparkline(
        ...     [5, -3, 8, -2, 10],
        ...     style='win_loss',
        ...     width=80,
        ...     height=25,
        ... )
    """
    # Convert string style to enum
    if isinstance(style, str):
        try:
            style = SparklineStyle(style.lower())
        except ValueError:
            logger.warning(f"Unknown sparkline style '{style}', using LINE")
            style = SparklineStyle.LINE

    # Create options
    options = SparklineOptions(
        width=width,
        height=height,
        style=style,
        line_color=line_color,
        fill_color=fill_color,
        highlight_min=highlight_min,
        highlight_max=highlight_max,
        show_reference_line=show_reference_line,
        show_endpoints=show_endpoints,
        auto_color_trend=auto_color_trend,
        **kwargs,
    )

    # Create the drawing
    drawing = Drawing(width, height)

    # Handle empty data
    if not data:
        logger.warning("Empty data provided for sparkline")
        return drawing

    # Filter out None values and convert to floats
    clean_data = []
    for v in data:
        if v is not None:
            try:
                clean_data.append(float(v))
            except (TypeError, ValueError):
                logger.warning(f"Skipping non-numeric value: {v}")

    if not clean_data:
        logger.warning("No valid numeric data for sparkline")
        return drawing

    # Calculate metrics
    metrics = calculate_sparkline_metrics(clean_data)

    # Draw reference line first (behind the data)
    _draw_reference_line(drawing, options, metrics)

    # Draw the sparkline based on style
    style_handlers = {
        SparklineStyle.LINE: _draw_line_sparkline,
        SparklineStyle.AREA: _draw_area_sparkline,
        SparklineStyle.BARS: _draw_bar_sparkline,
        SparklineStyle.WIN_LOSS: _draw_win_loss_sparkline,
        SparklineStyle.DOTS: _draw_dots_sparkline,
    }

    handler = style_handlers.get(options.style, _draw_line_sparkline)
    handler(drawing, clean_data, options, metrics)

    return drawing


def create_sparkline_with_label(
    data: List[float],
    label: str = "",
    label_position: str = "right",  # 'left', 'right', 'none'
    **kwargs,
) -> "Drawing":
    """Create a sparkline with an optional label showing the last value.

    Args:
        data: List of numeric values.
        label: Custom label text. If empty, shows the last value.
        label_position: Where to place the label ('left', 'right', 'none').
        **kwargs: Additional arguments passed to create_sparkline.

    Returns:
        Drawing with sparkline and label.
    """
    width = kwargs.get('width', DEFAULT_SPARKLINE_WIDTH)
    height = kwargs.get('height', DEFAULT_SPARKLINE_HEIGHT)

    if label_position == 'none':
        return create_sparkline(data, **kwargs)

    # Add space for label
    label_width = 25
    total_width = width + label_width if label_position in ('left', 'right') else width

    drawing = Drawing(total_width, height)

    # Create the sparkline
    sparkline_kwargs = {**kwargs}
    sparkline_kwargs['width'] = width
    sparkline = create_sparkline(data, **sparkline_kwargs)

    # Calculate offset for sparkline
    if label_position == 'left':
        sparkline_x = label_width
        label_x = 2
    else:  # right
        sparkline_x = 0
        label_x = width + 3

    # Add sparkline contents with offset using a Group
    group = Group()
    for item in sparkline.contents:
        group.add(item)
    group.transform = (1, 0, 0, 1, sparkline_x, 0)  # Translate transform
    drawing.add(group)

    # Add label
    label_text = label if label else (f"{data[-1]:.0f}" if data else "")
    if label_text:
        text = String(
            label_x,
            height / 2 - 3,
            label_text,
            fontName='Helvetica',
            fontSize=7,
            textAnchor='start' if label_position == 'left' else 'start',
            fillColor=colors.black
        )
        drawing.add(text)

    return drawing


def get_sparkline_metrics(data: List[float]) -> Dict[str, Any]:
    """Get summary metrics for sparkline data as a dictionary.

    Args:
        data: List of numeric values.

    Returns:
        Dictionary with min, max, mean, trend, and change percentage.

    Example:
        >>> metrics = get_sparkline_metrics([10, 15, 12, 18, 14, 20])
        >>> print(metrics['trend'])  # 'up'
        >>> print(metrics['change_percent'])  # 100.0
    """
    metrics = calculate_sparkline_metrics(data)
    return metrics.to_dict()


# =============================================================================
# Preset Sparkline Functions
# =============================================================================

def create_inline_sparkline(data: List[float], **kwargs) -> "Drawing":
    """Create a sparkline optimized for inline text use.

    Uses smaller dimensions suitable for embedding within paragraphs.
    """
    defaults = {
        'width': INLINE_SPARKLINE_WIDTH,
        'height': INLINE_SPARKLINE_HEIGHT,
        'stroke_width': 1.0,
    }
    return create_sparkline(data, **{**defaults, **kwargs})


def create_table_sparkline(data: List[float], **kwargs) -> "Drawing":
    """Create a sparkline optimized for table cells.

    Uses dimensions suitable for embedding within table cells.
    """
    defaults = {
        'width': TABLE_SPARKLINE_WIDTH,
        'height': TABLE_SPARKLINE_HEIGHT,
        'highlight_max': True,
        'stroke_width': 1.5,
    }
    return create_sparkline(data, **{**defaults, **kwargs})


def create_dashboard_sparkline(data: List[float], **kwargs) -> "Drawing":
    """Create a sparkline optimized for dashboard widgets.

    Uses larger dimensions suitable for dashboard display.
    """
    defaults = {
        'width': DASHBOARD_SPARKLINE_WIDTH,
        'height': DASHBOARD_SPARKLINE_HEIGHT,
        'highlight_min': True,
        'highlight_max': True,
        'show_endpoints': True,
        'stroke_width': 2.0,
    }
    return create_sparkline(data, **{**defaults, **kwargs})


# =============================================================================
# Demo / Test Function
# =============================================================================

if __name__ == "__main__":
    """Demo usage of the sparkline generator module."""
    import os
    from reportlab.graphics import renderPDF
    from reportlab.lib.pagesizes import letter
    from reportlab.pdfgen import canvas

    print("Sparkline Generator - Demo")
    print("=" * 60)

    # Generate sample data sets
    upward_trend = [10, 12, 11, 15, 14, 18, 17, 22, 21, 25]
    downward_trend = [25, 23, 24, 20, 21, 17, 18, 14, 15, 10]
    volatile_data = [15, 25, 10, 30, 5, 28, 12, 22, 8, 20]
    flat_data = [15, 16, 14, 15, 16, 15, 14, 16, 15, 15]
    win_loss_data = [5, -3, 8, -2, 10, -5, 7, -1, 12, -4]

    print("\nSample data sets generated:")
    print(f"  Upward trend: {upward_trend}")
    print(f"  Downward trend: {downward_trend}")
    print(f"  Volatile: {volatile_data}")
    print(f"  Flat: {flat_data}")
    print(f"  Win/Loss: {win_loss_data}")

    # Calculate and display metrics
    print("\n" + "-" * 40)
    print("Metrics for upward trend data:")
    metrics = get_sparkline_metrics(upward_trend)
    for key, value in metrics.items():
        print(f"  {key}: {value}")

    # Create sparklines in different styles
    print("\n" + "-" * 40)
    print("Creating sparklines in different styles...")

    sparklines = {}

    # Line sparkline
    sparklines['line'] = create_sparkline(upward_trend, style='line')
    print(f"  Line sparkline: {sparklines['line'].width}x{sparklines['line'].height}")

    # Area sparkline
    sparklines['area'] = create_sparkline(upward_trend, style='area', highlight_max=True)
    print(f"  Area sparkline: {sparklines['area'].width}x{sparklines['area'].height}")

    # Bar sparkline
    sparklines['bars'] = create_sparkline(upward_trend, style='bars', width=80)
    print(f"  Bar sparkline: {sparklines['bars'].width}x{sparklines['bars'].height}")

    # Win/Loss sparkline
    sparklines['win_loss'] = create_sparkline(win_loss_data, style='win_loss', width=80, height=25)
    print(f"  Win/Loss sparkline: {sparklines['win_loss'].width}x{sparklines['win_loss'].height}")

    # Dots sparkline
    sparklines['dots'] = create_sparkline(volatile_data, style='dots')
    print(f"  Dots sparkline: {sparklines['dots'].width}x{sparklines['dots'].height}")

    # Preset functions
    print("\n" + "-" * 40)
    print("Testing preset sparkline functions...")

    inline = create_inline_sparkline(upward_trend)
    print(f"  Inline sparkline: {inline.width}x{inline.height}")

    table = create_table_sparkline(upward_trend)
    print(f"  Table sparkline: {table.width}x{table.height}")

    dashboard = create_dashboard_sparkline(downward_trend)
    print(f"  Dashboard sparkline: {dashboard.width}x{dashboard.height}")

    # Save a demo PDF
    print("\n" + "-" * 40)
    try:
        output_path = os.path.join(os.path.dirname(__file__), "sparkline_demo.pdf")

        c = canvas.Canvas(output_path, pagesize=letter)
        page_width, page_height = letter

        # Title
        c.setFont("Helvetica-Bold", 16)
        c.drawString(50, page_height - 50, "Sparkline Generator Demo")

        c.setFont("Helvetica", 10)
        y_pos = page_height - 80

        demos = [
            ("Line Sparkline (Upward Trend)", create_sparkline(upward_trend, style='line', width=100, height=30)),
            ("Line Sparkline (Downward Trend)", create_sparkline(downward_trend, style='line', width=100, height=30)),
            ("Area Sparkline with Highlights", create_sparkline(upward_trend, style='area', width=100, height=30, highlight_min=True, highlight_max=True)),
            ("Bar Sparkline", create_sparkline(upward_trend, style='bars', width=100, height=30)),
            ("Win/Loss Sparkline", create_sparkline(win_loss_data, style='win_loss', width=100, height=30)),
            ("Dots Sparkline (Volatile Data)", create_sparkline(volatile_data, style='dots', width=100, height=30)),
            ("Dashboard Sparkline", create_dashboard_sparkline(upward_trend)),
            ("Table Sparkline", create_table_sparkline(flat_data)),
            ("With Reference Line", create_sparkline(volatile_data, style='line', width=100, height=30, show_reference_line=True)),
            ("With Endpoints", create_sparkline(upward_trend, style='line', width=100, height=30, show_endpoints=True)),
        ]

        for label, sparkline in demos:
            c.drawString(50, y_pos, label + ":")
            renderPDF.draw(sparkline, c, 200, y_pos - 10)
            y_pos -= 50

            if y_pos < 100:
                c.showPage()
                y_pos = page_height - 50

        c.save()
        print(f"Demo PDF saved to: {output_path}")
    except Exception as e:
        print(f"Could not save demo PDF: {e}")

    print("\n" + "=" * 60)
    print("Sparkline Generator module loaded successfully!")
    print("=" * 60)
