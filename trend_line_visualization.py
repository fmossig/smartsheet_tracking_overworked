"""
Trend Line Visualization Module

Provides functions for drawing trend line overlays on time-series bar charts.
Uses dashed lines and different colors to distinguish trend lines from actual data.

This module integrates with the existing chart generation system in smartsheet_report.py
and uses the trend_line_calculator module for computing trend data.

Usage:
    from trend_line_visualization import (
        make_time_series_bar_chart,
        make_group_activity_trend_chart,
        draw_trend_line_overlay,
        add_trend_line_legend,
        TREND_LINE_COLORS,
        TREND_LINE_STYLES,
    )
"""

import logging
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from reportlab.lib.colors import Color
    from reportlab.graphics.shapes import Drawing

from reportlab.lib import colors
from reportlab.graphics.shapes import Drawing, String, Line
from reportlab.graphics.charts.barcharts import VerticalBarChart

logger = logging.getLogger(__name__)


# =============================================================================
# Trend Line Style Constants
# =============================================================================

# Trend line colors - WCAG AA compliant
TREND_LINE_COLORS = {
    "regression": colors.HexColor("#E53935"),     # Red 600 - Linear regression line
    "sma": colors.HexColor("#1E88E5"),            # Blue 600 - Simple Moving Average
    "ema": colors.HexColor("#43A047"),            # Green 600 - Exponential Moving Average
}

# Trend line dash patterns and widths
TREND_LINE_STYLES = {
    "regression": {
        "strokeDashArray": [6, 3],  # Dashed line
        "strokeWidth": 2,
    },
    "sma": {
        "strokeDashArray": [4, 2],  # Short dashes
        "strokeWidth": 1.5,
    },
    "ema": {
        "strokeDashArray": [2, 2],  # Dotted line
        "strokeWidth": 1.5,
    },
}


# =============================================================================
# Trend Line Drawing Functions
# =============================================================================

def draw_trend_line_overlay(
    drawing: "Drawing",
    trend_points: List[Tuple[float, float]],
    chart_x: float,
    chart_y: float,
    chart_width: float,
    chart_height: float,
    line_color: "Color" = None,
    stroke_width: float = 2,
    stroke_dash_array: Optional[List[int]] = None,
) -> None:
    """Draw a trend line overlay on an existing chart.

    Draws a polyline connecting all trend points on top of the chart area.
    The line can be customized with color, width, and dash patterns.

    Args:
        drawing: The ReportLab Drawing object to add the trend line to.
        trend_points: List of (x, y) tuples in chart coordinates.
            x values should be in the range [0, num_bars-1]
            y values should be in the original data scale
        chart_x: X coordinate of the chart origin.
        chart_y: Y coordinate of the chart origin.
        chart_width: Width of the chart area.
        chart_height: Height of the chart area.
        line_color: ReportLab color for the line. Defaults to red.
        stroke_width: Line thickness. Defaults to 2.
        stroke_dash_array: Dash pattern like [6, 3] for dashed lines.

    Returns:
        None. Modifies the drawing in place.
    """
    if not trend_points or len(trend_points) < 2:
        return

    if line_color is None:
        line_color = TREND_LINE_COLORS["regression"]

    # Draw lines connecting each consecutive pair of points
    for i in range(len(trend_points) - 1):
        x1, y1 = trend_points[i]
        x2, y2 = trend_points[i + 1]

        line = Line(x1, y1, x2, y2)
        line.strokeColor = line_color
        line.strokeWidth = stroke_width
        if stroke_dash_array:
            line.strokeDashArray = stroke_dash_array

        drawing.add(line)


def add_trend_line_legend(
    drawing: "Drawing",
    x: float,
    y: float,
    show_regression: bool = True,
    show_sma: bool = False,
    show_ema: bool = False,
    trend_direction: Optional[str] = None,
    r_squared: Optional[float] = None,
) -> float:
    """Add a trend line legend to a chart.

    Displays legend entries for the trend lines shown on the chart,
    including optional trend direction and R-squared values.

    Args:
        drawing: The ReportLab Drawing object to add the legend to.
        x: X coordinate for the legend start.
        y: Y coordinate for the legend start.
        show_regression: Show linear regression line in legend.
        show_sma: Show Simple Moving Average line in legend.
        show_ema: Show Exponential Moving Average line in legend.
        trend_direction: Optional trend direction text (e.g., "up", "down", "flat").
        r_squared: Optional R-squared value (0-1) showing trend fit quality.

    Returns:
        The y-offset used (height of the legend) for positioning.
    """
    legend_items = []

    if show_regression:
        legend_items.append(("Trend Line", TREND_LINE_COLORS["regression"], TREND_LINE_STYLES["regression"]))
    if show_sma:
        legend_items.append(("SMA (7-day)", TREND_LINE_COLORS["sma"], TREND_LINE_STYLES["sma"]))
    if show_ema:
        legend_items.append(("EMA (7-day)", TREND_LINE_COLORS["ema"], TREND_LINE_STYLES["ema"]))

    if not legend_items:
        return 0

    # Calculate legend dimensions
    legend_width = 100
    legend_height = len(legend_items) * 12 + 8
    if trend_direction or r_squared is not None:
        legend_height += 14

    # Draw legend entries
    current_y = y
    for label, color, style in legend_items:
        # Draw line sample
        line_sample = Line(x + 5, current_y + 4, x + 25, current_y + 4)
        line_sample.strokeColor = color
        line_sample.strokeWidth = style["strokeWidth"]
        line_sample.strokeDashArray = style["strokeDashArray"]
        drawing.add(line_sample)

        # Draw label
        drawing.add(String(
            x + 30,
            current_y,
            label,
            fontName='Helvetica',
            fontSize=7,
            textAnchor='start',
            fillColor=colors.black
        ))
        current_y -= 12

    # Add trend info if provided
    if trend_direction or r_squared is not None:
        info_parts = []
        if trend_direction:
            direction_symbols = {
                "strong_up": "↑↑",
                "up": "↑",
                "flat": "→",
                "down": "↓",
                "strong_down": "↓↓",
            }
            symbol = direction_symbols.get(trend_direction, "")
            info_parts.append(f"Trend: {symbol}")
        if r_squared is not None:
            info_parts.append(f"R²: {r_squared:.2f}")

        info_text = " | ".join(info_parts)
        drawing.add(String(
            x + 5,
            current_y - 2,
            info_text,
            fontName='Helvetica',
            fontSize=6,
            textAnchor='start',
            fillColor=colors.gray
        ))

    return legend_height


# =============================================================================
# Time-Series Bar Chart with Trend Lines
# =============================================================================

def make_time_series_bar_chart(
    daily_counts: Dict[str, int],
    title: str,
    width: int = 400,
    height: int = 250,
    show_labels: bool = True,
    show_trend_line: bool = True,
    show_sma: bool = False,
    show_ema: bool = False,
    ma_window_size: int = 7,
    bar_color: "Color" = None,
    max_bars: int = 14,
) -> "Drawing":
    """Create a time-series bar chart with optional trend line overlays.

    Generates a ReportLab Drawing with a bar chart displaying daily activity
    counts over time, with optional linear regression trend line and moving
    average overlays to visualize activity trends.

    Args:
        daily_counts: Dictionary mapping date strings (YYYY-MM-DD) to counts.
            Example: {"2026-01-01": 10, "2026-01-02": 15, "2026-01-03": 12}
        title: Chart title displayed at the top center.
        width: Chart width in points. Defaults to 400.
        height: Chart height in points. Defaults to 250.
        show_labels: If True, display value labels on top of bars. Defaults to True.
        show_trend_line: If True, overlay a linear regression trend line. Defaults to True.
        show_sma: If True, overlay a simple moving average line. Defaults to False.
        show_ema: If True, overlay an exponential moving average line. Defaults to False.
        ma_window_size: Window size for moving average calculations. Defaults to 7.
        bar_color: ReportLab color for bars. Defaults to steelblue.
        max_bars: Maximum number of bars to display. Shows most recent dates. Defaults to 14.

    Returns:
        reportlab.graphics.shapes.Drawing: A Drawing object containing the
            time-series bar chart with trend lines. Can be added to a ReportLab story.

    Note:
        - Dates are sorted chronologically
        - If more than max_bars dates exist, only the most recent are shown
        - Trend line is dashed red; SMA is dashed blue; EMA is dotted green
        - Legend is added when any trend line is shown

    Example:
        >>> from historical_data_loader import get_daily_change_counts
        >>> counts = get_daily_change_counts(changes)
        >>> chart = make_time_series_bar_chart(counts, "Daily Activity Trend")
        >>> story.append(chart)
    """
    # Import trend line calculator functions
    from trend_line_calculator import (
        create_data_points_from_daily_counts,
        get_trend_visualization_data,
    )

    drawing = Drawing(width, height)

    # Add title
    drawing.add(String(
        width / 2, height - 15, title,
        fontName='Helvetica-Bold', fontSize=12, textAnchor='middle'
    ))

    # Handle empty data
    if not daily_counts:
        drawing.add(String(
            width / 2, height / 2,
            "No data available",
            fontName='Helvetica',
            fontSize=10,
            textAnchor='middle',
            fillColor=colors.gray
        ))
        return drawing

    # Sort dates and limit to max_bars (most recent)
    sorted_dates = sorted(daily_counts.keys())
    if len(sorted_dates) > max_bars:
        sorted_dates = sorted_dates[-max_bars:]

    # Prepare data for the chart
    data_values = [daily_counts[d] for d in sorted_dates]

    # Create short date labels (MM/DD)
    date_labels = []
    for d in sorted_dates:
        try:
            dt = datetime.strptime(d, "%Y-%m-%d")
            date_labels.append(dt.strftime("%m/%d"))
        except ValueError:
            date_labels.append(d[-5:])  # Fallback: last 5 chars

    # Set default bar color
    if bar_color is None:
        bar_color = colors.steelblue

    # Create the bar chart
    chart = VerticalBarChart()
    chart.x = 45
    chart.y = 45
    chart.height = height - 100
    chart.width = width - 80

    chart.data = [data_values]
    chart.categoryAxis.categoryNames = date_labels

    # Style the category axis (dates)
    num_bars = len(sorted_dates)
    if num_bars > 10:
        chart.categoryAxis.labels.fontSize = 6
        chart.categoryAxis.labels.angle = 45
        chart.categoryAxis.labels.dy = -12
    else:
        chart.categoryAxis.labels.fontSize = 7
        chart.categoryAxis.labels.angle = 0
        chart.categoryAxis.labels.dy = -8
    chart.categoryAxis.labels.boxAnchor = 'n'

    # Style the value axis
    max_val = max(data_values) if data_values else 10
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = max_val * 1.15  # Extra room for trend line
    chart.valueAxis.valueStep = max(1, int(max_val / 5))
    chart.valueAxis.labels.fontSize = 7

    # Set bar colors
    chart.bars[0].fillColor = bar_color

    drawing.add(chart)

    # Calculate trend line data if needed
    trend_data = None
    if show_trend_line or show_sma or show_ema:
        # Create subset of daily_counts for visible dates only
        visible_counts = {d: daily_counts[d] for d in sorted_dates}
        data_points = create_data_points_from_daily_counts(visible_counts)

        if len(data_points) >= 2:
            trend_data = get_trend_visualization_data(
                data_points,
                include_regression=show_trend_line,
                include_sma=show_sma,
                include_ema=show_ema,
                ma_window_size=ma_window_size
            )

    # Draw trend lines
    if trend_data:
        bar_width = chart.width / num_bars

        # Helper function to convert data points to chart coordinates
        def get_chart_points(values: List[Optional[float]]) -> List[Tuple[float, float]]:
            points = []
            for i, val in enumerate(values):
                if val is not None:
                    x = chart.x + (i + 0.5) * bar_width
                    y = chart.y + (val / max_val) * chart.height
                    points.append((x, y))
            return points

        # Draw linear regression trend line
        if show_trend_line and trend_data["chart_data"]["trend_line"]:
            trend_points = get_chart_points(trend_data["chart_data"]["trend_line"])
            draw_trend_line_overlay(
                drawing,
                trend_points,
                chart.x, chart.y, chart.width, chart.height,
                line_color=TREND_LINE_COLORS["regression"],
                stroke_width=TREND_LINE_STYLES["regression"]["strokeWidth"],
                stroke_dash_array=TREND_LINE_STYLES["regression"]["strokeDashArray"],
            )

        # Draw SMA line
        if show_sma and trend_data["chart_data"]["sma"]:
            sma_points = get_chart_points(trend_data["chart_data"]["sma"])
            draw_trend_line_overlay(
                drawing,
                sma_points,
                chart.x, chart.y, chart.width, chart.height,
                line_color=TREND_LINE_COLORS["sma"],
                stroke_width=TREND_LINE_STYLES["sma"]["strokeWidth"],
                stroke_dash_array=TREND_LINE_STYLES["sma"]["strokeDashArray"],
            )

        # Draw EMA line
        if show_ema and trend_data["chart_data"]["ema"]:
            ema_points = get_chart_points(trend_data["chart_data"]["ema"])
            draw_trend_line_overlay(
                drawing,
                ema_points,
                chart.x, chart.y, chart.width, chart.height,
                line_color=TREND_LINE_COLORS["ema"],
                stroke_width=TREND_LINE_STYLES["ema"]["strokeWidth"],
                stroke_dash_array=TREND_LINE_STYLES["ema"]["strokeDashArray"],
            )

        # Add legend
        legend_x = chart.x + chart.width - 90
        legend_y = chart.y + chart.height - 5
        add_trend_line_legend(
            drawing,
            legend_x, legend_y,
            show_regression=show_trend_line,
            show_sma=show_sma,
            show_ema=show_ema,
            trend_direction=trend_data["metrics"].get("trend_direction") if trend_data["metrics"] else None,
            r_squared=trend_data["metrics"].get("r_squared") if trend_data["metrics"] else None,
        )

    # Add data labels on top of bars
    if show_labels:
        bar_width = chart.width / num_bars
        for i, value in enumerate(data_values):
            if value > 0:
                bar_x = chart.x + (i + 0.5) * bar_width
                bar_y = chart.y + (value / max_val) * chart.height

                drawing.add(String(
                    bar_x,
                    bar_y + 3,
                    str(value),
                    fontName='Helvetica-Bold',
                    fontSize=6,
                    textAnchor='middle',
                    fillColor=colors.black
                ))

    return drawing


def make_group_activity_trend_chart(
    changes: List[Dict[str, Any]],
    title: str = "Group Activity Trend",
    width: int = 450,
    height: int = 280,
    days: int = 14,
    show_trend_line: bool = True,
    show_sma: bool = False,
    group: Optional[str] = None,
    group_colors: Optional[Dict[str, "Color"]] = None,
) -> "Drawing":
    """Create a group activity trend chart from change records.

    Convenience function that processes change records to create a time-series
    bar chart showing activity over time for a specific group or all groups.

    Args:
        changes: List of change records from historical_data_loader.
        title: Chart title. Defaults to "Group Activity Trend".
        width: Chart width in points. Defaults to 450.
        height: Chart height in points. Defaults to 280.
        days: Number of days to show. Defaults to 14.
        show_trend_line: If True, show linear regression trend line.
        show_sma: If True, show 7-day simple moving average.
        group: Optional group code to filter by (e.g., "NF", "NA").
        group_colors: Optional dictionary mapping group codes to colors.

    Returns:
        reportlab.graphics.shapes.Drawing: A Drawing object containing the chart.

    Example:
        >>> from historical_data_loader import load_last_n_days
        >>> changes = load_last_n_days(14)
        >>> chart = make_group_activity_trend_chart(changes, "NF Activity Trend", group="NF")
        >>> story.append(chart)
    """
    from historical_data_loader import get_daily_change_counts, filter_by_group

    # Filter by group if specified
    if group:
        changes = filter_by_group(changes, group)
        if not title.endswith(" Trend"):
            title = f"{title} ({group})"

    # Get daily counts
    daily_counts = get_daily_change_counts(changes)

    # Get the group color if specified
    bar_color = colors.steelblue
    if group and group_colors:
        bar_color = group_colors.get(group, colors.steelblue)

    return make_time_series_bar_chart(
        daily_counts,
        title,
        width=width,
        height=height,
        show_trend_line=show_trend_line,
        show_sma=show_sma,
        bar_color=bar_color,
        max_bars=days,
    )


# =============================================================================
# Demo / Test Function
# =============================================================================

if __name__ == "__main__":
    """Demo usage of the trend line visualization module."""
    from datetime import date, timedelta
    import os

    print("Trend Line Visualization - Demo")
    print("=" * 60)

    # Generate sample data with an upward trend
    base_date = date(2026, 1, 1)
    sample_counts = {}
    for i in range(14):
        d = base_date + timedelta(days=i)
        # Create upward trend with some noise
        value = 50 + i * 3 + (i % 5 - 2) * 5
        sample_counts[d.strftime("%Y-%m-%d")] = max(10, value)

    print(f"Generated {len(sample_counts)} sample data points")
    print(f"Date range: {min(sample_counts.keys())} to {max(sample_counts.keys())}")
    print(f"Value range: {min(sample_counts.values())} to {max(sample_counts.values())}")

    # Create a chart with trend line
    print("\nCreating time-series bar chart with trend line...")
    chart = make_time_series_bar_chart(
        sample_counts,
        "Daily Activity Trend (Demo)",
        show_trend_line=True,
        show_sma=True,
    )

    print(f"Chart created: {chart.width}x{chart.height} points")
    print(f"Chart contents: {len(chart.contents)} elements")

    # Save the chart as a standalone PDF
    try:
        from reportlab.graphics import renderPDF
        output_path = os.path.join(os.path.dirname(__file__), "trend_line_demo.pdf")
        renderPDF.drawToFile(chart, output_path)
        print(f"\nDemo chart saved to: {output_path}")
    except Exception as e:
        print(f"\nCould not save demo PDF: {e}")

    print("\n" + "=" * 60)
    print("Trend Line Visualization module loaded successfully.")
