"""
Pie Chart Label Positioning Utilities

This module provides intelligent label positioning for pie and donut charts
to avoid overlaps and maintain readability. Labels are positioned using
collision detection and repositioning strategies.
"""

import math
from reportlab.lib import colors
from reportlab.graphics.shapes import String, Line
from reportlab.pdfbase.pdfmetrics import stringWidth

from constants import PieChartConstants, DEFAULT_LEADER_LINE_COLOR


def calculate_pie_label_positions(chart_data, total_value, pie_cx, pie_cy, pie_radius,
                                   min_percentage=5.0, font_name='Helvetica-Bold', font_size=8,
                                   label_format='percentage', show_hours=False):
    """Calculate intelligent label positions for pie chart segments to avoid overlap.

    This function computes optimal label positions for pie chart percentage labels,
    using collision detection and repositioning strategies to ensure readability.

    Args:
        chart_data: List of (label, value) tuples for each slice
        total_value: Total value across all slices (for percentage calculation)
        pie_cx: X coordinate of pie center
        pie_cy: Y coordinate of pie center
        pie_radius: Radius of the pie chart
        min_percentage: Minimum percentage threshold for displaying labels (default: 5.0)
        font_name: Font name for label text (default: 'Helvetica-Bold')
        font_size: Font size for label text (default: 8)
        label_format: Format for labels - 'percentage' (default), 'hours', or 'both'
        show_hours: If True, include hours value in label (deprecated, use label_format='both')

    Returns:
        List of dictionaries containing label positioning info:
        [
            {
                'index': int,           # Index in chart_data
                'percentage': float,    # Percentage value
                'value': float,         # Raw value (hours)
                'label_x': float,       # Final X position
                'label_y': float,       # Final Y position
                'text': str,            # Label text (e.g., "25%" or "25% (5.0h)")
                'has_leader_line': bool,# True if label was moved outside with leader line
                'leader_start_x': float,# Leader line start X (if applicable)
                'leader_start_y': float,# Leader line start Y (if applicable)
                'mid_angle': float,     # Midpoint angle in degrees
            },
            ...
        ]
    """
    if total_value <= 0:
        return []

    label_info = []
    cumulative_angle = PieChartConstants.PIE_START_ANGLE  # Pie charts start at 90 degrees (top)

    # First pass: calculate initial positions for all qualifying labels
    for i, (label, value) in enumerate(chart_data):
        if value <= 0:
            slice_angle = 0
            cumulative_angle -= slice_angle
            continue

        percentage = (value / total_value) * PieChartConstants.PERCENTAGE_MULTIPLIER
        slice_angle = (value / total_value) * PieChartConstants.FULL_CIRCLE_DEGREES

        # Only create labels for slices meeting minimum threshold
        if percentage >= min_percentage:
            mid_angle = cumulative_angle - (slice_angle / 2)
            mid_angle_rad = math.radians(mid_angle)

            # Initial position: 60% of radius from center (inside the slice)
            label_radius = pie_radius * PieChartConstants.INITIAL_LABEL_RADIUS_RATIO
            label_x = pie_cx + label_radius * math.cos(mid_angle_rad)
            label_y = pie_cy + label_radius * math.sin(mid_angle_rad)

            # Format label text based on format option
            if label_format == 'both' or show_hours:
                label_text = f"{percentage:.0f}% ({value:.1f}h)"
            elif label_format == 'hours':
                label_text = f"{value:.1f}h"
            else:  # 'percentage' (default)
                label_text = f"{percentage:.0f}%"

            label_width = stringWidth(label_text, font_name, font_size)
            label_height = font_size

            label_info.append({
                'index': i,
                'percentage': percentage,
                'value': value,
                'label_x': label_x,
                'label_y': label_y,
                'text': label_text,
                'width': label_width,
                'height': label_height,
                'has_leader_line': False,
                'leader_start_x': None,
                'leader_start_y': None,
                'mid_angle': mid_angle,
                'mid_angle_rad': mid_angle_rad,
                'slice_angle': slice_angle,
            })

        cumulative_angle -= slice_angle

    # Second pass: detect and resolve overlaps
    label_info = _resolve_label_overlaps(label_info, pie_cx, pie_cy, pie_radius, font_size)

    return label_info


def _get_label_bounds(label):
    """Get bounding box for a label as (x_min, y_min, x_max, y_max)."""
    half_width = label['width'] / 2
    half_height = label['height'] / 2
    return (
        label['label_x'] - half_width,
        label['label_y'] - half_height,
        label['label_x'] + half_width,
        label['label_y'] + half_height
    )


def _labels_overlap(label1, label2, padding=PieChartConstants.LABEL_OVERLAP_PADDING):
    """Check if two labels overlap with optional padding."""
    b1 = _get_label_bounds(label1)
    b2 = _get_label_bounds(label2)

    # Add padding to bounds
    b1 = (b1[0] - padding, b1[1] - padding, b1[2] + padding, b1[3] + padding)

    # Check for overlap
    return not (b1[2] < b2[0] or  # label1 is left of label2
                b1[0] > b2[2] or  # label1 is right of label2
                b1[3] < b2[1] or  # label1 is below label2
                b1[1] > b2[3])    # label1 is above label2


def _resolve_label_overlaps(label_info, pie_cx, pie_cy, pie_radius, font_size):
    """Resolve overlapping labels by repositioning them intelligently.

    Strategy:
    1. For small slices that overlap, try moving to outer position
    2. Use leader lines for labels moved outside the slice
    3. Apply vertical offset for remaining overlaps
    """
    if len(label_info) <= 1:
        return label_info

    # Sort by percentage (largest first) - larger slices keep preferred positions
    sorted_labels = sorted(label_info, key=lambda x: x['percentage'], reverse=True)
    resolved_labels = []

    for label in sorted_labels:
        needs_repositioning = False

        # Check if this label overlaps with any already-placed label
        for placed_label in resolved_labels:
            if _labels_overlap(label, placed_label):
                needs_repositioning = True
                break

        if needs_repositioning:
            # Try repositioning strategies
            label = _reposition_label(label, resolved_labels, pie_cx, pie_cy, pie_radius, font_size)

        resolved_labels.append(label)

    # Restore original order by index
    return sorted(resolved_labels, key=lambda x: x['index'])


def _reposition_label(label, placed_labels, pie_cx, pie_cy, pie_radius, font_size):
    """Reposition a label to avoid overlap with placed labels."""
    mid_angle_rad = label['mid_angle_rad']

    # Strategy 1: Move to outer position with leader line
    outer_radius = pie_radius * PieChartConstants.OUTER_LABEL_RADIUS_RATIO
    outer_x = pie_cx + outer_radius * math.cos(mid_angle_rad)
    outer_y = pie_cy + outer_radius * math.sin(mid_angle_rad)

    outer_label = label.copy()
    outer_label['label_x'] = outer_x
    outer_label['label_y'] = outer_y
    outer_label['has_leader_line'] = True
    outer_label['leader_start_x'] = pie_cx + (pie_radius * PieChartConstants.LEADER_LINE_START_RATIO) * math.cos(mid_angle_rad)
    outer_label['leader_start_y'] = pie_cy + (pie_radius * PieChartConstants.LEADER_LINE_START_RATIO) * math.sin(mid_angle_rad)

    # Check if outer position works
    overlaps_outer = False
    for placed in placed_labels:
        if _labels_overlap(outer_label, placed):
            overlaps_outer = True
            break

    if not overlaps_outer:
        return outer_label

    # Strategy 2: Apply vertical offset to outer position
    for offset_multiplier in [1, -1, 2, -2]:
        offset = font_size * PieChartConstants.LABEL_OFFSET_MULTIPLIER * offset_multiplier
        offset_label = outer_label.copy()
        offset_label['label_y'] = outer_y + offset

        overlaps = False
        for placed in placed_labels:
            if _labels_overlap(offset_label, placed):
                overlaps = True
                break

        if not overlaps:
            return offset_label

    # Strategy 3: Move even further out
    far_outer_radius = pie_radius * PieChartConstants.FAR_OUTER_RADIUS_RATIO
    far_outer_x = pie_cx + far_outer_radius * math.cos(mid_angle_rad)
    far_outer_y = pie_cy + far_outer_radius * math.sin(mid_angle_rad)

    far_outer_label = label.copy()
    far_outer_label['label_x'] = far_outer_x
    far_outer_label['label_y'] = far_outer_y
    far_outer_label['has_leader_line'] = True
    far_outer_label['leader_start_x'] = pie_cx + (pie_radius * PieChartConstants.LEADER_LINE_START_RATIO) * math.cos(mid_angle_rad)
    far_outer_label['leader_start_y'] = pie_cy + (pie_radius * PieChartConstants.LEADER_LINE_START_RATIO) * math.sin(mid_angle_rad)

    return far_outer_label


def draw_pie_labels(drawing, label_positions, get_text_color_func, leader_line_color=None):
    """Draw percentage labels on a pie chart with optional leader lines.

    Args:
        drawing: ReportLab Drawing object to add labels to
        label_positions: List of label position dicts from calculate_pie_label_positions
        get_text_color_func: Function that takes label index and returns text color
        leader_line_color: Color for leader lines (default: dark gray)
    """
    if leader_line_color is None:
        leader_line_color = colors.HexColor(DEFAULT_LEADER_LINE_COLOR)

    for label in label_positions:
        # Draw leader line if needed
        if label['has_leader_line'] and label['leader_start_x'] is not None:
            drawing.add(Line(
                label['leader_start_x'],
                label['leader_start_y'],
                label['label_x'],
                label['label_y'],
                strokeColor=leader_line_color,
                strokeWidth=0.5
            ))

        # Get text color from the provided function
        text_color = get_text_color_func(label['index'])

        # If label has been moved outside, use dark text color for readability
        if label['has_leader_line']:
            text_color = colors.HexColor("#333333")

        # Draw the label
        drawing.add(String(
            label['label_x'],
            label['label_y'],
            label['text'],
            fontName='Helvetica-Bold',
            fontSize=8,
            textAnchor='middle',
            fillColor=text_color
        ))


def calculate_gauge_label_positions(status_values, total_value, cx, cy, radius,
                                     min_percentage=10.0, font_name='Helvetica-Bold', font_size=8):
    """Calculate intelligent label positions for gauge chart segments.

    Similar to calculate_pie_label_positions but for half-circle gauge charts.

    Args:
        status_values: Dictionary mapping status names to values
        total_value: Total value across all statuses
        cx: X coordinate of gauge center
        cy: Y coordinate of gauge center
        radius: Radius of the gauge
        min_percentage: Minimum percentage threshold for displaying labels (default: 10.0)
        font_name: Font name for label text
        font_size: Font size for label text

    Returns:
        List of label position dictionaries similar to calculate_pie_label_positions
    """
    if total_value <= 0:
        return []

    label_info = []
    start_angle = 180  # Gauge starts from the left

    for i, (status, value) in enumerate(status_values.items()):
        if value <= 0:
            continue

        percentage = (value / total_value) * 100
        angle_extent = (value / total_value) * 180

        if percentage >= min_percentage:
            mid_angle = start_angle - (angle_extent / 2)
            mid_angle_rad = math.radians(mid_angle)

            # Initial position: 60% of radius from center
            label_radius = radius * 0.6
            label_x = cx + label_radius * math.cos(mid_angle_rad)
            label_y = cy + label_radius * math.sin(mid_angle_rad)

            label_text = f"{percentage:.0f}%"
            label_width = stringWidth(label_text, font_name, font_size)
            label_height = font_size

            label_info.append({
                'index': i,
                'status': status,
                'percentage': percentage,
                'label_x': label_x,
                'label_y': label_y,
                'text': label_text,
                'width': label_width,
                'height': label_height,
                'has_leader_line': False,
                'leader_start_x': None,
                'leader_start_y': None,
                'mid_angle': mid_angle,
                'mid_angle_rad': mid_angle_rad,
                'angle_extent': angle_extent,
            })

        start_angle -= angle_extent

    # Resolve overlaps using the same algorithm
    label_info = _resolve_label_overlaps(label_info, cx, cy, radius, font_size)

    return label_info
