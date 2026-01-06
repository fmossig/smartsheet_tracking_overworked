"""
Special Activities Comparison Display Module

Creates visual comparison sections showing period-over-period changes for special activities.
Uses arrows and percentage changes for key metrics in the PDF report.

This module provides functions to create reportlab Flowables that display:
- Summary metrics with trend arrows and percentage changes
- Top category movers (increases and decreases)
- New and dropped category information
"""

import logging
from typing import List, TYPE_CHECKING

if TYPE_CHECKING:
    from reportlab.platypus import Flowable

from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import Paragraph, Spacer, Table, TableStyle
from reportlab.lib.styles import getSampleStyleSheet

from special_activities_category_breakdown import SpecialActivitiesPeriodComparison
from period_comparison_calculator import (
    TrendDirection,
    format_percent_change,
)

logger = logging.getLogger(__name__)


# Define colors for trends - WCAG compliant
TREND_UP_COLOR = colors.HexColor("#059669")    # Emerald green - good contrast
TREND_DOWN_COLOR = colors.HexColor("#DC2626")  # Red - good contrast
TREND_FLAT_COLOR = colors.HexColor("#6B7280")  # Gray - neutral

# Table styling colors - WCAG compliant with good visual hierarchy
TABLE_HEADER_BG = colors.HexColor("#1F2937")         # Dark grey header background
TABLE_HEADER_TEXT = colors.white                      # White text for header
TABLE_HEADER_SECONDARY_BG = colors.HexColor("#374151")  # Slightly lighter header
TABLE_ROW_ALT_BG = colors.HexColor("#F9FAFB")         # Light grey for alternating rows
TABLE_ROW_EVEN_BG = colors.white                      # White for even rows
TABLE_FOOTER_BG = colors.HexColor("#E5E7EB")          # Light grey for footer/total rows
TABLE_BORDER_COLOR = colors.HexColor("#D1D5DB")       # Medium grey for borders


def get_trend_color(trend: TrendDirection):
    """Get the appropriate color for a trend direction.

    Args:
        trend: TrendDirection enum value

    Returns:
        reportlab Color object for the trend
    """
    if trend == TrendDirection.UP:
        return TREND_UP_COLOR
    elif trend == TrendDirection.DOWN:
        return TREND_DOWN_COLOR
    else:
        return TREND_FLAT_COLOR


def get_arrow_symbol(trend: TrendDirection) -> str:
    """Get arrow symbol for trend direction.

    Args:
        trend: TrendDirection enum value

    Returns:
        Unicode arrow string
    """
    if trend == TrendDirection.UP:
        return "↑"
    elif trend == TrendDirection.DOWN:
        return "↓"
    else:
        return "→"


def apply_alternating_row_colors(table_style: list, num_data_rows: int, start_row: int = 1) -> list:
    """Apply alternating row colors to a table for improved readability.

    Args:
        table_style: List of TableStyle commands to append to
        num_data_rows: Number of data rows (excluding header)
        start_row: Row index to start alternating (default 1, skips header)

    Returns:
        Modified table_style list with alternating colors added
    """
    for i in range(num_data_rows):
        row_idx = start_row + i
        if i % 2 == 0:  # Even data rows (0, 2, 4...)
            table_style.append(('BACKGROUND', (0, row_idx), (-1, row_idx), TABLE_ROW_EVEN_BG))
        else:  # Odd data rows (1, 3, 5...)
            table_style.append(('BACKGROUND', (0, row_idx), (-1, row_idx), TABLE_ROW_ALT_BG))
    return table_style


def get_enhanced_table_base_style(has_footer: bool = False, num_rows: int = 0) -> list:
    """Get base table style with enhanced visual appearance.

    Args:
        has_footer: Whether the table has a footer/total row
        num_rows: Total number of rows including header

    Returns:
        List of TableStyle commands for base styling
    """
    style = [
        # Grid with softer border color
        ('GRID', (0, 0), (-1, -1), 0.5, TABLE_BORDER_COLOR),
        # Thicker line below header
        ('LINEBELOW', (0, 0), (-1, 0), 1.5, TABLE_HEADER_BG),
        # Header styling
        ('BACKGROUND', (0, 0), (-1, 0), TABLE_HEADER_BG),
        ('TEXTCOLOR', (0, 0), (-1, 0), TABLE_HEADER_TEXT),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10),
        # Data row styling
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        # Consistent padding
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 8),
        ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        # Vertical alignment
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]

    if has_footer and num_rows > 0:
        # Footer/total row styling
        style.extend([
            ('LINEABOVE', (0, -1), (-1, -1), 1.5, TABLE_HEADER_BG),
            ('BACKGROUND', (0, -1), (-1, -1), TABLE_FOOTER_BG),
            ('FONTNAME', (0, -1), (-1, -1), 'Helvetica-Bold'),
        ])

    return style


def create_special_activities_comparison_section(
    comparison: SpecialActivitiesPeriodComparison,
    width: int = 500,
) -> List["Flowable"]:
    """Create a comparison section showing period-over-period changes for special activities.

    This function creates a visual comparison section with:
    - Summary metrics with arrows and percentage changes
    - Metric cards showing current vs previous period
    - Top movers (categories with biggest increases/decreases)

    Args:
        comparison: SpecialActivitiesPeriodComparison object with period data
        width: Width of the section in points

    Returns:
        List of reportlab Flowables to add to the story

    Example:
        >>> comparison = compare_special_activities_periods(current, previous, ...)
        >>> flowables = create_special_activities_comparison_section(comparison)
        >>> story.extend(flowables)
    """
    flowables = []

    # Get styles
    styles = getSampleStyleSheet()
    subheading_style = styles['Heading3']
    normal_style = styles['Normal']

    # Section header
    flowables.append(Paragraph("Period-over-Period Comparison", subheading_style))
    flowables.append(Spacer(1, 3*mm))

    # Period info text
    period_text = (
        f"Comparing {comparison.current_period_start.strftime('%b %d')} - "
        f"{comparison.current_period_end.strftime('%b %d')} vs "
        f"{comparison.previous_period_start.strftime('%b %d')} - "
        f"{comparison.previous_period_end.strftime('%b %d')}"
    )
    flowables.append(Paragraph(period_text, normal_style))
    flowables.append(Spacer(1, 5*mm))

    # Get trend info for formatting
    hours_arrow = get_arrow_symbol(comparison.hours_trend)
    hours_color = get_trend_color(comparison.hours_trend)
    items_arrow = get_arrow_symbol(comparison.items_trend)
    items_color = get_trend_color(comparison.items_trend)

    # Format change strings
    hours_change_str = format_percent_change(comparison.hours_percent_change)
    items_change_str = format_percent_change(comparison.items_percent_change)

    # Create summary metrics table with arrows and percentages
    summary_data = [
        ["Metric", "Current", "Previous", "Change", "Trend"],
        [
            "Total Hours",
            f"{comparison.current_total_hours:.1f}h",
            f"{comparison.previous_total_hours:.1f}h",
            f"{comparison.hours_change:+.1f}h",
            f"{hours_arrow} {hours_change_str}"
        ],
        [
            "Activities",
            str(comparison.current_total_items),
            str(comparison.previous_total_items),
            f"{comparison.items_change:+d}",
            f"{items_arrow} {items_change_str}"
        ],
        [
            "Categories",
            str(comparison.current_category_count),
            str(comparison.previous_category_count),
            f"{comparison.current_category_count - comparison.previous_category_count:+d}",
            ""
        ],
    ]

    # Enhanced column widths for better proportions
    summary_table = Table(summary_data, colWidths=[45*mm, 28*mm, 28*mm, 28*mm, 32*mm])

    # Build enhanced table style with alternating rows
    table_style = get_enhanced_table_base_style(has_footer=False, num_rows=len(summary_data))

    # Apply alternating row colors for data rows (3 data rows after header)
    apply_alternating_row_colors(table_style, num_data_rows=3, start_row=1)

    # Column alignment: Metric label left, numeric data right-aligned for readability
    table_style.extend([
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),      # Metric column left-aligned
        ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),    # All numeric columns right-aligned
        ('ALIGN', (1, 0), (-1, 0), 'CENTER'),    # Header row centered
        # Color the trend column based on direction
        ('TEXTCOLOR', (4, 1), (4, 1), hours_color),
        ('TEXTCOLOR', (4, 2), (4, 2), items_color),
        ('FONTNAME', (4, 1), (4, -1), 'Helvetica-Bold'),
    ])

    summary_table.setStyle(TableStyle(table_style))
    flowables.append(summary_table)
    flowables.append(Spacer(1, 8*mm))

    # Top Movers section (categories with biggest changes)
    if comparison.top_increasing_categories or comparison.top_decreasing_categories:
        flowables.append(Paragraph("Top Category Changes", subheading_style))
        flowables.append(Spacer(1, 3*mm))

        movers_data = [["Category", "Current", "Previous", "Change", "% Change"]]
        mover_colors = []  # Track which rows are increases vs decreases

        # Add increasing categories (top 3)
        for cat in comparison.top_increasing_categories[:3]:
            movers_data.append([
                f"↑ {cat['category']}",
                f"{cat['current_hours']:.1f}h",
                f"{cat['previous_hours']:.1f}h",
                f"+{cat['hours_change']:.1f}h",
                format_percent_change(cat['hours_percent_change'])
            ])
            mover_colors.append(TREND_UP_COLOR)

        # Add decreasing categories (top 3)
        for cat in comparison.top_decreasing_categories[:3]:
            movers_data.append([
                f"↓ {cat['category']}",
                f"{cat['current_hours']:.1f}h",
                f"{cat['previous_hours']:.1f}h",
                f"{cat['hours_change']:.1f}h",
                format_percent_change(cat['hours_percent_change'])
            ])
            mover_colors.append(TREND_DOWN_COLOR)

        if len(movers_data) > 1:  # Only create table if we have data
            # Enhanced column widths for better proportions
            movers_table = Table(movers_data, colWidths=[55*mm, 24*mm, 24*mm, 24*mm, 28*mm])

            # Use secondary header color for differentiation from main summary table
            movers_style = get_enhanced_table_base_style(has_footer=False, num_rows=len(movers_data))

            # Override header with secondary color for visual distinction
            movers_style.append(('BACKGROUND', (0, 0), (-1, 0), TABLE_HEADER_SECONDARY_BG))
            movers_style.append(('LINEBELOW', (0, 0), (-1, 0), 1.5, TABLE_HEADER_SECONDARY_BG))

            # Apply alternating row colors for data rows
            num_data_rows = len(movers_data) - 1  # Exclude header
            apply_alternating_row_colors(movers_style, num_data_rows=num_data_rows, start_row=1)

            # Column alignment
            movers_style.extend([
                ('ALIGN', (0, 0), (0, -1), 'LEFT'),      # Category column left-aligned
                ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),    # All numeric columns right-aligned
                ('ALIGN', (1, 0), (-1, 0), 'CENTER'),    # Header row centered
            ])

            # Add row-specific colors for category names (trend indicators)
            for i, color in enumerate(mover_colors, start=1):
                movers_style.append(('TEXTCOLOR', (0, i), (0, i), color))
                movers_style.append(('FONTNAME', (0, i), (0, i), 'Helvetica-Bold'))

            movers_table.setStyle(TableStyle(movers_style))
            flowables.append(movers_table)
            flowables.append(Spacer(1, 5*mm))

    # New and Dropped Categories section
    if comparison.new_categories or comparison.dropped_categories:
        changes_text_parts = []

        if comparison.new_categories:
            new_cats = ", ".join(comparison.new_categories[:5])
            suffix = f" (+{len(comparison.new_categories) - 5} more)" if len(comparison.new_categories) > 5 else ""
            changes_text_parts.append(f"<b>New:</b> {new_cats}{suffix}")

        if comparison.dropped_categories:
            dropped_cats = ", ".join(comparison.dropped_categories[:5])
            suffix = f" (+{len(comparison.dropped_categories) - 5} more)" if len(comparison.dropped_categories) > 5 else ""
            changes_text_parts.append(f"<b>No longer active:</b> {dropped_cats}{suffix}")

        if changes_text_parts:
            changes_text = " | ".join(changes_text_parts)
            flowables.append(Paragraph(changes_text, normal_style))
            flowables.append(Spacer(1, 3*mm))

    logger.info(f"Created comparison section with {len(flowables)} flowables")
    return flowables


if __name__ == "__main__":
    # Demo usage
    from datetime import date

    print("Special Activities Comparison Display - Demo")
    print("=" * 60)

    # Create a mock comparison for testing
    mock_comparison = SpecialActivitiesPeriodComparison(
        current_period_start=date(2026, 1, 1),
        current_period_end=date(2026, 1, 7),
        previous_period_start=date(2025, 12, 25),
        previous_period_end=date(2025, 12, 31),
        current_total_hours=45.5,
        previous_total_hours=38.0,
        current_total_items=12,
        previous_total_items=10,
        current_category_count=5,
        previous_category_count=4,
        category_comparisons={
            "Meetings": {
                "current_hours": 20.0,
                "previous_hours": 15.0,
                "hours_change": 5.0,
                "hours_percent_change": 33.3,
            },
            "Research": {
                "current_hours": 10.0,
                "previous_hours": 12.0,
                "hours_change": -2.0,
                "hours_percent_change": -16.7,
            },
        },
        new_categories=["Training"],
        dropped_categories=[],
        top_increasing_categories=[
            {"category": "Meetings", "current_hours": 20.0, "previous_hours": 15.0, "hours_change": 5.0, "hours_percent_change": 33.3}
        ],
        top_decreasing_categories=[
            {"category": "Research", "current_hours": 10.0, "previous_hours": 12.0, "hours_change": -2.0, "hours_percent_change": -16.7}
        ],
    )

    print(f"\nHours Trend: {get_arrow_symbol(mock_comparison.hours_trend)} {mock_comparison.hours_percent_change:.1f}%")
    print(f"Items Trend: {get_arrow_symbol(mock_comparison.items_trend)} {mock_comparison.items_percent_change:.1f}%")

    # Create flowables (won't render without PDF context)
    flowables = create_special_activities_comparison_section(mock_comparison)
    print(f"\nCreated {len(flowables)} flowables for PDF rendering")
