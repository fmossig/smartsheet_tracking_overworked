import os
import csv
import json
from datetime import datetime, timedelta, date
from collections import defaultdict, Counter
import logging
import math
import argparse
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
    TYPE_CHECKING,
)
from unicode_utilities import (
    normalize_unicode,
    safe_truncate,
    prepare_for_pdf,
)

if TYPE_CHECKING:
    from reportlab.lib.styles import StyleSheet1
    from reportlab.graphics.shapes import Drawing
    from reportlab.platypus import Table, Flowable
    from reportlab.lib.colors import Color
    from reportlab.pdfgen.canvas import Canvas
    from reportlab.platypus import SimpleDocTemplate

import smartsheet
from dotenv import load_dotenv
from smartsheet_retry import SmartsheetRetryClient, execute_with_retry
from date_utilities import (
    parse_date,
    parse_date_argument,
    format_report_timestamp,
    SUPPORTED_DATE_FORMATS,
)
from phase_field_utilities import (
    PHASE_FIELDS,
    EXPECTED_DATE_COLUMNS,
    resolve_column_name,
    detect_missing_columns,
)
from group_health_scorer import (
    HealthStatus,
    GroupHealthScore,
    HealthScoreConfig,
    calculate_group_health_scores,
    calculate_health_scores_for_range,
    get_health_color,
    get_health_summary,
)
from completion_rate_calculator import (
    CompletionRateMetrics,
    GroupCompletionRate,
    calculate_completion_rate,
    calculate_completion_by_group,
    get_completion_visualization_data,
)
from phase_progression_funnel_calculator import (
    calculate_funnel_metrics,
    get_funnel_visualization_data,
    get_funnel_summary,
    identify_bottlenecks,
    BottleneckSeverity,
)
from date_range_filter import (
    DateRangeFilter,
    DateRangePreset,
    create_date_range,
    get_preset_range,
    validate_date_range,
    filter_data_by_range,
    get_date_range_options,
)
from accessibility_colors import (
    get_reportlab_group_colors,
    get_reportlab_phase_colors,
    get_reportlab_severity_colors,
    get_reportlab_error_category_colors,
    get_reportlab_chart_palette,
    get_reportlab_donut_palette,
    get_reportlab_user_colors,
    get_reportlab_base_colors,
    get_reportlab_overdue_status_colors,
    get_accessible_text_color,
    ACCESSIBLE_CHART_PALETTE,
    ENHANCED_DONUT_PALETTE,
    ACCESSIBLE_USER_COLORS,
    ACCESSIBLE_BASE_COLORS,
)
from special_activities_hours_distribution import (
    EfficiencyLevel,
    HoursDistributionSummary,
    calculate_hours_distribution,
    get_distribution_visualization_data,
)
from special_activities_category_breakdown import (
    SpecialActivitiesPeriodComparison,
    compare_special_activities_periods,
    format_period_comparison_table,
    get_period_comparison_summary,
)
from special_activities_comparison_display import (
    create_special_activities_comparison_section,
    get_enhanced_table_base_style,
    apply_alternating_row_colors,
    TABLE_HEADER_BG,
    TABLE_BORDER_COLOR,
)
from average_items_per_period_calculator import (
    AverageItemsMetrics,
    UserAverageItems,
    PerformanceLevel,
    calculate_average_items_per_user,
    calculate_average_items_all_users,
    get_average_items_summary,
)
from trend_line_visualization import (
    make_time_series_bar_chart,
    make_group_activity_trend_chart,
    draw_trend_line_overlay,
    add_trend_line_legend,
    TREND_LINE_COLORS,
    TREND_LINE_STYLES,
)
from period_comparison_calculator import (
    TrendDirection,
    get_trend_indicator,
    format_percent_change,
)
from user_contribution_calculator import (
    UserTeamContribution,
    TeamContributionSummary,
    calculate_all_user_contributions,
    get_contribution_summary,
    get_contribution_visualization_data,
)
from pie_label_utilities import (
    calculate_pie_label_positions,
    draw_pie_labels,
    calculate_gauge_label_positions,
)
from phase_distribution_per_user import (
    PHASE_NAMES as USER_PHASE_NAMES,
    ALL_PHASES,
    calculate_user_phase_distribution,
    UserPhaseDistribution,
    SpecializationLevel,
)
from error_collector import (
    get_global_collector,
    ErrorType,
    ErrorSeverity as CollectorSeverity,
    CollectedError,
)
from logging_config import (
    configure_logging,
    add_log_level_argument,
    get_module_logger,
)
from performance_timing import (
    timed_operation,
    PerformanceTimer,
    time_data_processing,
    time_pdf_generation,
    log_timing,
)
from reportlab.lib.pagesizes import A4, landscape
from reportlab.lib import colors
from reportlab.lib.units import mm
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image, Flowable
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from table_pagination import (
    create_paginated_table,
    PaginationConfig,
    calculate_optimal_rows_per_page,
    ensure_minimum_rows,
)
from constants import LandscapeMode
from reportlab.graphics.shapes import Drawing, String
from reportlab.graphics.charts.barcharts import VerticalBarChart, HorizontalBarChart
from reportlab.graphics.charts.legends import Legend
from reportlab.graphics.widgets.markers import makeMarker
from reportlab.graphics.shapes import Line, Rect
from reportlab.graphics.shapes import Path, Circle
from reportlab.graphics.shapes import Wedge
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas
from sparkline_generator import (
    create_table_sparkline,
    create_sparkline,
    SparklineStyle,
    TABLE_SPARKLINE_WIDTH,
    TABLE_SPARKLINE_HEIGHT,
)

# Logger will be configured in main() after parsing args
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
token = os.getenv("SMARTSHEET_TOKEN")
if not token:
    logger.error("SMARTSHEET_TOKEN not found in environment or .env file")
    exit(1)

# Constants
SHEET_IDS = {
    "NA": 6141179298008964,
    "NF": 615755411312516,
    "NH": 123340632051588,
    "NP": 3009924800925572,
    "NT": 2199739350077316,
    "NV": 8955413669040004,
    "NM": 4275419734822788,
    "SPECIAL": 5261724614610820,  # Adding special activities sheet ID
    "BUNDLE_FAN": 7412589630147852,  # Bundle sheet for FAN products (estimated 2,497 products)
    "BUNDLE_COOLER": 3698521470258963,  # Bundle sheet for COOLER products (estimated 121 products)
}

SPECIAL_ACTIVITIES_SHEET_ID = SHEET_IDS.get("SPECIAL")
# Color scheme - WCAG AA compliant colors for accessibility
# All colors meet minimum 3:1 contrast ratio against white background
# See accessibility_colors.py for contrast ratio documentation
GROUP_COLORS = get_reportlab_group_colors()

# Phase names for reference (now using Phase 1, Phase 2, etc. for display)
PHASE_NAMES = {
    "1": "Phase 1",
    "2": "Phase 2",
    "3": "Phase 3",
    "4": "Phase 4",
    "5": "Phase 5",
}

# PHASE_FIELDS and EXPECTED_DATE_COLUMNS are now imported from
# phase_field_utilities module to eliminate code duplication.

# Phase colors - WCAG AA compliant
PHASE_COLORS = get_reportlab_phase_colors()

# Severity colors for error/warning report - WCAG AA compliant
SEVERITY_COLORS = get_reportlab_severity_colors()

SEVERITY_LABELS = {
    "error": "Error",
    "warning": "Warning",
    "info": "Info",
}

# Error category colors - WCAG AA compliant
ERROR_CATEGORY_COLORS = get_reportlab_error_category_colors()

ERROR_CATEGORY_LABELS = {
    "data_quality": "Data Quality",
    "missing_data": "Missing Data",
    "invalid_format": "Invalid Format",
    "api_error": "API Error",
    "permission": "Permission Issue",
    "other": "Other",
}

# Error rate severity colors - WCAG AA compliant
# Maps error rate severity levels to visual indicators (green/yellow/red)
ERROR_RATE_SEVERITY_COLORS = {
    "low": colors.HexColor("#1B5E20"),       # Green 900 - < 5% error rate (healthy)
    "moderate": colors.HexColor("#8B6914"),  # Dark Gold - 5-15% error rate (caution)
    "high": colors.HexColor("#C65102"),      # Burnt Orange - 15-30% error rate (warning)
    "critical": colors.HexColor("#B71C1C"),  # Red 900 - > 30% error rate (critical)
}

ERROR_RATE_SEVERITY_LABELS = {
    "low": "Healthy",
    "moderate": "Needs Attention",
    "high": "Concerning",
    "critical": "Critical",
}

ERROR_RATE_SEVERITY_BG_COLORS = {
    "low": colors.HexColor("#E8F5E9"),       # Light green background
    "moderate": colors.HexColor("#FFF8E1"),  # Light amber background
    "high": colors.HexColor("#FFF3E0"),      # Light orange background
    "critical": colors.HexColor("#FFEBEE"),  # Light red background
}

ERROR_RATE_SEVERITY_BORDER_COLORS = {
    "low": colors.HexColor("#1B5E20"),       # Green border
    "moderate": colors.HexColor("#8B6914"),  # Gold border
    "high": colors.HexColor("#C65102"),      # Orange border
    "critical": colors.HexColor("#B71C1C"),  # Red border
}

# Fixed total product counts for each group
TOTAL_PRODUCTS = {
    "NA": 1779,
    "NF": 1716,
    "NM": 391,
    "NH": 893,
    "NP": 394,
    "NT": 119,
    "NV": 0,  # Adding NV with 0 since it wasn't provided
    "BUNDLE_FAN": 2497,  # Estimated product count for FAN bundle
    "BUNDLE_COOLER": 121,  # Estimated product count for COOLER bundle
}

# User colors (will be generated dynamically)
USER_COLORS = {}

# Bundle group display names for shorter presentation
BUNDLE_DISPLAY_NAMES = {
    "BUNDLE_FAN": "Bundle FAN",
    "BUNDLE_COOLER": "Bundle COOLER",
}


def get_group_display_name(group: str, short: bool = False) -> str:
    """Get a display name for a group, with optional short format for charts.

    Converts internal group identifiers to human-readable display names.
    Bundle groups can be abbreviated for use in charts where space is limited.

    Args:
        group (str): The group identifier (e.g., "NA", "NF", "BUNDLE_FAN", "BUNDLE_COOLER").
        short (bool, optional): If True, return abbreviated format for charts. Defaults to False.

    Returns:
        str: The display name for the group.
            - For regular groups: Returns the group identifier unchanged (e.g., "NA")
            - For bundle groups with short=False: Returns full name (e.g., "Bundle FAN")
            - For bundle groups with short=True: Returns abbreviated name (e.g., "B.FAN")

    Examples:
        >>> get_group_display_name("NA")
        'NA'
        >>> get_group_display_name("BUNDLE_FAN")
        'Bundle FAN'
        >>> get_group_display_name("BUNDLE_FAN", short=True)
        'B.FAN'
        >>> get_group_display_name("BUNDLE_COOLER", short=True)
        'B.COOLER'
    """
    if group.startswith("BUNDLE_"):
        if short:
            return "B." + group.replace("BUNDLE_", "")
        return BUNDLE_DISPLAY_NAMES.get(group, group)
    return group


# Directories
DATA_DIR = "tracking_data"
REPORTS_DIR = "reports"
os.makedirs(REPORTS_DIR, exist_ok=True)


# resolve_column_name and detect_missing_columns are now imported from
# phase_field_utilities module to eliminate code duplication.

# --- NEW: Constants for PDF Upload and Metadata ---
# TODO: Fill this with the ID of the sheet you want to update
REPORT_METADATA_SHEET_ID = 7888169555939204  # e.g., 8391667138243332

# Row IDs for attaching the actual PDF files
MONTHLY_REPORT_ATTACHMENT_ROW_ID = 5089581251235716
WEEKLY_REPORT_ATTACHMENT_ROW_ID = 1192484760260484

# Row IDs for writing the metadata (filename and date range)
MONTHLY_METADATA_ROW_ID = 5089581251235716  # TODO: Get this from Smartsheet for "Row 3"
WEEKLY_METADATA_ROW_ID = 1192484760260484    # TODO: Get this from Smartsheet for "Row 7"
# --- END NEW ---
# File paths
CHANGES_FILE = os.path.join(DATA_DIR, "change_history.csv")

# Note: parse_date is now imported from date_utilities module


# --- Section Tracking System for PDF Reports ---
# Global tracker for section names mapped to page numbers
_section_tracker = {
    'sections': {},  # Maps page number to section name
    'current_section': None,  # Current section being built
    'report_type': None,  # Type of report (Weekly, Monthly, Custom)
}

# --- PDF Bookmark System for Outline Navigation ---
# Global tracker for PDF bookmarks (outline entries)
_bookmark_tracker = {
    'bookmarks': [],  # List of (key, title, level, page) tuples for bookmarks
    'bookmark_counter': 0,  # Counter for generating unique bookmark keys
}


def reset_section_tracker(report_type: Optional[str] = None) -> None:
    """Reset the section tracker for a new document.

    Args:
        report_type (str, optional): The type of report being generated
    """
    global _section_tracker
    _section_tracker = {
        'sections': {},
        'current_section': None,
        'report_type': report_type,
    }


def reset_bookmark_tracker() -> None:
    """Reset the bookmark tracker for a new document.

    Should be called at the start of each report generation to ensure
    bookmarks are fresh and unique keys are generated properly.
    """
    global _bookmark_tracker
    _bookmark_tracker = {
        'bookmarks': [],
        'bookmark_counter': 0,
    }


def register_bookmark(title: str, level: int = 0) -> str:
    """Register a bookmark entry and return its unique key.

    Args:
        title (str): The display title for the bookmark in the outline panel
        level (int): The hierarchy level (0 = top level, 1 = sub-section, etc.)

    Returns:
        str: A unique key for this bookmark
    """
    global _bookmark_tracker
    _bookmark_tracker['bookmark_counter'] += 1
    key = f"bookmark_{_bookmark_tracker['bookmark_counter']}"
    # Store bookmark info (key, title, level) - page will be added when drawn
    _bookmark_tracker['bookmarks'].append({
        'key': key,
        'title': title,
        'level': level,
        'page': None,  # Will be set when bookmark flowable is drawn
    })
    return key


def get_all_bookmarks() -> List[Dict[str, Any]]:
    """Get all registered bookmarks.

    Returns:
        List of bookmark dictionaries with key, title, level, and page
    """
    return _bookmark_tracker['bookmarks']


def set_current_section(section_name: str) -> None:
    """Set the current section name for tracking.

    Args:
        section_name (str): The name of the current section
    """
    global _section_tracker
    _section_tracker['current_section'] = section_name


def get_section_for_page(page_number: int) -> Optional[str]:
    """Get the section name for a given page number.

    Args:
        page_number (int): The page number to look up

    Returns:
        str: The section name for that page, or None if not found
    """
    sections = _section_tracker['sections']
    # Find the most recent section that started at or before this page
    applicable_sections = [(p, s) for p, s in sections.items() if p <= page_number]
    if applicable_sections:
        # Return the section with the highest page number <= current page
        return max(applicable_sections, key=lambda x: x[0])[1]
    return None


class SectionMarker(Flowable):
    """A flowable that marks the start of a new report section.

    This invisible flowable records when a new section begins, allowing
    the page header to display the current section name. It also creates
    a PDF bookmark for the section, enabling quick navigation in PDF readers.
    It takes no space in the layout.

    Attributes:
        section_name (str): The name of the section being marked
        bookmark_level (int): The hierarchy level for the bookmark (0 = top level)
        bookmark_key (str): The unique key for this section's bookmark

    Example:
        story.append(SectionMarker("Executive Summary"))
        story.append(Paragraph("Summary content...", normal_style))

        # For sub-sections (nested in outline):
        story.append(SectionMarker("User Statistics", bookmark_level=1))
    """

    def __init__(self, section_name: str, bookmark_level: int = 0) -> None:
        """Initialize the section marker.

        Args:
            section_name (str): The display name for this section
            bookmark_level (int): The hierarchy level for the PDF bookmark
                (0 = top level, 1 = sub-section, etc.)
        """
        Flowable.__init__(self)
        self.section_name = section_name
        self.bookmark_level = bookmark_level
        self.bookmark_key = register_bookmark(section_name, bookmark_level)
        self.width = 0
        self.height = 0

    def wrap(self, available_width: float, available_height: float) -> Tuple[float, float]:
        """Return the dimensions of this flowable (zero, since it's invisible)."""
        return (0, 0)

    def draw(self) -> None:
        """Record the section marker and create PDF bookmark when drawn on the canvas."""
        # Get the current page number from the canvas
        canvas = self.canv
        page_number = canvas._pageNumber

        # Record this section for the current page (for header display)
        _section_tracker['sections'][page_number] = self.section_name
        _section_tracker['current_section'] = self.section_name

        # Create PDF bookmark for this section
        # bookmarkPage creates a destination, addOutlineEntry adds it to the outline panel
        canvas.bookmarkPage(self.bookmark_key)
        canvas.addOutlineEntry(
            self.section_name,
            self.bookmark_key,
            level=self.bookmark_level,
            closed=False  # Keep sections expanded by default
        )
# --- End Section Tracking System ---


# --- Page Orientation System for Landscape Mode Support ---
from enum import Enum
from dataclasses import dataclass


class PageOrientation(Enum):
    """Enum defining page orientation modes."""
    PORTRAIT = "portrait"
    LANDSCAPE = "landscape"


@dataclass
class OrientationConfig:
    """Configuration for page orientation settings.

    This class provides configuration options for controlling page orientation
    in PDF reports, enabling landscape mode for data-heavy sections.

    Attributes:
        default_orientation: The default orientation for the document (PORTRAIT or LANDSCAPE)
        data_heavy_sections: List of section names that should use landscape mode
        margin_portrait: Margin settings for portrait pages (left, right, top, bottom in mm)
        margin_landscape: Margin settings for landscape pages (left, right, top, bottom in mm)

    Example:
        config = OrientationConfig(
            default_orientation=PageOrientation.PORTRAIT,
            data_heavy_sections=["User Activity", "Error Report"],
            margin_portrait=(25, 25, 20, 20),
            margin_landscape=(15, 15, 15, 15)
        )
    """
    default_orientation: PageOrientation = PageOrientation.PORTRAIT
    data_heavy_sections: List[str] = None
    margin_portrait: Tuple[float, float, float, float] = (25, 25, 20, 20)  # left, right, top, bottom in mm
    margin_landscape: Tuple[float, float, float, float] = (15, 15, 15, 15)

    def __post_init__(self):
        if self.data_heavy_sections is None:
            # Default data-heavy sections that benefit from landscape mode
            self.data_heavy_sections = [
                "User Activity",
                "Error Report",
                "Data Quality & Issues Report",
                "User Details",
                "Cleanup Suggestions",
                "Missing Data Warnings",
            ]


# Global orientation tracker for mixed-orientation PDFs
_orientation_tracker = {
    'current_orientation': PageOrientation.PORTRAIT,
    'page_orientations': {},  # Maps page number to orientation
    'config': None,
}


def reset_orientation_tracker(config: Optional[OrientationConfig] = None) -> None:
    """Reset the orientation tracker for a new document.

    Args:
        config: Optional OrientationConfig to use for the document
    """
    global _orientation_tracker
    _orientation_tracker = {
        'current_orientation': PageOrientation.PORTRAIT,
        'page_orientations': {},
        'config': config or OrientationConfig(),
    }


def get_orientation_for_page(page_number: int) -> PageOrientation:
    """Get the orientation for a given page number.

    Args:
        page_number: The page number to look up

    Returns:
        PageOrientation: The orientation for that page
    """
    return _orientation_tracker['page_orientations'].get(
        page_number,
        _orientation_tracker['config'].default_orientation if _orientation_tracker['config'] else PageOrientation.PORTRAIT
    )


def get_current_orientation() -> PageOrientation:
    """Get the current orientation being used."""
    return _orientation_tracker['current_orientation']


def set_current_orientation(orientation: PageOrientation) -> None:
    """Set the current orientation."""
    _orientation_tracker['current_orientation'] = orientation


def should_use_landscape(section_name: str) -> bool:
    """Check if a section should use landscape orientation.

    Args:
        section_name: The name of the section to check

    Returns:
        bool: True if the section should use landscape mode
    """
    config = _orientation_tracker.get('config')
    if config is None:
        return False
    return section_name in config.data_heavy_sections


def get_landscape_page_width() -> float:
    """Get the width of a landscape A4 page in points."""
    return landscape(A4)[0]


def get_landscape_page_height() -> float:
    """Get the height of a landscape A4 page in points."""
    return landscape(A4)[1]


def get_portrait_page_width() -> float:
    """Get the width of a portrait A4 page in points."""
    return A4[0]


def get_portrait_page_height() -> float:
    """Get the height of a portrait A4 page in points."""
    return A4[1]


def get_available_width(orientation: PageOrientation, margins: Optional[Tuple[float, float, float, float]] = None) -> float:
    """Calculate available content width for given orientation.

    Args:
        orientation: The page orientation
        margins: Tuple of (left, right, top, bottom) margins in mm. If None, uses defaults.

    Returns:
        float: Available width in points
    """
    if margins is None:
        config = _orientation_tracker.get('config')
        if config:
            margins = config.margin_landscape if orientation == PageOrientation.LANDSCAPE else config.margin_portrait
        else:
            margins = (15, 15, 15, 15) if orientation == PageOrientation.LANDSCAPE else (25, 25, 20, 20)

    if orientation == PageOrientation.LANDSCAPE:
        page_width = get_landscape_page_width()
    else:
        page_width = get_portrait_page_width()

    left_margin, right_margin = margins[0] * mm, margins[1] * mm
    return page_width - left_margin - right_margin


def calculate_landscape_column_widths(
    portrait_widths: List[float],
    preserve_proportions: bool = True
) -> List[float]:
    """Calculate column widths optimized for landscape orientation.

    This function takes column widths designed for portrait mode and
    adjusts them for landscape mode, providing more space for data-heavy tables.

    Args:
        portrait_widths: Original column widths in portrait mode (in points or mm*mm)
        preserve_proportions: If True, scale proportionally. If False, distribute extra space evenly.

    Returns:
        List of adjusted column widths for landscape mode
    """
    portrait_available = get_available_width(PageOrientation.PORTRAIT)
    landscape_available = get_available_width(PageOrientation.LANDSCAPE)

    total_portrait_width = sum(portrait_widths)
    scale_factor = landscape_available / portrait_available

    if preserve_proportions:
        # Scale all columns proportionally
        return [w * scale_factor for w in portrait_widths]
    else:
        # Distribute extra space evenly
        extra_space = landscape_available - total_portrait_width
        extra_per_column = extra_space / len(portrait_widths)
        return [w + extra_per_column for w in portrait_widths]


class LandscapeSectionMarker(Flowable):
    """A flowable that marks the start of a landscape-oriented section.

    This flowable triggers a page break and orientation change, enabling
    landscape mode for data-heavy sections within a single PDF document.

    When this marker is encountered during PDF building, it:
    1. Records the orientation change for the upcoming page
    2. Can optionally include a section name for header tracking

    Attributes:
        section_name: Optional name for the section (for header display)
        orientation: The orientation for the section (default: LANDSCAPE)

    Example:
        # Start a landscape section for wide tables
        story.append(LandscapeSectionMarker("User Activity Data"))
        story.append(wide_data_table)

        # Return to portrait for next section
        story.append(PortraitSectionMarker("Summary"))
    """

    def __init__(
        self,
        section_name: Optional[str] = None,
        orientation: PageOrientation = PageOrientation.LANDSCAPE
    ) -> None:
        """Initialize the landscape section marker.

        Args:
            section_name: Optional display name for this section
            orientation: The orientation for this section (default: LANDSCAPE)
        """
        Flowable.__init__(self)
        self.section_name = section_name
        self.orientation = orientation
        self.width = 0
        self.height = 0

    def wrap(self, available_width: float, available_height: float) -> Tuple[float, float]:
        """Return the dimensions of this flowable (zero, since it's invisible)."""
        return (0, 0)

    def draw(self) -> None:
        """Record the orientation change when drawn on the canvas."""
        canvas = self.canv
        page_number = canvas._pageNumber

        # Record orientation for this page
        _orientation_tracker['page_orientations'][page_number] = self.orientation
        _orientation_tracker['current_orientation'] = self.orientation

        # Also update section tracker if section_name provided
        if self.section_name:
            _section_tracker['sections'][page_number] = self.section_name
            _section_tracker['current_section'] = self.section_name


class PortraitSectionMarker(LandscapeSectionMarker):
    """Convenience marker to return to portrait orientation.

    Example:
        story.append(PortraitSectionMarker("Executive Summary"))
    """

    def __init__(self, section_name: Optional[str] = None) -> None:
        super().__init__(section_name=section_name, orientation=PageOrientation.PORTRAIT)


class OrientationAwarePageBreak(Flowable):
    """A page break that can change orientation for the next page.

    This flowable combines a page break with an orientation change,
    useful for transitioning between portrait and landscape sections.

    Attributes:
        next_orientation: The orientation for the page after the break

    Example:
        # Break to a landscape page
        story.append(OrientationAwarePageBreak(PageOrientation.LANDSCAPE))
        story.append(wide_table)
    """

    def __init__(self, next_orientation: PageOrientation = PageOrientation.PORTRAIT) -> None:
        Flowable.__init__(self)
        self.next_orientation = next_orientation

    def wrap(self, available_width: float, available_height: float) -> Tuple[float, float]:
        # Force a page break by requesting more height than available
        return (0, available_height + 1)

    def draw(self) -> None:
        """Record the orientation for the next page."""
        canvas = self.canv
        next_page = canvas._pageNumber + 1
        _orientation_tracker['page_orientations'][next_page] = self.next_orientation
        _orientation_tracker['current_orientation'] = self.next_orientation


def create_landscape_section(
    story: List["Flowable"],
    section_name: str,
    content_builder: Callable[[List["Flowable"]], None],
    return_to_portrait: bool = True
) -> None:
    """Helper function to create a landscape section with proper markers.

    This function wraps content in landscape orientation markers, handling
    the transition to and from landscape mode automatically.

    Args:
        story: The ReportLab story list to append content to
        section_name: Name of the section (displayed in page header)
        content_builder: A callable that takes the story list and adds content
        return_to_portrait: If True, adds a portrait marker after content

    Example:
        def add_user_table(story):
            story.append(user_data_table)
            story.append(Spacer(1, 10*mm))

        create_landscape_section(story, "User Activity", add_user_table)
    """
    # Add page break and landscape marker
    story.append(PageBreak())
    story.append(LandscapeSectionMarker(section_name))

    # Build the content
    content_builder(story)

    # Optionally return to portrait
    if return_to_portrait:
        story.append(PageBreak())
        story.append(PortraitSectionMarker())


# --- Column Width Presets for Landscape Mode ---
# These presets provide optimized column widths for common table types

LANDSCAPE_USER_ACTIVITY_WIDTHS = [
    20*mm,   # Rank
    100*mm,  # User (expanded for long names)
    35*mm,   # Changes
    35*mm,   # % of Total
    70*mm,   # Activity bar (expanded)
]

LANDSCAPE_ERROR_REPORT_WIDTHS = [
    25*mm,   # Severity
    50*mm,   # Category
    140*mm,  # Description (significantly expanded)
    45*mm,   # Timestamp
]

LANDSCAPE_USER_DETAILS_WIDTHS = [
    20*mm,   # Rank
    90*mm,   # User
    30*mm,   # Total
    30*mm,   # Phase 1
    30*mm,   # Phase 2
    30*mm,   # Phase 3
    30*mm,   # Phase 4
]


def get_landscape_table_style_adjustments() -> List[Tuple]:
    """Get additional table style commands optimized for landscape mode.

    Returns:
        List of TableStyle commands for landscape optimization
    """
    return [
        ('FONTSIZE', (0, 1), (-1, -1), 9),  # Slightly larger font for readability
        ('ROWHEIGHT', (0, 0), (-1, -1), 18),  # More row height for comfort
        ('LEFTPADDING', (0, 0), (-1, -1), 6),
        ('RIGHTPADDING', (0, 0), (-1, -1), 6),
    ]


# --- Automatic Landscape Detection for Wide Tables ---

def should_use_landscape_for_table(
    table_data: List[List[Any]],
    col_widths: Optional[List[float]] = None,
    force_landscape: Optional[bool] = None
) -> bool:
    """Determine if a table should use landscape orientation based on column count and width.

    This function automatically detects when a table has many columns or would be
    too wide for portrait orientation, and recommends landscape mode.

    Args:
        table_data: 2D list of table data (rows x columns)
        col_widths: Optional list of column widths in points (None for auto-detection)
        force_landscape: If provided, override automatic detection

    Returns:
        bool: True if the table should use landscape orientation

    Examples:
        >>> data = [["A", "B", "C", "D", "E", "F", "G"]]  # 7 columns
        >>> should_use_landscape_for_table(data)
        True
        >>> data = [["A", "B", "C"]]  # 3 columns
        >>> should_use_landscape_for_table(data)
        False
    """
    # Allow explicit override
    if force_landscape is not None:
        return force_landscape

    if not table_data:
        return False

    # Get number of columns from first row
    num_columns = len(table_data[0]) if table_data else 0

    # Check column count threshold
    if num_columns >= LandscapeMode.COLUMN_THRESHOLD:
        return True

    # Check total width if column widths are provided
    if col_widths:
        # Convert to mm for comparison (assuming widths are in points)
        total_width_mm = sum(col_widths) / mm
        if total_width_mm >= LandscapeMode.MIN_TOTAL_WIDTH_MM:
            return True

    return False


def calculate_landscape_widths_for_table(
    portrait_widths: List[float],
    preserve_proportions: bool = True
) -> List[float]:
    """Calculate optimized column widths for landscape orientation.

    Takes column widths designed for portrait mode and scales them
    appropriately for landscape mode, utilizing the extra horizontal space.

    Args:
        portrait_widths: Original column widths (in points or mm values)
        preserve_proportions: If True, scale proportionally; if False, distribute evenly

    Returns:
        List of adjusted column widths for landscape mode
    """
    if not portrait_widths:
        return portrait_widths

    # Get available widths
    portrait_available = LandscapeMode.PORTRAIT_AVAILABLE_WIDTH_MM * mm
    landscape_available = LandscapeMode.LANDSCAPE_AVAILABLE_WIDTH_MM * mm

    total_portrait_width = sum(portrait_widths)
    scale_factor = landscape_available / portrait_available

    if preserve_proportions:
        # Scale all columns proportionally
        return [w * scale_factor for w in portrait_widths]
    else:
        # Distribute extra space evenly
        extra_space = landscape_available - total_portrait_width
        if extra_space > 0:
            extra_per_column = extra_space / len(portrait_widths)
            return [w + extra_per_column for w in portrait_widths]
        return portrait_widths


class LandscapeTableWrapper(Flowable):
    """A flowable wrapper that automatically handles landscape orientation for wide tables.

    This wrapper detects when a table needs landscape orientation and handles:
    1. Inserting page breaks before and after the landscape section
    2. Adding landscape orientation markers
    3. Scaling column widths for landscape mode
    4. Ensuring proper headers are displayed in landscape mode
    5. Returning to portrait mode after the table

    Usage:
        # Wrap any table for automatic landscape handling
        wrapper = LandscapeTableWrapper(
            table_data=data,
            col_widths=widths,
            style_commands=styles,
            section_name="Wide Data Table"
        )
        story.append(wrapper)

    Attributes:
        table_data: The raw table data
        col_widths: Column widths (will be scaled for landscape if needed)
        style_commands: TableStyle commands to apply
        section_name: Section name for page headers
        use_landscape: Whether to use landscape (auto-detected if None)
        pagination_config: Configuration for table pagination
    """

    def __init__(
        self,
        table_data: List[List[Any]],
        col_widths: Optional[List[float]] = None,
        style_commands: Optional[List[Tuple]] = None,
        section_name: Optional[str] = None,
        use_landscape: Optional[bool] = None,
        pagination_config: Optional[PaginationConfig] = None,
        return_to_portrait: bool = True,
        **table_kwargs
    ):
        """Initialize the landscape table wrapper.

        Args:
            table_data: 2D list of table data
            col_widths: List of column widths (optional)
            style_commands: List of TableStyle commands
            section_name: Name for section header display
            use_landscape: Override auto-detection (None = auto-detect)
            pagination_config: Pagination configuration for the table
            return_to_portrait: Whether to return to portrait after this section
            **table_kwargs: Additional kwargs passed to Table constructor
        """
        Flowable.__init__(self)
        self.table_data = table_data
        self.original_col_widths = col_widths
        self.style_commands = style_commands or []
        self.section_name = section_name
        self.pagination_config = pagination_config or PaginationConfig()
        self.return_to_portrait = return_to_portrait
        self.table_kwargs = table_kwargs

        # Auto-detect landscape if not specified
        if use_landscape is None:
            self.use_landscape = should_use_landscape_for_table(table_data, col_widths)
        else:
            self.use_landscape = use_landscape

        # Calculate appropriate column widths
        if self.use_landscape and col_widths:
            self.col_widths = calculate_landscape_widths_for_table(col_widths)
            # Add landscape style adjustments
            self.style_commands = self.style_commands + get_landscape_table_style_adjustments()
        else:
            self.col_widths = col_widths

        # Build the table
        self._build_table()

    def _build_table(self) -> None:
        """Build the internal table with pagination support."""
        self.table = create_paginated_table(
            data=self.table_data,
            col_widths=self.col_widths,
            style_commands=self.style_commands,
            config=self.pagination_config,
            **self.table_kwargs
        )

    def wrap(self, available_width: float, available_height: float) -> Tuple[float, float]:
        """Calculate the space needed for this flowable."""
        return self.table.wrap(available_width, available_height)

    def split(self, available_width: float, available_height: float) -> List[Flowable]:
        """Split the flowable if it doesn't fit."""
        return self.table.split(available_width, available_height)

    def draw(self) -> None:
        """Draw the table on the canvas."""
        self.table.drawOn(self.canv, 0, 0)

    def get_flowables(self) -> List[Flowable]:
        """Get the list of flowables to add to the story for landscape mode.

        Returns a list that includes:
        - PageBreak (if landscape)
        - LandscapeSectionMarker (if landscape and section_name provided)
        - The table itself
        - PageBreak (if landscape and return_to_portrait)
        - PortraitSectionMarker (if landscape and return_to_portrait)

        For non-landscape tables, returns just the table.
        """
        flowables = []

        if self.use_landscape:
            # Add page break before landscape section
            flowables.append(PageBreak())

            # Add landscape marker with section name
            if self.section_name:
                flowables.append(LandscapeSectionMarker(self.section_name))
            else:
                flowables.append(LandscapeSectionMarker())

            # Add the table
            flowables.append(self.table)

            # Return to portrait if requested
            if self.return_to_portrait:
                flowables.append(PageBreak())
                flowables.append(PortraitSectionMarker())
        else:
            # Just return the table for portrait mode
            flowables.append(self.table)

        return flowables


def create_landscape_aware_table(
    data: List[List[Any]],
    col_widths: Optional[List[float]] = None,
    style_commands: Optional[List[Tuple]] = None,
    section_name: Optional[str] = None,
    pagination_config: Optional[PaginationConfig] = None,
    force_landscape: Optional[bool] = None,
    return_flowables: bool = True,
    **table_kwargs
) -> Union[Table, List[Flowable]]:
    """Create a table with automatic landscape detection and handling.

    This is the main entry point for creating tables that automatically
    switch to landscape orientation when they have many columns.

    Args:
        data: 2D list of table data
        col_widths: Column widths (optional, will be scaled for landscape)
        style_commands: TableStyle commands to apply
        section_name: Section name for page headers in landscape mode
        pagination_config: Pagination configuration
        force_landscape: Override auto-detection (None = auto, True/False = explicit)
        return_flowables: If True, returns list of flowables including page breaks;
                         If False, returns just the table
        **table_kwargs: Additional kwargs for Table constructor

    Returns:
        Either a Table object or a list of Flowables including orientation markers

    Examples:
        # Auto-detect and get flowables for story
        flowables = create_landscape_aware_table(
            data=wide_data,
            col_widths=[20*mm, 30*mm, 40*mm, 30*mm, 25*mm, 35*mm, 40*mm],
            style_commands=style_commands,
            section_name="Wide Data Report"
        )
        story.extend(flowables)

        # Force portrait mode
        table = create_landscape_aware_table(
            data=data,
            col_widths=widths,
            force_landscape=False,
            return_flowables=False
        )
        story.append(table)
    """
    wrapper = LandscapeTableWrapper(
        table_data=data,
        col_widths=col_widths,
        style_commands=style_commands,
        section_name=section_name,
        use_landscape=force_landscape,
        pagination_config=pagination_config,
        **table_kwargs
    )

    if return_flowables:
        return wrapper.get_flowables()
    else:
        return wrapper.table


def add_landscape_table_section(
    story: List[Flowable],
    table_data: List[List[Any]],
    col_widths: Optional[List[float]] = None,
    style_commands: Optional[List[Tuple]] = None,
    section_name: Optional[str] = None,
    section_heading: Optional[str] = None,
    pagination_config: Optional[PaginationConfig] = None,
    force_landscape: Optional[bool] = None
) -> None:
    """Add a table section to the story with automatic landscape handling.

    This helper function adds a complete table section including:
    - Optional section heading
    - Automatic landscape detection and switching
    - Proper page breaks and orientation markers
    - Header repetition for multi-page tables

    Args:
        story: The reportlab story list to append elements to
        table_data: 2D list of table data
        col_widths: Column widths (optional)
        style_commands: TableStyle commands
        section_name: Name for page header display
        section_heading: Optional heading paragraph before the table
        pagination_config: Pagination configuration
        force_landscape: Override auto-detection

    Examples:
        add_landscape_table_section(
            story=story,
            table_data=error_data,
            col_widths=[25*mm, 50*mm, 140*mm, 45*mm],
            section_name="Error Report",
            section_heading="Detailed Error Analysis",
            pagination_config=PaginationConfig(repeat_header_rows=1)
        )
    """
    styles = getSampleStyleSheet()

    # Add section heading if provided
    if section_heading:
        story.append(Paragraph(section_heading, styles['Heading2']))
        story.append(Spacer(1, 5 * mm))

    # Get flowables with automatic landscape handling
    flowables = create_landscape_aware_table(
        data=table_data,
        col_widths=col_widths,
        style_commands=style_commands,
        section_name=section_name,
        pagination_config=pagination_config,
        force_landscape=force_landscape,
        return_flowables=True
    )

    # Add all flowables to story
    story.extend(flowables)

# --- End Automatic Landscape Detection ---

# --- End Page Orientation System ---


# --- Page Numbering System for PDF Reports ---
class NumberedCanvas(Canvas):
    """Custom canvas class that tracks page numbers for 'Page X of Y' format.

    This class extends ReportLab's Canvas to enable two-pass page numbering
    where the total number of pages is determined during the build process
    and then added to each page footer. It also draws section headers at the
    top of each page for easier navigation in long reports.

    The footer is excluded from the cover page (page 1) for a cleaner look,
    and optionally displays company information on the left side.

    Attributes:
        _saved_page_states (list): List of saved canvas states for each page
        _pagesize: The page size (A4 or landscape A4)
        _generation_timestamp (str): Optional timestamp string to display in footer
        _company_info (str): Optional company info to display in footer left side
        _exclude_footer_on_cover (bool): Whether to exclude footer from cover page
    """

    # Class-level attributes for footer configuration
    _generation_timestamp = None
    _company_info = None
    _exclude_footer_on_cover = True  # Default: exclude footer from cover page

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        Canvas.__init__(self, *args, **kwargs)
        self._saved_page_states: List[Dict[str, Any]] = []

    def showPage(self) -> None:
        """Save the current page state before moving to next page."""
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self) -> None:
        """Add page numbers, section headers, and PDF outline to all pages and save the document.

        Excludes footer from cover page (page 1) when _exclude_footer_on_cover is True.
        Shows the PDF bookmark outline panel when the document is opened.
        """
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            current_page = self._pageNumber

            # Draw section header on all pages except cover page
            if not (NumberedCanvas._exclude_footer_on_cover and current_page == 1):
                self._draw_section_header()

            # Draw footer on all pages except cover page (when exclusion is enabled)
            if not (NumberedCanvas._exclude_footer_on_cover and current_page == 1):
                self._draw_page_number(num_pages)

            Canvas.showPage(self)

        # Show the PDF outline/bookmarks panel when the document is opened
        # This makes the navigation panel visible by default in PDF readers
        self.showOutline()

        Canvas.save(self)

    def _draw_section_header(self) -> None:
        """Draw the section header at the top of the current page.

        Displays the current report section name in the page header area
        to help users navigate through long reports. Also adapts margins
        based on page orientation (portrait vs landscape).
        """
        # Get the section name for this page
        section_name = get_section_for_page(self._pageNumber)

        if not section_name:
            return  # No section to display

        self.saveState()

        # Get page dimensions
        page_width = self._pagesize[0]
        page_height = self._pagesize[1]

        # Determine orientation based on page dimensions
        is_landscape = page_width > page_height

        # Use appropriate margins based on orientation
        # Landscape uses 15mm margins, portrait uses 25mm
        margin = 15 * mm if is_landscape else 25 * mm

        # Header styling
        self.setFont("Helvetica-Oblique", 9)
        self.setFillColor(colors.HexColor("#7f8c8d"))  # Same grey as page numbers

        # Draw a subtle line below the header
        line_y = page_height - 12 * mm
        self.setStrokeColor(colors.HexColor("#bdc3c7"))
        self.setLineWidth(0.5)
        self.line(margin, line_y, page_width - margin, line_y)

        # Add report type prefix if available
        report_type = _section_tracker.get('report_type')
        if report_type:
            header_text = f"{report_type} Report — {section_name}"
        else:
            header_text = section_name

        # Add landscape indicator if in landscape mode
        if is_landscape:
            header_text = f"{header_text} [Landscape View]"

        # Position at top of page, aligned left within margins
        x_position = margin
        y_position = page_height - 10 * mm

        # Draw the section header
        self.drawString(x_position, y_position, header_text)

        self.restoreState()

    def _draw_page_number(self, page_count: int) -> None:
        """Draw the consistent footer with page number, generation timestamp, and company info.

        The footer layout is:
        - Left: Optional company info
        - Center: Page numbering ("Page X of Y")
        - Right: Generation timestamp ("Generated: DD.MM.YYYY at HH:MM")

        Args:
            page_count (int): Total number of pages in the document
        """
        self.saveState()
        self.setFont("Helvetica", 9)
        self.setFillColor(colors.HexColor("#7f8c8d"))

        # Get page dimensions
        page_width = self._pagesize[0]
        page_height = self._pagesize[1]

        # Position at bottom of page, 10mm from bottom edge
        y_position = 10 * mm

        # Draw optional company info on the left side
        if NumberedCanvas._company_info:
            company_x = 15 * mm  # Left margin
            self.drawString(company_x, y_position, NumberedCanvas._company_info)

        # Create page number text in "Page X of Y" format
        page_text = f"Page {self._pageNumber} of {page_count}"

        # Calculate text width to center it
        text_width = stringWidth(page_text, "Helvetica", 9)

        # Position at bottom center of page
        x_position = (page_width - text_width) / 2

        # Draw the page number
        self.drawString(x_position, y_position, page_text)

        # Draw generation timestamp on the right side of footer
        if NumberedCanvas._generation_timestamp:
            timestamp_text = f"Generated: {NumberedCanvas._generation_timestamp}"
            timestamp_width = stringWidth(timestamp_text, "Helvetica", 9)
            # Position on right side with 15mm margin from right edge
            timestamp_x = page_width - timestamp_width - 15 * mm
            self.drawString(timestamp_x, y_position, timestamp_text)

        self.restoreState()


def create_numbered_canvas_with_timestamp(
    generation_timestamp: str,
    company_info: Optional[str] = None,
    exclude_footer_on_cover: bool = True
) -> type:
    """Factory function to create a NumberedCanvas class with footer configuration.

    This function configures the NumberedCanvas class with generation timestamp,
    optional company info, and cover page exclusion setting before returning it,
    allowing consistent footer display on every page in the generated PDF.

    Args:
        generation_timestamp (str): The formatted timestamp string to display
        company_info (str, optional): Company name or info to display on left side
            of footer. Defaults to None.
        exclude_footer_on_cover (bool, optional): Whether to exclude footer from
            cover page (page 1). Defaults to True.

    Returns:
        class: The NumberedCanvas class configured with footer settings

    Example:
        >>> canvas_class = create_numbered_canvas_with_timestamp(
        ...     "06.01.2025 at 14:30",
        ...     company_info="Acme Corporation",
        ...     exclude_footer_on_cover=True
        ... )
        >>> doc.build(story, canvasmaker=canvas_class)
    """
    NumberedCanvas._generation_timestamp = generation_timestamp
    NumberedCanvas._company_info = company_info
    NumberedCanvas._exclude_footer_on_cover = exclude_footer_on_cover
    return NumberedCanvas


def add_page_footer(canvas: Any, doc: Any) -> None:
    """Callback function to add consistent footer elements to each page.

    This function is called by ReportLab's document builder for each page
    and adds footer content. When used with NumberedCanvas, page numbers
    are added separately in the 'Page X of Y' format.

    Args:
        canvas: The ReportLab canvas object
        doc: The document template object

    Note:
        Page numbers in 'Page X of Y' format are handled by NumberedCanvas,
        not by this function. This function can be extended to add other
        footer elements like document title or generation date if needed.
    """
    canvas.saveState()
    # Footer elements can be added here if needed in addition to page numbers
    # Page numbers are handled by NumberedCanvas for "Page X of Y" format
    canvas.restoreState()
# --- End Page Numbering System ---


# ============================================================================
# COVER PAGE SECTION
# ============================================================================

def create_cover_page(
    story: List["Flowable"],
    report_title: str,
    period_str: str,
    generation_timestamp: str,
    company_name: Optional[str] = "Automaker Financial Accounts",
    report_type: str = "Weekly"
) -> None:
    """Create a professional cover page for the report.

    This function generates a clean, professional cover page with:
    - Report title with professional styling
    - Date range for the report period
    - Generation timestamp
    - Company branding and logo placeholder
    - Visual separator and consistent styling

    The cover page serves as the first page of the report and provides
    immediate context about the report contents and when it was generated.

    Args:
        story: The reportlab story list to append elements to
        report_title: The main title for the report (e.g., "Weekly Activity Report")
        period_str: String describing the report period (e.g., "06.01.2025 - 12.01.2025")
        generation_timestamp: Formatted timestamp when report was generated
        company_name: Company name for branding (default: "Automaker Financial Accounts")
        report_type: Type of report for section tracking ("Weekly", "Monthly", "Custom")

    Returns:
        None (modifies story in place)

    Example:
        >>> story = []
        >>> create_cover_page(
        ...     story,
        ...     report_title="Weekly Activity Report",
        ...     period_str="06.01.2025 - 12.01.2025",
        ...     generation_timestamp="06.01.2025 at 14:30",
        ...     company_name="Acme Corporation"
        ... )
        >>> len(story) > 0
        True
    """
    from accessibility_colors import (
        get_reportlab_header_colors,
        get_reportlab_text_colors,
        get_reportlab_background_colors,
        get_reportlab_border_colors,
        ACCESSIBLE_HEADER_COLORS,
        ACCESSIBLE_TEXT_COLORS,
    )

    # Add section marker for cover page (won't show in header due to exclude_footer_on_cover)
    story.append(SectionMarker("Cover Page"))

    styles = getSampleStyleSheet()

    # Get accessible colors
    header_colors = get_reportlab_header_colors()
    text_colors = get_reportlab_text_colors()
    border_colors = get_reportlab_border_colors()

    # ========================================================================
    # COVER PAGE STYLES - Professional and consistent with report styling
    # ========================================================================

    # Company name style - subtle, professional header
    company_style = ParagraphStyle(
        'CoverCompanyName',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.HexColor(ACCESSIBLE_TEXT_COLORS["secondary"]),
        alignment=1,  # Center
        spaceAfter=15*mm,
        fontName='Helvetica'
    )

    # Main title style - large, prominent, professional
    title_style = ParagraphStyle(
        'CoverTitle',
        parent=styles['Title'],
        fontSize=32,
        textColor=colors.HexColor(ACCESSIBLE_HEADER_COLORS["title"]),
        alignment=1,  # Center
        spaceAfter=8*mm,
        spaceBefore=20*mm,
        fontName='Helvetica-Bold',
        leading=38  # Line height
    )

    # Report type subtitle style
    report_type_style = ParagraphStyle(
        'CoverReportType',
        parent=styles['Normal'],
        fontSize=18,
        textColor=colors.HexColor(ACCESSIBLE_HEADER_COLORS["section"]),
        alignment=1,  # Center
        spaceAfter=15*mm,
        fontName='Helvetica'
    )

    # Date range style - prominent but secondary to title
    date_range_style = ParagraphStyle(
        'CoverDateRange',
        parent=styles['Normal'],
        fontSize=14,
        textColor=colors.HexColor(ACCESSIBLE_TEXT_COLORS["primary"]),
        alignment=1,  # Center
        spaceAfter=6*mm,
        fontName='Helvetica-Bold'
    )

    # Period label style
    period_label_style = ParagraphStyle(
        'CoverPeriodLabel',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor(ACCESSIBLE_TEXT_COLORS["secondary"]),
        alignment=1,  # Center
        spaceAfter=2*mm,
        fontName='Helvetica'
    )

    # Generation timestamp style - subtle footer info
    timestamp_style = ParagraphStyle(
        'CoverTimestamp',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor(ACCESSIBLE_TEXT_COLORS["muted"]),
        alignment=1,  # Center
        spaceBefore=25*mm,
        fontName='Helvetica-Oblique'
    )

    # Confidentiality notice style
    confidential_style = ParagraphStyle(
        'CoverConfidential',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor(ACCESSIBLE_TEXT_COLORS["muted"]),
        alignment=1,  # Center
        spaceBefore=10*mm,
        fontName='Helvetica'
    )

    # ========================================================================
    # BUILD COVER PAGE CONTENT
    # ========================================================================

    # Top spacing to push content down for visual balance
    story.append(Spacer(1, 25*mm))

    # Company name / branding at top
    if company_name:
        story.append(Paragraph(company_name, company_style))

    # Decorative line above title
    line_drawing = Drawing(160*mm, 2)
    line_drawing.add(Line(0, 1, 160*mm, 1,
                         strokeColor=colors.HexColor(ACCESSIBLE_HEADER_COLORS["section"]),
                         strokeWidth=2))
    # Center the line
    line_table = Table([[line_drawing]], colWidths=[160*mm])
    line_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    story.append(line_table)

    # Main report title
    story.append(Paragraph(report_title, title_style))

    # Report type indicator
    story.append(Paragraph(f"{report_type} Report", report_type_style))

    # Another decorative line below subtitle
    story.append(line_table)

    # Spacer before date section
    story.append(Spacer(1, 15*mm))

    # Reporting period section
    story.append(Paragraph("Reporting Period", period_label_style))
    story.append(Paragraph(period_str, date_range_style))

    # Create a styled box for the date range
    date_box_data = [[Paragraph(period_str, date_range_style)]]
    date_box = Table(date_box_data, colWidths=[120*mm])
    date_box.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor(ACCESSIBLE_HEADER_COLORS["subsection"])),
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F5F5F5")),  # Light gray background
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
    ]))

    # Center the date box
    centered_date = Table([[date_box]], colWidths=[160*mm])
    centered_date.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))
    story.append(centered_date)

    # Push generation info toward bottom
    story.append(Spacer(1, 40*mm))

    # Generation timestamp
    story.append(Paragraph(f"Report Generated: {generation_timestamp}", timestamp_style))

    # Confidentiality notice
    story.append(Paragraph("CONFIDENTIAL - For Internal Use Only", confidential_style))

    # Page break after cover page
    story.append(PageBreak())


def load_changes(
    start_date: Optional[date] = None,
    end_date: Optional[date] = None
) -> List[Dict[str, Any]]:
    """Load change history records from the CSV file, optionally filtered by date range.

    Reads the change_history.csv file containing tracked changes from Smartsheet
    and returns records that fall within the specified date range. Each record
    includes parsed date information for downstream processing.

    Args:
        start_date (date, optional): Start of the date range filter (inclusive).
            If None, no start date filter is applied.
        end_date (date, optional): End of the date range filter (inclusive).
            If None, no end date filter is applied.

    Returns:
        list: A list of dictionaries, where each dictionary represents a change
            record with keys from the CSV columns plus 'ParsedDate' containing
            the parsed date.date object. Returns empty list if file not found
            or on error.

    File Format Expected:
        CSV with columns including 'Timestamp', 'Date', 'Group', 'Phase', 'User',
        and other tracking information.

    Side Effects:
        - Logs error if file not found
        - Logs warning for rows with parsing errors
        - Logs info message with count of loaded changes

    Examples:
        >>> from datetime import date
        >>> changes = load_changes()  # Load all changes
        >>> changes = load_changes(date(2024, 1, 1), date(2024, 1, 31))  # January only
        >>> len(changes)
        42
    """
    import time as time_module
    load_start_time = time_module.perf_counter()

    if not os.path.exists(CHANGES_FILE):
        logger.error(f"Changes file not found: {CHANGES_FILE}")
        return []

    changes = []
    try:
        with open(CHANGES_FILE, 'r', newline='', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            for row in reader:
                # Parse the timestamp
                try:
                    ts = datetime.strptime(row['Timestamp'], "%Y-%m-%d %H:%M:%S").date()

                    # Apply date filter if specified
                    if start_date and end_date:
                        if start_date <= ts <= end_date:
                            # Also parse the date field for later use
                            row['ParsedDate'] = parse_date(row['Date'])
                            changes.append(row)
                    else:
                        # No date filter, include all changes
                        row['ParsedDate'] = parse_date(row['Date'])
                        changes.append(row)
                except (ValueError, KeyError) as e:
                    logger.warning(f"Error parsing row: {row} - {e}")
                    continue
    except Exception as e:
        logger.error(f"Error reading changes file: {e}")

    # Log timing for data loading
    load_duration = time_module.perf_counter() - load_start_time
    log_timing("data_loading.load_changes", load_duration, records_loaded=len(changes))

    if start_date and end_date:
        logger.info(f"Loaded {len(changes)} changes between {start_date} and {end_date}")
    else:
        logger.info(f"Loaded {len(changes)} total changes")
    return changes


def load_changes_with_filter(date_range_filter: DateRangeFilter) -> List[Dict[str, Any]]:
    """Load changes from the CSV file using a DateRangeFilter object.

    This is the preferred method for loading changes when using custom date ranges,
    as it ensures all filtering is done consistently through the DateRangeFilter.
    Wraps load_changes() with DateRangeFilter integration.

    Args:
        date_range_filter (DateRangeFilter): A DateRangeFilter object specifying
            the date range. Must have start_date and end_date attributes.

    Returns:
        list: A list of change record dictionaries within the specified date range.
            Same format as load_changes() return value.

    See Also:
        load_changes: The underlying function that performs the actual loading.
        DateRangeFilter: The filter class from date_range_filter module.

    Examples:
        >>> from date_range_filter import DateRangeFilter, DateRangePreset
        >>> filter = DateRangeFilter(preset=DateRangePreset.LAST_7_DAYS)
        >>> changes = load_changes_with_filter(filter)
        >>> len(changes)
        15
    """
    return load_changes(
        start_date=date_range_filter.start_date,
        end_date=date_range_filter.end_date
    )


def collect_metrics_for_range(
    date_range_filter: DateRangeFilter,
    unavailable_sheets: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Collect metrics from change history data for a specific date range.

    Combines loading and metric collection into a single operation for custom
    date ranges. This is the high-level entry point for generating metrics
    that respect date range constraints. Adds additional date range metadata
    to the returned metrics.

    Args:
        date_range_filter (DateRangeFilter): A DateRangeFilter object specifying
            the date range for metric collection. Contains start_date, end_date,
            label, preset, and days_in_range attributes.
        unavailable_sheets (dict, optional): Dictionary mapping sheet group names
            (str) to error reasons (str) for sheets that couldn't be accessed.
            Used for graceful degradation in reports. Defaults to None.

    Returns:
        dict: A dictionary containing all collected metrics plus date range info:
            - 'total_changes' (int): Total number of changes in the range
            - 'groups' (dict): Change counts by group
            - 'phases' (dict): Change counts by phase
            - 'users' (dict): Change counts by user
            - 'group_phase_user' (dict): Nested dict for detailed breakdowns
            - 'unavailable_sheets' (dict): Passed-through unavailable sheets info
            - 'date_range_info' (dict): Contains start_date, end_date, label,
              preset, days_in_range, and daily_average

    Examples:
        >>> from date_range_filter import DateRangeFilter, DateRangePreset
        >>> filter = DateRangeFilter(preset=DateRangePreset.LAST_30_DAYS)
        >>> metrics = collect_metrics_for_range(filter)
        >>> metrics['date_range_info']['days_in_range']
        30
        >>> metrics['date_range_info']['daily_average']
        5.2
    """
    changes = load_changes_with_filter(date_range_filter)
    metrics = collect_metrics(changes, unavailable_sheets)

    # Add date range information to metrics
    metrics["date_range_info"] = {
        "start_date": date_range_filter.start_date.isoformat(),
        "end_date": date_range_filter.end_date.isoformat(),
        "label": date_range_filter.label,
        "preset": date_range_filter.preset.value if hasattr(date_range_filter.preset, 'value') else str(date_range_filter.preset),
        "days_in_range": date_range_filter.days_in_range,
        "daily_average": len(changes) / max(date_range_filter.days_in_range, 1),
    }

    return metrics

def collect_metrics(
    changes: List[Dict[str, Any]],
    unavailable_sheets: Optional[Dict[str, str]] = None
) -> Dict[str, Any]:
    """Aggregate change records into metrics for reporting.

    Processes a list of change records and calculates summary statistics
    including counts by group, phase, and user. Creates a hierarchical
    data structure for detailed group-phase-user breakdowns.

    If no changes are provided, generates sample data to demonstrate
    report layout and functionality.

    Args:
        changes (list): List of change record dictionaries, each containing
            'Group', 'Phase', and 'User' keys. Typically from load_changes().
        unavailable_sheets (dict, optional): Dictionary mapping sheet group
            names to error reasons for sheets that couldn't be accessed.
            Included in returned metrics for display in reports.
            Defaults to None (empty dict).

    Returns:
        dict: A dictionary containing aggregated metrics:
            - 'total_changes' (int): Total number of change records
            - 'groups' (defaultdict): Change counts keyed by group name
            - 'phases' (defaultdict): Change counts keyed by phase number
            - 'users' (defaultdict): Change counts keyed by username
            - 'group_phase_user' (defaultdict): Nested dict structure
              [group][phase][user] -> count for detailed breakdowns
            - 'unavailable_sheets' (dict): Passed-through unavailable sheets

    Note:
        When changes list is empty, sample data is returned with realistic
        group names (NA, NF, NH, NP, NT, NV, NM, BUNDLE_FAN, BUNDLE_COOLER)
        and sample user initials to demonstrate report formatting.

    Examples:
        >>> changes = [
        ...     {'Group': 'NA', 'Phase': '1', 'User': 'DM'},
        ...     {'Group': 'NA', 'Phase': '2', 'User': 'EK'},
        ... ]
        >>> metrics = collect_metrics(changes)
        >>> metrics['total_changes']
        2
        >>> metrics['groups']['NA']
        2
    """
    import time as time_module
    metrics_start_time = time_module.perf_counter()

    metrics = {
        "total_changes": len(changes),
        "groups": defaultdict(int),
        "phases": defaultdict(int),
        "users": defaultdict(int),
        "group_phase_user": defaultdict(lambda: defaultdict(lambda: defaultdict(int))),
        "unavailable_sheets": unavailable_sheets or {},  # Track unavailable sheets for report display
    }
    
    # Add sample data if no changes
    if not changes:
        # Sample data for empty report - includes bundle groups
        metrics["groups"] = {
            "NA": 5, "NF": 3, "NH": 2, "NP": 1, "NT": 4, "NV": 2, "NM": 3,
            "BUNDLE_FAN": 6, "BUNDLE_COOLER": 2  # Include bundle groups in sample data
        }
        metrics["phases"] = {"1": 3, "2": 4, "3": 2, "4": 1, "5": 3}

        # Sample user data for each group - MODIFIED: Use actual user initials with counts
        sample_users = {"DM": 8, "EK": 7, "HI": 6, "SM": 5, "JHU": 9, "LK": 4}
        metrics["users"] = sample_users  # Add sample users to metrics

        # Create random distribution of users across phases
        for group in metrics["groups"]:
            for phase in metrics["phases"]:
                for user, count in sample_users.items():
                    # Create random distribution of users across phases
                    if (ord(group[-1]) + ord(phase) + ord(user[-1])) % 3 == 0:
                        metrics["group_phase_user"][group][phase][user] = (ord(group[-1]) + ord(phase) + ord(user[-1])) % 5 + 1

        return metrics
    
    # Process real data
    for change in changes:
        group = change.get('Group', '')
        phase = change.get('Phase', '')
        user = change.get('User', '')
        
        metrics["groups"][group] += 1
        metrics["phases"][phase] += 1
        metrics["users"][user] += 1
        
        # Detailed metrics for group-phase-user breakdown
        if group and phase and user:
            metrics["group_phase_user"][group][phase][user] += 1

    # Log timing for metrics collection
    metrics_duration = time_module.perf_counter() - metrics_start_time
    log_timing("data_processing.collect_metrics", metrics_duration,
               changes_processed=len(changes),
               groups_count=len(metrics["groups"]),
               users_count=len(metrics["users"]))

    return metrics

def get_column_map(sheet_id: int) -> Optional[Dict[str, int]]:
    """Fetch a mapping of column names to column IDs for a Smartsheet.

    Connects to the Smartsheet API and retrieves column metadata for the
    specified sheet. Uses retry logic for resilience against transient
    API failures.

    Args:
        sheet_id (int): The unique Smartsheet ID of the sheet to query.
            Can be found in SHEET_IDS constant or from Smartsheet URL.

    Returns:
        dict or None: A dictionary mapping column titles (str) to column
            IDs (int). Returns None if the sheet cannot be retrieved
            (e.g., network error, permission denied, sheet not found).

    Side Effects:
        - Makes HTTP requests to Smartsheet API
        - Logs discovered column titles for debugging
        - Logs success/failure messages

    Raises:
        No exceptions are raised; errors are caught and logged, returning None.

    Examples:
        >>> col_map = get_column_map(6141179298008964)
        >>> col_map['Kontrolle']
        1234567890123456
        >>> 'BE am' in col_map
        True
        >>> get_column_map(invalid_id)  # Returns None on error
        None

    See Also:
        SmartsheetRetryClient: Provides retry logic for API calls.
    """
    try:
        base_client = smartsheet.Smartsheet(token)
        base_client.errors_as_exceptions(True)
        client = SmartsheetRetryClient(
            base_client,
            max_retries=3,
            base_delay=1.0,
            max_delay=30.0,
            continue_on_failure=True
        )
        sheet = client.get_sheet(sheet_id, include=['columns'])

        # Check if sheet retrieval failed after retries
        if sheet is None:
            logger.warning(f"Sheet {sheet_id} not available after retries, returning empty column map")
            return None

        # --- NEW: Log the discovered column titles for debugging ---
        discovered_columns = [col.title for col in sheet.columns]
        logger.info(f"Discovered columns in sheet {sheet_id}: {discovered_columns}")
        # --- END NEW ---

        column_map = {col.title: col.id for col in sheet.columns}
        logger.info(f"Successfully created column map for sheet ID {sheet_id}.")
        return column_map
    except Exception as e:
        logger.error(f"Failed to create column map for sheet ID {sheet_id}: {e}", exc_info=True)
        return None

def get_sheet_summary_data(sheet_id: int) -> Optional[Dict[str, str]]:
    """Fetch summary field data from a Smartsheet's summary section.

    Retrieves the summary fields (displayed at the top of a Smartsheet)
    which typically contain aggregate statistics, KPIs, or metadata about
    the sheet contents. Uses retry logic for API resilience.

    Args:
        sheet_id (int): The unique Smartsheet ID of the sheet whose summary
            should be retrieved.

    Returns:
        dict or None: A dictionary mapping summary field titles (str) to their
            display values (str). Returns None if summary cannot be retrieved
            (e.g., sheet not found, no permission, network error).

    Side Effects:
        - Makes HTTP requests to Smartsheet API
        - Logs info/error messages about the operation

    Note:
        Summary fields are different from regular sheet columns. They appear
        in the summary pane of a Smartsheet and are used for high-level metrics.

    Examples:
        >>> summary = get_sheet_summary_data(6141179298008964)
        >>> summary['Total Products']
        '1,779'
        >>> summary['Overdue Count']
        '42'
        >>> get_sheet_summary_data(invalid_id)  # Returns None on error
        None
    """
    try:
        base_client = smartsheet.Smartsheet(token)
        base_client.errors_as_exceptions(True)
        client = SmartsheetRetryClient(
            base_client,
            max_retries=3,
            base_delay=1.0,
            max_delay=30.0,
            continue_on_failure=True
        )
        logger.info(f"Fetching sheet summary for sheet ID {sheet_id}...")

        # This is the core API call to get the summary (with retry logic)
        summary = client.get_sheet_summary(sheet_id)

        # Check if summary retrieval failed after retries
        if summary is None:
            logger.warning(f"Sheet summary for {sheet_id} not available after retries")
            return None

        # Convert the list of summary fields into a simple dictionary
        summary_data = {field.title: field.display_value for field in summary.fields}

        logger.info(f"Successfully fetched {len(summary_data)} summary fields.")
        return summary_data

    except Exception as e:
        logger.error(f"Failed to fetch sheet summary for sheet ID {sheet_id}: {e}", exc_info=True)
        return None


def check_sheet_availability() -> Dict[str, str]:
    """Check availability of all configured sheets before report generation.

    Proactively tests access to each sheet defined in SHEET_IDS to identify
    any that are unavailable. This enables graceful degradation in reports
    by showing placeholder sections for inaccessible sheets rather than
    failing the entire report.

    Returns:
        dict: A dictionary mapping sheet group names (str) to error reasons (str)
            for any unavailable sheets. An empty dict indicates all sheets are
            accessible. Possible error reasons include:
            - "Sheet not accessible after retries"
            - "Sheet not found"
            - "Permission denied"
            - "Rate limited"
            - "Error: <truncated error message>"

    Side Effects:
        - Makes HTTP requests to Smartsheet API for each sheet in SHEET_IDS
        - Logs warnings for each unavailable sheet
        - Logs summary of availability check results

    Note:
        The "SPECIAL" sheet is skipped during availability checks as it's
        handled separately for special activities.

    Examples:
        >>> unavailable = check_sheet_availability()
        >>> len(unavailable)
        0  # All sheets available
        >>> unavailable = check_sheet_availability()
        >>> unavailable.get('NV')
        'Permission denied'

    See Also:
        create_data_unavailable_section: Creates placeholder for unavailable groups.
    """
    unavailable_sheets = {}

    try:
        base_client = smartsheet.Smartsheet(token)
        base_client.errors_as_exceptions(True)
        client = SmartsheetRetryClient(
            base_client,
            max_retries=2,  # Fewer retries for availability check
            base_delay=1.0,
            max_delay=15.0,
            continue_on_failure=True
        )

        for group, sheet_id in SHEET_IDS.items():
            if group == "SPECIAL":
                continue  # Skip special activities sheet for this check

            try:
                # Try to get minimal sheet info
                sheet = client.get_sheet(sheet_id, include=['columns'])

                if sheet is None:
                    unavailable_sheets[group] = "Sheet not accessible after retries"
                    logger.warning(f"Sheet {group} (ID: {sheet_id}) is unavailable for report generation")
            except Exception as e:
                error_msg = str(e)
                if "not found" in error_msg.lower():
                    unavailable_sheets[group] = "Sheet not found"
                elif "permission" in error_msg.lower() or "access" in error_msg.lower():
                    unavailable_sheets[group] = "Permission denied"
                elif "rate limit" in error_msg.lower():
                    unavailable_sheets[group] = "Rate limited"
                else:
                    unavailable_sheets[group] = f"Error: {error_msg[:50]}"
                logger.warning(f"Sheet {group} unavailable: {unavailable_sheets[group]}")

    except Exception as e:
        logger.error(f"Error during sheet availability check: {e}")
        # Don't mark all sheets as unavailable on general error - let individual checks fail

    if unavailable_sheets:
        logger.info(f"Sheet availability check complete: {len(unavailable_sheets)} sheet(s) unavailable: {list(unavailable_sheets.keys())}")
    else:
        logger.info("Sheet availability check complete: All sheets available")

    return unavailable_sheets


def create_data_unavailable_section(
    group: str,
    reason: str,
    styles: Any
) -> List[Any]:
    """Create a styled placeholder section for groups whose data is unavailable.

    Generates ReportLab flowable elements that display a graceful "data unavailable"
    message when a sheet cannot be accessed. Uses muted styling to visually
    distinguish from available groups while maintaining consistent layout.

    Args:
        group (str): The group name (e.g., "NA", "NF", "BUNDLE_FAN") for
            which data is unavailable.
        reason (str): Human-readable explanation of why data is unavailable
            (e.g., "Permission denied", "Sheet not found").
        styles (reportlab.lib.styles.StyleSheet1): ReportLab stylesheet object
            for creating paragraph styles. Typically from getSampleStyleSheet().

    Returns:
        list: A list of ReportLab flowables (Table, Spacer, Paragraph) that
            can be appended to a document story. Includes:
            - A muted grey header with group name
            - A styled message box with warning icon
            - Reason for unavailability
            - Note about report continuation

    Examples:
        >>> from reportlab.lib.styles import getSampleStyleSheet
        >>> styles = getSampleStyleSheet()
        >>> elements = create_data_unavailable_section("NA", "Permission denied", styles)
        >>> len(elements)
        5
        >>> # Add to report story
        >>> story.extend(elements)

    See Also:
        check_sheet_availability: Detects which sheets are unavailable.
    """
    from reportlab.platypus import Paragraph, Spacer, Table, TableStyle

    elements = []

    # Get group color (greyed out version for unavailable)
    group_color = GROUP_COLORS.get(group, colors.steelblue)
    # Create a muted/greyed version of the color
    muted_color = colors.HexColor("#9CA3AF")  # Grey color for unavailable

    # Create header similar to available groups but with muted styling
    display_name = get_group_display_name(group)
    group_header_data = [[f"{display_name} Details"]]
    group_header = Table(group_header_data, colWidths=[150*mm], rowHeights=[10*mm])
    group_header.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), muted_color),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 14),
    ]))
    elements.append(group_header)
    elements.append(Spacer(1, 5*mm))

    # Create "Data unavailable" message box
    unavailable_style = ParagraphStyle(
        'DataUnavailable',
        parent=styles['Normal'],
        fontSize=12,
        textColor=colors.HexColor("#6B7280"),
        alignment=1,  # Center
        spaceAfter=10,
    )

    warning_style = ParagraphStyle(
        'WarningText',
        parent=styles['Normal'],
        fontSize=10,
        textColor=colors.HexColor("#9CA3AF"),
        alignment=1,  # Center
    )

    # Create a styled box for the unavailable message
    message_data = [[
        Paragraph("⚠️ Data Unavailable", unavailable_style),
    ], [
        Paragraph(f"The {display_name} sheet could not be accessed.", warning_style),
    ], [
        Paragraph(f"Reason: {reason}", warning_style),
    ], [
        Paragraph("Report generation continued with available data.", warning_style),
    ]]

    message_table = Table(message_data, colWidths=[150*mm])
    message_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#F3F4F6")),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#D1D5DB")),
        ('TOPPADDING', (0, 0), (-1, -1), 15),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 15),
        ('LEFTPADDING', (0, 0), (-1, -1), 20),
        ('RIGHTPADDING', (0, 0), (-1, -1), 20),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))

    elements.append(Spacer(1, 20*mm))
    elements.append(message_table)
    elements.append(Spacer(1, 20*mm))

    return elements


def create_overdue_status_chart(summary_data: Dict[str, Any], width: int = 500, height: int = 80, show_labels: bool = True) -> "Drawing":
    """Create a horizontal stacked bar chart visualizing product overdue status.

    Generates a ReportLab Drawing showing the distribution of products across
    different overdue status categories. Uses WCAG AA compliant colors for
    accessibility and includes a legend with values and percentages.

    Args:
        summary_data (dict): Dictionary containing overdue status values.
            Expected keys are status category names (e.g., "On Time", "Overdue").
            Values should be numeric or numeric strings.
        width (int, optional): Chart width in points. Defaults to 500.
        height (int, optional): Chart height in points. Defaults to 80.
        show_labels (bool, optional): If True, display value and percentage
            labels directly on bar segments. Defaults to True.

    Returns:
        reportlab.graphics.shapes.Drawing: A Drawing object containing the
            stacked bar chart with title and legend. Can be added to a
            ReportLab story or rendered directly.

    Note:
        - Returns a drawing with "No overdue status data available" message
          if total products is 0
        - Labels are only shown on segments wider than 30 points
        - Percentage is included if segment is wider than 60 points

    Examples:
        >>> summary_data = {"On Time": 100, "Warning": 20, "Overdue": 5}
        >>> chart = create_overdue_status_chart(summary_data)
        >>> # Add to report
        >>> story.append(chart)

    See Also:
        get_reportlab_overdue_status_colors: Provides the color scheme.
    """

    drawing = Drawing(width, height)

    # Define the categories and their WCAG AA compliant colors
    status_categories = get_reportlab_overdue_status_colors()

    # Extract values from summary_data, handling potential errors
    status_values = {}
    for cat in status_categories:
        try:
            # Ensure the value is treated as a number, default to 0
            value = summary_data.get(cat, '0') or '0'
            status_values[cat] = int(str(value).replace('.', ''))
        except (ValueError, TypeError):
            status_values[cat] = 0

    total_products = sum(status_values.values())

    if total_products == 0:
        drawing.add(String(width/2, height/2, "No overdue status data available.", textAnchor='middle'))
        return drawing

    # Chart dimensions
    chart_x = 50
    chart_width = width - 100
    bar_height = 25
    y_pos = 30

    # Draw the stacked bar segments with enhanced labels
    x_start = chart_x
    for category, color in status_categories.items():
        value = status_values[category]
        if value > 0:
            segment_width = (value / total_products) * chart_width
            percentage = (value / total_products * 100)

            rect = Rect(x_start, y_pos, segment_width, bar_height, fillColor=color, strokeColor=colors.black, strokeWidth=0.5)
            drawing.add(rect)

            # Add labels if enabled and segment is wide enough
            if show_labels and segment_width > 30:
                # Get accessible text color based on segment background
                segment_color_hex = "#{:02x}{:02x}{:02x}".format(
                    int(color.red * 255),
                    int(color.green * 255),
                    int(color.blue * 255)
                )
                text_color_hex = get_accessible_text_color(segment_color_hex)
                text_color = colors.HexColor(text_color_hex)

                # Show value and percentage if segment is wide enough
                if segment_width > 60:
                    # Show both value and percentage
                    label_text = f"{value} ({percentage:.0f}%)"
                else:
                    # Show just value
                    label_text = str(value)

                label = String(
                    x_start + segment_width/2,
                    y_pos + bar_height/2,
                    label_text,
                    textAnchor='middle',
                    fontName='Helvetica-Bold',
                    fontSize=8,
                    fillColor=text_color
                )
                drawing.add(label)

            x_start += segment_width

    drawing.add(String(width/2, height - 20, "Product Overdue Status Breakdown", textAnchor='middle', fontName='Helvetica-Bold'))

    # Add a legend below the bar with values and percentages
    legend_y = 10
    x_start = chart_x
    for category, color in status_categories.items():
        value = status_values[category]
        percentage = (value / total_products * 100) if total_products > 0 else 0

        drawing.add(Rect(x_start, legend_y, 8, 8, fillColor=color))
        legend_text = f"{category}: {value} ({percentage:.1f}%)"
        drawing.add(String(x_start + 12, legend_y, legend_text, fontName='Helvetica', fontSize=8))
        x_start += stringWidth(legend_text, 'Helvetica', 8) + 20

    return drawing


def get_special_activities(start_date: date, end_date: date) -> Tuple[Dict[str, Dict[str, Any]], int, float]:
    """Fetch and process special activities from the dedicated Smartsheet.

    Retrieves special activity records (non-standard work items like meetings,
    compliance tasks, etc.) from the SPECIAL_ACTIVITIES_SHEET_ID sheet and
    aggregates them by user.

    Args:
        start_date (date): Start of the date range to query (inclusive).
        end_date (date): End of the date range to query (inclusive).

    Returns:
        tuple: A tuple containing three elements:
            - user_activity (dict): Dictionary mapping usernames to their activity
              data. Each user entry contains:
                - 'count' (int): Number of activities
                - 'hours' (float): Total hours spent
                - 'categories' (dict): Hours by activity category
            - total_activities (int): Total count of activities in the range
            - total_hours (float): Sum of all hours across all activities

    Side Effects:
        - Makes HTTP requests to Smartsheet API
        - Logs progress and error messages

    Note:
        Returns ({}, 0, 0) if SPECIAL_ACTIVITIES_SHEET_ID is not set or
        if the sheet cannot be accessed.

    Examples:
        >>> from datetime import date
        >>> activities, count, hours = get_special_activities(
        ...     date(2024, 1, 1), date(2024, 1, 31)
        ... )
        >>> activities['DM']['hours']
        45.5
        >>> count
        25
    """
    if not SPECIAL_ACTIVITIES_SHEET_ID:
        logger.warning("SPECIAL_ACTIVITIES_SHEET_ID not set. Skipping.")
        return {}, 0, 0

    try:
        base_client = smartsheet.Smartsheet(token)
        base_client.errors_as_exceptions(True)
        client = SmartsheetRetryClient(
            base_client,
            max_retries=3,
            base_delay=1.0,
            max_delay=30.0,
            continue_on_failure=True
        )
        sheet = client.get_sheet(SPECIAL_ACTIVITIES_SHEET_ID)

        # Check if sheet retrieval failed after retries
        if sheet is None:
            logger.warning("Special activities sheet not available after retries")
            return {}, 0, 0

        # --- NEW: Validate the response from the API ---
        if isinstance(sheet, smartsheet.models.Error):
            logger.error(f"Error fetching special activities sheet: {sheet.message}")
            # Return empty values to prevent crashing the report
            return {}, 0, 0
        # --- END NEW ---

        logger.info(f"Retrieved special activities sheet with {len(sheet.rows)} rows")

        # Get column IDs
        col_map = {col.title: col.id for col in sheet.columns}
        user_col_id = col_map.get("Wer")
        date_col_id = col_map.get("Datum")
        category_col_id = col_map.get("Kategorie")
        duration_col_id = col_map.get("Zeitaufwand (Stunden)")

        if not all([user_col_id, date_col_id, category_col_id, duration_col_id]):
            logger.error("One or more required columns (Wer, Datum, Kategorie, Zeitaufwand) not found in special activities sheet.")
            return {}, 0, 0

        user_activity = {}
        total_activities = 0
        total_hours = 0

        for row in sheet.rows:
            date_cell = row.get_column(date_col_id)
            if date_cell and date_cell.value:
                try:
                    activity_date = datetime.strptime(date_cell.value, '%Y-%m-%d').date()
                    if start_date <= activity_date <= end_date:
                        user_cell = row.get_column(user_col_id)
                        category_cell = row.get_column(category_col_id)
                        duration_cell = row.get_column(duration_col_id)

                        user = user_cell.value if user_cell else "Unassigned"
                        category = category_cell.value if category_cell else "Uncategorized"
                        
                        duration = 0
                        if duration_cell and duration_cell.value:
                            try:
                                # Handle both comma and dot as decimal separators
                                duration_str = str(duration_cell.value).replace(',', '.')
                                duration = float(duration_str)
                            except (ValueError, TypeError):
                                duration = 0 # Ignore invalid duration values

                        if user not in user_activity:
                            user_activity[user] = {"count": 0, "hours": 0, "categories": {}}
                        
                        user_activity[user]["count"] += 1
                        user_activity[user]["hours"] += duration
                        user_activity[user]["categories"][category] = user_activity[user]["categories"].get(category, 0) + duration

                        total_activities += 1
                        total_hours += duration

                except (ValueError, TypeError):
                    continue # Skip rows with invalid date format

        logger.info(f"Processed {total_activities} special activities in the date range")
        logger.info(f"Found {len(user_activity)} categories with {total_hours:.1f} total hours")
        return user_activity, total_activities, total_hours

    except Exception as e:
        logger.error(f"Failed to get special activities: {e}", exc_info=True)
        return {}, 0, 0
def get_user_special_activities(user_name: str, days: int = 30) -> Tuple[List[Tuple[str, float]], float]:
    """
    Fetch special activities for a specific user.
    
    Args:
        user_name: The name of the user to filter for
        days: Number of days to look back (default 30)
    
    Returns:
        Tuple of (sorted_category_hours, total_hours) containing activity data
    """
    client = smartsheet.Smartsheet(token)
    
    try:
        # Get the special activities sheet
        sheet_id = SHEET_IDS.get("SPECIAL")
        sheet = client.Sheets.get_sheet(sheet_id)
        
        # Current date for calculating the date range
        current_date = datetime.now().date()
        cutoff_date = current_date - timedelta(days=days)
        
        # Map column titles to IDs
        col_map = {col.title: col.id for col in sheet.columns}
        
        # Check if required columns exist
        required_columns = ["Mitarbeiter", "Datum", "Kategorie", "Arbeitszeit in Std"]
        missing_columns = [col for col in required_columns if col not in col_map]
        if missing_columns:
            logger.warning(f"Missing columns in special activities sheet: {missing_columns}")
            return [], 0
        
        # Extract data from rows for this specific user
        activities = []
        
        for row in sheet.rows:
            activity = {}
            user_match = False
            
            # Extract cell values
            for cell in row.cells:
                for col_name, col_id in col_map.items():
                    if cell.column_id == col_id and cell.value is not None:
                        activity[col_name] = cell.value
                        
                        # Check if this is the right user
                        if col_name == "Mitarbeiter" and cell.value == user_name:
                            user_match = True
            
            # Skip if not the user we're looking for or missing key data
            if not user_match or not activity.get("Kategorie") or not activity.get("Arbeitszeit in Std"):
                continue
                
            # Parse date
            if "Datum" in activity:
                try:
                    activity_date = parse_date(activity["Datum"])
                    if activity_date and activity_date >= cutoff_date:
                        # Convert hours from string to float (handling comma as decimal separator)
                        if "Arbeitszeit in Std" in activity:
                            hours_str = str(activity["Arbeitszeit in Std"]).replace(',', '.')
                            try:
                                activity["Hours"] = float(hours_str)
                            except ValueError:
                                activity["Hours"] = 0
                        else:
                            activity["Hours"] = 0
                            
                        activities.append(activity)
                except Exception as e:
                    logger.debug(f"Error parsing date in special activities: {e}")
        
        # Group by category and sum hours
        category_hours = defaultdict(float)
        for activity in activities:
            category = activity.get("Kategorie", "Unknown")
            hours = activity.get("Hours", 0)
            category_hours[category] += hours
        
        # Sort by hours descending
        sorted_categories = sorted(category_hours.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate total hours
        total_hours = sum(category_hours.values())
        
        logger.info(f"User {user_name}: Found {len(activities)} special activities in the last {days} days")
        logger.info(f"User {user_name}: Found {len(category_hours)} categories with {total_hours:.1f} total hours")
        
        return sorted_categories, total_hours
        
    except Exception as e:
        logger.error(f"Error fetching special activities for user {user_name}: {e}")
        return [], 0

def generate_sample_special_activities() -> Tuple[List[Tuple[str, float]], float]:
    """Generate sample special activities data for testing and fallback."""
    # Sample data similar to the example image
    sample_data = [
        ("Compliance", 17.25),
        ("Meetings", 23.25),
        ("Meeting Vor- & Nachbereitung", 9.75),
        ("Organisatorische Aufgaben", 20.25),
        ("Primary Case/Gerd (Technical Account Manager)", 17.25),
        ("Produkte anlegen", 10.50),
        ("Research", 1.50),
        ("Search Suppressed", 1.25),
        ("Feed File Upload", 1.00),
        ("Anderes", 9.00),
        ("A+", 3.00),
    ]
    
    # Calculate total hours
    total_hours = sum(hours for _, hours in sample_data)
    
    return sample_data, total_hours

def create_activities_pie_chart(category_hours: List[Tuple[str, float]], total_hours: float, width: int = 500, height: int = 400, show_labels: bool = True) -> "Drawing":
    """Create an enhanced donut chart showing hours by activity category with modern styling.

    This function creates a modern donut-style chart with:
    - Vibrant WCAG-compliant color palette
    - Data labels showing both percentage AND hours
    - Center display with total hours
    - Enhanced legend with category details
    - Better visual hierarchy and modern appearance

    Args:
        category_hours: List of (category_name, hours) tuples
        total_hours: Total hours across all categories
        width: Chart width in points
        height: Chart height in points
        show_labels: If True, display value labels on pie slices (default: True)
    """
    from reportlab.graphics.shapes import Wedge, Circle

    drawing = Drawing(width, height)

    # Add title with modern styling
    drawing.add(String(width/2, height-20,
                      f"Summe Stunden Sonderaktivitäten letzte 30T",
                      fontName='Helvetica-Bold', fontSize=14, textAnchor='middle'))

    # Donut chart dimensions
    chart_center_x = width * 0.28
    chart_center_y = height / 2.2
    outer_radius = min(width, height) * 0.28
    inner_radius = outer_radius * 0.55  # Creates the donut hole

    # Limit to top categories if there are too many
    max_slices = 12
    if len(category_hours) > max_slices:
        top_categories = category_hours[:max_slices-1]
        other_hours = sum(hours for _, hours in category_hours[max_slices-1:])
        if other_hours > 0:
            chart_data = top_categories + [("Other", other_hours)]
        else:
            chart_data = top_categories
    else:
        chart_data = category_hours

    # Use enhanced donut color palette for more vibrant appearance
    donut_palette = get_reportlab_donut_palette()

    # Draw donut segments using Wedge shapes
    start_angle = 90  # Start from top
    for i, (category, hours) in enumerate(chart_data):
        if hours <= 0:
            continue

        percentage = (hours / total_hours) * 100 if total_hours > 0 else 0
        sweep_angle = (hours / total_hours) * 360 if total_hours > 0 else 0

        # Calculate end angle (going clockwise, so subtract)
        end_angle = start_angle - sweep_angle

        # Get color for this segment
        color = donut_palette[i % len(donut_palette)]

        # Draw outer wedge
        outer_wedge = Wedge(
            chart_center_x, chart_center_y, outer_radius,
            end_angle, start_angle,
            fillColor=color,
            strokeColor=colors.white,
            strokeWidth=2
        )
        drawing.add(outer_wedge)

        start_angle = end_angle

    # Draw inner circle (donut hole) - white to create the donut effect
    inner_circle = Circle(
        chart_center_x, chart_center_y, inner_radius,
        fillColor=colors.white,
        strokeColor=colors.white,
        strokeWidth=0
    )
    drawing.add(inner_circle)

    # Add center content - total hours display
    drawing.add(String(
        chart_center_x, chart_center_y + 8,
        f"{total_hours:.1f}",
        fontName='Helvetica-Bold', fontSize=16, textAnchor='middle',
        fillColor=colors.HexColor("#1F2937")
    ))
    drawing.add(String(
        chart_center_x, chart_center_y - 8,
        "Stunden",
        fontName='Helvetica', fontSize=10, textAnchor='middle',
        fillColor=colors.HexColor("#6B7280")
    ))

    # Add data labels on donut segments for slices >= 5%
    if show_labels and total_hours > 0:
        # Calculate label positions using the enhanced label format
        label_positions = calculate_pie_label_positions(
            chart_data, total_hours, chart_center_x, chart_center_y,
            outer_radius,  # Use outer radius for label positioning
            min_percentage=5.0, font_name='Helvetica-Bold', font_size=8,
            label_format='both'  # Show both percentage and hours
        )

        # Adjust label positions for donut (place them on the donut ring)
        for label in label_positions:
            mid_angle_rad = label['mid_angle_rad']
            # Position labels at the middle of the donut ring
            label_radius = (outer_radius + inner_radius) / 2
            label['label_x'] = chart_center_x + label_radius * math.cos(mid_angle_rad)
            label['label_y'] = chart_center_y + label_radius * math.sin(mid_angle_rad)

        # Create function to get text color for each slice
        def get_slice_text_color(index: int) -> "Color":
            slice_color = donut_palette[index % len(donut_palette)]
            slice_color_hex = "#{:02x}{:02x}{:02x}".format(
                int(slice_color.red * 255),
                int(slice_color.green * 255),
                int(slice_color.blue * 255)
            )
            text_color_hex = get_accessible_text_color(slice_color_hex)
            return colors.HexColor(text_color_hex)

        # Draw labels with intelligent positioning
        draw_pie_labels(drawing, label_positions, get_slice_text_color)

    # Add enhanced legend - positioned to the right with better formatting
    legend_x = width * 0.55
    legend_y = height * 0.82
    legend_font_size = 9
    line_height = legend_font_size * 1.8

    # Legend title
    drawing.add(String(
        legend_x, legend_y + 15,
        "Kategorien",
        fontName='Helvetica-Bold', fontSize=10
    ))

    for i, (category, hours) in enumerate(chart_data):
        if hours <= 0:
            continue

        y_pos = legend_y - (i * line_height)
        color_idx = i % len(donut_palette)

        # Add rounded color indicator
        drawing.add(Rect(
            legend_x,
            y_pos - 2,
            10,
            10,
            fillColor=donut_palette[color_idx],
            strokeColor=None,
            strokeWidth=0,
            rx=2,  # Rounded corners
            ry=2
        ))

        # Add category name
        percentage = (hours / total_hours * 100) if total_hours > 0 else 0
        drawing.add(String(
            legend_x + 15,
            y_pos,
            f"{category}",
            fontName='Helvetica-Bold',
            fontSize=legend_font_size - 1
        ))

        # Add hours and percentage on separate line for clarity
        drawing.add(String(
            legend_x + 15,
            y_pos - 10,
            f"{hours:.1f}h ({percentage:.1f}%)",
            fontName='Helvetica',
            fontSize=legend_font_size - 2,
            fillColor=colors.HexColor("#6B7280")
        ))

    # Add total hours summary at bottom
    drawing.add(String(
        width/2,
        25,
        f"Gesamtstunden: {total_hours:.1f}",
        fontName='Helvetica-Bold',
        fontSize=12,
        textAnchor='middle',
        fillColor=colors.HexColor("#1F2937")
    ))

    return drawing

def create_special_activities_breakdown(category_hours: List[Tuple[str, float]], total_hours: float) -> "Table":
    """Create a detailed table showing the breakdown of special activities.

    Enhanced styling with:
    - Improved column widths for better readability
    - Alternating row colors for visual clarity
    - Proper header styling with dark background
    - Footer row for totals with distinct styling
    - Right-aligned numeric columns
    """
    # Create table data with header
    table_data = [["Category", "Hours", "% of Total"]]

    # Add data rows sorted by hours descending
    for category, hours in category_hours:
        percentage = (hours / total_hours * 100) if total_hours > 0 else 0
        table_data.append([category, f"{hours:.1f}h", f"{percentage:.1f}%"])

    # Add total row
    table_data.append(["Total", f"{total_hours:.1f}h", "100.0%"])

    # Create the table with improved column widths
    # Category column wider for long names, numeric columns properly sized
    table = Table(table_data, colWidths=[110*mm, 32*mm, 32*mm])

    # Build enhanced table style with footer
    table_style = get_enhanced_table_base_style(has_footer=True, num_rows=len(table_data))

    # Apply alternating row colors for data rows (exclude header and footer)
    num_data_rows = len(table_data) - 2  # Exclude header and footer
    if num_data_rows > 0:
        apply_alternating_row_colors(table_style, num_data_rows=num_data_rows, start_row=1)

    # Column alignment
    table_style.extend([
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),      # Category column left-aligned
        ('ALIGN', (1, 0), (-1, -1), 'RIGHT'),    # Numeric columns right-aligned
        ('ALIGN', (1, 0), (-1, 0), 'CENTER'),    # Header row centered
    ])

    table.setStyle(TableStyle(table_style))

    return table

def get_marketplace_activity(group_name: str, sheet_id: int, start_date: date, end_date: date) -> Dict[str, Any]:
    """
    Analyzes a sheet to get activity metrics per marketplace, and determines
    most and least active marketplaces based on the last activity date.
    """
    try:
        logger.info(f"Processing sheet {group_name} for marketplace activity")
        base_client = smartsheet.Smartsheet(token)
        base_client.errors_as_exceptions(True)
        client = SmartsheetRetryClient(
            base_client,
            max_retries=3,
            base_delay=1.0,
            max_delay=30.0,
            continue_on_failure=True
        )
        sheet = client.get_sheet(sheet_id)

        # Check if sheet retrieval failed after retries
        if sheet is None:
            logger.warning(f"Sheet {group_name} not available after retries for marketplace activity")
            return [], []

        if isinstance(sheet, smartsheet.models.Error):
            logger.error(f"Error processing sheet {group_name} for marketplace activity: {sheet.message}")
            return [], []

        col_map = {col.title: col.id for col in sheet.columns}

        # Detect missing columns with detailed logging (graceful degradation)
        column_status = detect_missing_columns(col_map, group_name)

        # Check for required columns
        marketplace_col_id = col_map.get("Amazon")
        date_cols = {title: col_id for title, col_id in col_map.items() if " am" in title or "Kontrolle" in title}

        if not marketplace_col_id or not date_cols:
            logger.warning(f"Missing required columns in sheet {group_name}. Skipping marketplace analysis.")
            return [], []

        # This dictionary will store the last activity date for each product (row)
        product_last_activity = {}

        for row in sheet.rows:
            last_date = None
            for cell in row.cells:
                if cell.column_id in date_cols.values():
                    try:
                        cell_date = parse_date(cell.value)
                        if cell_date and (last_date is None or cell_date > last_date):
                            last_date = cell_date
                    except (ValueError, TypeError):
                        continue
            
            if last_date:
                product_last_activity[row.id] = last_date

        # This dictionary will store activity counts per marketplace
        marketplace_counts = defaultdict(int)
        # This dictionary will store the days since last activity for each marketplace
        marketplace_days_since_activity = defaultdict(list)
        
        today = datetime.now().date()

        for row in sheet.rows:
            if row.id in product_last_activity:
                marketplace_cell = row.get_column(marketplace_col_id)
                if marketplace_cell and marketplace_cell.value:
                    marketplace_code = marketplace_cell.value.strip().upper()
                    marketplace_counts[marketplace_code] += 1
                    
                    days_diff = (today - product_last_activity[row.id]).days
                    marketplace_days_since_activity[marketplace_code].append(days_diff)

        # Calculate average days since last activity
        avg_days = {}
        for mp, days_list in marketplace_days_since_activity.items():
            avg_days[mp] = sum(days_list) / len(days_list) if days_list else 0

        # Combine the data
        combined_data = []
        for mp, count in marketplace_counts.items():
            # Here, we use the raw 'mp' code directly
            combined_data.append((mp, avg_days.get(mp, 0), count))

        # Sort to find most and least active
        # Most active: lower average days since activity
        most_active = sorted(combined_data, key=lambda x: x[1])[:5]
        # Most inactive: higher average days since activity
        most_inactive = sorted(combined_data, key=lambda x: x[1], reverse=True)[:5]
        
        logger.info(f"Found {len(most_active)} most active and {len(most_inactive)} most inactive marketplaces for {group_name}")
        
        return most_active, most_inactive

    except Exception as e:
        logger.error(f"Error getting marketplace activity for group {group_name}: {e}", exc_info=True)
        return [], []


def generate_sample_marketplace_data() -> Tuple[List[Tuple[str, int, int]], List[Tuple[str, int, int]]]:
    """Generate sample marketplace activity data."""
    # Sample marketplaces with different activity levels (marketplace, days, count)
    sample_marketplaces = [
        ("Amazon DE", 12, 45),
        ("Amazon FR", 8, 32),
        ("Amazon UK", 5, 28),
        ("Amazon IT", 18, 25),
        ("Amazon ES", 22, 20),
        ("Amazon US", 3, 50),
        ("Amazon CA", 7, 15),
        ("Amazon JP", 30, 10),
        ("Amazon AU", 15, 8),
        ("Amazon BR", 25, 12)
    ]
    
    # Create sample active and inactive lists
    active_samples = [(m, d, c) for m, d, c in sample_marketplaces if d < 20][:5]
    inactive_samples = [(m, d, c) for m, d, c in sample_marketplaces if d >= 20][:5]
    
    return active_samples, inactive_samples

def create_activity_table(activity_data: List[Tuple[str, float, int]], title: str) -> "Table":
    """Create a table showing marketplace activity data."""
    if not activity_data:
        # Instead of using a custom paragraph style with Helvetica-Italic,
        # create a simple table with a "No data" message
        table_data = [["No marketplace data available"]]
        table = Table(table_data, colWidths=[70*mm])  # Reduced width
        table.setStyle(TableStyle([
            ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica'),
            ('BACKGROUND', (0,0), (-1,-1), colors.lightgrey),
        ]))
        return table
    
    # Create table data with header - SHORTENED HEADERS
    table_data = [["Country", "Avg Days", "Products"]]  # Shorter header text
    
    # Add data rows
    for country, avg_days, count in activity_data:
        table_data.append([country, f"{avg_days:.1f}", str(count)])
    
    # Create the table with REDUCED COLUMN WIDTHS
    table = Table(table_data, colWidths=[40*mm, 20*mm, 15*mm])  # Narrower columns
    
    # Style the table
    table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('ALIGN', (1,0), (2,-1), 'RIGHT'),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('FONTSIZE', (0,0), (-1,-1), 8),  # Reduced font size
    ]))
    
    return table

# Generate user colors with WCAG AA compliant colors for accessibility
def generate_user_colors(users: Dict[str, Any]) -> Dict[str, "Color"]:
    """Generate consistent WCAG AA compliant colors for user visualization.

    Assigns colors to users for use in charts and reports. Predefined users
    receive custom colors, while others are assigned from a base palette.
    All colors meet WCAG AA accessibility standards (minimum 3:1 contrast
    ratio against white background).

    Args:
        users (dict): Dictionary with usernames as keys. Values are typically
            counts but are not used for color assignment - only keys matter.

    Returns:
        dict: The global USER_COLORS dictionary, updated with color assignments.
            Maps username (str) to reportlab color object.

    Side Effects:
        - Modifies the global USER_COLORS dictionary
        - Clears any existing color assignments before reassigning

    Note:
        Some users have predefined custom colors (from get_reportlab_user_colors),
        while others receive colors from a rotating base palette.

    Examples:
        >>> users = {"DM": 10, "EK": 8, "HI": 6}
        >>> colors = generate_user_colors(users)
        >>> colors["DM"]
        Color(.545, .271, .075, 1)  # Custom brown color for DM
    """
    # WCAG AA compliant custom colors for specific users
    custom_colors = get_reportlab_user_colors()

    # WCAG AA compliant base colors for other users
    base_colors = get_reportlab_base_colors()

    # Clear existing colors
    USER_COLORS.clear()

    # First assign the custom colors
    for user in users.keys():
        if user in custom_colors:
            USER_COLORS[user] = custom_colors[user]

    # Then assign colors to any remaining users
    color_index = 0
    for user in sorted(users.keys()):
        if user and user not in USER_COLORS:
            USER_COLORS[user] = base_colors[color_index % len(base_colors)]
            color_index += 1

    return USER_COLORS

def make_group_bar_chart(data_dict: Dict[str, int], title: str, width: int = 250, height: int = 200, show_labels: bool = True) -> "Drawing":
    """Create a vertical bar chart showing activity counts by group.

    Generates a ReportLab Drawing with a bar chart displaying counts for each
    group. Bars are colored according to GROUP_COLORS. Labels show counts
    above bars and percentages inside bars (when tall enough).

    Args:
        data_dict (dict): Dictionary mapping group names (str) to count values
            (int). Example: {"NA": 150, "NF": 120, "BUNDLE_FAN": 80}
        title (str): Chart title displayed at the top center.
        width (int, optional): Chart width in points. Defaults to 250.
        height (int, optional): Chart height in points. Defaults to 200.
        show_labels (bool, optional): If True, display value labels on top
            of bars and percentage labels inside bars. Defaults to True.

    Returns:
        reportlab.graphics.shapes.Drawing: A Drawing object containing the
            bar chart. Can be added to a ReportLab story.

    Note:
        - Empty data_dict defaults to {"Sample": 1} for placeholder display
        - For more than 7 groups, labels are angled 45° and font size reduced
        - Bundle group names are abbreviated (e.g., "BUNDLE_FAN" -> "B.FAN")
        - Percentage labels only shown on bars taller than 25 pixels

    Examples:
        >>> data = {"NA": 100, "NF": 80, "NH": 60}
        >>> chart = make_group_bar_chart(data, "Changes by Group")
        >>> story.append(chart)
    """
    drawing = Drawing(width, height)

    # Add title
    drawing.add(String(width/2, height-15, title,
                      fontName='Helvetica-Bold', fontSize=12, textAnchor='middle'))

    # If data is empty, add sample data
    if not data_dict:
        data_dict = {"Sample": 1}

    # Create the bar chart
    chart = VerticalBarChart()
    chart.x = 30
    chart.y = 30
    chart.height = 130
    chart.width = width - 60

    # Sort groups alphabetically
    sorted_keys = sorted(data_dict.keys())
    # Create short display names for bundle groups using helper function
    display_names = [get_group_display_name(key, short=True) for key in sorted_keys]
    chart.categoryAxis.categoryNames = display_names
    chart.data = [list(data_dict[k] for k in sorted_keys)]

    # Add colors for each group
    for i, key in enumerate(sorted_keys):
        if key in GROUP_COLORS:
            chart.bars[0].fillColor = GROUP_COLORS[key]

    # Adjust labels - use smaller font and angle if many groups
    num_groups = len(sorted_keys)
    if num_groups > 7:
        chart.categoryAxis.labels.fontSize = 6
        chart.categoryAxis.labels.angle = 45
        chart.categoryAxis.labels.dy = -15
    else:
        chart.categoryAxis.labels.fontSize = 8
        chart.categoryAxis.labels.angle = 0
        chart.categoryAxis.labels.dy = -10
    chart.categoryAxis.labels.boxAnchor = 'n'

    # Value axis
    max_val = max(data_dict.values()) if data_dict else 10
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = max_val * 1.1
    chart.valueAxis.valueStep = max(1, int(max_val / 5)) if data_dict else 2
    chart.valueAxis.labels.fontSize = 8

    # Add group colors
    for i, group in enumerate(sorted_keys):
        if group in GROUP_COLORS:
            chart.bars[(0, i)].fillColor = GROUP_COLORS.get(group, colors.steelblue)

    drawing.add(chart)

    # Add data labels on top of each bar
    if show_labels and data_dict:
        bar_width = chart.width / len(sorted_keys)
        for i, key in enumerate(sorted_keys):
            value = data_dict[key]
            if value > 0:  # Only show label for non-zero values
                # Calculate bar center x position
                bar_x = chart.x + (i + 0.5) * bar_width
                # Calculate bar top y position (proportional to value)
                bar_y = chart.y + (value / max_val) * chart.height

                # Calculate total for percentage
                total = sum(data_dict.values())
                percentage = (value / total * 100) if total > 0 else 0

                # Determine if we have room for percentage (check bar height)
                bar_height_px = (value / max_val) * chart.height

                # Add value label on top of bar
                label_y_offset = 5  # Position above the bar

                # Get accessible text color based on bar background
                bar_color_hex = "#4682B4"  # Default steelblue
                if key in GROUP_COLORS:
                    bar_color = GROUP_COLORS[key]
                    # Convert reportlab color to hex for contrast calculation
                    bar_color_hex = "#{:02x}{:02x}{:02x}".format(
                        int(bar_color.red * 255),
                        int(bar_color.green * 255),
                        int(bar_color.blue * 255)
                    )

                # Primary label: value on top of bar
                drawing.add(String(
                    bar_x,
                    bar_y + label_y_offset,
                    str(value),
                    fontName='Helvetica-Bold',
                    fontSize=8,
                    textAnchor='middle',
                    fillColor=colors.black
                ))

                # Secondary label: percentage inside bar if tall enough (>25px)
                if bar_height_px > 25:
                    text_color_hex = get_accessible_text_color(bar_color_hex)
                    text_color = colors.HexColor(text_color_hex)
                    drawing.add(String(
                        bar_x,
                        bar_y - 12,  # Inside the bar
                        f"{percentage:.0f}%",
                        fontName='Helvetica',
                        fontSize=7,
                        textAnchor='middle',
                        fillColor=text_color
                    ))

    return drawing

def make_phase_bar_chart(data_dict: Dict[str, int], title: str, width: int = 250, height: int = 200, show_labels: bool = True) -> "Drawing":
    """Create a vertical bar chart showing activity counts by workflow phase.

    Generates a ReportLab Drawing with a bar chart displaying counts for each
    phase in the workflow (Phase 1-5). Bars are colored according to PHASE_COLORS.
    Labels show counts above bars and percentages inside bars when tall enough.

    Args:
        data_dict (dict): Dictionary mapping phase numbers (str) to count values
            (int). Keys should be "1", "2", "3", "4", "5" representing workflow phases.
            Example: {"1": 50, "2": 80, "3": 45, "4": 30, "5": 25}
        title (str): Chart title displayed at the top center.
        width (int, optional): Chart width in points. Defaults to 250.
        height (int, optional): Chart height in points. Defaults to 200.
        show_labels (bool, optional): If True, display value labels on top
            of bars and percentage labels inside bars. Defaults to True.

    Returns:
        reportlab.graphics.shapes.Drawing: A Drawing object containing the
            bar chart. Can be added to a ReportLab story.

    Note:
        - Phase numbers are displayed using names from PHASE_NAMES constant
          (e.g., "Phase 1", "Phase 2")
        - Empty data_dict defaults to {"Sample": 1} for placeholder display
        - Percentage labels only shown on bars taller than 25 pixels

    Examples:
        >>> phases = {"1": 100, "2": 150, "3": 80, "4": 50, "5": 30}
        >>> chart = make_phase_bar_chart(phases, "Activity by Phase")
        >>> story.append(chart)
    """
    drawing = Drawing(width, height)

    # Add title
    drawing.add(String(width/2, height-15, title,
                      fontName='Helvetica-Bold', fontSize=12, textAnchor='middle'))

    # If data is empty, add sample data
    if not data_dict:
        data_dict = {"1": 3, "2": 4, "3": 2, "4": 1, "5": 3}

    # Create the bar chart
    chart = VerticalBarChart()
    chart.x = 30
    chart.y = 30
    chart.height = 130
    chart.width = width - 60

    # Sort phases numerically
    sorted_keys = sorted(data_dict.keys(), key=lambda x: int(x) if x.isdigit() else 999)

    # Use phase names for display
    chart.categoryAxis.categoryNames = [f"{PHASE_NAMES.get(k, '')}" for k in sorted_keys]
    chart.data = [list(data_dict[k] for k in sorted_keys)]

    # Adjust labels
    chart.categoryAxis.labels.fontSize = 8
    chart.categoryAxis.labels.boxAnchor = 'n'
    chart.categoryAxis.labels.angle = 0
    chart.categoryAxis.labels.dy = -10

    # Value axis
    max_val = max(data_dict.values()) if data_dict else 10
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = max_val * 1.1
    chart.valueAxis.valueStep = max(1, int(max_val / 5)) if data_dict else 2
    chart.valueAxis.labels.fontSize = 8

    # Add phase colors
    for i, phase in enumerate(sorted_keys):
        chart.bars[(0, i)].fillColor = PHASE_COLORS.get(phase, colors.steelblue)

    drawing.add(chart)

    # Add data labels on top of each bar
    if show_labels and data_dict:
        bar_width = chart.width / len(sorted_keys)
        total = sum(data_dict.values())

        for i, key in enumerate(sorted_keys):
            value = data_dict[key]
            if value > 0:  # Only show label for non-zero values
                # Calculate bar center x position
                bar_x = chart.x + (i + 0.5) * bar_width
                # Calculate bar top y position (proportional to value)
                bar_y = chart.y + (value / max_val) * chart.height

                # Calculate percentage
                percentage = (value / total * 100) if total > 0 else 0

                # Determine if we have room for percentage (check bar height)
                bar_height_px = (value / max_val) * chart.height

                # Get accessible text color based on bar background
                phase_color = PHASE_COLORS.get(key, colors.steelblue)
                bar_color_hex = "#{:02x}{:02x}{:02x}".format(
                    int(phase_color.red * 255),
                    int(phase_color.green * 255),
                    int(phase_color.blue * 255)
                )

                # Add value label on top of bar
                label_y_offset = 5  # Position above the bar
                drawing.add(String(
                    bar_x,
                    bar_y + label_y_offset,
                    str(value),
                    fontName='Helvetica-Bold',
                    fontSize=8,
                    textAnchor='middle',
                    fillColor=colors.black
                ))

                # Secondary label: percentage inside bar if tall enough (>25px)
                if bar_height_px > 25:
                    text_color_hex = get_accessible_text_color(bar_color_hex)
                    text_color = colors.HexColor(text_color_hex)
                    drawing.add(String(
                        bar_x,
                        bar_y - 12,  # Inside the bar
                        f"{percentage:.0f}%",
                        fontName='Helvetica',
                        fontSize=7,
                        textAnchor='middle',
                        fillColor=text_color
                    ))

    return drawing



# Efficiency level colors for hours distribution chart - WCAG AA compliant
EFFICIENCY_LEVEL_COLORS = {
    EfficiencyLevel.HIGHLY_EFFICIENT: colors.HexColor("#1B5E20"),    # Green 900 - Contrast: 7.8:1
    EfficiencyLevel.EFFICIENT: colors.HexColor("#43A047"),           # Green 600 - Contrast: 3.5:1
    EfficiencyLevel.AVERAGE: colors.HexColor("#8B6914"),             # Dark Gold - Contrast: 4.5:1
    EfficiencyLevel.INEFFICIENT: colors.HexColor("#C65102"),         # Burnt Orange - Contrast: 4.8:1
    EfficiencyLevel.HIGHLY_INEFFICIENT: colors.HexColor("#B71C1C"),  # Red 900 - Contrast: 6.9:1
}


def make_hours_distribution_chart(
    viz_data: Dict[str, Any],
    title: str = "Hours Distribution by Category",
    width: int = 500,
    height: int = 300,
    show_labels: bool = True,
    show_average_line: bool = True,
    color_by_efficiency: bool = True
) -> Drawing:
    """Create vertical bar chart showing hours distribution across special activity categories.

    Generates a ReportLab Drawing with a bar chart displaying hours per item for each
    activity category. Includes an optional horizontal reference line showing the average
    hours per item across all categories. Bars can be colored by efficiency level.

    Args:
        viz_data: Visualization data from get_distribution_visualization_data().
            Required fields:
            - bar_chart: List of dicts with category, hours_per_item, average
            - summary_metrics: Dict with avg_hours_per_item
        title: Chart title displayed at the top center.
        width: Chart width in points. Defaults to 500.
        height: Chart height in points. Defaults to 300.
        show_labels: If True, display value labels on top of bars. Defaults to True.
        show_average_line: If True, display horizontal average reference line. Defaults to True.
        color_by_efficiency: If True, color bars by efficiency level. Defaults to True.

    Returns:
        reportlab.graphics.shapes.Drawing: A Drawing object containing the bar chart.

    Note:
        - Categories are displayed sorted by hours per item (descending)
        - Average line is drawn as a dashed red line with label
        - Efficiency-based coloring: green (efficient) to red (inefficient)
        - For more than 7 categories, labels are angled 45 degrees and font size reduced

    Examples:
        >>> from special_activities_hours_distribution import (
        ...     calculate_hours_distribution, get_distribution_visualization_data
        ... )
        >>> summary = calculate_hours_distribution(user_data, start_date, end_date)
        >>> viz_data = get_distribution_visualization_data(summary)
        >>> chart = make_hours_distribution_chart(viz_data)
        >>> story.append(chart)
    """
    drawing = Drawing(width, height)

    # Add title
    drawing.add(String(
        width / 2, height - 15, title,
        fontName='Helvetica-Bold', fontSize=12, textAnchor='middle'
    ))

    # Extract bar chart data
    bar_data = viz_data.get("bar_chart", [])
    summary_metrics = viz_data.get("summary_metrics", {})
    avg_hours_per_item = summary_metrics.get("avg_hours_per_item", 0)

    # Handle empty data
    if not bar_data:
        drawing.add(String(
            width / 2, height / 2, "No data available",
            fontName='Helvetica', fontSize=12, textAnchor='middle'
        ))
        return drawing

    # Sort by hours_per_item descending for visual impact
    sorted_data = sorted(bar_data, key=lambda x: x.get("hours_per_item", 0), reverse=True)

    # Create the bar chart
    chart = VerticalBarChart()
    chart.x = 50
    chart.y = 50
    chart.height = height - 100
    chart.width = width - 100

    # Extract category names and values
    category_names = [item.get("category", "Unknown")[:15] for item in sorted_data]  # Truncate long names
    hours_values = [item.get("hours_per_item", 0) for item in sorted_data]

    chart.categoryAxis.categoryNames = category_names
    chart.data = [hours_values]

    # Adjust labels - use smaller font and angle if many categories
    num_categories = len(category_names)
    if num_categories > 7:
        chart.categoryAxis.labels.fontSize = 6
        chart.categoryAxis.labels.angle = 45
        chart.categoryAxis.labels.dy = -15
    else:
        chart.categoryAxis.labels.fontSize = 8
        chart.categoryAxis.labels.angle = 0
        chart.categoryAxis.labels.dy = -10
    chart.categoryAxis.labels.boxAnchor = 'n'

    # Value axis
    max_val = max(hours_values) if hours_values else 1
    chart.valueAxis.valueMin = 0
    chart.valueAxis.valueMax = max_val * 1.15  # Extra space for labels
    chart.valueAxis.valueStep = max(0.5, round(max_val / 5, 1)) if max_val > 0 else 0.5
    chart.valueAxis.labels.fontSize = 8
    chart.valueAxis.labelTextFormat = "%.1f"

    # Assign colors to bars
    if color_by_efficiency:
        # Color based on hours_per_item relative to average
        for i, item in enumerate(sorted_data):
            hours_per_item = item.get("hours_per_item", 0)
            efficiency_level = _classify_efficiency_for_chart(hours_per_item, avg_hours_per_item)
            bar_color = EFFICIENCY_LEVEL_COLORS.get(efficiency_level, colors.steelblue)
            chart.bars[(0, i)].fillColor = bar_color
    else:
        # Use default chart palette
        palette = get_reportlab_chart_palette()
        for i in range(len(sorted_data)):
            chart.bars[(0, i)].fillColor = palette[i % len(palette)]

    drawing.add(chart)

    # Add data labels on top of each bar
    if show_labels and hours_values:
        bar_width = chart.width / len(category_names)
        for i, hours_val in enumerate(hours_values):
            if hours_val > 0:
                # Calculate bar center x position
                bar_x = chart.x + (i + 0.5) * bar_width
                # Calculate bar top y position (proportional to value)
                bar_y = chart.y + (hours_val / max_val) * chart.height

                # Add value label on top of bar
                drawing.add(String(
                    bar_x,
                    bar_y + 5,
                    f"{hours_val:.1f}",
                    fontName='Helvetica-Bold',
                    fontSize=7,
                    textAnchor='middle',
                    fillColor=colors.black
                ))

    # Add average reference line
    if show_average_line and avg_hours_per_item > 0 and avg_hours_per_item <= max_val * 1.15:
        avg_y = chart.y + (avg_hours_per_item / (max_val * 1.15)) * chart.height

        # Draw dashed average line
        line = Line(
            chart.x, avg_y,
            chart.x + chart.width, avg_y
        )
        line.strokeColor = colors.HexColor("#C41E3A")  # Cardinal Red
        line.strokeWidth = 1.5
        line.strokeDashArray = [5, 3]  # Dashed line
        drawing.add(line)

        # Add average label
        drawing.add(String(
            chart.x + chart.width + 5, avg_y,
            f"Avg: {avg_hours_per_item:.1f}",
            fontName='Helvetica-Bold',
            fontSize=7,
            textAnchor='start',
            fillColor=colors.HexColor("#C41E3A")
        ))

    # Add subtitle with summary metrics
    total_hours = summary_metrics.get("total_hours", 0)
    total_items = summary_metrics.get("total_items", 0)
    total_categories = summary_metrics.get("categories", 0)

    subtitle = f"Total: {total_hours:.1f} hrs across {total_items} items in {total_categories} categories"
    drawing.add(String(
        width / 2, 25, subtitle,
        fontName='Helvetica', fontSize=9, textAnchor='middle',
        fillColor=colors.HexColor("#546E7A")  # Blue Grey 600
    ))

    return drawing


def _classify_efficiency_for_chart(hours_per_item: float, average: float) -> EfficiencyLevel:
    """Classify efficiency level for chart coloring.

    Args:
        hours_per_item: Hours per item for this category
        average: Overall average hours per item

    Returns:
        EfficiencyLevel classification
    """
    if average <= 0:
        return EfficiencyLevel.AVERAGE

    ratio = hours_per_item / average

    if ratio <= 0.5:
        return EfficiencyLevel.HIGHLY_EFFICIENT
    elif ratio <= 0.8:
        return EfficiencyLevel.EFFICIENT
    elif ratio <= 1.2:
        return EfficiencyLevel.AVERAGE
    elif ratio <= 1.8:
        return EfficiencyLevel.INEFFICIENT
    else:
        return EfficiencyLevel.HIGHLY_INEFFICIENT

def create_funnel_chart(funnel_data: Union[Dict[str, Any], Any], title: str = "Phase Progression Funnel", width: int = 500, height: int = 350, show_labels: bool = True) -> "Drawing":
    """Create a funnel chart visualization showing items progressing through phases.

    Generates a ReportLab Drawing with a funnel chart displaying item flow through
    the 5 workflow phases (Kontrolle → BE → K → C → Reopen). Uses progressive width
    reduction to visually represent drop-offs at each phase transition.

    Args:
        funnel_data: Either a FunnelMetrics object or a dict from get_funnel_visualization_data().
            Required fields in dict:
            - phases: List of phase data with phase_number, phase_name, items_entered,
                     completion_rate, drop_off_rate, width_percent, color
            - max_value: Maximum items_entered value for scaling
            - efficiency: Funnel efficiency score (0-100)
            - overall_completion: Overall completion rate (%)
        title (str, optional): Chart title. Defaults to "Phase Progression Funnel".
        width (int, optional): Chart width in points. Defaults to 500.
        height (int, optional): Chart height in points. Defaults to 350.
        show_labels (bool, optional): If True, display labels with item counts,
            completion rates, and drop-off percentages. Defaults to True.

    Returns:
        reportlab.graphics.shapes.Drawing: A Drawing object containing the funnel
            chart. Can be added to a ReportLab story.

    Note:
        - Each funnel segment is drawn as a trapezoid with width proportional to items_entered
        - Colors are based on completion rate (green ≥80%, gold ≥60%, orange ≥40%, red <40%)
        - Drop-off arrows show items lost between phases
        - Efficiency score is displayed prominently

    Examples:
        >>> from phase_progression_funnel_calculator import calculate_funnel_metrics, get_funnel_visualization_data
        >>> funnel = calculate_funnel_metrics(changes)
        >>> viz_data = get_funnel_visualization_data(funnel)
        >>> chart = create_funnel_chart(viz_data, "Q4 Funnel Analysis")
        >>> story.append(chart)
    """
    drawing = Drawing(width, height)

    # Handle FunnelMetrics object or dict from get_funnel_visualization_data()
    if hasattr(funnel_data, 'phases') and hasattr(funnel_data, 'funnel_efficiency'):
        # It's a FunnelMetrics object - import and convert
        try:
            from phase_progression_funnel_calculator import get_funnel_visualization_data
            funnel_data = get_funnel_visualization_data(funnel_data)
        except ImportError:
            logger.warning("Could not import get_funnel_visualization_data, using raw data")

    # Extract data from the visualization dict
    phases = funnel_data.get("phases", [])
    max_value = funnel_data.get("max_value", 0)
    efficiency = funnel_data.get("efficiency", 0)
    overall_completion = funnel_data.get("overall_completion", 0)

    # Handle empty data
    if not phases or max_value == 0:
        drawing.add(String(
            width / 2, height / 2,
            "No funnel data available",
            fontName='Helvetica',
            fontSize=12,
            textAnchor='middle',
            fillColor=colors.gray
        ))
        return drawing

    # Add title
    drawing.add(String(
        width / 2, height - 15,
        title,
        fontName='Helvetica-Bold',
        fontSize=14,
        textAnchor='middle',
        fillColor=colors.black
    ))

    # Define funnel dimensions
    funnel_top_y = height - 50
    funnel_bottom_y = 80
    funnel_height = funnel_top_y - funnel_bottom_y
    center_x = width / 2
    max_half_width = (width - 100) / 2  # Leave margins

    # Calculate segment heights
    num_phases = len(phases)
    segment_height = funnel_height / num_phases
    segment_gap = 3  # Small gap between segments

    # Draw efficiency score badge in top-right corner
    badge_x = width - 60
    badge_y = height - 35
    badge_radius = 22

    # Efficiency color based on score
    if efficiency >= 80:
        efficiency_color = colors.HexColor("#1B5E20")  # Green
    elif efficiency >= 60:
        efficiency_color = colors.HexColor("#8B6914")  # Gold
    elif efficiency >= 40:
        efficiency_color = colors.HexColor("#E65100")  # Orange
    else:
        efficiency_color = colors.HexColor("#B71C1C")  # Red

    # Draw efficiency circle
    drawing.add(Circle(badge_x, badge_y, badge_radius, fillColor=efficiency_color, strokeColor=None))

    # Efficiency text
    efficiency_text_color = colors.HexColor(get_accessible_text_color(
        "#{:02x}{:02x}{:02x}".format(
            int(efficiency_color.red * 255),
            int(efficiency_color.green * 255),
            int(efficiency_color.blue * 255)
        )
    ))
    drawing.add(String(
        badge_x, badge_y + 5,
        f"{efficiency:.0f}%",
        fontName='Helvetica-Bold',
        fontSize=11,
        textAnchor='middle',
        fillColor=efficiency_text_color
    ))
    drawing.add(String(
        badge_x, badge_y - 8,
        "Efficiency",
        fontName='Helvetica',
        fontSize=6,
        textAnchor='middle',
        fillColor=efficiency_text_color
    ))

    # Draw overall completion rate in top-left
    drawing.add(String(
        50, height - 35,
        f"Overall Completion: {overall_completion:.1f}%",
        fontName='Helvetica-Bold',
        fontSize=9,
        textAnchor='start',
        fillColor=colors.black
    ))

    # Draw funnel segments (top to bottom)
    for i, phase in enumerate(phases):
        phase_number = phase.get("phase_number", i + 1)
        phase_name = phase.get("phase_name", f"Phase {phase_number}")
        items_entered = phase.get("items_entered", 0)
        completion_rate = phase.get("completion_rate", 0)
        drop_off_rate = phase.get("drop_off_rate", 0)
        width_percent = phase.get("width_percent", 100)
        phase_color_hex = phase.get("color", "#4682B4")

        # Calculate widths for this segment (trapezoid)
        # Top width is current phase width
        top_half_width = (width_percent / 100) * max_half_width

        # Bottom width is the next phase's width (or smaller if last phase)
        if i < num_phases - 1:
            next_width_percent = phases[i + 1].get("width_percent", width_percent * 0.8)
            bottom_half_width = (next_width_percent / 100) * max_half_width
        else:
            # Last phase - taper to represent completion
            bottom_half_width = top_half_width * (completion_rate / 100) if completion_rate > 0 else top_half_width * 0.3
            bottom_half_width = max(bottom_half_width, 15)  # Minimum width

        # Segment Y positions
        segment_top_y = funnel_top_y - (i * segment_height)
        segment_bottom_y = segment_top_y - segment_height + segment_gap

        # Create trapezoid path
        phase_color = colors.HexColor(phase_color_hex)
        path = Path(
            fillColor=phase_color,
            strokeColor=colors.HexColor("#333333"),
            strokeWidth=1
        )

        # Draw trapezoid: top-left, top-right, bottom-right, bottom-left
        path.moveTo(center_x - top_half_width, segment_top_y)
        path.lineTo(center_x + top_half_width, segment_top_y)
        path.lineTo(center_x + bottom_half_width, segment_bottom_y)
        path.lineTo(center_x - bottom_half_width, segment_bottom_y)
        path.closePath()

        drawing.add(path)

        # Add labels if enabled
        if show_labels:
            segment_center_y = (segment_top_y + segment_bottom_y) / 2

            # Get accessible text color
            text_color_hex = get_accessible_text_color(phase_color_hex)
            text_color = colors.HexColor(text_color_hex)

            # Phase name and items - inside the segment
            label_text = f"{phase_name}: {items_entered}"
            drawing.add(String(
                center_x, segment_center_y + 6,
                label_text,
                fontName='Helvetica-Bold',
                fontSize=9,
                textAnchor='middle',
                fillColor=text_color
            ))

            # Completion rate - inside segment below phase name
            completion_text = f"{completion_rate:.0f}% completion"
            drawing.add(String(
                center_x, segment_center_y - 6,
                completion_text,
                fontName='Helvetica',
                fontSize=7,
                textAnchor='middle',
                fillColor=text_color
            ))

            # Drop-off indicator on the right side (for phases after the first)
            if i > 0 and drop_off_rate > 0:
                # Draw drop-off arrow and text on the right
                arrow_x = center_x + max_half_width + 20
                arrow_y = segment_top_y + 5

                # Determine drop-off severity color
                if drop_off_rate >= 50:
                    dropoff_color = colors.HexColor("#B71C1C")  # Critical - Red
                elif drop_off_rate >= 30:
                    dropoff_color = colors.HexColor("#E65100")  # High - Orange
                elif drop_off_rate >= 15:
                    dropoff_color = colors.HexColor("#F57C00")  # Medium - Light Orange
                else:
                    dropoff_color = colors.HexColor("#8B6914")  # Low - Gold

                # Drop-off label
                drawing.add(String(
                    arrow_x, arrow_y,
                    f"↓{drop_off_rate:.0f}%",
                    fontName='Helvetica-Bold',
                    fontSize=8,
                    textAnchor='start',
                    fillColor=dropoff_color
                ))

    # Add phase flow legend at the bottom
    legend_y = 55
    phase_names_short = ["Kontrolle", "BE", "K", "C", "Reopen"]

    # Calculate arrow positions
    arrow_spacing = (width - 100) / (len(phase_names_short) - 1) if len(phase_names_short) > 1 else width - 100
    start_x = 50

    for i, name in enumerate(phase_names_short):
        x_pos = start_x + (i * arrow_spacing)

        # Phase name
        drawing.add(String(
            x_pos, legend_y,
            name,
            fontName='Helvetica-Bold',
            fontSize=8,
            textAnchor='middle',
            fillColor=colors.black
        ))

        # Arrow between phases (except after last)
        if i < len(phase_names_short) - 1:
            arrow_start_x = x_pos + 25
            arrow_end_x = start_x + ((i + 1) * arrow_spacing) - 25

            # Draw arrow line
            drawing.add(Line(
                arrow_start_x, legend_y + 3,
                arrow_end_x - 5, legend_y + 3,
                strokeColor=colors.gray,
                strokeWidth=1
            ))

            # Arrow head
            drawing.add(Line(
                arrow_end_x - 5, legend_y + 3,
                arrow_end_x - 10, legend_y + 6,
                strokeColor=colors.gray,
                strokeWidth=1
            ))
            drawing.add(Line(
                arrow_end_x - 5, legend_y + 3,
                arrow_end_x - 10, legend_y,
                strokeColor=colors.gray,
                strokeWidth=1
            ))

    # Add color legend for completion rates
    legend_items = [
        ("≥80%", "#1B5E20"),
        ("≥60%", "#8B6914"),
        ("≥40%", "#E65100"),
        ("<40%", "#B71C1C"),
    ]

    legend_start_x = 50
    legend_item_y = 25
    legend_box_size = 10

    drawing.add(String(
        legend_start_x, legend_item_y + 12,
        "Completion Rate:",
        fontName='Helvetica',
        fontSize=7,
        textAnchor='start',
        fillColor=colors.gray
    ))

    for j, (label, color_hex) in enumerate(legend_items):
        item_x = legend_start_x + 75 + (j * 65)

        # Color box
        drawing.add(Rect(
            item_x, legend_item_y,
            legend_box_size, legend_box_size,
            fillColor=colors.HexColor(color_hex),
            strokeColor=colors.black,
            strokeWidth=0.5
        ))

        # Label
        drawing.add(String(
            item_x + legend_box_size + 3, legend_item_y + 2,
            label,
            fontName='Helvetica',
            fontSize=7,
            textAnchor='start',
            fillColor=colors.black
        ))

    return drawing


def make_group_detail_chart(group: str, phase_user_data: Dict[str, Dict[str, int]], title: str, width: int = 500, height: int = 200, show_labels: bool = True) -> "Drawing":
    """Create a horizontal stacked bar chart showing user contributions per phase.

    Args:
        group: Group identifier
        phase_user_data: Dictionary mapping phases to user contribution data
        title: Chart title
        width: Chart width in points
        height: Chart height in points
        show_labels: If True, display value and percentage labels on segments (default: True)
    """
    from reportlab.graphics.shapes import Rect

    drawing = Drawing(width, height)

    # Add title
    drawing.add(String(width/2, height-10, title, # Reduced from height-15
                      fontName='Helvetica-Bold', fontSize=12, textAnchor='middle'))

    # Sort phases numerically
    phases = sorted(phase_user_data.keys(), key=lambda x: int(x) if x.isdigit() else 999)

    # Get all users across all phases
    all_users = set()
    for phase_data in phase_user_data.values():
        all_users.update(phase_data.keys())
    all_users = sorted(all_users)

    # Generate consistent colors for users
    user_colors = generate_user_colors({user: 1 for user in all_users})

    # Chart dimensions - FURTHER REDUCED
    chart_x = 120
    chart_y = 30  # Reduced from 40
    chart_width = 320
    chart_height = 130  # Reduced from 160
    bar_height = 14     # Reduced from 16
    spacing = 6         # Reduced from 8

    # Calculate maximum total for scale
    max_total = 1  # Minimum value to avoid division by zero
    for phase in phases:
        phase_total = sum(phase_user_data.get(phase, {}).values())
        if phase_total > max_total:
            max_total = phase_total

    # Calculate grand total for percentage calculations
    grand_total = sum(sum(phase_user_data.get(phase, {}).values()) for phase in phases)

    # Draw each phase as a stacked bar
    for i, phase in enumerate(phases):
        y_position = chart_y + (bar_height + spacing) * i

        # Add phase label
        drawing.add(String(
            chart_x - 10,
            y_position + bar_height/2,
            PHASE_NAMES.get(phase, f"Phase {phase}"),
            fontName='Helvetica',
            fontSize=8,  # Reduced from 9
            textAnchor='end'
        ))

        # Get user data for this phase
        phase_data = phase_user_data.get(phase, {})

        # Calculate total for this phase
        phase_total = sum(phase_data.values())

        # Starting position for first segment
        x_start = chart_x

        # Draw each user's contribution as a colored segment
        for user in all_users:
            value = phase_data.get(user, 0)
            if value > 0:
                # Calculate width of this segment proportional to its value
                segment_width = (value / max_total) * chart_width

                # Get user color
                user_color = user_colors.get(user, colors.steelblue)

                # Draw segment
                rect = Rect(
                    x_start,
                    y_position,
                    segment_width,
                    bar_height,
                    fillColor=user_color,
                    strokeColor=colors.black,
                    strokeWidth=0.5
                )
                drawing.add(rect)

                # Add value label if segment is wide enough and labels are enabled
                if show_labels and segment_width > 20:
                    # Get accessible text color based on segment background
                    user_color_hex = "#{:02x}{:02x}{:02x}".format(
                        int(user_color.red * 255),
                        int(user_color.green * 255),
                        int(user_color.blue * 255)
                    )
                    text_color_hex = get_accessible_text_color(user_color_hex)
                    text_color = colors.HexColor(text_color_hex)

                    # Show value label
                    drawing.add(String(
                        x_start + segment_width/2,
                        y_position + bar_height/2,
                        str(value),
                        fontName='Helvetica-Bold',
                        fontSize=6,  # Reduced from 7
                        textAnchor='middle',
                        fillColor=text_color
                    ))

                # Move x position for next segment
                x_start += segment_width

        # Add phase total at the end of the bar (if labels enabled)
        if show_labels and phase_total > 0:
            percentage = (phase_total / grand_total * 100) if grand_total > 0 else 0
            drawing.add(String(
                x_start + 5,
                y_position + bar_height/2,
                f"{phase_total} ({percentage:.0f}%)",
                fontName='Helvetica',
                fontSize=6,
                textAnchor='start',
                fillColor=colors.black
            ))
    
    # Draw axis line
    drawing.add(Line(
        chart_x, chart_y - 6,  # Reduced from chart_y - 8
        chart_x + chart_width, chart_y - 6,
        strokeWidth=1,
        strokeColor=colors.black
    ))
    
    # Add scale markers
    scale_steps = 5
    for i in range(scale_steps + 1):
        x_pos = chart_x + (i / scale_steps) * chart_width
        value = int((i / scale_steps) * max_total)
        
        # Tick mark
        drawing.add(Line(
            x_pos, chart_y - 6,
            x_pos, chart_y - 10,
            strokeWidth=1,
            strokeColor=colors.black
        ))
        
        # Value label
        drawing.add(String(
            x_pos, chart_y - 18,  # Reduced from chart_y - 20
            str(value),
            fontName='Helvetica',
            fontSize=6,  # Reduced from 7
            textAnchor='middle'
        ))
    
    # Return the chart and legend data
    return drawing, [(user_colors.get(user, colors.steelblue), user) for user in all_users]
    
def create_horizontal_legend(color_name_pairs: List[Tuple["Color", str]], width: int = 500, height: int = 24, min_font: int = 6, max_font: int = 10) -> "Drawing":
    """
    Draw a single-line horizontal legend that adapts spacing:
      - width: total drawing width (use A4[0] - 50*mm to use full printable width)
      - height: drawing height
      - min_font/max_font: font size clamp
    The function will reduce font size to fit all items on one row and truncate labels if needed.
    """
    drawing = Drawing(width, height)
    num_items = len(color_name_pairs)
    if num_items == 0:
        return drawing

    # Padding around edges and between items
    left_pad = 6
    right_pad = 6
    box_label_gap = 4   # gap between color box and label
    item_gap = 10       # minimal gap between items

    usable_width = width - left_pad - right_pad
    # start with a reasonable font size depending on item count
    font_size = max(min(max_font, int( max_font - (num_items - 6) * 1 )), min_font)

    # box size scales with font size
    box_size = max(6, int(font_size * 0.9))

    # compute label widths and adjust font size down until everything fits
    def total_needed_width(fs: int) -> float:
        bs = max(6, int(fs * 0.9))
        total: float = 0
        for _, name in color_name_pairs:
            lbl_w = stringWidth(name, 'Helvetica', fs)
            total += bs + box_label_gap + lbl_w + item_gap
        return total

    # reduce font size until it fits or reach min_font
    while font_size > min_font and total_needed_width(font_size) > usable_width:
        font_size -= 1

    # Recalculate box_size after final font_size is determined
    box_size = max(6, int(font_size * 0.9))

    # If at min_font and still doesn't fit, truncate labels
    truncated_names = []
    if total_needed_width(font_size) > usable_width:
        # compute average available width per item for label
        avg_space_per_item = (usable_width - num_items * (box_size + box_label_gap + item_gap)) / num_items
        avg_space_per_item = max(10, avg_space_per_item)  # at least 10pt
        for _, name in color_name_pairs:
            # Normalize unicode for consistent PDF display
            name = prepare_for_pdf(name)
            # measure full name; if larger than avg_space_per_item, truncate
            w = stringWidth(name, 'Helvetica', font_size)
            if w <= avg_space_per_item:
                truncated_names.append(name)
            else:
                # Unicode-aware truncation that respects grapheme clusters
                # Start with a reasonable estimate based on character count
                max_chars = max(1, int(len(name) * avg_space_per_item / max(w, 1)))
                truncated = safe_truncate(name, max_chars, '…')
                # Fine-tune: reduce until it fits the pixel width
                while truncated and len(truncated) > 1 and stringWidth(truncated, 'Helvetica', font_size) > avg_space_per_item:
                    max_chars -= 1
                    truncated = safe_truncate(name, max_chars, '…')
                truncated_names.append(truncated if truncated else '…')
    else:
        # Normalize all names for PDF output
        truncated_names = [prepare_for_pdf(n) for _, n in color_name_pairs]

    # Now render items evenly from left_pad
    # Calculate vertical center for both boxes and text
    center_y = height / 2
    
    x = left_pad

    for i, (color, name) in enumerate(color_name_pairs):
        lbl = truncated_names[i]

        # Draw color box - centered vertically
        box_y = center_y - box_size / 2
        drawing.add(Rect(x, box_y, box_size, box_size,
                         fillColor=color, strokeColor=colors.black, strokeWidth=0.5))
        x += box_size + box_label_gap

        # Draw label - centered vertically to align with box center
        # String baseline needs to be adjusted so text center aligns with box center
        text_y = center_y - font_size * 0.35  # Adjust baseline for visual centering
        drawing.add(String(x, text_y, lbl,
                           fontName='Helvetica', fontSize=font_size, fillColor=colors.black))
        lbl_w = stringWidth(lbl, 'Helvetica', font_size)
        x += lbl_w + item_gap

        # if we run out of space break (safety)
        if x > width - right_pad:
            break

    return drawing
    
def collect_user_group_data(metrics: Dict[str, Any], target_user: str) -> Dict[str, Dict[str, int]]:
    """Collect data about a user's activity across different product groups."""
    user_data = defaultdict(lambda: defaultdict(int))
    
    # Go through the group_phase_user data and flip it for user-centric view
    for group, phase_data in metrics["group_phase_user"].items():
        for phase, user_data_dict in phase_data.items():
            if target_user in user_data_dict:
                user_data[group][phase] = user_data_dict[target_user]
    
    return user_data

def make_user_detail_chart(user: str, group_phase_data: Dict[str, Dict[str, int]], width: int = 500, height: int = 200, show_labels: bool = True) -> "Drawing":
    """Create a horizontal stacked bar chart showing user's work across phases with groups as segments.

    Args:
        user: User name for the chart
        group_phase_data: Dictionary mapping groups to phase data
        width: Chart width in points
        height: Chart height in points
        show_labels: If True, display value and percentage labels on segments (default: True)
    """
    drawing = Drawing(width, height)

    # Add title - normalize user name for PDF
    safe_user = prepare_for_pdf(user)
    drawing.add(String(width/2, height-10,
                      f"Activity by Phase for {safe_user}",
                      fontName='Helvetica-Bold', fontSize=12, textAnchor='middle'))

    # Get all phases across all groups
    all_phases = set()
    for phase_data in group_phase_data.values():
        all_phases.update(phase_data.keys())
    all_phases = sorted(all_phases, key=lambda x: int(x) if x.isdigit() else 999)

    # Sort groups alphabetically
    groups = sorted(group_phase_data.keys())

    # Chart dimensions
    chart_x = 120
    chart_y = 30
    chart_width = 320
    chart_height = 130
    bar_height = 14
    spacing = 6

    # Calculate maximum total for scale
    max_total = 1  # Minimum value to avoid division by zero
    for phase in all_phases:
        phase_total = 0
        for group in groups:
            phase_total += group_phase_data.get(group, {}).get(phase, 0)
        if phase_total > max_total:
            max_total = phase_total

    # Calculate grand total for percentage calculations
    grand_total = sum(
        sum(group_phase_data.get(group, {}).get(phase, 0) for group in groups)
        for phase in all_phases
    )

    # Draw each phase as a stacked bar
    for i, phase in enumerate(all_phases):
        y_position = chart_y + (bar_height + spacing) * i

        # Add phase label
        drawing.add(String(
            chart_x - 10,
            y_position + bar_height/2,
            PHASE_NAMES.get(phase, f"Phase {phase}"),
            fontName='Helvetica-Bold',
            fontSize=8,
            textAnchor='end'
        ))

        # Calculate total for this phase
        phase_total = sum(group_phase_data.get(group, {}).get(phase, 0) for group in groups)

        # Starting position for first segment
        x_start = chart_x

        # Draw each group contribution as a colored segment
        for group in groups:
            value = group_phase_data.get(group, {}).get(phase, 0)
            if value > 0:
                # Calculate width of this segment proportional to its value
                segment_width = (value / max_total) * chart_width

                # Get group color
                group_color = GROUP_COLORS.get(group, colors.steelblue)

                # Draw segment
                rect = Rect(
                    x_start,
                    y_position,
                    segment_width,
                    bar_height,
                    fillColor=group_color,
                    strokeColor=colors.black,
                    strokeWidth=0.5
                )
                drawing.add(rect)

                # Add value label if segment is wide enough and labels are enabled
                if show_labels and segment_width > 20:
                    # Get accessible text color based on segment background
                    group_color_hex = "#{:02x}{:02x}{:02x}".format(
                        int(group_color.red * 255),
                        int(group_color.green * 255),
                        int(group_color.blue * 255)
                    )
                    text_color_hex = get_accessible_text_color(group_color_hex)
                    text_color = colors.HexColor(text_color_hex)

                    # Show value label
                    drawing.add(String(
                        x_start + segment_width/2,
                        y_position + bar_height/2,
                        str(value),
                        fontName='Helvetica-Bold',
                        fontSize=6,
                        textAnchor='middle',
                        fillColor=text_color
                    ))

                # Move x position for next segment
                x_start += segment_width

        # Add phase total at the end of the bar (if labels enabled)
        if show_labels and phase_total > 0:
            percentage = (phase_total / grand_total * 100) if grand_total > 0 else 0
            drawing.add(String(
                x_start + 5,
                y_position + bar_height/2,
                f"{phase_total} ({percentage:.0f}%)",
                fontName='Helvetica',
                fontSize=6,
                textAnchor='start',
                fillColor=colors.black
            ))
    
    # Draw axis line
    drawing.add(Line(
        chart_x, chart_y - 6,
        chart_x + chart_width, chart_y - 6,
        strokeWidth=1,
        strokeColor=colors.black
    ))
    
    # Add scale markers
    scale_steps = 5
    for i in range(scale_steps + 1):
        x_pos = chart_x + (i / scale_steps) * chart_width
        value = int((i / scale_steps) * max_total)
        
        # Tick mark
        drawing.add(Line(
            x_pos, chart_y - 6,
            x_pos, chart_y - 10,
            strokeWidth=1,
            strokeColor=colors.black
        ))
        
        # Value label
        drawing.add(String(
            x_pos, chart_y - 18,
            str(value),
            fontName='Helvetica',
            fontSize=6,
            textAnchor='middle'
        ))
    
    # Create legend data for groups - use shorter names for bundle groups
    legend_data = []
    for group in groups:
        if group.startswith("BUNDLE_"):
            display_name = get_group_display_name(group, short=True)
        else:
            display_name = f"Group {group}"
        legend_data.append((GROUP_COLORS.get(group, colors.steelblue), display_name))

    return drawing, legend_data

def create_user_group_distribution_chart(group_phase_data: Dict[str, Dict[str, int]], user: str, width: int = 500, height: int = 250, show_labels: bool = True) -> "Drawing":
    """Create a pie chart showing distribution of user's work across product groups with data labels.

    Args:
        group_phase_data: Dictionary mapping groups to phase data
        user: User name for the chart title
        width: Chart width in points
        height: Chart height in points
        show_labels: If True, display value labels on pie slices (default: True)
    """
    drawing = Drawing(width, height)

    # Add title
    drawing.add(String(width/2, height-20,
                      f"Group Distribution for {user} (Last 30 Days)",
                      fontName='Helvetica-Bold', fontSize=12, textAnchor='middle'))

    # Calculate totals for each group
    group_totals = {}
    for group, phase_data in group_phase_data.items():
        group_totals[group] = sum(phase_data.values())

    # Skip if no data
    if not group_totals or sum(group_totals.values()) == 0:
        drawing.add(String(width/2, height/2, "No data available",
                          fontName='Helvetica-Italic', fontSize=12, textAnchor='middle'))
        return drawing

    # Calculate total changes
    total_changes = sum(group_totals.values())

    # Create pie chart data
    from reportlab.graphics.charts.piecharts import Pie

    pie = Pie()
    pie.x = width * 0.3  # Position left of center
    pie.y = height / 2
    pie.width = min(width, height) * 0.4
    pie.height = min(width, height) * 0.4

    # Sort groups by count
    sorted_groups = sorted(group_totals.items(), key=lambda x: x[1], reverse=True)

    # Set data
    pie.data = [count for _, count in sorted_groups]

    # Set colors
    for i, (group, _) in enumerate(sorted_groups):
        if i < len(pie.slices):
            pie.slices[i].fillColor = GROUP_COLORS.get(group, colors.steelblue)

    # Add the pie to the drawing
    drawing.add(pie)

    # Add data labels on pie slices for slices >= 5% (to avoid clutter)
    if show_labels and total_changes > 0:
        # Calculate pie center and radius
        pie_cx = pie.x + pie.width / 2
        pie_cy = pie.y + pie.height / 2
        pie_radius = pie.width / 2

        # Calculate intelligent label positions with overlap detection
        label_positions = calculate_pie_label_positions(
            sorted_groups, total_changes, pie_cx, pie_cy, pie_radius,
            min_percentage=5.0, font_name='Helvetica-Bold', font_size=8
        )

        # Create function to get text color for each slice
        def get_group_text_color(index: int) -> "Color":
            group = sorted_groups[index][0]
            slice_color = GROUP_COLORS.get(group, colors.steelblue)
            slice_color_hex = "#{:02x}{:02x}{:02x}".format(
                int(slice_color.red * 255),
                int(slice_color.green * 255),
                int(slice_color.blue * 255)
            )
            text_color_hex = get_accessible_text_color(slice_color_hex)
            return colors.HexColor(text_color_hex)

        # Draw labels with intelligent positioning
        draw_pie_labels(drawing, label_positions, get_group_text_color)

    # Add legend manually
    legend_x = width * 0.65  # Position to the right of pie
    legend_y = height - 50
    legend_font_size = 8
    line_height = legend_font_size * 1.5

    for i, (group, count) in enumerate(sorted_groups):
        y_pos = legend_y - (i * line_height)

        # Add color box
        drawing.add(Rect(
            legend_x,
            y_pos,
            8,
            8,
            fillColor=GROUP_COLORS.get(group, colors.steelblue),
            strokeColor=colors.black,
            strokeWidth=0.5
        ))

        # Add group name with value and percentage
        percentage = (count / total_changes * 100) if total_changes > 0 else 0
        drawing.add(String(
            legend_x + 12,
            y_pos + 4,
            f"Group {group}: {count} ({percentage:.1f}%)",
            fontName='Helvetica',
            fontSize=legend_font_size
        ))

    # Add total at the bottom
    drawing.add(String(
        width/2,
        30,
        f"Total changes: {total_changes}",
        fontName='Helvetica-Bold',
        fontSize=10,
        textAnchor='middle'
    ))

    return drawing


def create_user_phase_distribution_chart(
    user: str,
    phase_distribution: UserPhaseDistribution,
    width: int = 400,
    height: int = 280
) -> "Drawing":
    """Create a donut chart showing user's phase distribution.

    This chart visualizes how a user's work is distributed across different
    workflow phases (Kontrolle, BE, K2, C, Reopen C2), helping identify
    user specialization patterns and workload balance.

    Args:
        user: User name/identifier for the chart title
        phase_distribution: UserPhaseDistribution object containing phase data
        width: Chart width in points
        height: Chart height in points

    Returns:
        ReportLab Drawing object containing the donut chart
    """
    from reportlab.graphics.shapes import Wedge, Circle

    drawing = Drawing(width, height)

    # Normalize user name for PDF display
    safe_user = prepare_for_pdf(user)

    # Add title with modern styling
    drawing.add(String(
        width / 2, height - 20,
        f"Phasenverteilung für {safe_user}",
        fontName='Helvetica-Bold', fontSize=12, textAnchor='middle'
    ))

    # Donut chart dimensions
    chart_center_x = width * 0.30
    chart_center_y = height / 2.1
    outer_radius = min(width, height) * 0.28
    inner_radius = outer_radius * 0.55  # Creates the donut hole

    # Get phase colors (using the accessible phase colors)
    phase_colors = get_reportlab_phase_colors()

    # Prepare chart data - list of (phase_name, count) tuples
    chart_data = []
    total_items = phase_distribution.total_items

    if total_items == 0:
        # No data - display message
        drawing.add(String(
            width / 2, height / 2,
            "Keine Phasendaten verfügbar",
            fontName='Helvetica-Italic', fontSize=12, textAnchor='middle'
        ))
        return drawing

    # Collect phase data in order
    for phase in ALL_PHASES:
        phase_dist = phase_distribution.phases.get(phase)
        if phase_dist and phase_dist.count > 0:
            # Use short phase name for chart
            short_name = USER_PHASE_NAMES.get(phase, f"Phase {phase}")
            # Extract just the first part of the name before parenthesis
            if "(" in short_name:
                short_name = short_name.split("(")[0].strip()
            chart_data.append((f"Phase {phase}: {short_name}", phase_dist.count, phase))

    # Draw donut segments using Wedge shapes
    start_angle = 90  # Start from top
    slice_colors = []  # Store colors for label generation

    for i, (label, count, phase) in enumerate(chart_data):
        if count <= 0:
            continue

        sweep_angle = (count / total_items) * 360 if total_items > 0 else 0

        # Calculate end angle (going clockwise, so subtract)
        end_angle = start_angle - sweep_angle

        # Get color for this phase
        color = phase_colors.get(str(phase), colors.steelblue)
        slice_colors.append((i, color))

        # Draw outer wedge
        outer_wedge = Wedge(
            chart_center_x, chart_center_y, outer_radius,
            end_angle, start_angle,
            fillColor=color,
            strokeColor=colors.white,
            strokeWidth=2
        )
        drawing.add(outer_wedge)

        start_angle = end_angle

    # Draw inner circle (donut hole) - white to create the donut effect
    inner_circle = Circle(
        chart_center_x, chart_center_y, inner_radius,
        fillColor=colors.white,
        strokeColor=colors.white,
        strokeWidth=0
    )
    drawing.add(inner_circle)

    # Add center content - total items and specialization level
    drawing.add(String(
        chart_center_x, chart_center_y + 10,
        f"{total_items}",
        fontName='Helvetica-Bold', fontSize=18, textAnchor='middle',
        fillColor=colors.HexColor("#1F2937")
    ))
    drawing.add(String(
        chart_center_x, chart_center_y - 6,
        "Items",
        fontName='Helvetica', fontSize=10, textAnchor='middle',
        fillColor=colors.HexColor("#6B7280")
    ))

    # Add specialization level indicator
    specialization_text = {
        SpecializationLevel.SPECIALIST: "Spezialist",
        SpecializationLevel.FOCUSED: "Fokussiert",
        SpecializationLevel.BALANCED: "Ausgewogen",
        SpecializationLevel.GENERALIST: "Generalist",
    }.get(phase_distribution.specialization_level, "")

    if specialization_text:
        drawing.add(String(
            chart_center_x, chart_center_y - 20,
            f"({specialization_text})",
            fontName='Helvetica', fontSize=8, textAnchor='middle',
            fillColor=colors.HexColor("#9CA3AF")
        ))

    # Add data labels on donut segments for slices >= 8%
    if total_items > 0:
        # Prepare data for label positioning (just (label, count) tuples)
        label_data = [(label, count) for label, count, phase in chart_data]

        label_positions = calculate_pie_label_positions(
            label_data, total_items, chart_center_x, chart_center_y,
            outer_radius,
            min_percentage=8.0, font_name='Helvetica-Bold', font_size=8,
            label_format='percentage'
        )

        # Adjust label positions for donut (place them on the donut ring)
        for label in label_positions:
            mid_angle_rad = label['mid_angle_rad']
            # Position labels at the middle of the donut ring
            label_radius = (outer_radius + inner_radius) / 2
            label['label_x'] = chart_center_x + label_radius * math.cos(mid_angle_rad)
            label['label_y'] = chart_center_y + label_radius * math.sin(mid_angle_rad)

        # Create function to get text color for each slice
        def get_phase_text_color(index: int) -> "Color":
            if index < len(slice_colors):
                _, slice_color = slice_colors[index]
                slice_color_hex = "#{:02x}{:02x}{:02x}".format(
                    int(slice_color.red * 255),
                    int(slice_color.green * 255),
                    int(slice_color.blue * 255)
                )
                text_color_hex = get_accessible_text_color(slice_color_hex)
                return colors.HexColor(text_color_hex)
            return colors.white

        # Draw labels with intelligent positioning
        draw_pie_labels(drawing, label_positions, get_phase_text_color)

    # Add enhanced legend - positioned to the right with better formatting
    legend_x = width * 0.58
    legend_y = height * 0.80
    legend_font_size = 9
    line_height = legend_font_size * 2.0

    # Legend title
    drawing.add(String(
        legend_x, legend_y + 15,
        "Phasen",
        fontName='Helvetica-Bold', fontSize=10
    ))

    for i, (label, count, phase) in enumerate(chart_data):
        if count <= 0:
            continue

        y_pos = legend_y - (i * line_height)
        color = phase_colors.get(str(phase), colors.steelblue)

        # Add rounded color indicator
        drawing.add(Rect(
            legend_x,
            y_pos - 2,
            10,
            10,
            fillColor=color,
            strokeColor=None,
            strokeWidth=0,
            rx=2,  # Rounded corners
            ry=2
        ))

        # Add phase name (short version)
        short_name = USER_PHASE_NAMES.get(phase, f"Phase {phase}")
        if "(" in short_name:
            short_name = short_name.split("(")[0].strip()
        drawing.add(String(
            legend_x + 15,
            y_pos,
            f"P{phase}: {short_name}",
            fontName='Helvetica-Bold',
            fontSize=legend_font_size - 1
        ))

        # Add count and percentage on separate line
        percentage = (count / total_items * 100) if total_items > 0 else 0
        drawing.add(String(
            legend_x + 15,
            y_pos - 10,
            f"{count} items ({percentage:.1f}%)",
            fontName='Helvetica',
            fontSize=legend_font_size - 2,
            fillColor=colors.HexColor("#6B7280")
        ))

    # Add primary phase indicator at bottom
    if phase_distribution.primary_phase:
        primary_phase_name = USER_PHASE_NAMES.get(
            phase_distribution.primary_phase, f"Phase {phase_distribution.primary_phase}"
        )
        if "(" in primary_phase_name:
            primary_phase_name = primary_phase_name.split("(")[0].strip()
        drawing.add(String(
            width / 2,
            20,
            f"Primäre Phase: {primary_phase_name} ({phase_distribution.primary_phase_percentage:.0f}%)",
            fontName='Helvetica-Bold',
            fontSize=10,
            textAnchor='middle',
            fillColor=colors.HexColor("#1F2937")
        ))

    return drawing


def create_user_contribution_pie_chart(
    contribution_summary: TeamContributionSummary,
    width: int = 400,
    height: int = 280
) -> "Drawing":
    """Create a pie chart showing each user's contribution percentage to total team activity.

    Args:
        contribution_summary: TeamContributionSummary object with user contribution data
        width: Chart width in points
        height: Chart height in points

    Returns:
        Drawing: A ReportLab Drawing object containing the pie chart
    """
    from reportlab.graphics.charts.piecharts import Pie

    drawing = Drawing(width, height)

    # Add title
    drawing.add(String(
        width / 2, height - 15,
        "Team Contribution Distribution",
        fontName='Helvetica-Bold',
        fontSize=12,
        textAnchor='middle'
    ))

    # Get user contributions sorted by percentage
    if not contribution_summary.user_contributions:
        drawing.add(String(
            width / 2, height / 2,
            "No contribution data available",
            fontName='Helvetica-Italic',
            fontSize=10,
            textAnchor='middle'
        ))
        return drawing

    sorted_users = sorted(
        contribution_summary.user_contributions.items(),
        key=lambda x: x[1].percentage_of_team_total,
        reverse=True
    )

    # Create pie chart
    pie = Pie()
    pie.x = width * 0.15
    pie.y = height * 0.2
    pie.width = min(width, height) * 0.4
    pie.height = min(width, height) * 0.4

    # Set data and colors
    pie.data = [contrib.total_items for _, contrib in sorted_users]

    # Get user colors from accessible palette
    user_color_palette = get_reportlab_user_colors()
    user_color_list = list(user_color_palette.values())

    for i, (user, _) in enumerate(sorted_users):
        if i < len(pie.slices):
            # Use user-specific color or cycle through palette
            if user in user_color_palette:
                pie.slices[i].fillColor = user_color_palette[user]
            else:
                pie.slices[i].fillColor = user_color_list[i % len(user_color_list)]

    drawing.add(pie)

    # Calculate pie center for label positioning
    pie_cx = pie.x + pie.width / 2
    pie_cy = pie.y + pie.height / 2
    pie_radius = pie.width / 2

    # Add labels on larger slices
    total_items = contribution_summary.total_team_activity
    if total_items > 0:
        # Format data for pie label utilities
        label_data = [(user, contrib.total_items) for user, contrib in sorted_users]

        label_positions = calculate_pie_label_positions(
            label_data, total_items, pie_cx, pie_cy, pie_radius,
            min_percentage=5.0, font_name='Helvetica-Bold', font_size=8
        )

        def get_user_text_color(index: int) -> "Color":
            user = sorted_users[index][0]
            if user in user_color_palette:
                slice_color = user_color_palette[user]
            else:
                slice_color = user_color_list[index % len(user_color_list)]
            slice_color_hex = "#{:02x}{:02x}{:02x}".format(
                int(slice_color.red * 255),
                int(slice_color.green * 255),
                int(slice_color.blue * 255)
            )
            text_color_hex = get_accessible_text_color(slice_color_hex)
            return colors.HexColor(text_color_hex)

        draw_pie_labels(drawing, label_positions, get_user_text_color)

    # Add legend on the right side
    legend_x = width * 0.6
    legend_y = height - 45
    legend_font_size = 8
    line_height = 14

    for i, (user, contrib) in enumerate(sorted_users[:8]):  # Show top 8 users
        y_pos = legend_y - (i * line_height)

        # Get user color
        if user in user_color_palette:
            user_color = user_color_palette[user]
        else:
            user_color = user_color_list[i % len(user_color_list)]

        # Color box
        drawing.add(Rect(
            legend_x, y_pos, 10, 10,
            fillColor=user_color,
            strokeColor=colors.black,
            strokeWidth=0.5
        ))

        # Prepare user name for PDF
        safe_user = prepare_for_pdf(user) if user else "Unknown"

        # User name with percentage
        drawing.add(String(
            legend_x + 14, y_pos + 3,
            f"{safe_user}: {contrib.percentage_of_team_total:.1f}%",
            fontName='Helvetica',
            fontSize=legend_font_size
        ))

    # Add total at the bottom
    drawing.add(String(
        width / 2, 15,
        f"Total Team Activity: {contribution_summary.total_team_activity} items",
        fontName='Helvetica-Bold',
        fontSize=9,
        textAnchor='middle'
    ))

    return drawing


def create_user_contribution_bars(
    contribution_summary: TeamContributionSummary,
    width: int = 450,
    height: int = 180,
    max_users: int = 8
) -> "Drawing":
    """Create horizontal percentage bars showing each user's contribution percentage.

    Displays a visual progress-bar style representation of each user's
    contribution to team activity with percentage and rank indicators.

    Args:
        contribution_summary: TeamContributionSummary object with user contribution data
        width: Chart width in points
        height: Chart height in points
        max_users: Maximum number of users to display

    Returns:
        Drawing: A ReportLab Drawing object containing the percentage bars
    """
    drawing = Drawing(width, height)

    # Add title
    drawing.add(String(
        width / 2, height - 12,
        "User Contribution Rankings",
        fontName='Helvetica-Bold',
        fontSize=11,
        textAnchor='middle'
    ))

    if not contribution_summary.user_contributions:
        drawing.add(String(
            width / 2, height / 2,
            "No contribution data available",
            fontName='Helvetica-Italic',
            fontSize=10,
            textAnchor='middle'
        ))
        return drawing

    # Sort users by contribution percentage
    sorted_users = sorted(
        contribution_summary.user_contributions.items(),
        key=lambda x: x[1].percentage_of_team_total,
        reverse=True
    )[:max_users]

    # Get user colors
    user_color_palette = get_reportlab_user_colors()
    user_color_list = list(user_color_palette.values())

    # Bar dimensions
    bar_start_x = 65  # Leave room for user name
    bar_width = width - bar_start_x - 80  # Leave room for percentage
    bar_height = 12
    bar_spacing = 16
    start_y = height - 35

    # Draw bars
    for i, (user, contrib) in enumerate(sorted_users):
        y_pos = start_y - (i * bar_spacing)

        # Prepare user name for PDF
        safe_user = prepare_for_pdf(user) if user else "Unknown"
        truncated_name = safe_user[:8] if len(safe_user) > 8 else safe_user

        # User name (left-aligned)
        drawing.add(String(
            5, y_pos + bar_height / 2 - 3,
            truncated_name,
            fontName='Helvetica-Bold',
            fontSize=8
        ))

        # Background bar (light gray)
        drawing.add(Rect(
            bar_start_x, y_pos, bar_width, bar_height,
            fillColor=colors.HexColor("#E0E0E0"),
            strokeColor=colors.HexColor("#BDBDBD"),
            strokeWidth=0.5
        ))

        # Get user color
        if user in user_color_palette:
            bar_color = user_color_palette[user]
        else:
            bar_color = user_color_list[i % len(user_color_list)]

        # Filled bar (proportional to percentage)
        filled_width = (contrib.percentage_of_team_total / 100) * bar_width
        if filled_width > 0:
            drawing.add(Rect(
                bar_start_x, y_pos, filled_width, bar_height,
                fillColor=bar_color,
                strokeColor=None
            ))

        # Add rank badge for top 3 contributors
        if i < 3:
            badge_colors = [
                colors.HexColor("#FFD700"),  # Gold
                colors.HexColor("#C0C0C0"),  # Silver
                colors.HexColor("#CD7F32"),  # Bronze
            ]
            badge_x = bar_start_x + filled_width + 5
            if badge_x > bar_start_x + bar_width - 15:
                badge_x = bar_start_x + bar_width - 15

            # Draw rank badge
            drawing.add(Circle(
                badge_x + 6, y_pos + bar_height / 2, 6,
                fillColor=badge_colors[i],
                strokeColor=colors.black,
                strokeWidth=0.5
            ))
            drawing.add(String(
                badge_x + 6, y_pos + bar_height / 2 - 2.5,
                str(i + 1),
                fontName='Helvetica-Bold',
                fontSize=7,
                textAnchor='middle'
            ))

        # Percentage text (right-aligned)
        pct_text = f"{contrib.percentage_of_team_total:.1f}%"
        drawing.add(String(
            width - 35, y_pos + bar_height / 2 - 3,
            pct_text,
            fontName='Helvetica-Bold',
            fontSize=9
        ))

        # Items count (small, right-aligned)
        drawing.add(String(
            width - 5, y_pos + bar_height / 2 - 3,
            f"({contrib.total_items})",
            fontName='Helvetica',
            fontSize=7,
            textAnchor='end'
        ))

    return drawing


def create_top_contributors_highlight(
    contribution_summary: TeamContributionSummary,
    top_n: int = 3
) -> List[List[str]]:
    """Create data for highlighting top contributors in a table format.

    Args:
        contribution_summary: TeamContributionSummary object with user contribution data
        top_n: Number of top contributors to highlight

    Returns:
        List of lists containing formatted table data for top contributors
    """
    if not contribution_summary.top_contributors:
        return []

    table_data = [["Rank", "User", "Items", "% of Team", "Groups"]]

    for i, contributor in enumerate(contribution_summary.top_contributors[:top_n]):
        user = contributor.get("user", "Unknown")
        safe_user = prepare_for_pdf(user) if user else "Unknown"

        rank_display = str(i + 1)

        table_data.append([
            rank_display,
            safe_user,
            str(contributor.get("total_items", 0)),
            f"{contributor.get('percentage', 0):.1f}%",
            str(contributor.get("unique_groups", 0))
        ])

    return table_data


def add_user_contribution_section(
    story: List["Flowable"],
    changes: List[Dict[str, Any]],
    title: str = "Team Contribution Overview"
) -> None:
    """Add a section showing user contribution percentages with visual indicators.

    This adds a pie chart and percentage bars showing each user's contribution
    to total team activity, along with a table highlighting top contributors.

    Args:
        story: List of ReportLab flowables to append to
        changes: List of change records from historical data
        title: Section title
    """
    styles = getSampleStyleSheet()
    heading_style = styles['Heading2']
    normal_style = styles['Normal']

    # Calculate contribution summary
    contribution_summary = get_contribution_summary(changes, top_n=5)

    if contribution_summary.total_team_activity == 0:
        story.append(Paragraph("No contribution data available for this period.", normal_style))
        return

    # Add section header
    story.append(Paragraph(title, heading_style))
    story.append(Spacer(1, 5 * mm))

    # Add summary text
    story.append(Paragraph(
        f"<b>Total Activity:</b> {contribution_summary.total_team_activity} items by "
        f"{contribution_summary.total_users} team members",
        normal_style
    ))
    story.append(Spacer(1, 8 * mm))

    # Add pie chart
    pie_chart = create_user_contribution_pie_chart(contribution_summary)
    story.append(pie_chart)
    story.append(Spacer(1, 10 * mm))

    # Add percentage bars
    bars_chart = create_user_contribution_bars(contribution_summary)
    story.append(bars_chart)
    story.append(Spacer(1, 8 * mm))

    # Add top contributors table with highlighting
    if contribution_summary.top_contributors:
        story.append(Paragraph("Top Contributors", styles['Heading3']))
        story.append(Spacer(1, 3 * mm))

        table_data = create_top_contributors_highlight(contribution_summary, top_n=5)
        if table_data:
            top_table = Table(table_data, colWidths=[30, 60, 50, 55, 40])
            top_table.setStyle(TableStyle([
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1565C0")),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, -1), 9),
                ('ALIGN', (0, 0), (0, -1), 'CENTER'),
                ('ALIGN', (2, 0), (-1, -1), 'CENTER'),
                # Highlight top 3 rows
                ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor("#FFF8E1")),  # Gold tint
                ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor("#ECEFF1")),  # Silver tint
                ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor("#EFEBE9")),  # Bronze tint
            ]))
            story.append(top_table)


def query_smartsheet_data(group: Optional[str] = None) -> Dict[str, Any]:
    """Query raw Smartsheet data to get real-time activity metrics.

    Connects directly to Smartsheet API to retrieve and calculate current
    activity metrics from the configured sheets. This provides live data
    as opposed to using cached change history.

    Args:
        group (str, optional): If specified, only query the sheet for this
            group (e.g., "NA", "NF", "BUNDLE_FAN"). If None, queries all
            sheets in SHEET_IDS. Defaults to None.

    Returns:
        dict: A dictionary containing activity metrics:
            - 'total_items' (int): Total number of items across queried sheets
            - 'recent_activity_items' (int): Items with activity in last 30 days
            - Other metrics calculated from sheet data

    Side Effects:
        - Makes HTTP requests to Smartsheet API
        - Uses retry logic via SmartsheetRetryClient
        - Logs progress and errors

    Note:
        This function queries live data from Smartsheet, which may be
        slower than using cached change history. Use for real-time dashboards
        or when fresh data is required.

    Examples:
        >>> metrics = query_smartsheet_data()  # All groups
        >>> metrics['total_items']
        5292
        >>> metrics = query_smartsheet_data(group="NA")  # NA group only
        >>> metrics['total_items']
        1779
    """
    base_client = smartsheet.Smartsheet(token)
    base_client.errors_as_exceptions(True)
    client = SmartsheetRetryClient(
        base_client,
        max_retries=3,
        base_delay=1.0,
        max_delay=30.0,
        continue_on_failure=True
    )

    # Track counts
    total_items = 0
    recent_activity_items = 0
    thirty_days_ago = datetime.now() - timedelta(days=30)

    # Process sheets
    if group and group in SHEET_IDS:
        # Process only the specified group
        sheet_ids_to_process = {group: SHEET_IDS[group]}
    else:
        # Process all groups
        sheet_ids_to_process = SHEET_IDS

    # Process each sheet
    for sheet_group, sheet_id in sheet_ids_to_process.items():
        if sheet_group == "SPECIAL":  # Skip special activities sheet for this query
            continue

        try:
            # Get the sheet with all rows and columns (with retry logic)
            sheet = client.get_sheet(sheet_id)

            # Check if sheet retrieval failed after retries
            if sheet is None:
                logger.warning(f"Sheet {sheet_group} not available after retries, skipping")
                continue

            logger.info(f"Processing sheet {sheet_group} for activity metrics")
            
            # Map column titles to IDs for the phase columns
            phase_cols = {}
            for col in sheet.columns:
                if col.title in ["Kontrolle", "BE am", "K am", "C am", "Reopen C2 am"]:
                    phase_cols[col.title] = col.id
            
            # Process each row
            for row in sheet.rows:
                total_items += 1
                
                # Find the most recent date across all phase columns
                most_recent_date = None
                
                for col_title, col_id in phase_cols.items():
                    for cell in row.cells:
                        if cell.column_id == col_id and cell.value:
                            try:
                                date_val = parse_date(cell.value)
                                if date_val and (most_recent_date is None or date_val > most_recent_date):
                                    most_recent_date = date_val
                            except:
                                pass
                
                # Check if the most recent date is within the last 30 days
                if most_recent_date and most_recent_date >= thirty_days_ago.date():
                    recent_activity_items += 1
                    
        except Exception as e:
            logger.error(f"Error processing sheet {sheet_group} for metrics: {e}")
    
    # Calculate percentage
    recent_percentage = (recent_activity_items / total_items * 100) if total_items > 0 else 0
    
    return {
        "total_items": total_items,
        "recent_activity_items": recent_activity_items,
        "recent_percentage": recent_percentage
    }

def draw_half_circle_gauge(value_percentage: float, total_value: int, label: str, width: int = 250, height: int = 150,
                          color: "Color" = colors.steelblue, empty_color: "Color" = colors.lightgrey, show_labels: bool = True) -> "Drawing":
    """Draw a half-circle gauge chart showing a percentage with clear data labels.

    Args:
        value_percentage: Percentage value to display (0-100)
        total_value: Total count value to display
        label: Title label for the gauge
        width: Chart width in points
        height: Chart height in points
        color: Fill color for the gauge arc
        empty_color: Background color for empty portion
        show_labels: If True, display data labels (default: True)
    """
    from reportlab.graphics.shapes import Wedge, String

    drawing = Drawing(width, height)

    # Center of the half-circle
    cx = width / 2
    cy = height * 0.2  # Position near the bottom to leave room for labels

    # Radius of the half-circle
    radius = min(width, height) * 0.7 / 2

    # Create the background (empty) half-circle using a Wedge
    background = Wedge(cx, cy, radius, 0, 180, fillColor=empty_color, strokeColor=colors.grey, strokeWidth=1)
    drawing.add(background)

    # Calculate angle for the filled portion (0 to 180 degrees)
    filled_angle = min(180, value_percentage * 1.8)  # 100% = 180 degrees

    # Create the filled portion using a Wedge
    if filled_angle > 0:
        filled = Wedge(cx, cy, radius, 0, filled_angle,
                     fillColor=color, strokeColor=colors.black, strokeWidth=0.5)
        drawing.add(filled)

    # Add labels
    if show_labels:
        # Title label at top
        drawing.add(String(cx, cy + radius * 1.2, label,
                         fontName='Helvetica-Bold', fontSize=12, textAnchor='middle'))

        # Main percentage value - prominent display
        drawing.add(String(cx, cy - radius * 0.2, f"{value_percentage:.1f}%",
                         fontName='Helvetica-Bold', fontSize=16, textAnchor='middle',
                         fillColor=colors.black))

        # Total items count
        drawing.add(String(cx, cy - radius * 0.55, f"{total_value} items",
                         fontName='Helvetica', fontSize=10, textAnchor='middle'))

        # Add 0% and 100% markers at gauge endpoints
        drawing.add(String(cx - radius - 5, cy, "0%",
                         fontName='Helvetica', fontSize=7, textAnchor='end'))
        drawing.add(String(cx + radius + 5, cy, "100%",
                         fontName='Helvetica', fontSize=7, textAnchor='start'))

    return drawing

def create_stacked_gauge_chart(summary_data: Dict[str, Any], width: int = 500, height: int = 150, show_labels: bool = True) -> "Drawing":
    """Creates a custom stacked half-circle gauge chart for overdue statuses with data labels.

    Args:
        summary_data: Dictionary containing overdue status values
        width: Chart width in points
        height: Chart height in points
        show_labels: If True, display data labels on segments (default: True)
    """

    drawing = Drawing(width, height)

    # Define the categories and their WCAG AA compliant colors
    status_categories = get_reportlab_overdue_status_colors()

    # Extract and clean values
    status_values = {}
    for cat in status_categories:
        try:
            value_str = str(summary_data.get(cat, '0') or '0').replace('.', '')
            status_values[cat] = int(value_str)
        except (ValueError, TypeError):
            status_values[cat] = 0

    total_products = sum(status_values.values())

    if total_products == 0:
        drawing.add(String(width/2, height/2, "No overdue status data available.", textAnchor='middle'))
        return drawing

    # Gauge properties
    cx, cy = width / 2, height * 0.4
    radius = min(width, height) * 0.8 / 2

    # Draw the segments first
    start_angle = 180  # Start from the left
    for category, color in status_categories.items():
        value = status_values[category]
        if value > 0:
            percentage = value / total_products
            angle_extent = percentage * 180  # The size of this segment in degrees

            # Draw the wedge for this segment
            wedge = Wedge(cx, cy, radius, start_angle - angle_extent, start_angle, fillColor=color)
            drawing.add(wedge)

            # Update the start angle for the next segment
            start_angle -= angle_extent

    # Add labels with intelligent positioning to avoid overlap
    if show_labels:
        # Calculate intelligent label positions using gauge-specific function
        label_positions = calculate_gauge_label_positions(
            status_values, total_products, cx, cy, radius,
            min_percentage=10.0, font_name='Helvetica-Bold', font_size=8
        )

        # Create list of status categories for color lookup
        status_list = list(status_categories.items())

        # Create function to get text color for each segment
        def get_segment_text_color(index: int) -> "Color":
            if index < len(status_list):
                color = status_list[index][1]
                segment_color_hex = "#{:02x}{:02x}{:02x}".format(
                    int(color.red * 255),
                    int(color.green * 255),
                    int(color.blue * 255)
                )
                text_color_hex = get_accessible_text_color(segment_color_hex)
                return colors.HexColor(text_color_hex)
            return colors.black

        # Draw labels with intelligent positioning
        draw_pie_labels(drawing, label_positions, get_segment_text_color)

    # Add a title and total count in the center
    drawing.add(String(cx, cy - 15, f"Total: {total_products}", textAnchor='middle', fontName='Helvetica-Bold'))
    drawing.add(String(cx, height - 20, "Product Overdue Status", textAnchor='middle', fontName='Helvetica-Bold'))

    # Add a legend below the gauge with values and percentages
    legend_y = 10
    x_start = 50
    legend_items = []
    for category, value in status_values.items():
        percentage = (value / total_products * 100) if total_products > 0 else 0
        legend_items.append(f"{category}: {value} ({percentage:.1f}%)")

    # Draw legend
    for i, (category, color) in enumerate(status_categories.items()):
        drawing.add(Rect(x_start, legend_y, 8, 8, fillColor=color))
        drawing.add(String(x_start + 12, legend_y, legend_items[i], fontName='Helvetica', fontSize=8))
        x_start += stringWidth(legend_items[i], 'Helvetica', 8) + 25

    return drawing

def draw_full_gauge(total_value: int, label: str, width: int = 250, height: int = 150, color: "Color" = colors.steelblue) -> "Drawing":
    """Draw a decorative half-circle gauge showing the total count."""
    # This is essentially draw_half_circle_gauge with 100% value
    return draw_half_circle_gauge(100, total_value, label, width, height, color)


def draw_completion_rate_gauge(completion_rate: float, completed_count: int, total_count: int, label: str = "Completion Rate",
                               width: int = 250, height: int = 150, target_rate: float = 80.0, show_target: bool = True) -> "Drawing":
    """Draw a completion rate gauge with target comparison indicator.

    Creates a half-circle gauge showing completion rate percentage with a visual
    indicator comparing against a target completion rate. The gauge uses color
    coding to indicate performance level:
    - Green: >= 80% (excellent)
    - Yellow/Gold: 60-79% (good)
    - Orange: 40-59% (average)
    - Red: < 40% (poor)

    Args:
        completion_rate: Completion rate percentage (0-100)
        completed_count: Number of completed items
        total_count: Total number of items
        label: Title label for the gauge (default: "Completion Rate")
        width: Chart width in points (default: 250)
        height: Chart height in points (default: 150)
        target_rate: Target completion rate to compare against (default: 80.0)
        show_target: If True, display target indicator line (default: True)

    Returns:
        ReportLab Drawing object with the completion rate gauge
    """
    from reportlab.graphics.shapes import Wedge, String, Line

    drawing = Drawing(width, height)

    # Center of the half-circle
    cx = width / 2
    cy = height * 0.2  # Position near the bottom to leave room for labels

    # Radius of the half-circle
    radius = min(width, height) * 0.7 / 2

    # Determine color based on completion rate performance level
    if completion_rate >= 80:
        gauge_color = colors.HexColor("#1B5E20")  # Green 900 - excellent
        performance_text = "Excellent"
    elif completion_rate >= 60:
        gauge_color = colors.HexColor("#8B6914")  # Dark Gold - good
        performance_text = "Good"
    elif completion_rate >= 40:
        gauge_color = colors.HexColor("#E65100")  # Orange 800 - average
        performance_text = "Average"
    else:
        gauge_color = colors.HexColor("#B71C1C")  # Red 900 - poor
        performance_text = "Needs Improvement"

    # Create the background (empty) half-circle using a Wedge
    background = Wedge(cx, cy, radius, 0, 180, fillColor=colors.lightgrey, strokeColor=colors.grey, strokeWidth=1)
    drawing.add(background)

    # Calculate angle for the filled portion (0 to 180 degrees)
    filled_angle = min(180, completion_rate * 1.8)  # 100% = 180 degrees

    # Create the filled portion using a Wedge
    if filled_angle > 0:
        filled = Wedge(cx, cy, radius, 0, filled_angle,
                     fillColor=gauge_color, strokeColor=colors.black, strokeWidth=0.5)
        drawing.add(filled)

    # Add target indicator line if requested
    if show_target:
        target_angle_rad = (target_rate * 1.8) * (3.14159 / 180)  # Convert to radians
        target_x = cx + radius * 0.95 * math.cos(target_angle_rad)
        target_y = cy + radius * 0.95 * math.sin(target_angle_rad)
        target_x_inner = cx + radius * 0.6 * math.cos(target_angle_rad)
        target_y_inner = cy + radius * 0.6 * math.sin(target_angle_rad)

        # Draw target line
        target_line = Line(target_x_inner, target_y_inner, target_x, target_y,
                          strokeColor=colors.HexColor("#2c3e50"), strokeWidth=2)
        drawing.add(target_line)

        # Add small "T" label near the target line
        target_label_x = cx + radius * 1.05 * math.cos(target_angle_rad)
        target_label_y = cy + radius * 1.05 * math.sin(target_angle_rad)
        drawing.add(String(target_label_x, target_label_y, "Target",
                         fontName='Helvetica', fontSize=6, textAnchor='middle',
                         fillColor=colors.HexColor("#2c3e50")))

    # Title label at top
    drawing.add(String(cx, cy + radius * 1.25, label,
                     fontName='Helvetica-Bold', fontSize=12, textAnchor='middle'))

    # Main percentage value - prominent display
    drawing.add(String(cx, cy - radius * 0.15, f"{completion_rate:.1f}%",
                     fontName='Helvetica-Bold', fontSize=16, textAnchor='middle',
                     fillColor=colors.black))

    # Completed/Total count
    drawing.add(String(cx, cy - radius * 0.45, f"{completed_count}/{total_count} completed",
                     fontName='Helvetica', fontSize=9, textAnchor='middle'))

    # Performance label
    drawing.add(String(cx, cy - radius * 0.70, performance_text,
                     fontName='Helvetica-Bold', fontSize=8, textAnchor='middle',
                     fillColor=gauge_color))

    # Add 0% and 100% markers at gauge endpoints
    drawing.add(String(cx - radius - 5, cy, "0%",
                     fontName='Helvetica', fontSize=7, textAnchor='end'))
    drawing.add(String(cx + radius + 5, cy, "100%",
                     fontName='Helvetica', fontSize=7, textAnchor='start'))

    return drawing


def draw_completion_progress_bar(completion_rate: float, completed_count: int, total_count: int,
                                 label: str = "Completion Rate", width: int = 220, height: int = 60,
                                 target_rate: float = 80.0, show_target: bool = True) -> "Drawing":
    """Draw a horizontal progress bar showing completion rate with target indicator.

    Creates a horizontal progress bar visualization of completion rate with
    color coding based on performance level and optional target marker.

    Args:
        completion_rate: Completion rate percentage (0-100)
        completed_count: Number of completed items
        total_count: Total number of items
        label: Title label for the progress bar (default: "Completion Rate")
        width: Chart width in points (default: 220)
        height: Chart height in points (default: 60)
        target_rate: Target completion rate to compare against (default: 80.0)
        show_target: If True, display target indicator line (default: True)

    Returns:
        ReportLab Drawing object with the completion progress bar
    """
    from reportlab.graphics.shapes import Rect, String, Line

    drawing = Drawing(width, height)

    # Bar dimensions
    bar_width = width - 40
    bar_height = 18
    bar_x = 20
    bar_y = height / 2 - bar_height / 2

    # Determine color based on completion rate performance level
    if completion_rate >= 80:
        bar_color = colors.HexColor("#1B5E20")  # Green 900 - excellent
        performance_text = "Excellent"
    elif completion_rate >= 60:
        bar_color = colors.HexColor("#8B6914")  # Dark Gold - good
        performance_text = "Good"
    elif completion_rate >= 40:
        bar_color = colors.HexColor("#E65100")  # Orange 800 - average
        performance_text = "Average"
    else:
        bar_color = colors.HexColor("#B71C1C")  # Red 900 - poor
        performance_text = "Needs Improvement"

    # Draw background bar
    background_bar = Rect(bar_x, bar_y, bar_width, bar_height,
                         fillColor=colors.lightgrey, strokeColor=colors.grey, strokeWidth=1)
    drawing.add(background_bar)

    # Draw filled portion
    filled_width = bar_width * (completion_rate / 100)
    if filled_width > 0:
        filled_bar = Rect(bar_x, bar_y, filled_width, bar_height,
                         fillColor=bar_color, strokeColor=None)
        drawing.add(filled_bar)

    # Add target marker if requested
    if show_target:
        target_x = bar_x + bar_width * (target_rate / 100)
        target_line = Line(target_x, bar_y - 3, target_x, bar_y + bar_height + 3,
                          strokeColor=colors.HexColor("#2c3e50"), strokeWidth=2)
        drawing.add(target_line)

        # Add small triangle marker above the target line
        drawing.add(String(target_x, bar_y + bar_height + 7, "▼",
                         fontName='Helvetica', fontSize=6, textAnchor='middle',
                         fillColor=colors.HexColor("#2c3e50")))

    # Label at top
    drawing.add(String(bar_x, height - 5, label,
                     fontName='Helvetica-Bold', fontSize=9, textAnchor='start'))

    # Percentage value at right of label
    drawing.add(String(bar_x + bar_width, height - 5, f"{completion_rate:.1f}%",
                     fontName='Helvetica-Bold', fontSize=9, textAnchor='end',
                     fillColor=bar_color))

    # Completed/Total and performance text below bar
    drawing.add(String(bar_x, bar_y - 8, f"{completed_count}/{total_count} ({performance_text})",
                     fontName='Helvetica', fontSize=7, textAnchor='start',
                     fillColor=colors.HexColor("#5d6d7e")))

    # Target label
    if show_target:
        drawing.add(String(bar_x + bar_width, bar_y - 8, f"Target: {target_rate:.0f}%",
                         fontName='Helvetica', fontSize=7, textAnchor='end',
                         fillColor=colors.HexColor("#5d6d7e")))

    return drawing


def create_sample_image(title: str, message: str, width: int = 500, height: int = 200) -> "Drawing":
    """Create a placeholder image with text."""
    from reportlab.lib.colors import lightgrey, black
    from reportlab.graphics.shapes import Rect
    
    drawing = Drawing(width, height)
    
    # Add a background rectangle
    drawing.add(Rect(0, 0, width, height, fillColor=lightgrey))
    
    # Add title
    drawing.add(String(width/2, height-30, title,
                     fontName='Helvetica-Bold', fontSize=14, textAnchor='middle'))
    
    # Add message
    drawing.add(String(width/2, height/2, message,
                     fontName='Helvetica', fontSize=12, textAnchor='middle'))

    return drawing



# ============================================================================
# EXECUTIVE SUMMARY PAGE LAYOUT
# ============================================================================

def create_overall_activity_summary(metrics: Dict[str, Any], period_str: str, width: int = 500, height: int = 120) -> "Drawing":
    """Create a prominent overall activity summary section with large typography.

    Displays total changes across all groups, active user count, and date range
    in a visually prominent layout suitable for executive summary pages.

    Args:
        metrics: Dictionary containing total_changes, groups, users data
        period_str: String describing the report period (e.g., "Jan 1 - Jan 7, 2026")
        width: Width of the section in points
        height: Height of the section in points

    Returns:
        Drawing object with the overall activity summary
    """
    drawing = Drawing(width, height)

    # Background panel with subtle gradient effect
    # Shadow layer
    drawing.add(Rect(3, 0, width - 3, height - 3,
                     fillColor=colors.HexColor("#d5d8dc"), strokeColor=None))

    # Main background
    drawing.add(Rect(0, 3, width - 3, height - 3,
                     fillColor=colors.HexColor("#f8f9fa"),
                     strokeColor=colors.HexColor("#dee2e6"),
                     strokeWidth=1))

    # Top accent bar with gradient-like effect (primary brand color)
    accent_color = colors.HexColor("#2c3e50")
    drawing.add(Rect(0, height - 5, width - 3, 8,
                     fillColor=accent_color, strokeColor=None))

    # Calculate positions for three main metrics
    section_width = (width - 60) / 3
    base_y = height / 2

    # ===== METRIC 1: Total Changes =====
    metric1_x = 30
    total_changes = metrics.get("total_changes", 0)

    # Large value - prominent display
    drawing.add(String(metric1_x + section_width / 2, base_y + 20,
                       str(total_changes),
                       fontName='Helvetica-Bold', fontSize=36, textAnchor='middle',
                       fillColor=colors.HexColor("#2c3e50")))

    # Label below the value
    drawing.add(String(metric1_x + section_width / 2, base_y - 10,
                       "TOTAL CHANGES",
                       fontName='Helvetica-Bold', fontSize=11, textAnchor='middle',
                       fillColor=colors.HexColor("#5d6d7e")))

    # Subtitle
    drawing.add(String(metric1_x + section_width / 2, base_y - 25,
                       "across all groups",
                       fontName='Helvetica', fontSize=9, textAnchor='middle',
                       fillColor=colors.HexColor("#7f8c8d")))

    # Vertical divider
    divider1_x = metric1_x + section_width + 10
    drawing.add(Line(divider1_x, 20, divider1_x, height - 20,
                     strokeColor=colors.HexColor("#bdc3c7"), strokeWidth=1))

    # ===== METRIC 2: Active Users =====
    metric2_x = divider1_x + 10
    active_users = len(metrics.get("users", {}))

    # Large value - prominent display
    drawing.add(String(metric2_x + section_width / 2, base_y + 20,
                       str(active_users),
                       fontName='Helvetica-Bold', fontSize=36, textAnchor='middle',
                       fillColor=colors.HexColor("#9b59b6")))

    # Label below the value
    drawing.add(String(metric2_x + section_width / 2, base_y - 10,
                       "ACTIVE USERS",
                       fontName='Helvetica-Bold', fontSize=11, textAnchor='middle',
                       fillColor=colors.HexColor("#5d6d7e")))

    # Subtitle
    drawing.add(String(metric2_x + section_width / 2, base_y - 25,
                       "contributors",
                       fontName='Helvetica', fontSize=9, textAnchor='middle',
                       fillColor=colors.HexColor("#7f8c8d")))

    # Vertical divider
    divider2_x = metric2_x + section_width + 10
    drawing.add(Line(divider2_x, 20, divider2_x, height - 20,
                     strokeColor=colors.HexColor("#bdc3c7"), strokeWidth=1))

    # ===== METRIC 3: Date Range =====
    metric3_x = divider2_x + 10

    # Calendar icon placeholder (using a simple box with text)
    icon_size = 24
    icon_x = metric3_x + section_width / 2 - icon_size / 2
    icon_y = base_y + 15
    drawing.add(Rect(icon_x, icon_y, icon_size, icon_size,
                     fillColor=colors.HexColor("#3498db"),
                     strokeColor=None,
                     rx=3, ry=3))
    drawing.add(String(icon_x + icon_size / 2, icon_y + 8, "📅",
                       fontName='Helvetica', fontSize=10, textAnchor='middle',
                       fillColor=colors.white))

    # Label below the icon
    drawing.add(String(metric3_x + section_width / 2, base_y - 5,
                       "REPORTING PERIOD",
                       fontName='Helvetica-Bold', fontSize=11, textAnchor='middle',
                       fillColor=colors.HexColor("#5d6d7e")))

    # Date range value - slightly smaller but still prominent
    drawing.add(String(metric3_x + section_width / 2, base_y - 22,
                       period_str,
                       fontName='Helvetica-Bold', fontSize=12, textAnchor='middle',
                       fillColor=colors.HexColor("#3498db")))

    # Decorative corner accents
    corner_size = 15
    # Bottom-left corner
    corner_path = Path()
    corner_path.moveTo(0, 3)
    corner_path.lineTo(0, 3 + corner_size)
    corner_path.lineTo(corner_size, 3)
    corner_path.closePath()
    corner_path.fillColor = accent_color
    corner_path.strokeColor = None
    drawing.add(corner_path)

    # Bottom-right corner
    corner_path2 = Path()
    corner_path2.moveTo(width - 3 - corner_size, 3)
    corner_path2.lineTo(width - 3, 3)
    corner_path2.lineTo(width - 3, 3 + corner_size)
    corner_path2.closePath()
    corner_path2.fillColor = accent_color
    corner_path2.strokeColor = None
    drawing.add(corner_path2)

    return drawing


def create_executive_summary_kpi_card(value: Any, label: str, subtitle: Optional[str] = None, width: int = 80, height: int = 60,
                                       accent_color: "Color" = colors.HexColor("#2ecc71")) -> "Drawing":
    """Create a single KPI card with a value, label, and optional subtitle.

    Args:
        value: The main metric value to display
        label: The label describing the metric
        subtitle: Optional additional context
        width: Card width in points
        height: Card height in points
        accent_color: Color for the accent bar

    Returns:
        Drawing object representing the KPI card
    """
    drawing = Drawing(width, height)

    # Background card with subtle shadow effect
    shadow_offset = 2
    drawing.add(Rect(shadow_offset, 0, width - shadow_offset, height - shadow_offset,
                     fillColor=colors.HexColor("#e0e0e0"), strokeColor=None))
    drawing.add(Rect(0, shadow_offset, width - shadow_offset, height - shadow_offset,
                     fillColor=colors.white, strokeColor=colors.HexColor("#d0d0d0"), strokeWidth=0.5))

    # Accent bar on the left
    accent_width = 4
    drawing.add(Rect(0, shadow_offset, accent_width, height - shadow_offset,
                     fillColor=accent_color, strokeColor=None))

    # Main value - large and bold
    value_str = str(value)
    value_y = height - 22
    drawing.add(String(width/2, value_y, value_str,
                      fontName='Helvetica-Bold', fontSize=18, textAnchor='middle',
                      fillColor=colors.HexColor("#2c3e50")))

    # Label - smaller below the value
    label_y = value_y - 18
    drawing.add(String(width/2, label_y, label,
                      fontName='Helvetica', fontSize=9, textAnchor='middle',
                      fillColor=colors.HexColor("#7f8c8d")))

    # Subtitle if provided
    if subtitle:
        subtitle_y = label_y - 12
        drawing.add(String(width/2, subtitle_y, subtitle,
                          fontName='Helvetica', fontSize=7, textAnchor='middle',
                          fillColor=colors.HexColor("#95a5a6")))

    return drawing


def create_executive_summary_metrics_row(metrics: Dict[str, Any], width: int = 500) -> "Table":
    """Create a row of KPI cards showing overall metrics.

    Args:
        metrics: Dictionary containing total_changes, groups, phases, users counts
        width: Total width available for the row

    Returns:
        Table containing the KPI cards
    """
    # Calculate card dimensions
    num_cards = 4
    card_spacing = 10
    card_width = (width - (num_cards - 1) * card_spacing) / num_cards
    card_height = 65

    # Create individual KPI cards
    total_changes_card = create_executive_summary_kpi_card(
        value=metrics.get("total_changes", 0),
        label="Total Changes",
        subtitle="this period",
        width=card_width,
        height=card_height,
        accent_color=colors.HexColor("#3498db")  # Blue
    )

    groups_card = create_executive_summary_kpi_card(
        value=len(metrics.get("groups", {})),
        label="Active Groups",
        subtitle="with activity",
        width=card_width,
        height=card_height,
        accent_color=colors.HexColor("#2ecc71")  # Green
    )

    users_card = create_executive_summary_kpi_card(
        value=len(metrics.get("users", {})),
        label="Active Users",
        subtitle="contributors",
        width=card_width,
        height=card_height,
        accent_color=colors.HexColor("#9b59b6")  # Purple
    )

    phases_card = create_executive_summary_kpi_card(
        value=len(metrics.get("phases", {})),
        label="Phases Active",
        subtitle="in workflow",
        width=card_width,
        height=card_height,
        accent_color=colors.HexColor("#e67e22")  # Orange
    )

    # Arrange cards in a table
    cards_data = [[total_changes_card, groups_card, users_card, phases_card]]
    cards_table = Table(cards_data, colWidths=[card_width] * num_cards)
    cards_table.setStyle(TableStyle([
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
    ]))

    return cards_table


def create_executive_highlights_section(metrics: Dict[str, Any], width: int = 500, height: int = 120) -> "Drawing":
    """Create a highlights section showing top performers and key insights.

    Args:
        metrics: Dictionary containing groups, users, phases data
        width: Width of the section
        height: Height of the section

    Returns:
        Drawing object with highlights visualization
    """
    drawing = Drawing(width, height)

    # Calculate positions for three highlight boxes
    box_width = (width - 40) / 3
    box_height = 80
    box_y = 15

    # Get top performers
    top_group = max(metrics.get("groups", {"N/A": 0}).items(), key=lambda x: x[1], default=("N/A", 0))
    top_user = max(metrics.get("users", {"N/A": 0}).items(), key=lambda x: x[1], default=("N/A", 0))
    top_phase = max(metrics.get("phases", {"N/A": 0}).items(), key=lambda x: x[1], default=("N/A", 0))

    highlights = [
        ("Most Active Group", top_group[0], f"{top_group[1]} changes", GROUP_COLORS.get(top_group[0], colors.HexColor("#3498db"))),
        ("Top Contributor", top_user[0], f"{top_user[1]} changes", colors.HexColor("#9b59b6")),
        ("Busiest Phase", f"Phase {top_phase[0]}", f"{top_phase[1]} changes", PHASE_COLORS.get(top_phase[0], colors.HexColor("#e67e22"))),
    ]

    for i, (title, value, detail, accent_color) in enumerate(highlights):
        box_x = 10 + i * (box_width + 10)

        # Box background
        drawing.add(Rect(box_x, box_y, box_width, box_height,
                        fillColor=colors.HexColor("#f8f9fa"),
                        strokeColor=colors.HexColor("#dee2e6"),
                        strokeWidth=0.5))

        # Accent bar at top
        drawing.add(Rect(box_x, box_y + box_height - 4, box_width, 4,
                        fillColor=accent_color, strokeColor=None))

        # Title
        drawing.add(String(box_x + box_width/2, box_y + box_height - 20, title,
                          fontName='Helvetica', fontSize=8, textAnchor='middle',
                          fillColor=colors.HexColor("#6c757d")))

        # Value - main highlight
        drawing.add(String(box_x + box_width/2, box_y + box_height/2 - 5, value,
                          fontName='Helvetica-Bold', fontSize=16, textAnchor='middle',
                          fillColor=colors.HexColor("#2c3e50")))

        # Detail
        drawing.add(String(box_x + box_width/2, box_y + 12, detail,
                          fontName='Helvetica', fontSize=9, textAnchor='middle',
                          fillColor=colors.HexColor("#6c757d")))

    return drawing


def create_group_status_grid(metrics: Dict[str, Any], width: int = 500, height: int = 220, health_scores: Optional[Dict[str, Any]] = None) -> "Drawing":
    """Create a grid showing all product groups with health status indicators.

    Displays a visual grid of all product groups with:
    - Health status indicators (green/yellow/red circles)
    - Group name with activity count
    - Mini metrics: total activity, completion rate
    - Activity progress bar

    Args:
        metrics: Dictionary containing groups and group_phase_user data
        width: Width of the grid
        height: Height of the grid
        health_scores: Optional dictionary mapping group names to GroupHealthScore objects.
                      If not provided, health status is estimated from activity metrics.

    Returns:
        Drawing object with the group status grid
    """
    drawing = Drawing(width, height)

    # Get all groups including bundle groups (use predefined list to ensure consistent order)
    all_groups = ["NA", "NF", "NH", "NM", "NP", "NT", "NV", "BUNDLE_FAN", "BUNDLE_COOLER"]
    groups_data = metrics.get("groups", {})

    # Health status colors for indicators
    health_status_colors = {
        "green": colors.HexColor("#2ca02c"),   # Green - Healthy
        "yellow": colors.HexColor("#ff7f0e"),  # Yellow/Orange - Caution
        "red": colors.HexColor("#d62728"),     # Red - Critical
    }

    # Calculate estimated health status if not provided
    def estimate_health_status(group: str, changes: int, total_products: int) -> Tuple[str, int, float]:
        """Estimate health status based on activity metrics."""
        if total_products == 0:
            return "yellow", 50, 0.0  # status, score, completion_rate

        # Calculate activity rate (changes per product)
        activity_rate = changes / total_products if total_products > 0 else 0

        # Estimate completion rate based on activity (higher activity suggests progress)
        # This is a rough estimate; actual values would come from sheet data
        completion_rate = min(0.95, 0.50 + activity_rate * 2.0)

        # Calculate estimated health score
        # Activity contributes 35%, completion 40%, no overdue penalty for estimation (25%)
        activity_score = min(100, activity_rate * 1000)  # Scale activity
        completion_score = completion_rate * 100
        estimated_score = (0.35 * activity_score + 0.40 * completion_score + 0.25 * 75)  # Assume 75% overdue score

        # Determine status from score
        if estimated_score >= 70:
            return "green", estimated_score, completion_rate
        elif estimated_score >= 40:
            return "yellow", estimated_score, completion_rate
        else:
            return "red", estimated_score, completion_rate

    # Calculate grid layout - adjusted for 9 groups (3 rows x 3 cols)
    cols = 3
    rows = 3
    cell_width = (width - 40) / cols
    cell_height = 65  # Slightly taller to accommodate mini metrics
    grid_start_y = height - 15
    grid_start_x = 20

    # Add legend at top
    legend_y = height - 8
    legend_items = [
        ("green", "Healthy"),
        ("yellow", "Caution"),
        ("red", "Critical"),
    ]
    legend_x = 20
    for status, label in legend_items:
        status_color = health_status_colors.get(status, colors.gray)
        drawing.add(Circle(legend_x + 5, legend_y, 4,
                          fillColor=status_color, strokeColor=colors.white, strokeWidth=0.5))
        drawing.add(String(legend_x + 12, legend_y - 3, label,
                          fontName='Helvetica', fontSize=7,
                          fillColor=colors.HexColor("#6c757d")))
        legend_x += 55

    for idx, group in enumerate(all_groups):
        row = idx // cols
        col = idx % cols

        x = grid_start_x + col * cell_width
        y = grid_start_y - row * (cell_height + 8) - 10  # Adjusted for legend

        changes = groups_data.get(group, 0)
        group_color = GROUP_COLORS.get(group, colors.steelblue)
        total_products = TOTAL_PRODUCTS.get(group, 0)

        # Get health status - from provided scores or estimate
        if health_scores and group in health_scores:
            score_obj = health_scores[group]
            health_status = score_obj.status.value  # "green", "yellow", or "red"
            health_score = score_obj.overall_score
            completion_rate = score_obj.completed_products / score_obj.total_products if score_obj.total_products > 0 else 0
        else:
            health_status, health_score, completion_rate = estimate_health_status(group, changes, total_products)

        status_color = health_status_colors.get(health_status, colors.gray)

        # Cell background with subtle border matching health status
        drawing.add(Rect(x + 2, y - cell_height + 2, cell_width - 4, cell_height - 4,
                        fillColor=colors.white,
                        strokeColor=status_color,
                        strokeWidth=1.5))

        # Health status indicator circle (top-left)
        circle_radius = 7
        drawing.add(Circle(x + 15, y - 12, circle_radius,
                          fillColor=status_color, strokeColor=colors.white, strokeWidth=1))

        # Health score inside the circle
        score_text = f"{int(health_score)}"
        font_size_score = 6 if health_score >= 100 else 7
        drawing.add(String(x + 15, y - 14, score_text,
                          fontName='Helvetica-Bold', fontSize=font_size_score, textAnchor='middle',
                          fillColor=colors.white))

        # Group name - use shorter display names for bundle groups
        display_name = get_group_display_name(group, short=True)
        font_size = 10 if group.startswith("BUNDLE_") else 12
        drawing.add(String(x + 28, y - 15, display_name,
                          fontName='Helvetica-Bold', fontSize=font_size,
                          fillColor=colors.HexColor("#2c3e50")))

        # Mini metrics row - Total Activity
        mini_y = y - 28
        drawing.add(String(x + 10, mini_y, "Activity:",
                          fontName='Helvetica', fontSize=7,
                          fillColor=colors.HexColor("#6c757d")))
        drawing.add(String(x + cell_width - 10, mini_y, str(changes),
                          fontName='Helvetica-Bold', fontSize=8, textAnchor='end',
                          fillColor=group_color))

        # Mini metrics row - Completion Rate
        mini_y2 = y - 39
        completion_pct = int(completion_rate * 100)
        drawing.add(String(x + 10, mini_y2, "Completion:",
                          fontName='Helvetica', fontSize=7,
                          fillColor=colors.HexColor("#6c757d")))
        drawing.add(String(x + cell_width - 10, mini_y2, f"{completion_pct}%",
                          fontName='Helvetica-Bold', fontSize=8, textAnchor='end',
                          fillColor=health_status_colors.get("green" if completion_pct >= 70 else "yellow" if completion_pct >= 40 else "red")))

        # Products count (smaller, bottom of cell)
        if total_products > 0:
            drawing.add(String(x + 10, y - cell_height + 14, f"{total_products:,} products",
                              fontName='Helvetica', fontSize=7,
                              fillColor=colors.HexColor("#95a5a6")))

        # Activity indicator bar (proportion of changes)
        max_changes = max(groups_data.values()) if groups_data else 1
        bar_max_width = cell_width - 24
        bar_width = ((changes / max_changes) * bar_max_width) if max_changes > 0 else 0
        bar_height = 3
        bar_y = y - cell_height + 6

        # Background bar
        drawing.add(Rect(x + 10, bar_y, bar_max_width, bar_height,
                        fillColor=colors.HexColor("#e9ecef"), strokeColor=None))
        # Filled bar with group color
        if bar_width > 0:
            drawing.add(Rect(x + 10, bar_y, bar_width, bar_height,
                            fillColor=group_color, strokeColor=None))

    return drawing


def create_health_score_grid(health_scores: Dict[str, Any], width: int = 500, height: int = 280) -> "Drawing":
    """Create a grid showing health scores for all groups with colored status indicators.

    Displays each group's health score with:
    - Color-coded status indicator (green/yellow/red circle)
    - Group name
    - Overall health score
    - Score breakdown bars (activity, completion, overdue)
    - Trend indicator

    Args:
        health_scores: Dictionary mapping group names to GroupHealthScore objects
        width: Width of the grid
        height: Height of the grid

    Returns:
        Drawing object with the health score grid
    """
    drawing = Drawing(width, height)

    if not health_scores:
        # Show placeholder if no data
        drawing.add(String(width/2, height/2, "No health score data available",
                          fontName='Helvetica', fontSize=12, textAnchor='middle',
                          fillColor=colors.HexColor("#7f8c8d")))
        return drawing

    # Health status colors
    status_colors = {
        HealthStatus.GREEN: colors.HexColor("#2ca02c"),   # Green
        HealthStatus.YELLOW: colors.HexColor("#ff7f0e"),  # Orange/Yellow
        HealthStatus.RED: colors.HexColor("#d62728"),     # Red
    }

    # Get all groups and sort by score (highest first)
    sorted_scores = sorted(
        health_scores.values(),
        key=lambda x: x.overall_score,
        reverse=True
    )

    # Calculate grid layout - 3 columns for groups
    cols = 3
    rows = (len(sorted_scores) + cols - 1) // cols
    cell_width = (width - 40) / cols
    cell_height = 75
    grid_start_y = height - 25
    grid_start_x = 20

    # Add title
    drawing.add(String(width/2, height - 8, "Group Health Scores",
                      fontName='Helvetica-Bold', fontSize=11, textAnchor='middle',
                      fillColor=colors.HexColor("#2c3e50")))

    # Add legend at top
    legend_y = height - 22
    legend_items = [
        (HealthStatus.GREEN, "Healthy (70+)"),
        (HealthStatus.YELLOW, "Caution (40-69)"),
        (HealthStatus.RED, "Critical (<40)"),
    ]

    legend_x = 30
    for status, label in legend_items:
        status_color = status_colors.get(status, colors.gray)
        drawing.add(Circle(legend_x, legend_y, 4,
                          fillColor=status_color, strokeColor=colors.white, strokeWidth=0.5))
        drawing.add(String(legend_x + 8, legend_y - 3, label,
                          fontName='Helvetica', fontSize=7,
                          fillColor=colors.HexColor("#6c757d")))
        legend_x += 85

    for idx, score in enumerate(sorted_scores):
        row = idx // cols
        col = idx % cols

        x = grid_start_x + col * cell_width
        y = grid_start_y - row * (cell_height + 8)

        status_color = status_colors.get(score.status, colors.gray)
        group_color = GROUP_COLORS.get(score.group, colors.steelblue)

        # Cell background with rounded corners simulation
        drawing.add(Rect(x + 2, y - cell_height + 2, cell_width - 4, cell_height - 4,
                        fillColor=colors.white,
                        strokeColor=colors.HexColor("#dee2e6"),
                        strokeWidth=1))

        # Status indicator circle (left side)
        circle_radius = 10
        drawing.add(Circle(x + 18, y - 18, circle_radius,
                          fillColor=status_color, strokeColor=colors.white, strokeWidth=1))

        # Score inside the circle
        score_text = f"{int(score.overall_score)}"
        drawing.add(String(x + 18, y - 21, score_text,
                          fontName='Helvetica-Bold', fontSize=8, textAnchor='middle',
                          fillColor=colors.white))

        # Group name with group color accent
        display_name = get_group_display_name(score.group, short=True)
        font_size = 10 if score.group.startswith("BUNDLE_") else 12
        drawing.add(String(x + 35, y - 16, display_name,
                          fontName='Helvetica-Bold', fontSize=font_size,
                          fillColor=colors.HexColor("#2c3e50")))

        # Status label
        drawing.add(String(x + 35, y - 28, score.status_label,
                          fontName='Helvetica', fontSize=7,
                          fillColor=status_color))

        # Trend indicator
        trend_symbols = {"up": "+", "down": "-", "flat": "="}
        trend_colors = {
            "up": colors.HexColor("#2ca02c"),
            "down": colors.HexColor("#d62728"),
            "flat": colors.HexColor("#6c757d"),
        }
        trend_symbol = trend_symbols.get(score.trend, "=")
        trend_color = trend_colors.get(score.trend, colors.gray)
        drawing.add(String(x + cell_width - 15, y - 14, trend_symbol,
                          fontName='Helvetica-Bold', fontSize=14, textAnchor='end',
                          fillColor=trend_color))

        # Score breakdown bars
        bar_width_max = cell_width - 30
        bar_height = 4
        bar_y_start = y - 40
        bar_spacing = 10

        score_components = [
            ("Activity", score.activity_score, colors.HexColor("#3498db")),
            ("Completion", score.completion_score, colors.HexColor("#27ae60")),
            ("Overdue", score.overdue_score, colors.HexColor("#e74c3c")),
        ]

        for i, (label, value, bar_color) in enumerate(score_components):
            bar_y = bar_y_start - i * bar_spacing

            # Label
            drawing.add(String(x + 8, bar_y - 3, label[:3],
                              fontName='Helvetica', fontSize=6,
                              fillColor=colors.HexColor("#95a5a6")))

            # Background bar
            bg_bar_x = x + 25
            bg_bar_width = bar_width_max - 35
            drawing.add(Rect(bg_bar_x, bar_y, bg_bar_width, bar_height,
                            fillColor=colors.HexColor("#e9ecef"), strokeColor=None))

            # Filled bar (proportional to score)
            filled_width = (value / 100.0) * bg_bar_width if value > 0 else 0
            if filled_width > 0:
                drawing.add(Rect(bg_bar_x, bar_y, filled_width, bar_height,
                                fillColor=bar_color, strokeColor=None))

            # Value text
            drawing.add(String(x + cell_width - 12, bar_y - 3, f"{int(value)}",
                              fontName='Helvetica', fontSize=6, textAnchor='end',
                              fillColor=colors.HexColor("#7f8c8d")))

    return drawing


def create_health_summary_callout(health_scores: Dict[str, Any], width: int = 500, height: int = 60) -> "Drawing":
    """Create a summary callout showing overall health status.

    Args:
        health_scores: Dictionary mapping group names to GroupHealthScore objects
        width: Width of the callout
        height: Height of the callout

    Returns:
        Drawing object with the health summary
    """
    drawing = Drawing(width, height)

    if not health_scores:
        return drawing

    summary = get_health_summary(health_scores)

    # Background
    drawing.add(Rect(10, 5, width - 20, height - 10,
                    fillColor=colors.HexColor("#f8f9fa"),
                    strokeColor=colors.HexColor("#dee2e6"),
                    strokeWidth=1))

    # Title
    drawing.add(String(width/2, height - 18, "Health Score Summary",
                      fontName='Helvetica-Bold', fontSize=10, textAnchor='middle',
                      fillColor=colors.HexColor("#2c3e50")))

    # Status counts with colored circles
    y_pos = height - 38
    x_positions = [60, 170, 280, 420]

    # Average score
    avg_color = colors.HexColor("#2ca02c") if summary['average_score'] >= 70 else \
                colors.HexColor("#ff7f0e") if summary['average_score'] >= 40 else \
                colors.HexColor("#d62728")
    drawing.add(String(x_positions[0], y_pos, f"Avg: {summary['average_score']}",
                      fontName='Helvetica-Bold', fontSize=11, textAnchor='middle',
                      fillColor=avg_color))

    # Green count
    drawing.add(Circle(x_positions[1] - 20, y_pos + 3, 6,
                      fillColor=colors.HexColor("#2ca02c"), strokeColor=None))
    drawing.add(String(x_positions[1], y_pos, f"{summary['green_count']} Healthy",
                      fontName='Helvetica', fontSize=9,
                      fillColor=colors.HexColor("#2c3e50")))

    # Yellow count
    drawing.add(Circle(x_positions[2] - 20, y_pos + 3, 6,
                      fillColor=colors.HexColor("#ff7f0e"), strokeColor=None))
    drawing.add(String(x_positions[2], y_pos, f"{summary['yellow_count']} Caution",
                      fontName='Helvetica', fontSize=9,
                      fillColor=colors.HexColor("#2c3e50")))

    # Red count
    drawing.add(Circle(x_positions[3] - 20, y_pos + 3, 6,
                      fillColor=colors.HexColor("#d62728"), strokeColor=None))
    drawing.add(String(x_positions[3], y_pos, f"{summary['red_count']} Critical",
                      fontName='Helvetica', fontSize=9,
                      fillColor=colors.HexColor("#2c3e50")))

    return drawing


def create_key_metrics_callout_section(metrics: Dict[str, Any], special_activity_hours: float, width: int = 500, height: int = 100) -> "Drawing":
    """Create attention-grabbing callout boxes for key metrics.

    Displays three prominent callout boxes showing:
    - Most active group (with group color)
    - Most active user (top contributor)
    - Total special activity hours

    Args:
        metrics: Dictionary containing groups, users data
        special_activity_hours: Total hours from special activities
        width: Width of the section
        height: Height of the section

    Returns:
        Drawing object with three callout boxes
    """
    drawing = Drawing(width, height)

    # Calculate positions for three callout boxes
    box_width = (width - 40) / 3
    box_height = 80
    box_y = 10

    # Get key metrics data
    groups_data = metrics.get("groups", {"N/A": 0})
    users_data = metrics.get("users", {"N/A": 0})

    # Find most active group
    if groups_data:
        top_group = max(groups_data.items(), key=lambda x: x[1])
    else:
        top_group = ("N/A", 0)

    # Find most active user
    if users_data:
        top_user = max(users_data.items(), key=lambda x: x[1])
    else:
        top_user = ("N/A", 0)

    # Prepare callout data: (title, main_value, detail_text, accent_color, icon_symbol)
    callouts = [
        (
            "MOST ACTIVE GROUP",
            top_group[0],
            f"{top_group[1]} changes",
            GROUP_COLORS.get(top_group[0], colors.HexColor("#E63946")),
            "★"
        ),
        (
            "TOP CONTRIBUTOR",
            top_user[0],
            f"{top_user[1]} changes",
            colors.HexColor("#9b59b6"),  # Purple
            "👤"
        ),
        (
            "SPECIAL ACTIVITIES",
            f"{special_activity_hours:.1f}h",
            "total hours logged",
            colors.HexColor("#f39c12"),  # Orange/Gold
            "⏱"
        ),
    ]

    for i, (title, value, detail, accent_color, icon) in enumerate(callouts):
        box_x = 10 + i * (box_width + 10)

        # Main box background with gradient effect (simulated with layered rects)
        # Shadow layer
        drawing.add(Rect(box_x + 2, box_y - 2, box_width, box_height,
                        fillColor=colors.HexColor("#d0d0d0"), strokeColor=None))

        # Main white background
        drawing.add(Rect(box_x, box_y, box_width, box_height,
                        fillColor=colors.white,
                        strokeColor=colors.HexColor("#e0e0e0"),
                        strokeWidth=1))

        # Thick accent bar at top (attention-grabbing)
        accent_bar_height = 6
        drawing.add(Rect(box_x, box_y + box_height - accent_bar_height, box_width, accent_bar_height,
                        fillColor=accent_color, strokeColor=None))

        # Left accent stripe
        drawing.add(Rect(box_x, box_y, 4, box_height - accent_bar_height,
                        fillColor=accent_color, strokeColor=None))

        # Title text (uppercase, bold, smaller)
        title_y = box_y + box_height - 20
        drawing.add(String(box_x + box_width/2, title_y, title,
                          fontName='Helvetica-Bold', fontSize=8, textAnchor='middle',
                          fillColor=colors.HexColor("#6c757d")))

        # Main value - large, bold, and attention-grabbing
        value_y = box_y + box_height/2 - 2
        drawing.add(String(box_x + box_width/2, value_y, str(value),
                          fontName='Helvetica-Bold', fontSize=20, textAnchor='middle',
                          fillColor=accent_color))

        # Detail text at bottom
        detail_y = box_y + 12
        drawing.add(String(box_x + box_width/2, detail_y, detail,
                          fontName='Helvetica', fontSize=9, textAnchor='middle',
                          fillColor=colors.HexColor("#7f8c8d")))

        # Add decorative corner accent (small triangle)
        corner_size = 12
        corner_path = Path()
        corner_path.moveTo(box_x + box_width - corner_size, box_y + box_height - accent_bar_height)
        corner_path.lineTo(box_x + box_width, box_y + box_height - accent_bar_height)
        corner_path.lineTo(box_x + box_width, box_y + box_height - accent_bar_height - corner_size)
        corner_path.closePath()
        corner_path.fillColor = accent_color
        corner_path.strokeColor = None
        drawing.add(corner_path)

    return drawing


# ============================================================================
# PERIOD COMPARISON SECTION
# ============================================================================

# Import period comparison module
from period_comparison_calculator import (
    PeriodComparison,
    TrendDirection,
    calculate_week_over_week,
    calculate_month_over_month,
    get_dimensional_trends,
    format_percent_change,
    get_trend_indicator,
    get_comparison_summary,
)

# Colors for trend indicators
TREND_COLORS = {
    "up": colors.HexColor("#28a745"),      # Green for increase (positive)
    "down": colors.HexColor("#dc3545"),    # Red for decrease (negative)
    "flat": colors.HexColor("#6c757d"),    # Gray for flat/no change
    "no_data": colors.HexColor("#adb5bd"), # Light gray for no data
}


def get_trend_arrow_symbol(trend: str) -> str:
    """Get Unicode arrow symbol for trend direction.

    Args:
        trend: Trend direction string ("up", "down", "flat", "no_data")

    Returns:
        Unicode arrow character
    """
    arrows = {
        "up": "▲",      # Up triangle
        "down": "▼",    # Down triangle
        "flat": "●",    # Circle for flat
        "no_data": "○", # Empty circle for no data
    }
    return arrows.get(trend, "○")


def create_comparison_metrics_table(comparison: PeriodComparison, dimension: str = "group",
                                     title: str = "Period Comparison", width: float = 160) -> "Table":
    """Create a styled table displaying comparison metrics with color-coded trends.

    Args:
        comparison: PeriodComparison object containing comparison data
        dimension: Dimension to display ("group", "phase", "user", "marketplace")
        title: Title for the table
        width: Table width in mm

    Returns:
        A ReportLab Table object containing the comparison table
    """
    # Get dimensional trends (sorted by percent change)
    trends = get_dimensional_trends(comparison, dimension)

    if not trends:
        # Return empty message table if no data
        table_data = [["No comparison data available"]]
        table = Table(table_data, colWidths=[width*mm])
        table.setStyle(TableStyle([
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.grey),
        ]))
        return table

    # Build table data with headers
    table_data = [
        ["Item", "Current", "Previous", "Change", "% Change", "Trend"]
    ]

    # Row-level trend colors for styling
    row_trends = []

    for trend in trends[:10]:  # Limit to top 10 items
        # Format the data
        item_name = trend["key"]
        if dimension == "group" and item_name.startswith("BUNDLE_"):
            item_name = get_group_display_name(item_name, short=True)
        elif dimension == "phase" and item_name in PHASE_NAMES:
            item_name = PHASE_NAMES[item_name]

        current = str(trend["current"])
        previous = str(trend["previous"])
        abs_change = trend["absolute_change"]
        change_str = f"+{abs_change}" if abs_change > 0 else str(abs_change)
        pct_change = trend["formatted_change"]
        trend_arrow = get_trend_arrow_symbol(trend["trend"])

        table_data.append([
            item_name,
            current,
            previous,
            change_str,
            pct_change,
            trend_arrow,
        ])
        row_trends.append(trend["trend"])

    # Calculate column widths
    col_widths = [
        35*mm,   # Item
        20*mm,   # Current
        20*mm,   # Previous
        20*mm,   # Change
        22*mm,   # % Change
        13*mm,   # Trend
    ]

    # Create the table
    table = Table(table_data, colWidths=col_widths)

    # Build style list
    style_commands = [
        # Grid and borders
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BOX', (0, 0), (-1, -1), 1, colors.HexColor("#34495e")),

        # Header styling
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#34495e")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 9),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),

        # Data row styling
        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 1), (-1, -1), 8),
        ('ALIGN', (0, 1), (0, -1), 'LEFT'),    # Item names left aligned
        ('ALIGN', (1, 1), (4, -1), 'RIGHT'),   # Numbers right aligned
        ('ALIGN', (5, 1), (5, -1), 'CENTER'),  # Trend arrows centered
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),

        # Alternating row backgrounds
        ('ROWBACKGROUNDS', (0, 1), (-1, -1), [colors.white, colors.HexColor("#f8f9fa")]),
    ]

    # Add color coding for trend column and % change column
    for row_idx, trend in enumerate(row_trends, start=1):
        trend_color = TREND_COLORS.get(trend, TREND_COLORS["no_data"])
        # Color the trend arrow
        style_commands.append(('TEXTCOLOR', (5, row_idx), (5, row_idx), trend_color))
        # Color the % change text
        style_commands.append(('TEXTCOLOR', (4, row_idx), (4, row_idx), trend_color))
        # Bold positive/negative changes
        if trend in ("up", "down"):
            style_commands.append(('FONTNAME', (4, row_idx), (5, row_idx), 'Helvetica-Bold'))

    table.setStyle(TableStyle(style_commands))

    return table


def create_comparison_summary_card(comparison: PeriodComparison, comparison_type: str = "Week-over-Week",
                                    width: float = 160, height: float = 80) -> "Drawing":
    """Create a summary card showing overall comparison metrics.

    Args:
        comparison: PeriodComparison object
        comparison_type: "Week-over-Week" or "Month-over-Month"
        width: Card width in mm
        height: Card height

    Returns:
        A ReportLab Drawing object
    """
    drawing = Drawing(width*mm, height)

    # Background card
    card_width = width*mm - 10
    card_height = height - 10
    card_x = 5
    card_y = 5

    # Draw card background
    card = Rect(card_x, card_y, card_width, card_height,
                fillColor=colors.HexColor("#ffffff"),
                strokeColor=colors.HexColor("#dee2e6"),
                strokeWidth=1)
    drawing.add(card)

    # Title bar
    title_height = 20
    title_bar = Rect(card_x, card_y + card_height - title_height, card_width, title_height,
                     fillColor=colors.HexColor("#34495e"),
                     strokeColor=None)
    drawing.add(title_bar)

    # Title text
    drawing.add(String(card_x + card_width/2, card_y + card_height - 14,
                      f"{comparison_type} Comparison",
                      fontName='Helvetica-Bold', fontSize=11, textAnchor='middle',
                      fillColor=colors.white))

    # Get trend color and arrow
    trend_str = comparison.trend.value if hasattr(comparison.trend, 'value') else str(comparison.trend)
    trend_color = TREND_COLORS.get(trend_str, TREND_COLORS["no_data"])
    trend_arrow = get_trend_arrow_symbol(trend_str)

    # Current period info
    current = comparison.current_period
    previous = comparison.previous_period

    # Layout for metrics - three columns
    col_width = card_width / 3
    y_center = card_y + (card_height - title_height) / 2

    # Current period
    drawing.add(String(card_x + col_width/2, y_center + 15,
                      "Current Period",
                      fontName='Helvetica', fontSize=8, textAnchor='middle',
                      fillColor=colors.HexColor("#6c757d")))
    drawing.add(String(card_x + col_width/2, y_center - 5,
                      str(current.total_changes),
                      fontName='Helvetica-Bold', fontSize=18, textAnchor='middle',
                      fillColor=colors.HexColor("#2c3e50")))
    drawing.add(String(card_x + col_width/2, y_center - 20,
                      f"({current.start_date.strftime('%m/%d')} - {current.end_date.strftime('%m/%d')})",
                      fontName='Helvetica', fontSize=7, textAnchor='middle',
                      fillColor=colors.HexColor("#adb5bd")))

    # Change indicator (middle column)
    pct_change_str = format_percent_change(comparison.percent_change)
    drawing.add(String(card_x + col_width + col_width/2, y_center + 15,
                      "Change",
                      fontName='Helvetica', fontSize=8, textAnchor='middle',
                      fillColor=colors.HexColor("#6c757d")))
    drawing.add(String(card_x + col_width + col_width/2, y_center - 5,
                      f"{trend_arrow} {pct_change_str}",
                      fontName='Helvetica-Bold', fontSize=16, textAnchor='middle',
                      fillColor=trend_color))
    drawing.add(String(card_x + col_width + col_width/2, y_center - 20,
                      f"({comparison.absolute_change:+d} changes)",
                      fontName='Helvetica', fontSize=7, textAnchor='middle',
                      fillColor=colors.HexColor("#adb5bd")))

    # Previous period
    drawing.add(String(card_x + 2*col_width + col_width/2, y_center + 15,
                      "Previous Period",
                      fontName='Helvetica', fontSize=8, textAnchor='middle',
                      fillColor=colors.HexColor("#6c757d")))
    drawing.add(String(card_x + 2*col_width + col_width/2, y_center - 5,
                      str(previous.total_changes),
                      fontName='Helvetica-Bold', fontSize=18, textAnchor='middle',
                      fillColor=colors.HexColor("#2c3e50")))
    drawing.add(String(card_x + 2*col_width + col_width/2, y_center - 20,
                      f"({previous.start_date.strftime('%m/%d')} - {previous.end_date.strftime('%m/%d')})",
                      fontName='Helvetica', fontSize=7, textAnchor='middle',
                      fillColor=colors.HexColor("#adb5bd")))

    return drawing


def add_period_comparison_section(story: List["Flowable"], report_type: str = "Weekly",
                                   start_date: Optional[date] = None, end_date: Optional[date] = None) -> None:
    """Add the period comparison section to the report story.

    This creates a section showing:
    - Overall comparison summary (current vs previous period)
    - Group-level comparison table with trends
    - Phase-level comparison table with trends

    Args:
        story: The reportlab story list to append elements to
        report_type: "Weekly" or "Monthly"
        start_date: Start date of current period (optional)
        end_date: End date of current period (optional)

    Returns:
        None (modifies story in place)
    """
    # Add section marker for page header navigation
    story.append(SectionMarker("Period Comparison"))

    styles = getSampleStyleSheet()

    # Section header style
    section_header_style = ParagraphStyle(
        'ComparisonSectionHeader',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor("#34495e"),
        spaceBefore=5*mm,
        spaceAfter=3*mm
    )

    subsection_style = ParagraphStyle(
        'ComparisonSubsection',
        parent=styles['Heading3'],
        fontSize=11,
        textColor=colors.HexColor("#495057"),
        spaceBefore=3*mm,
        spaceAfter=2*mm
    )

    # Calculate comparison based on report type
    try:
        if report_type == "Monthly":
            comparison = calculate_month_over_month(reference_date=end_date)
            comparison_type = "Month-over-Month"
        else:
            comparison = calculate_week_over_week(reference_date=end_date)
            comparison_type = "Week-over-Week"

        # Section title
        story.append(Paragraph(f"Period Comparison ({comparison_type})", section_header_style))

        # Overall summary card
        summary_card = create_comparison_summary_card(comparison, comparison_type)
        story.append(summary_card)
        story.append(Spacer(1, 5*mm))

        # Group comparison table
        story.append(Paragraph("Comparison by Group", subsection_style))
        group_table = create_comparison_metrics_table(comparison, dimension="group",
                                                       title="Group Comparison")
        story.append(group_table)
        story.append(Spacer(1, 5*mm))

        # Phase comparison table
        story.append(Paragraph("Comparison by Phase", subsection_style))
        phase_table = create_comparison_metrics_table(comparison, dimension="phase",
                                                       title="Phase Comparison")
        story.append(phase_table)
        story.append(Spacer(1, 5*mm))

        # Top user changes (if there are users)
        if comparison.by_user:
            story.append(Paragraph("Top User Activity Changes", subsection_style))
            user_table = create_comparison_metrics_table(comparison, dimension="user",
                                                          title="User Comparison")
            story.append(user_table)
            story.append(Spacer(1, 5*mm))

        logger.info(f"Period comparison section added: {comparison_type}")

    except Exception as e:
        logger.warning(f"Could not generate period comparison section: {e}")
        # Add a note that comparison data is not available
        note_style = ParagraphStyle(
            'ComparisonNote',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.HexColor("#6c757d"),
            fontName='Helvetica-Oblique'
        )
        story.append(Paragraph("Period Comparison", section_header_style))
        story.append(Paragraph(
            f"<i>Period comparison data is not available. This may be due to insufficient historical data.</i>",
            note_style
        ))
        story.append(Spacer(1, 5*mm))


def add_executive_summary_section(story: List["Flowable"], metrics: Dict[str, Any], period_str: str, report_type: str = "Weekly",
                                   start_date: Optional[date] = None, end_date: Optional[date] = None, generation_timestamp: Optional[str] = None) -> None:
    """Add the executive summary section to the report story.

    This creates a clean, executive-friendly first page with:
    - Overall activity summary (total changes, active users, date range)
    - Key metrics callout boxes (most active group, user, special hours)
    - Overall metrics in KPI cards
    - Key highlights section
    - Group status grid
    - Report generation timestamp for version tracking

    Args:
        story: The reportlab story list to append elements to
        metrics: Dictionary containing all report metrics
        period_str: String describing the report period
        report_type: "Weekly" or "Monthly"
        start_date: Start date for fetching special activities (optional)
        end_date: End date for fetching special activities (optional)
        generation_timestamp: Formatted timestamp string showing when report was generated

    Returns:
        None (modifies story in place)
    """
    # Add section marker for page header navigation
    story.append(SectionMarker("Executive Summary"))

    styles = getSampleStyleSheet()

    # Fetch special activities hours if dates are provided
    special_activity_hours = 0
    if start_date and end_date:
        try:
            _, _, special_activity_hours = get_special_activities(start_date, end_date)
        except Exception as e:
            logger.warning(f"Could not fetch special activities for executive summary: {e}")
            special_activity_hours = 0

    # Executive Summary title with professional styling
    exec_title_style = ParagraphStyle(
        'ExecutiveTitle',
        parent=styles['Title'],
        fontSize=24,
        spaceAfter=5*mm,
        textColor=colors.HexColor("#2c3e50")
    )

    exec_subtitle_style = ParagraphStyle(
        'ExecutiveSubtitle',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor("#7f8c8d"),
        alignment=1  # Center alignment
    )

    # Style for generation timestamp
    timestamp_style = ParagraphStyle(
        'GenerationTimestamp',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor("#95a5a6"),
        alignment=1  # Center alignment
    )

    # Title
    story.append(Paragraph(f"{report_type} Executive Summary", exec_title_style))
    story.append(Paragraph(f"Smartsheet Activity Report | {period_str}", exec_subtitle_style))

    # Add generation timestamp on cover page
    if generation_timestamp:
        story.append(Spacer(1, 2*mm))
        story.append(Paragraph(f"Report generated: {generation_timestamp}", timestamp_style))

    story.append(Spacer(1, 6*mm))

    # Section header style
    section_header_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading2'],
        fontSize=12,
        textColor=colors.HexColor("#34495e"),
        spaceBefore=3*mm,
        spaceAfter=3*mm
    )

    # Overall Activity Summary Section (prominent display of key totals)
    story.append(Paragraph("Activity Summary", section_header_style))
    overall_summary = create_overall_activity_summary(
        metrics, period_str, width=160*mm, height=120
    )
    story.append(overall_summary)
    story.append(Spacer(1, 6*mm))

    # Key Metrics Callout Section (attention-grabbing)
    story.append(Paragraph("Key Metrics at a Glance", section_header_style))
    key_metrics_callouts = create_key_metrics_callout_section(
        metrics, special_activity_hours, width=160*mm, height=100
    )
    story.append(key_metrics_callouts)
    story.append(Spacer(1, 6*mm))

    # Overall Metrics Section
    story.append(Paragraph("Overall Metrics", section_header_style))
    metrics_row = create_executive_summary_metrics_row(metrics, width=160*mm)
    story.append(metrics_row)
    story.append(Spacer(1, 6*mm))

    # Key Highlights Section
    story.append(Paragraph("Key Highlights", section_header_style))
    highlights = create_executive_highlights_section(metrics, width=160*mm, height=100)
    story.append(highlights)
    story.append(Spacer(1, 6*mm))

    # Calculate health scores for all groups (used in status grid and health section)
    health_scores = None
    try:
        # Prepare data for health score calculation
        from historical_data_loader import load_change_history

        # Load changes for health calculation (last 60 days for trend comparison)
        health_lookback_start = (end_date or date.today()) - timedelta(days=60) if end_date else date.today() - timedelta(days=60)
        health_reference_date = end_date or date.today()

        try:
            health_changes = load_change_history(
                start_date=health_lookback_start,
                end_date=health_reference_date
            )
        except FileNotFoundError:
            # Use existing metrics if no historical data
            health_changes = []
            logger.warning("Could not load historical data for health scores, using current metrics")

        # Estimate completion and overdue counts from metrics
        # In a real implementation, these would come from actual sheet data
        completed_by_group = {}
        overdue_by_group = {}

        for group in TOTAL_PRODUCTS.keys():
            total = TOTAL_PRODUCTS.get(group, 0)
            changes = metrics.get("groups", {}).get(group, 0)

            # Estimate completion based on activity (higher activity = higher completion estimate)
            # This is a rough estimate - actual values would come from sheet data
            if total > 0:
                activity_rate = changes / total if total > 0 else 0
                # Assume 60-90% completion range based on activity
                completion_estimate = int(total * min(0.90, 0.60 + activity_rate * 0.3))
                completed_by_group[group] = completion_estimate

                # Estimate overdue as inverse of activity (less activity = more likely overdue)
                overdue_estimate = int(total * max(0, 0.15 - activity_rate * 0.1))
                overdue_by_group[group] = overdue_estimate

        # Calculate health scores
        health_scores = calculate_group_health_scores(
            changes=health_changes,
            total_products=TOTAL_PRODUCTS,
            completed_by_group=completed_by_group,
            overdue_by_group=overdue_by_group,
            reference_date=health_reference_date
        )
    except Exception as e:
        logger.warning(f"Could not calculate health scores: {e}")
        health_scores = None

    # Group Status Grid with Health Indicators
    story.append(Paragraph("Group Status Overview", section_header_style))
    group_grid = create_group_status_grid(metrics, width=160*mm, height=240, health_scores=health_scores)
    story.append(group_grid)

    # Add a page break after executive summary
    story.append(PageBreak())

    # Group Health Scores Section (new page)
    story.append(Paragraph("Group Health Scores", exec_title_style))
    story.append(Paragraph("Health assessment based on activity, completion, and overdue metrics", exec_subtitle_style))
    story.append(Spacer(1, 6*mm))

    # Use already calculated health scores
    if health_scores:
        # Add health summary callout
        health_summary = create_health_summary_callout(health_scores, width=160*mm, height=60)
        story.append(health_summary)
        story.append(Spacer(1, 6*mm))

        # Add health score grid
        story.append(Paragraph("Group Health Details", section_header_style))
        health_grid = create_health_score_grid(health_scores, width=160*mm, height=280)
        story.append(health_grid)
    else:
        story.append(Paragraph("Health score data unavailable", styles['Normal']))

    # Add page break before main content
    story.append(PageBreak())


# ============================================================================
# ERROR AND WARNING REPORT SECTION
# ============================================================================

def generate_sample_errors() -> List[Dict[str, Any]]:
    """Generate sample error/warning data for demonstration purposes."""
    from datetime import datetime

    sample_errors = [
        {
            "severity": "error",
            "category": "data_quality",
            "sheet_name": "NA Products",
            "row_id": "1234567",
            "field_name": "Price",
            "message": "Invalid price format: contains non-numeric characters",
            "suggested_action": "Review and correct the price value to contain only numbers",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
        {
            "severity": "error",
            "category": "missing_data",
            "sheet_name": "NF Products",
            "row_id": "2345678",
            "field_name": "SKU",
            "message": "Required SKU field is empty",
            "suggested_action": "Add the missing SKU value for this product",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
        {
            "severity": "warning",
            "category": "invalid_format",
            "sheet_name": "NH Products",
            "row_id": "3456789",
            "field_name": "Date",
            "message": "Date format does not match expected pattern (YYYY-MM-DD)",
            "suggested_action": "Update date to use YYYY-MM-DD format",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
        {
            "severity": "warning",
            "category": "data_quality",
            "sheet_name": "NM Products",
            "row_id": "4567890",
            "field_name": "Description",
            "message": "Description exceeds recommended length (500+ characters)",
            "suggested_action": "Consider shortening the description for better readability",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
        {
            "severity": "info",
            "category": "other",
            "sheet_name": "NT Products",
            "row_id": "5678901",
            "field_name": "Status",
            "message": "Product status unchanged for 30+ days",
            "suggested_action": "Review if this product needs status update",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
        {
            "severity": "error",
            "category": "api_error",
            "sheet_name": "NV Products",
            "row_id": "N/A",
            "field_name": "N/A",
            "message": "API rate limit exceeded during data fetch",
            "suggested_action": "Retry operation after rate limit reset",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
        {
            "severity": "warning",
            "category": "missing_data",
            "sheet_name": "NA Products",
            "row_id": "6789012",
            "field_name": "Category",
            "message": "Product category not assigned",
            "suggested_action": "Assign appropriate category to this product",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
        {
            "severity": "info",
            "category": "permission",
            "sheet_name": "NP Products",
            "row_id": "7890123",
            "field_name": "N/A",
            "message": "User access level changed for this sheet",
            "suggested_action": "Verify user permissions are appropriate",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
    ]

    return sample_errors


def create_error_summary_table(errors: List[Dict[str, Any]]) -> "Table":
    """Create a summary table showing counts by severity and category."""
    styles = getSampleStyleSheet()

    # Count by severity
    severity_counts = {"error": 0, "warning": 0, "info": 0}
    category_counts = {}

    for error in errors:
        severity = error.get("severity", "info")
        category = error.get("category", "other")

        severity_counts[severity] = severity_counts.get(severity, 0) + 1
        category_counts[category] = category_counts.get(category, 0) + 1

    # Create summary data
    summary_data = [
        [
            Paragraph("<b>Severity</b>", styles['Normal']),
            Paragraph("<b>Count</b>", styles['Normal']),
            Paragraph("<b>Category</b>", styles['Normal']),
            Paragraph("<b>Count</b>", styles['Normal']),
        ]
    ]

    # Get sorted lists
    severity_list = sorted(severity_counts.items(), key=lambda x: ["error", "warning", "info"].index(x[0]))
    category_list = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)

    # Fill in rows (max of severity or category count)
    max_rows = max(len(severity_list), len(category_list))

    for i in range(max_rows):
        row = []

        # Severity column
        if i < len(severity_list):
            sev, count = severity_list[i]
            sev_color = SEVERITY_COLORS.get(sev, colors.grey)
            # Create colored indicator
            sev_label = SEVERITY_LABELS.get(sev, sev.title())
            row.append(Paragraph(f'<font color="{sev_color.hexval()}">\u25cf</font> {sev_label}', styles['Normal']))
            row.append(str(count))
        else:
            row.extend(["", ""])

        # Category column
        if i < len(category_list):
            cat, count = category_list[i]
            cat_color = ERROR_CATEGORY_COLORS.get(cat, colors.grey)
            cat_label = ERROR_CATEGORY_LABELS.get(cat, cat.replace("_", " ").title())
            row.append(Paragraph(f'<font color="{cat_color.hexval()}">\u25cf</font> {cat_label}', styles['Normal']))
            row.append(str(count))
        else:
            row.extend(["", ""])

        summary_data.append(row)

    # Create table
    table = Table(summary_data, colWidths=[45*mm, 20*mm, 50*mm, 20*mm])
    table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#E8E8E8")),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('ALIGN', (3, 0), (3, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))

    return table


def create_severity_indicator(severity: str, width: int = 12, height: int = 12) -> "Drawing":
    """Create a small colored box indicating severity level."""
    drawing = Drawing(width, height)
    color = SEVERITY_COLORS.get(severity, colors.grey)
    drawing.add(Rect(0, 0, width - 2, height - 2, fillColor=color, strokeColor=None))
    return drawing


def create_error_details_table(errors: List[Dict[str, Any]], max_rows: int = 15, enable_pagination: bool = True) -> List["Flowable"]:
    """Create a detailed table showing individual errors and warnings.

    This function now supports enhanced pagination to avoid orphaned rows
    and ensure minimum rows per page when the table spans multiple pages.

    Args:
        errors: List of error dictionaries
        max_rows: Maximum number of rows to display (default: 15)
        enable_pagination: Enable enhanced pagination features (default: True)

    Returns:
        ReportLab Table element with pagination support
    """
    styles = getSampleStyleSheet()

    if not errors:
        # Return empty state message
        return Paragraph("<i>No errors or warnings to report.</i>", styles['Normal'])

    # Sort errors: errors first, then warnings, then info
    severity_order = {"error": 0, "warning": 1, "info": 2}
    sorted_errors = sorted(errors, key=lambda x: severity_order.get(x.get("severity", "info"), 3))

    # Limit to max_rows if needed
    display_errors = sorted_errors[:max_rows]
    truncated = len(sorted_errors) > max_rows

    # Create header row
    table_data = [[
        Paragraph("<b>Sev.</b>", styles['Normal']),
        Paragraph("<b>Sheet</b>", styles['Normal']),
        Paragraph("<b>Row ID</b>", styles['Normal']),
        Paragraph("<b>Field</b>", styles['Normal']),
        Paragraph("<b>Issue</b>", styles['Normal']),
        Paragraph("<b>Action</b>", styles['Normal']),
    ]]

    # Add data rows
    for error in display_errors:
        severity = error.get("severity", "info")
        sev_color = SEVERITY_COLORS.get(severity, colors.grey)
        sev_label = severity[0].upper()  # E, W, or I

        # Truncate long text
        # Use unicode-safe truncation for all text fields
        message = safe_truncate(prepare_for_pdf(error.get("message", "")), 60, "...")
        action = safe_truncate(prepare_for_pdf(error.get("suggested_action", "")), 50, "...")
        sheet_name = safe_truncate(prepare_for_pdf(error.get("sheet_name", "N/A")), 12)
        row_id = safe_truncate(prepare_for_pdf(str(error.get("row_id", "N/A"))), 10)
        field_name = safe_truncate(prepare_for_pdf(error.get("field_name", "N/A")), 12)

        row = [
            Paragraph(f'<font color="{sev_color.hexval()}"><b>{sev_label}</b></font>', styles['Normal']),
            Paragraph(sheet_name, styles['Normal']),
            Paragraph(row_id, styles['Normal']),
            Paragraph(field_name, styles['Normal']),
            Paragraph(message, styles['Normal']),
            Paragraph(action, styles['Normal']),
        ]
        table_data.append(row)

    # Add truncation notice if needed
    if truncated:
        remaining = len(sorted_errors) - max_rows
        table_data.append([
            "",
            Paragraph(f"<i>... and {remaining} more items</i>", styles['Normal']),
            "", "", "", ""
        ])

    # Style the table
    style_commands = [
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#E8E8E8")),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7),
        ('ALIGN', (0, 0), (0, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 3),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3),
    ]

    # Add alternating row colors for readability
    for i in range(1, len(table_data)):
        if i % 2 == 0:
            style_commands.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor("#F8F8F8")))

        # Color code based on severity
        if i <= len(display_errors):
            error = display_errors[i - 1]
            severity = error.get("severity", "info")
            if severity == "error":
                style_commands.append(('BACKGROUND', (0, i), (0, i), colors.HexColor("#FFEBEE")))
            elif severity == "warning":
                style_commands.append(('BACKGROUND', (0, i), (0, i), colors.HexColor("#FFF8E1")))

    col_widths = [12*mm, 25*mm, 20*mm, 22*mm, 45*mm, 36*mm]

    # Use landscape-aware table creation for automatic landscape handling
    # This table has 6 columns which meets the threshold for landscape mode
    if enable_pagination:
        pagination_config = PaginationConfig(
            repeat_header_rows=1,
            min_rows_per_page=3,
            avoid_orphans=True,
            orphan_threshold=2
        )
        # Use landscape-aware table which will auto-detect based on columns
        flowables = create_landscape_aware_table(
            data=table_data,
            col_widths=col_widths,
            style_commands=style_commands,
            section_name="Error Details",
            pagination_config=pagination_config,
            return_flowables=True
        )
        # Return the flowables list (includes page breaks for landscape if needed)
        return flowables
    else:
        # Fallback to standard table
        table = Table(table_data, colWidths=col_widths)
        table.setStyle(TableStyle(style_commands))

    return table


def create_error_legend() -> "Table":
    """Create a legend explaining severity levels."""
    styles = getSampleStyleSheet()

    legend_data = [[
        Paragraph('<font color="#DC3545">\u25cf</font> <b>E</b> = Error (Critical)', styles['Normal']),
        Paragraph('<font color="#FFC107">\u25cf</font> <b>W</b> = Warning (Attention needed)', styles['Normal']),
        Paragraph('<font color="#17A2B8">\u25cf</font> <b>I</b> = Info (For review)', styles['Normal']),
    ]]

    table = Table(legend_data, colWidths=[50*mm, 55*mm, 50*mm])
    table.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ]))

    return table


def create_data_quality_issues_table(issues: List[Dict[str, Any]], max_rows: int = 20, enable_pagination: bool = True) -> Union["Table", "Paragraph", List["Flowable"]]:
    """Create a focused table showing data quality issues with row identifiers, field names, and descriptions.

    This function creates a dedicated table for displaying data quality problems
    encountered during processing, making it easy to identify and fix specific issues.
    Now supports enhanced pagination to avoid orphaned rows and ensure minimum rows per page.

    Args:
        issues: List of data quality issue dictionaries with keys:
            - row_id: Row identifier where the issue was found
            - field_name: Name of the field with the issue
            - issue_description: Description of the data quality problem
            - sheet_name: Name of the sheet (optional)
            - severity: Severity level (error, warning, info)
            - suggested_action: Suggested fix (optional)
        max_rows: Maximum number of issues to display
        enable_pagination: Enable enhanced pagination features (default: True)

    Returns:
        ReportLab Table element or Paragraph if no issues
    """
    styles = getSampleStyleSheet()

    if not issues:
        return Paragraph("<i>No data quality issues found.</i>", styles['Normal'])

    # Filter for data quality specific issues
    data_quality_issues = [
        issue for issue in issues
        if issue.get("category") in ("data_quality", "missing_data", "invalid_format", "validation")
        or issue.get("category", "").startswith("data")
    ]

    if not data_quality_issues:
        return Paragraph(
            '<font color="#28A745">\u2714</font> <i>No data quality issues detected in the processed data.</i>',
            styles['Normal']
        )

    # Sort by severity (errors first) then by sheet name
    severity_order = {"error": 0, "warning": 1, "info": 2}
    sorted_issues = sorted(
        data_quality_issues,
        key=lambda x: (
            severity_order.get(x.get("severity", "info"), 3),
            x.get("sheet_name", ""),
            x.get("row_id", "")
        )
    )

    # Limit to max_rows
    display_issues = sorted_issues[:max_rows]
    truncated = len(sorted_issues) > max_rows

    # Create header row - focused on data quality specifics
    table_data = [[
        Paragraph("<b>Row ID</b>", styles['Normal']),
        Paragraph("<b>Field Name</b>", styles['Normal']),
        Paragraph("<b>Issue Description</b>", styles['Normal']),
        Paragraph("<b>Sheet</b>", styles['Normal']),
        Paragraph("<b>Sev.</b>", styles['Normal']),
    ]]

    for issue in display_issues:
        severity = issue.get("severity", "info")
        sev_color = SEVERITY_COLORS.get(severity, colors.grey)
        sev_label = severity[0].upper()  # E, W, or I

        # Get row ID - could be in different fields
        row_id = str(issue.get("row_id", issue.get("context", {}).get("row_id", "N/A")))[:12]

        # Get field name
        field_name = issue.get("field_name", issue.get("context", {}).get("field_name", "N/A"))[:15]

        # Get issue description - truncate if too long
        description = issue.get("message", issue.get("issue_description", ""))[:70]
        if len(issue.get("message", issue.get("issue_description", ""))) > 70:
            description += "..."

        # Get sheet name
        sheet_name = issue.get("sheet_name", issue.get("context", {}).get("sheet_name", ""))[:12]

        row = [
            Paragraph(row_id, styles['Normal']),
            Paragraph(field_name, styles['Normal']),
            Paragraph(description, styles['Normal']),
            Paragraph(sheet_name, styles['Normal']),
            Paragraph(f'<font color="{sev_color.hexval()}"><b>{sev_label}</b></font>', styles['Normal']),
        ]
        table_data.append(row)

    # Add truncation notice if needed
    if truncated:
        remaining = len(sorted_issues) - max_rows
        table_data.append([
            "",
            "",
            Paragraph(f"<i>... and {remaining} more data quality issues</i>", styles['Normal']),
            "",
            ""
        ])

    # Column widths: Row ID, Field Name, Description, Sheet, Severity
    col_widths = [25*mm, 30*mm, 65*mm, 25*mm, 15*mm]

    # Style the table
    style_commands = [
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#B71C1C")),  # Dark red for data quality header
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (0, 0), (0, -1), 'LEFT'),
        ('ALIGN', (4, 0), (4, -1), 'CENTER'),  # Severity column centered
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ('RIGHTPADDING', (0, 0), (-1, -1), 4),
    ]

    # Add alternating row colors and severity highlighting
    for i in range(1, len(table_data)):
        if i % 2 == 0:
            style_commands.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor("#FFF8F8")))  # Light pink

        # Highlight severity column based on level
        if i <= len(display_issues):
            issue = display_issues[i - 1]
            severity = issue.get("severity", "info")
            if severity == "error":
                style_commands.append(('BACKGROUND', (4, i), (4, i), colors.HexColor("#FFEBEE")))
            elif severity == "warning":
                style_commands.append(('BACKGROUND', (4, i), (4, i), colors.HexColor("#FFF8E1")))

    # Use paginated table for enhanced pagination support
    if enable_pagination:
        pagination_config = PaginationConfig(
            repeat_header_rows=1,
            min_rows_per_page=3,
            avoid_orphans=True,
            orphan_threshold=2,
            group_by_column=3  # Group by sheet name column
        )
        table = create_paginated_table(
            data=table_data,
            col_widths=col_widths,
            style_commands=style_commands,
            config=pagination_config
        )
    else:
        # Fallback to standard table
        table = Table(table_data, colWidths=col_widths)
        table.setStyle(TableStyle(style_commands))

    return table


def get_data_quality_issues_from_errors(errors: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Extract and format data quality issues from a list of errors.

    This helper function filters errors to find data quality specific issues
    and formats them for display in the data quality issues table.

    Args:
        errors: List of error dictionaries from generate_sample_errors() or real collection

    Returns:
        List of data quality issue dictionaries
    """
    if not errors:
        return []

    data_quality_categories = {
        "data_quality", "missing_data", "invalid_format", "validation",
        "null_value", "data_corruption", "duplicate_data"
    }

    data_quality_issues = []
    for error in errors:
        category = error.get("category", "").lower()

        # Check if this is a data quality related issue
        if category in data_quality_categories or category.startswith("data"):
            # Format for the data quality issues table
            issue = {
                "row_id": error.get("row_id", error.get("context", {}).get("row_id", "N/A")),
                "field_name": error.get("field_name", error.get("context", {}).get("field_name", "N/A")),
                "message": error.get("message", ""),
                "issue_description": error.get("message", ""),
                "sheet_name": error.get("sheet_name", error.get("context", {}).get("sheet_name", "")),
                "severity": error.get("severity", "info"),
                "category": category,
                "suggested_action": error.get("suggested_action", ""),
            }
            data_quality_issues.append(issue)

    return data_quality_issues


def add_data_quality_issues_section(story: List["Flowable"], errors: Optional[List[Dict[str, Any]]] = None) -> None:
    """Add a dedicated data quality issues section to the PDF story.

    This section focuses specifically on data quality problems, displaying:
    - Row identifiers where issues were found
    - Field names with problems
    - Detailed issue descriptions
    - Severity indicators

    Args:
        story: The ReportLab story list to append content to
        errors: List of error/warning dictionaries. If provided, data quality
               issues will be extracted from this list.
    """
    # Add section marker for page header navigation
    story.append(SectionMarker("Data Quality Issues"))

    styles = getSampleStyleSheet()

    # Create section subheading style
    subheading_style = ParagraphStyle(
        'DataQualitySubheading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor("#B71C1C"),  # Dark red for data quality emphasis
        spaceBefore=6*mm,
        spaceAfter=4*mm
    )

    normal_style = styles['Normal']

    # Add section header
    story.append(Paragraph("Data Quality Issues", subheading_style))

    # Add description
    story.append(Paragraph(
        "The following table shows data quality problems encountered during processing. "
        "Each issue includes the row identifier, field name, and a description of the problem.",
        normal_style
    ))
    story.append(Spacer(1, 4*mm))

    # Extract data quality issues from errors
    data_quality_issues = get_data_quality_issues_from_errors(errors) if errors else []

    # Create and add the data quality issues table
    issues_table = create_data_quality_issues_table(data_quality_issues)
    story.append(issues_table)
    story.append(Spacer(1, 4*mm))

    # Add summary count if there are issues
    if data_quality_issues:
        error_count = sum(1 for i in data_quality_issues if i.get("severity") == "error")
        warning_count = sum(1 for i in data_quality_issues if i.get("severity") == "warning")
        info_count = sum(1 for i in data_quality_issues if i.get("severity") == "info")

        summary_text = f"<b>Summary:</b> {error_count} error(s), {warning_count} warning(s), {info_count} info item(s) related to data quality."
        story.append(Paragraph(summary_text, normal_style))


# =============================================================================
# SKIPPED ROWS SECTION
# =============================================================================

# Mapping of skip reason codes to human-readable descriptions
SKIP_REASON_DESCRIPTIONS = {
    "missing_row_id": "Row has no ID - cannot track changes without row identifier",
    "missing_group": "Group code is missing or empty",
    "empty_date_value": "Date field is empty - no change to track",
    "null_like_date_value": "Date contains null-like value (null, none, n/a, -)",
    "missing_timestamp": "Required timestamp field is missing",
    "missing_phase": "Phase information is missing",
    "missing_datefield": "Date field name is missing",
    "missing_date": "Date value is missing from required field",
    "invalid_date_format": "Date format is invalid or unrecognizable",
    "row_processing_error": "Error occurred during row processing",
    "phase_processing_error": "Error occurred during phase processing",
    "cell_access_error": "Failed to access cell value",
    "validation_failed": "Row failed validation checks",
}

# Skip reason severity levels for display ordering
SKIP_REASON_SEVERITY = {
    "missing_row_id": "error",
    "missing_group": "error",
    "row_processing_error": "error",
    "phase_processing_error": "error",
    "cell_access_error": "error",
    "validation_failed": "error",
    "invalid_date_format": "warning",
    "null_like_date_value": "warning",
    "empty_date_value": "info",
    "missing_timestamp": "warning",
    "missing_phase": "warning",
    "missing_datefield": "warning",
    "missing_date": "warning",
}


def generate_sample_skipped_rows() -> List[Dict[str, Any]]:
    """Generate sample skipped rows data for demonstration.

    Returns:
        List of skipped row dictionaries with sheet_name, row_id, skip_reason, and details.
    """
    sample_data = [
        {
            "sheet_name": "NA",
            "row_id": "1234567890",
            "skip_reason": "missing_row_id",
            "details": "Row ID field is null",
            "phase": None,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
        {
            "sheet_name": "NF",
            "row_id": "2345678901",
            "skip_reason": "empty_date_value",
            "details": "Phase 2 date field is empty",
            "phase": "2",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
        {
            "sheet_name": "NH",
            "row_id": "3456789012",
            "skip_reason": "null_like_date_value",
            "details": "Date value contains 'N/A'",
            "phase": "3",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
        {
            "sheet_name": "NP",
            "row_id": "4567890123",
            "skip_reason": "invalid_date_format",
            "details": "Date '32/13/2024' is not valid",
            "phase": "1",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
        {
            "sheet_name": "BUNDLE_FAN",
            "row_id": "5678901234",
            "skip_reason": "cell_access_error",
            "details": "Could not read cell value at column 'Kontrolle'",
            "phase": "1",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
        {
            "sheet_name": "NA",
            "row_id": "6789012345",
            "skip_reason": "missing_group",
            "details": "Group code field is empty",
            "phase": None,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
        {
            "sheet_name": "BUNDLE_COOLER",
            "row_id": "7890123456",
            "skip_reason": "row_processing_error",
            "details": "Unexpected error: KeyError 'status'",
            "phase": "4",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
        {
            "sheet_name": "NT",
            "row_id": "8901234567",
            "skip_reason": "empty_date_value",
            "details": "Phase 5 date field is empty",
            "phase": "5",
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        },
    ]
    return sample_data


def get_skipped_rows_from_validation_stats(validation_stats: Any) -> List[Dict[str, Any]]:
    """Extract skipped rows data from ValidationStats.

    Args:
        validation_stats: ValidationStats object from validation module

    Returns:
        List of skipped row dictionaries formatted for display
    """
    if not validation_stats:
        return []

    skipped_rows = []

    # Get detailed error info from error_details
    for detail in getattr(validation_stats, 'error_details', []):
        skipped_row = {
            "sheet_name": detail.get("group", "N/A"),
            "row_id": str(detail.get("row_id", "N/A")),
            "skip_reason": detail.get("category", "unknown"),
            "details": detail.get("error_message", ""),
            "phase": detail.get("context", {}).get("phase"),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
        }
        skipped_rows.append(skipped_row)

    # Also process failed_row_ids that might not be in error_details
    for row_identifier in getattr(validation_stats, 'failed_row_ids', []):
        # Check if this row is already in skipped_rows
        if not any(sr.get("row_id") in row_identifier for sr in skipped_rows):
            parts = row_identifier.split(":", 1)
            group = parts[0] if len(parts) > 1 else "N/A"
            row_id = parts[1] if len(parts) > 1 else parts[0]

            skipped_row = {
                "sheet_name": group,
                "row_id": row_id,
                "skip_reason": "validation_failed",
                "details": "Row failed validation",
                "phase": None,
                "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
            }
            skipped_rows.append(skipped_row)

    return skipped_rows


def get_skipped_rows_from_error_collector() -> List[Dict[str, Any]]:
    """Extract skipped rows data from the global error collector.

    Returns:
        List of skipped row dictionaries formatted for display
    """
    collector = get_global_collector()

    # Get errors that indicate skipped rows
    skipped_categories = [
        "row_processing", "phase_processing", "cell_access",
        "missing_required", "null_value", "validation"
    ]

    skipped_rows = []

    for error in collector.get_all_errors():
        category_value = error.category.value if hasattr(error.category, 'value') else str(error.category)

        # Check if this error category indicates a skipped row
        if category_value in skipped_categories or "skip" in error.message.lower():
            skipped_row = {
                "sheet_name": error.context.get("group", error.context.get("sheet_name", "N/A")),
                "row_id": str(error.context.get("row_id", "N/A")),
                "skip_reason": category_value,
                "details": error.message,
                "phase": error.context.get("phase"),
                "timestamp": error.timestamp.strftime("%Y-%m-%d %H:%M") if error.timestamp else "",
            }
            skipped_rows.append(skipped_row)

    return skipped_rows


def create_skipped_rows_summary_table(skipped_rows: List[Dict[str, Any]]) -> Union["Table", "Paragraph"]:
    """Create a summary table showing counts of skipped rows by reason.

    Args:
        skipped_rows: List of skipped row dictionaries

    Returns:
        ReportLab Table element
    """
    styles = getSampleStyleSheet()

    if not skipped_rows:
        return Paragraph("<i>No rows were skipped during processing.</i>", styles['Normal'])

    # Count by skip reason
    reason_counts = {}
    sheet_counts = {}

    for row in skipped_rows:
        reason = row.get("skip_reason", "unknown")
        sheet = row.get("sheet_name", "unknown")

        reason_counts[reason] = reason_counts.get(reason, 0) + 1
        sheet_counts[sheet] = sheet_counts.get(sheet, 0) + 1

    # Sort by count (descending)
    sorted_reasons = sorted(reason_counts.items(), key=lambda x: x[1], reverse=True)
    sorted_sheets = sorted(sheet_counts.items(), key=lambda x: x[1], reverse=True)

    # Create summary data table
    summary_data = [
        [
            Paragraph("<b>Skip Reason</b>", styles['Normal']),
            Paragraph("<b>Count</b>", styles['Normal']),
            Paragraph("<b>Sheet</b>", styles['Normal']),
            Paragraph("<b>Count</b>", styles['Normal']),
        ]
    ]

    max_rows = max(len(sorted_reasons), len(sorted_sheets))

    for i in range(max_rows):
        row = []

        # Skip reason column
        if i < len(sorted_reasons):
            reason, count = sorted_reasons[i]
            severity = SKIP_REASON_SEVERITY.get(reason, "info")
            sev_color = SEVERITY_COLORS.get(severity, colors.grey)
            reason_label = SKIP_REASON_DESCRIPTIONS.get(reason, reason.replace("_", " ").title())[:40]
            row.append(Paragraph(f'<font color="{sev_color.hexval()}">\u25cf</font> {reason_label}', styles['Normal']))
            row.append(str(count))
        else:
            row.extend(["", ""])

        # Sheet column
        if i < len(sorted_sheets):
            sheet, count = sorted_sheets[i]
            sheet_color = GROUP_COLORS.get(sheet, colors.steelblue)
            sheet_label = get_group_display_name(sheet)
            row.append(Paragraph(f'<font color="{sheet_color.hexval()}">\u25cf</font> {sheet_label}', styles['Normal']))
            row.append(str(count))
        else:
            row.extend(["", ""])

        summary_data.append(row)

    # Create table
    table = Table(summary_data, colWidths=[70*mm, 20*mm, 45*mm, 20*mm])
    table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#6C757D")),  # Dark gray header
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('ALIGN', (3, 0), (3, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))

    return table


def create_skipped_rows_details_table(skipped_rows: List[Dict[str, Any]], max_rows: int = 25, enable_pagination: bool = True) -> Union["Table", "Paragraph", List["Flowable"]]:
    """Create a detailed table showing all skipped rows with reasons.

    Now supports enhanced pagination to avoid orphaned rows and ensure
    minimum rows per page when the table spans multiple pages.

    Args:
        skipped_rows: List of skipped row dictionaries
        max_rows: Maximum number of rows to display
        enable_pagination: Enable enhanced pagination features (default: True)

    Returns:
        ReportLab Table element
    """
    styles = getSampleStyleSheet()

    if not skipped_rows:
        return Paragraph(
            '<font color="#28A745">\u2714</font> <i>All rows processed successfully. No rows were skipped.</i>',
            styles['Normal']
        )

    # Sort by severity (errors first), then by sheet name
    severity_order = {"error": 0, "warning": 1, "info": 2}
    sorted_rows = sorted(
        skipped_rows,
        key=lambda x: (
            severity_order.get(SKIP_REASON_SEVERITY.get(x.get("skip_reason", ""), "info"), 3),
            x.get("sheet_name", ""),
            x.get("row_id", "")
        )
    )

    # Limit to max_rows
    display_rows = sorted_rows[:max_rows]
    truncated = len(sorted_rows) > max_rows

    # Create header row
    table_data = [[
        Paragraph("<b>Sheet</b>", styles['Normal']),
        Paragraph("<b>Row ID</b>", styles['Normal']),
        Paragraph("<b>Skip Reason</b>", styles['Normal']),
        Paragraph("<b>Details</b>", styles['Normal']),
        Paragraph("<b>Phase</b>", styles['Normal']),
    ]]

    for row in display_rows:
        skip_reason = row.get("skip_reason", "unknown")
        severity = SKIP_REASON_SEVERITY.get(skip_reason, "info")
        sev_color = SEVERITY_COLORS.get(severity, colors.grey)

        # Get sheet name with color
        sheet_name = row.get("sheet_name", "N/A")
        sheet_display = get_group_display_name(sheet_name)[:12]

        # Get row ID
        row_id = str(row.get("row_id", "N/A"))[:15]

        # Get skip reason description
        reason_desc = SKIP_REASON_DESCRIPTIONS.get(skip_reason, skip_reason.replace("_", " ").title())[:35]
        if len(reason_desc) >= 35:
            reason_desc += "..."

        # Get details
        details = row.get("details", "")[:45]
        if len(row.get("details", "")) > 45:
            details += "..."

        # Get phase (if applicable)
        phase = row.get("phase", "-")
        if phase:
            phase_display = f"Phase {phase}"
        else:
            phase_display = "-"

        table_row = [
            Paragraph(f'<font color="{sev_color.hexval()}">\u25cf</font> {sheet_display}', styles['Normal']),
            Paragraph(row_id, styles['Normal']),
            Paragraph(reason_desc, styles['Normal']),
            Paragraph(details, styles['Normal']),
            Paragraph(phase_display, styles['Normal']),
        ]
        table_data.append(table_row)

    # Add truncation notice if needed
    if truncated:
        remaining = len(sorted_rows) - max_rows
        table_data.append([
            "",
            "",
            Paragraph(f"<i>... and {remaining} more skipped rows</i>", styles['Normal']),
            "",
            ""
        ])

    # Create table with appropriate column widths
    # Sheet, Row ID, Skip Reason, Details, Phase
    col_widths = [25*mm, 28*mm, 45*mm, 50*mm, 18*mm]

    # Style the table
    style_commands = [
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#FFC107")),  # Amber header for skipped rows
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.HexColor("#212529")),  # Dark text on amber
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 7),
        ('ALIGN', (4, 0), (4, -1), 'CENTER'),  # Phase column centered
        ('VALIGN', (0, 0), (-1, -1), 'TOP'),
        ('TOPPADDING', (0, 0), (-1, -1), 3),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
        ('LEFTPADDING', (0, 0), (-1, -1), 3),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3),
    ]

    # Add alternating row colors and severity highlighting
    for i in range(1, len(table_data)):
        if i % 2 == 0:
            style_commands.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor("#FFFDE7")))  # Light yellow

        # Highlight based on severity
        if i <= len(display_rows):
            row = display_rows[i - 1]
            skip_reason = row.get("skip_reason", "")
            severity = SKIP_REASON_SEVERITY.get(skip_reason, "info")
            if severity == "error":
                style_commands.append(('BACKGROUND', (0, i), (0, i), colors.HexColor("#FFEBEE")))  # Light red
            elif severity == "warning":
                style_commands.append(('BACKGROUND', (0, i), (0, i), colors.HexColor("#FFF8E1")))  # Light amber

    # Use paginated table for header repetition on multi-page tables
    if enable_pagination:
        table = create_paginated_table(
            data=table_data,
            col_widths=col_widths,
            style_commands=style_commands,
            config=PaginationConfig(
                repeat_header_rows=1,
                min_rows_per_page=3,
                avoid_orphans=True
            )
        )
    else:
        table = Table(table_data, colWidths=col_widths)
        table.setStyle(TableStyle(style_commands))

    return table


def create_skipped_rows_legend() -> "Table":
    """Create a legend explaining skip reason severity levels."""
    styles = getSampleStyleSheet()

    legend_data = [[
        Paragraph('<font color="#DC3545">\u25cf</font> Critical (Row skipped entirely)', styles['Normal']),
        Paragraph('<font color="#FFC107">\u25cf</font> Warning (Phase/field skipped)', styles['Normal']),
        Paragraph('<font color="#17A2B8">\u25cf</font> Info (Expected skip)', styles['Normal']),
    ]]

    table = Table(legend_data, colWidths=[55*mm, 55*mm, 50*mm])
    table.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ]))

    return table


def add_skipped_rows_section(story: List["Flowable"], skipped_rows: Optional[List[Dict[str, Any]]] = None, show_sample: bool = False) -> None:
    """Add the skipped rows section to the PDF story.

    This section lists all rows that were skipped during processing, including
    the sheet name, row ID, and specific error that caused the skip.

    Args:
        story: The ReportLab story list to append content to
        skipped_rows: List of skipped row dictionaries. If None and show_sample is True,
                     sample data will be generated.
        show_sample: If True and no skipped_rows provided, generate sample data for demonstration
    """
    # Add section marker for page header navigation
    story.append(SectionMarker("Skipped Rows"))

    styles = getSampleStyleSheet()

    # Section header style
    subheading_style = ParagraphStyle(
        'SkippedRowsSubheading',
        parent=styles['Heading2'],
        fontSize=14,
        textColor=colors.HexColor("#FFC107"),  # Amber color for skipped rows emphasis
        spaceBefore=6*mm,
        spaceAfter=4*mm
    )

    normal_style = styles['Normal']

    # Add section header
    story.append(Paragraph("Skipped Rows During Processing", subheading_style))

    # Get skipped rows data
    if skipped_rows is None:
        if show_sample:
            skipped_rows = generate_sample_skipped_rows()
            story.append(Paragraph(
                "<i>Sample data shown for demonstration. In production, this section displays actual skipped rows from processing.</i>",
                normal_style
            ))
            story.append(Spacer(1, 3*mm))
        else:
            # Try to get from error collector
            skipped_rows = get_skipped_rows_from_error_collector()

    # Add description
    story.append(Paragraph(
        "The following table shows rows that were skipped during the change tracking process. "
        "Each entry includes the sheet name, row identifier, and the specific reason the row was skipped.",
        normal_style
    ))
    story.append(Spacer(1, 4*mm))

    if not skipped_rows:
        # No skipped rows - show success message
        success_data = [[
            Paragraph(
                '<font color="#28A745">\u2714</font> <b>All rows processed successfully</b>',
                styles['Normal']
            )
        ]]
        success_table = Table(success_data, colWidths=[160*mm])
        success_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#D4EDDA")),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ]))
        story.append(success_table)
        story.append(Spacer(1, 5*mm))
        story.append(Paragraph(
            "No rows were skipped during processing. All data was successfully tracked.",
            normal_style
        ))
        return

    # Count totals
    total_skipped = len(skipped_rows)
    error_count = sum(1 for r in skipped_rows if SKIP_REASON_SEVERITY.get(r.get("skip_reason", ""), "info") == "error")
    warning_count = sum(1 for r in skipped_rows if SKIP_REASON_SEVERITY.get(r.get("skip_reason", ""), "info") == "warning")
    info_count = sum(1 for r in skipped_rows if SKIP_REASON_SEVERITY.get(r.get("skip_reason", ""), "info") == "info")

    # Overview box
    overview_text = f"<b>{total_skipped}</b> row(s) were skipped: <b>{error_count}</b> critical, <b>{warning_count}</b> warnings, <b>{info_count}</b> expected skips."

    # Determine box color based on severity
    if error_count > 0:
        box_color = colors.HexColor("#FFF3CD")  # Light yellow/amber
        border_color = colors.HexColor("#FFC107")
    elif warning_count > 0:
        box_color = colors.HexColor("#FFF8E1")  # Lighter yellow
        border_color = colors.HexColor("#FFD54F")
    else:
        box_color = colors.HexColor("#E3F2FD")  # Light blue for info only
        border_color = colors.HexColor("#2196F3")

    overview_data = [[Paragraph(overview_text, normal_style)]]
    overview_table = Table(overview_data, colWidths=[158*mm])
    overview_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), box_color),
        ('BOX', (0, 0), (-1, -1), 2, border_color),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(overview_table)
    story.append(Spacer(1, 6*mm))

    # Add summary table
    story.append(Paragraph("Summary by Reason & Sheet", styles['Heading3']))
    story.append(Spacer(1, 3*mm))
    summary_table = create_skipped_rows_summary_table(skipped_rows)
    story.append(summary_table)
    story.append(Spacer(1, 6*mm))

    # Add legend
    legend = create_skipped_rows_legend()
    story.append(legend)
    story.append(Spacer(1, 3*mm))

    # Add details table
    story.append(Paragraph("Skipped Row Details", styles['Heading3']))
    story.append(Spacer(1, 3*mm))
    details_table = create_skipped_rows_details_table(skipped_rows)
    story.append(details_table)
    story.append(Spacer(1, 6*mm))

    # Add recommendations
    story.append(Paragraph("Recommendations", styles['Heading3']))
    story.append(Spacer(1, 2*mm))

    if error_count > 0:
        story.append(Paragraph(
            f'<font color="#DC3545">\u25cf</font> <b>High Priority:</b> Review {error_count} critical skip(s). These rows were entirely skipped due to missing required data.',
            normal_style
        ))
    if warning_count > 0:
        story.append(Paragraph(
            f'<font color="#FFC107">\u25cf</font> <b>Medium Priority:</b> Investigate {warning_count} warning(s). Some phases or fields may have incomplete data.',
            normal_style
        ))
    if info_count > 0:
        story.append(Paragraph(
            f'<font color="#17A2B8">\u25cf</font> <b>Low Priority:</b> {info_count} row(s) were skipped as expected (e.g., empty date fields indicating no change).',
            normal_style
        ))


def convert_collected_errors_to_report_format(collected_errors: List[CollectedError]) -> List[Dict[str, Any]]:
    """Convert CollectedError objects to the dictionary format used by report functions.

    This function takes a list of CollectedError objects from the error_collector module
    and converts them to the dictionary format expected by the error report functions.

    Args:
        collected_errors: List of CollectedError objects from error_collector

    Returns:
        List of error dictionaries with keys: severity, category, sheet_name, row_id,
        field_name, message, suggested_action, timestamp
    """
    if not collected_errors:
        return []

    converted_errors = []
    for error in collected_errors:
        # Map CollectorSeverity to string format
        severity_map = {
            CollectorSeverity.CRITICAL: "error",
            CollectorSeverity.ERROR: "error",
            CollectorSeverity.WARNING: "warning",
            CollectorSeverity.INFO: "info",
        }

        converted = {
            "severity": severity_map.get(error.severity, "info"),
            "category": error.category.value if hasattr(error.category, 'value') else str(error.category),
            "sheet_name": error.context.get("sheet_name", error.context.get("group", "N/A")),
            "row_id": str(error.context.get("row_id", "N/A")),
            "field_name": error.context.get("field_name", error.context.get("column_name", "N/A")),
            "message": error.message,
            "suggested_action": error.suggested_action or "",
            "timestamp": error.timestamp.strftime("%Y-%m-%d %H:%M") if error.timestamp else "",
            "context": error.context,
        }
        converted_errors.append(converted)

    return converted_errors


def get_collected_errors_for_report() -> List[Dict[str, Any]]:
    """Get all collected errors from the global error collector, formatted for reports.

    This function retrieves errors from the global error collector and converts them
    to the format expected by the error report functions.

    Returns:
        List of error dictionaries ready for use in add_error_report_section()
    """
    collector = get_global_collector()
    all_errors = collector.get_all_errors()
    return convert_collected_errors_to_report_format(all_errors)


def get_data_quality_errors_for_report() -> List[Dict[str, Any]]:
    """Get data quality specific errors from the global collector, formatted for reports.

    This function retrieves only data quality issues from the global error collector
    and converts them to the format expected by the data quality issues section.

    Returns:
        List of data quality error dictionaries
    """
    collector = get_global_collector()
    data_quality_errors = collector.get_data_quality_issues()
    return convert_collected_errors_to_report_format(data_quality_errors)


def add_executive_summary(story: List["Flowable"], metrics: Dict[str, Any], start_date: date, end_date: date, has_data: bool) -> None:
    """Add executive summary section for custom date range reports.

    This provides a high-level overview of activity within the custom date range,
    including key metrics, activity trends, and highlights.

    Args:
        story: The ReportLab story list to append content to
        metrics: Dictionary containing report metrics
        start_date: Start date of the report period
        end_date: End date of the report period
        has_data: Boolean indicating if there is actual data for the period
    """
    styles = getSampleStyleSheet()

    # Section header with date range context
    section_header_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor("#2c3e50"),
        spaceBefore=4*mm,
        spaceAfter=6*mm
    )

    # Summary box style
    summary_box_style = ParagraphStyle(
        'SummaryBox',
        parent=styles['Normal'],
        fontSize=11,
        textColor=colors.HexColor("#34495e"),
        spaceBefore=2*mm,
        spaceAfter=2*mm
    )

    story.append(Paragraph("Executive Summary", section_header_style))

    # Calculate summary statistics
    total_changes = metrics.get("total_changes", 0)
    total_groups = len(metrics.get("groups", {}))
    total_users = len(metrics.get("users", {}))
    days_in_range = (end_date - start_date).days + 1
    daily_avg = total_changes / days_in_range if days_in_range > 0 else 0

    # Create summary table with key metrics
    summary_data = [
        ["Metric", "Value", "Context"],
        ["Total Changes", str(total_changes), f"Across {days_in_range} days"],
        ["Daily Average", f"{daily_avg:.1f}", "Changes per day"],
        ["Active Groups", str(total_groups), "Groups with activity"],
        ["Active Users", str(total_users), "Users making changes"],
    ]

    summary_table = Table(summary_data, colWidths=[60*mm, 50*mm, 70*mm])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#3498db")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('FONTSIZE', (0, 1), (-1, -1), 10),
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWHEIGHT', (0, 0), (-1, -1), 20),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#ecf0f1")),
    ]))

    story.append(summary_table)
    story.append(Spacer(1, 6*mm))

    # Add data status note if no data
    if not has_data:
        note_style = ParagraphStyle(
            'DataNote',
            parent=styles['Normal'],
            fontSize=9,
            textColor=colors.HexColor("#e74c3c"),
            alignment=1
        )
        story.append(Paragraph(
            "Note: No data found for this date range. Sample data shown for demonstration.",
            note_style
        ))
        story.append(Spacer(1, 4*mm))


def get_group_activity_trends(
    groups: Dict[str, int],
    trend_days: int = 7,
    end_date: Optional[date] = None
) -> Dict[str, List[float]]:
    """Calculate daily activity trends for each group over a specified period.

    Generates daily activity counts for sparkline visualization in summary tables.
    Supports both 7-day and 30-day trend windows.

    Args:
        groups: Dictionary mapping group names to total change counts
        trend_days: Number of days to include in the trend (default: 7, max: 30)
        end_date: End date for the trend period. Defaults to today.

    Returns:
        Dictionary mapping group names to lists of daily activity counts.
        Each list contains `trend_days` values, one per day, oldest first.
        Empty groups or groups with no historical data return lists of zeros.

    Example:
        >>> groups = {"NA": 100, "NF": 80}
        >>> trends = get_group_activity_trends(groups, trend_days=7)
        >>> trends["NA"]  # [12, 15, 10, 18, 14, 16, 15]
    """
    from historical_data_loader import load_change_history, filter_by_group

    # Ensure trend_days is within valid range
    trend_days = max(1, min(30, trend_days))

    # Calculate date range
    reference_date = end_date or date.today()
    start = reference_date - timedelta(days=trend_days - 1)

    # Load historical change data
    try:
        all_changes = load_change_history(
            start_date=start,
            end_date=reference_date,
            validate=True,
            log_validation_stats=False
        )
    except FileNotFoundError:
        logger.warning("Historical data file not found for sparkline trends")
        # Return zero-filled data for all groups
        return {group: [0.0] * trend_days for group in groups.keys()}
    except Exception as e:
        logger.warning(f"Error loading historical data for sparklines: {e}")
        return {group: [0.0] * trend_days for group in groups.keys()}

    trends: Dict[str, List[float]] = {}

    for group in groups.keys():
        # Filter changes for this group
        group_changes = filter_by_group(all_changes, group)

        # Count changes per day
        daily_counts: Dict[str, int] = {}
        for change in group_changes:
            ts = change.get('ParsedTimestamp')
            if ts:
                date_str = ts.date().isoformat()
                daily_counts[date_str] = daily_counts.get(date_str, 0) + 1

        # Build the trend list (one value per day, oldest first)
        trend_data: List[float] = []
        for i in range(trend_days):
            day = start + timedelta(days=i)
            day_str = day.isoformat()
            trend_data.append(float(daily_counts.get(day_str, 0)))

        trends[group] = trend_data

    return trends


def create_sparkline_cell(
    data: List[float],
    width: int = TABLE_SPARKLINE_WIDTH,
    height: int = TABLE_SPARKLINE_HEIGHT,
    style: SparklineStyle = SparklineStyle.LINE,
) -> "Drawing":
    """Create a sparkline suitable for embedding in a table cell.

    Wrapper function that creates a properly sized sparkline Drawing
    for use in ReportLab tables. Handles empty data gracefully.

    Args:
        data: List of numeric values for the sparkline
        width: Width of the sparkline in points (default: TABLE_SPARKLINE_WIDTH)
        height: Height of the sparkline in points (default: TABLE_SPARKLINE_HEIGHT)
        style: Sparkline visualization style (default: LINE)

    Returns:
        A ReportLab Drawing object containing the sparkline chart.
        Returns an empty Drawing if data is empty or all zeros.

    Example:
        >>> trend = [5, 8, 3, 10, 7]
        >>> sparkline = create_sparkline_cell(trend)
        >>> # Use in table: table_data.append([..., sparkline])
    """
    # Handle empty or all-zero data
    if not data or all(v == 0 for v in data):
        # Return an empty drawing with "No data" indicator
        drawing = Drawing(width, height)
        from reportlab.graphics.shapes import String
        drawing.add(String(
            width / 2, height / 2 - 3,
            "—",
            fontName='Helvetica',
            fontSize=8,
            textAnchor='middle',
            fillColor=colors.HexColor("#95a5a6")
        ))
        return drawing

    return create_table_sparkline(
        data=data,
        width=width,
        height=height,
        style=style,
        highlight_max=True,
        highlight_min=False,
    )


def add_group_summary(story: List["Flowable"], metrics: Dict[str, Any], has_data: bool,
                      trend_days: int = 7, end_date: Optional[date] = None) -> None:
    """Add group activity summary section for custom date range reports.

    Displays a breakdown of changes by group with visual indicators
    showing the relative activity levels and sparkline trends.

    Args:
        story: The ReportLab story list to append content to
        metrics: Dictionary containing report metrics
        has_data: Boolean indicating if there is actual data for the period
        trend_days: Number of days for sparkline trend (7 or 30). Defaults to 7.
        end_date: End date for trend calculation. Defaults to today.
    """
    styles = getSampleStyleSheet()

    section_header_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor("#2c3e50"),
        spaceBefore=4*mm,
        spaceAfter=6*mm
    )

    story.append(Paragraph("Group Activity Summary", section_header_style))

    groups = metrics.get("groups", {})

    if not groups:
        story.append(Paragraph(
            "No group activity data available for this period.",
            styles['Normal']
        ))
        story.append(Spacer(1, 6*mm))
        return

    # Sort groups by activity (descending)
    sorted_groups = sorted(groups.items(), key=lambda x: x[1], reverse=True)
    max_changes = max(groups.values()) if groups.values() else 1

    # Get activity trends for sparklines
    activity_trends = get_group_activity_trends(
        groups=groups,
        trend_days=trend_days,
        end_date=end_date
    )

    # Determine trend label based on days
    trend_label = f"{trend_days}-Day Trend"

    # Create data for table with sparkline column
    table_data = [["Group", "Changes", trend_label, "Activity Level"]]

    for group, changes in sorted_groups[:10]:  # Top 10 groups
        # Calculate activity bar width (percentage of max)
        activity_pct = (changes / max_changes * 100) if max_changes > 0 else 0
        # Visual representation using Unicode blocks
        bar_length = int(activity_pct / 5)  # 20 max blocks
        activity_bar = "█" * bar_length + "░" * (20 - bar_length)

        # Create sparkline for this group's trend
        trend_data = activity_trends.get(group, [0.0] * trend_days)
        sparkline = create_sparkline_cell(
            data=trend_data,
            width=TABLE_SPARKLINE_WIDTH,
            height=TABLE_SPARKLINE_HEIGHT,
            style=SparklineStyle.LINE
        )

        table_data.append([group, str(changes), sparkline, f"{activity_bar} {activity_pct:.0f}%"])

    # Style commands for group activity table
    group_style_commands = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#27ae60")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('FONTNAME', (3, 1), (3, -1), 'Courier'),  # Monospace for activity bar (now column 3)
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('ALIGN', (2, 0), (2, -1), 'CENTER'),  # Center sparkline column
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWHEIGHT', (0, 0), (-1, -1), 28),  # Increased row height for sparklines
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#f8f9fa")),
    ]

    # Use paginated table for header repetition on multi-page tables
    # Adjusted column widths to accommodate sparkline column
    group_table = create_paginated_table(
        data=table_data,
        col_widths=[35*mm, 25*mm, 30*mm, 90*mm],  # Group, Changes, Trend, Activity Level
        style_commands=group_style_commands,
        config=PaginationConfig(
            repeat_header_rows=1,
            min_rows_per_page=3,
            avoid_orphans=True
        )
    )

    story.append(group_table)
    story.append(Spacer(1, 6*mm))


def add_phase_distribution(story: List["Flowable"], metrics: Dict[str, Any], has_data: bool) -> None:
    """Add phase distribution section for custom date range reports.

    Shows how changes are distributed across different phases
    (e.g., Planning, Development, Testing, Production).

    Args:
        story: The ReportLab story list to append content to
        metrics: Dictionary containing report metrics
        has_data: Boolean indicating if there is actual data for the period
    """
    styles = getSampleStyleSheet()

    section_header_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor("#2c3e50"),
        spaceBefore=4*mm,
        spaceAfter=6*mm
    )

    story.append(Paragraph("Phase Distribution", section_header_style))

    phases = metrics.get("phases", {})

    if not phases:
        story.append(Paragraph(
            "No phase distribution data available for this period.",
            styles['Normal']
        ))
        story.append(Spacer(1, 6*mm))
        return

    # Calculate totals for percentages
    total_changes = sum(phases.values())

    # Sort phases by count (descending)
    sorted_phases = sorted(phases.items(), key=lambda x: x[1], reverse=True)

    # Phase color mapping (using distinct colors for each phase)
    phase_colors = {
        "Planning": "#3498db",
        "Development": "#9b59b6",
        "Testing": "#e67e22",
        "Production": "#27ae60",
        "Review": "#f39c12",
        "Completed": "#1abc9c",
        "On Hold": "#95a5a6",
    }

    # Create data for table
    table_data = [["Phase", "Changes", "Percentage", "Distribution"]]

    for phase, count in sorted_phases:
        pct = (count / total_changes * 100) if total_changes > 0 else 0
        bar_length = int(pct / 5)  # 20 max blocks
        distribution_bar = "█" * bar_length + "░" * (20 - bar_length)
        table_data.append([phase or "Unknown", str(count), f"{pct:.1f}%", distribution_bar])

    # Style commands for phase distribution table
    phase_style_commands = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#9b59b6")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 11),
        ('FONTSIZE', (0, 1), (-1, -1), 9),
        ('FONTNAME', (3, 1), (3, -1), 'Courier'),  # Monospace for bar
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
        ('ALIGN', (1, 0), (2, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWHEIGHT', (0, 0), (-1, -1), 18),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#f5f6fa")),
    ]

    # Use paginated table for header repetition on multi-page tables
    phase_table = create_paginated_table(
        data=table_data,
        col_widths=[45*mm, 30*mm, 30*mm, 75*mm],
        style_commands=phase_style_commands,
        config=PaginationConfig(
            repeat_header_rows=1,
            min_rows_per_page=3,
            avoid_orphans=True
        )
    )

    story.append(phase_table)
    story.append(Spacer(1, 6*mm))


def add_user_activity(
    story: List["Flowable"],
    metrics: Dict[str, Any],
    has_data: bool,
    use_landscape: bool = False
) -> None:
    """Add user activity section for custom date range reports.

    Shows the most active users within the date range with their
    change counts and activity breakdown. Supports landscape mode
    for improved readability of wide data tables.

    Args:
        story: The ReportLab story list to append content to
        metrics: Dictionary containing report metrics
        has_data: Boolean indicating if there is actual data for the period
        use_landscape: If True, uses landscape-optimized column widths for wider tables
    """
    styles = getSampleStyleSheet()

    section_header_style = ParagraphStyle(
        'SectionHeader',
        parent=styles['Heading1'],
        fontSize=18,
        textColor=colors.HexColor("#2c3e50"),
        spaceBefore=4*mm,
        spaceAfter=6*mm
    )

    story.append(Paragraph("User Activity", section_header_style))

    users = metrics.get("users", {})

    if not users:
        story.append(Paragraph(
            "No user activity data available for this period.",
            styles['Normal']
        ))
        story.append(Spacer(1, 6*mm))
        return

    # Sort users by activity (descending)
    sorted_users = sorted(users.items(), key=lambda x: x[1], reverse=True)
    total_changes = sum(users.values())
    max_changes = max(users.values()) if users.values() else 1

    # In landscape mode, show more users and allow longer names
    max_users = 20 if use_landscape else 15
    name_truncate = 50 if use_landscape else 30
    bar_blocks = 15 if use_landscape else 10  # More blocks for wider activity bar

    # Create data for table
    table_data = [["Rank", "User", "Changes", "% of Total", "Activity"]]

    for rank, (user, changes) in enumerate(sorted_users[:max_users], 1):
        pct_total = (changes / total_changes * 100) if total_changes > 0 else 0
        activity_pct = (changes / max_changes * 100) if max_changes > 0 else 0
        bar_length = int(activity_pct / (100 / bar_blocks))  # Scale to bar_blocks
        activity_bar = "█" * bar_length + "░" * (bar_blocks - bar_length)
        table_data.append([
            str(rank),
            user[:name_truncate] if user else "Unknown",  # Truncate based on mode
            str(changes),
            f"{pct_total:.1f}%",
            activity_bar
        ])

    # Style commands for user activity table
    user_style_commands = [
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#e74c3c")),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, 0), 10 if use_landscape else 10),
        ('FONTSIZE', (0, 1), (-1, -1), 9 if use_landscape else 8),
        ('FONTNAME', (4, 1), (4, -1), 'Courier'),  # Monospace for bar
        ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#bdc3c7")),
        ('ALIGN', (0, 0), (0, -1), 'CENTER'),
        ('ALIGN', (2, 0), (3, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ROWHEIGHT', (0, 0), (-1, -1), 18 if use_landscape else 16),
        ('BACKGROUND', (0, 1), (-1, -1), colors.HexColor("#fdf2f2")),
    ]

    # Add landscape-specific styling
    if use_landscape:
        user_style_commands.extend([
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ])

    # Use appropriate column widths based on orientation
    if use_landscape:
        # Landscape mode: use wider columns for better readability
        col_widths = LANDSCAPE_USER_ACTIVITY_WIDTHS
    else:
        # Portrait mode: original column widths
        col_widths = [15*mm, 70*mm, 25*mm, 25*mm, 45*mm]

    # Use paginated table for header repetition on multi-page tables
    user_table = create_paginated_table(
        data=table_data,
        col_widths=col_widths,
        style_commands=user_style_commands,
        config=PaginationConfig(
            repeat_header_rows=1,
            min_rows_per_page=3,
            avoid_orphans=True
        )
    )

    story.append(user_table)
    story.append(Spacer(1, 6*mm))

    # Add summary note
    summary_style = ParagraphStyle(
        'UserSummary',
        parent=styles['Normal'],
        fontSize=9,
        textColor=colors.HexColor("#7f8c8d"),
        alignment=0
    )
    story.append(Paragraph(
        f"Showing top {min(15, len(sorted_users))} users of {len(users)} total active users.",
        summary_style
    ))


def get_error_rate_severity_level(error_rate: float) -> str:
    """Determine the severity level based on error rate percentage.

    Args:
        error_rate: Error rate as a percentage (0-100)

    Returns:
        str: Severity level ('low', 'moderate', 'high', or 'critical')
    """
    if error_rate < 5.0:
        return "low"
    elif error_rate < 15.0:
        return "moderate"
    elif error_rate < 30.0:
        return "high"
    else:
        return "critical"


def create_error_rate_summary_display(errors: List[Dict[str, Any]], total_rows: Optional[int] = None) -> "Drawing":
    """Create a high-level error statistics summary display with visual severity indicators.

    This function generates a visual summary box showing:
    - Total error count with percentage
    - Error rate severity indicator (green/yellow/orange/red)
    - Breakdown by severity type (errors, warnings, info)
    - Quick status assessment

    Args:
        errors: List of error/warning dictionaries
        total_rows: Optional total number of rows processed (for error rate calculation)

    Returns:
        ReportLab Table element containing the summary display
    """
    styles = getSampleStyleSheet()

    # Calculate statistics
    total_issues = len(errors) if errors else 0
    error_count = sum(1 for e in errors if e.get("severity") == "error") if errors else 0
    warning_count = sum(1 for e in errors if e.get("severity") == "warning") if errors else 0
    info_count = sum(1 for e in errors if e.get("severity") == "info") if errors else 0
    critical_count = sum(1 for e in errors if e.get("severity") == "critical") if errors else 0

    # Calculate error rate if total_rows provided
    if total_rows and total_rows > 0:
        error_rate = (total_issues / total_rows) * 100
    else:
        # Estimate based on typical report size if not provided
        # Use error count as a proxy for severity
        if error_count + critical_count == 0:
            error_rate = 0.0
        elif error_count + critical_count <= 2:
            error_rate = 2.5  # Low
        elif error_count + critical_count <= 5:
            error_rate = 8.0  # Moderate
        elif error_count + critical_count <= 10:
            error_rate = 20.0  # High
        else:
            error_rate = 35.0  # Critical

    # Determine severity level
    severity_level = get_error_rate_severity_level(error_rate)

    # Get colors based on severity
    indicator_color = ERROR_RATE_SEVERITY_COLORS.get(severity_level, colors.grey)
    bg_color = ERROR_RATE_SEVERITY_BG_COLORS.get(severity_level, colors.HexColor("#F5F5F5"))
    border_color = ERROR_RATE_SEVERITY_BORDER_COLORS.get(severity_level, colors.grey)
    status_label = ERROR_RATE_SEVERITY_LABELS.get(severity_level, "Unknown")

    # Create the summary content
    # Main header with large indicator
    if severity_level == "low":
        status_icon = "\u2714"  # Checkmark
    elif severity_level in ("moderate", "high"):
        status_icon = "\u26A0"  # Warning triangle
    else:
        status_icon = "\u2716"  # X mark

    # Build the summary display as a multi-row table
    header_style = ParagraphStyle(
        'SummaryHeader',
        parent=styles['Normal'],
        fontSize=14,
        fontName='Helvetica-Bold',
        textColor=indicator_color,
        alignment=1  # Center
    )

    stats_style = ParagraphStyle(
        'SummaryStats',
        parent=styles['Normal'],
        fontSize=10,
        alignment=1  # Center
    )

    detail_style = ParagraphStyle(
        'SummaryDetail',
        parent=styles['Normal'],
        fontSize=9,
        alignment=0  # Left
    )

    # Row 1: Status header with icon
    header_text = f'<font color="{indicator_color.hexval()}">{status_icon}</font> Error Rate Status: <b>{status_label}</b>'
    header_para = Paragraph(header_text, header_style)

    # Row 2: Key statistics
    if total_rows and total_rows > 0:
        rate_text = f"{error_rate:.1f}%"
    else:
        rate_text = f"~{error_rate:.0f}%" if error_rate > 0 else "0%"

    stats_text = f"<b>{total_issues}</b> total issues | Error Rate: <b>{rate_text}</b>"
    stats_para = Paragraph(stats_text, stats_style)

    # Row 3: Severity breakdown with colored indicators
    breakdown_parts = []
    if critical_count > 0:
        breakdown_parts.append(f'<font color="#B71C1C">\u25cf</font> Critical: {critical_count}')
    if error_count > 0:
        breakdown_parts.append(f'<font color="#B71C1C">\u25cf</font> Errors: {error_count}')
    if warning_count > 0:
        breakdown_parts.append(f'<font color="#8B6914">\u25cf</font> Warnings: {warning_count}')
    if info_count > 0:
        breakdown_parts.append(f'<font color="#0D47A1">\u25cf</font> Info: {info_count}')

    if breakdown_parts:
        breakdown_text = "  |  ".join(breakdown_parts)
    else:
        breakdown_text = '<font color="#1B5E20">\u2714</font> No issues detected'

    breakdown_para = Paragraph(breakdown_text, detail_style)

    # Create table structure
    summary_data = [
        [header_para],
        [stats_para],
        [breakdown_para],
    ]

    summary_table = Table(summary_data, colWidths=[158*mm])
    summary_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), bg_color),
        ('BOX', (0, 0), (-1, -1), 2, border_color),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, 0), 10),
        ('BOTTOMPADDING', (0, -1), (-1, -1), 10),
        ('TOPPADDING', (0, 1), (-1, -2), 5),
        ('BOTTOMPADDING', (0, 0), (-1, -2), 5),
        ('LEFTPADDING', (0, 0), (-1, -1), 15),
        ('RIGHTPADDING', (0, 0), (-1, -1), 15),
    ]))

    return summary_table


def add_error_report_section(
    story: List["Flowable"],
    errors: Optional[List[Dict[str, Any]]] = None,
    show_sample: bool = False,
    use_landscape: bool = False
) -> None:
    """Add the error and warning report section to the PDF story.

    Args:
        story: The ReportLab story list to append content to
        errors: List of error/warning dictionaries. If None and show_sample is True,
                sample data will be generated.
        show_sample: If True and no errors provided, generate sample data for demonstration
        use_landscape: If True, uses landscape-optimized layout for wider error tables
    """
    styles = getSampleStyleSheet()
    heading_style = styles['Heading1']
    subheading_style = styles['Heading2']
    normal_style = styles['Normal']

    # Add page break before error report section
    story.append(PageBreak())

    # Add section marker for page header navigation
    # Use LandscapeSectionMarker if landscape mode is requested
    if use_landscape:
        story.append(LandscapeSectionMarker("Error Report"))
    else:
        story.append(SectionMarker("Error Report"))

    # Determine header width based on orientation
    header_width = 260*mm if use_landscape else 160*mm

    # Section header with colored background
    header_data = [["Data Quality & Issues Report"]]
    header_table = Table(header_data, colWidths=[header_width], rowHeights=[12*mm])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#495057")),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 16),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 8*mm))

    # Get errors (use sample if needed)
    if errors is None:
        if show_sample:
            errors = generate_sample_errors()
            story.append(Paragraph(
                "<i>Sample data shown for demonstration. In production, this section will display actual data quality issues.</i>",
                normal_style
            ))
            story.append(Spacer(1, 5*mm))
        else:
            errors = []

    # Add high-level error statistics summary at top of error report
    # This provides visual indicators (green/yellow/red) for error rate severity
    error_summary_display = create_error_rate_summary_display(errors)
    story.append(error_summary_display)
    story.append(Spacer(1, 8*mm))

    if not errors:
        # Determine table width based on orientation
        table_width = 260*mm if use_landscape else 160*mm

        # No errors to report - show success message
        success_data = [[
            Paragraph(
                '<font color="#28A745">\u2714</font> <b>No issues detected</b>',
                styles['Normal']
            )
        ]]
        success_table = Table(success_data, colWidths=[table_width])
        success_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#D4EDDA")),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ]))
        story.append(success_table)
        story.append(Spacer(1, 5*mm))
        story.append(Paragraph(
            "All data quality checks passed. No errors or warnings to report.",
            normal_style
        ))
        return

    # Determine table width based on orientation
    table_width = 260*mm if use_landscape else 160*mm
    overview_width = 258*mm if use_landscape else 158*mm

    # Count totals
    error_count = sum(1 for e in errors if e.get("severity") == "error")
    warning_count = sum(1 for e in errors if e.get("severity") == "warning")
    info_count = sum(1 for e in errors if e.get("severity") == "info")

    # Summary overview box
    overview_text = f"Found <b>{error_count}</b> error(s), <b>{warning_count}</b> warning(s), and <b>{info_count}</b> informational item(s)."

    # Determine box color based on severity
    if error_count > 0:
        box_color = colors.HexColor("#F8D7DA")  # Light red
        border_color = colors.HexColor("#DC3545")
    elif warning_count > 0:
        box_color = colors.HexColor("#FFF3CD")  # Light yellow
        border_color = colors.HexColor("#FFC107")
    else:
        box_color = colors.HexColor("#D1ECF1")  # Light blue
        border_color = colors.HexColor("#17A2B8")

    overview_data = [[Paragraph(overview_text, normal_style)]]
    overview_table = Table(overview_data, colWidths=[overview_width])
    overview_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), box_color),
        ('BOX', (0, 0), (-1, -1), 2, border_color),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(overview_table)
    story.append(Spacer(1, 8*mm))

    # Summary by severity and category
    story.append(Paragraph("Summary by Severity & Category", subheading_style))
    story.append(Spacer(1, 3*mm))

    summary_table = create_error_summary_table(errors)
    story.append(summary_table)
    story.append(Spacer(1, 8*mm))

    # Add dedicated Data Quality Issues section
    add_data_quality_issues_section(story, errors)
    story.append(Spacer(1, 8*mm))

    # Add Skipped Rows section
    add_skipped_rows_section(story, skipped_rows=None, show_sample=show_sample)
    story.append(Spacer(1, 8*mm))

    # Detailed issues table (all issues)
    story.append(Paragraph("All Issue Details", subheading_style))
    story.append(Spacer(1, 3*mm))

    # Add legend
    legend = create_error_legend()
    story.append(legend)
    story.append(Spacer(1, 3*mm))

    # Add details table
    details_table = create_error_details_table(errors)
    story.append(details_table)
    story.append(Spacer(1, 8*mm))

    # Actionable recommendations
    story.append(Paragraph("Recommended Actions", subheading_style))
    story.append(Spacer(1, 3*mm))

    # Group by priority
    if error_count > 0:
        story.append(Paragraph(
            f'<font color="#DC3545">\u25cf</font> <b>High Priority:</b> Address {error_count} error(s) immediately to prevent data issues.',
            normal_style
        ))
    if warning_count > 0:
        story.append(Paragraph(
            f'<font color="#FFC107">\u25cf</font> <b>Medium Priority:</b> Review {warning_count} warning(s) at earliest convenience.',
            normal_style
        ))
    if info_count > 0:
        story.append(Paragraph(
            f'<font color="#17A2B8">\u25cf</font> <b>Low Priority:</b> Consider reviewing {info_count} informational item(s) during regular maintenance.',
            normal_style
        ))


# ============================================================================
# MISSING DATA WARNINGS SECTION
# ============================================================================

def generate_sample_missing_data_warnings() -> Dict[str, Dict[str, List[Dict[str, Any]]]]:
    """Generate sample missing/incomplete data warnings grouped by sheet and field.

    Returns:
        dict: Dictionary structured as {sheet_name: {field_name: [warnings]}}
    """
    from datetime import datetime

    # Sample data representing systematic issues across sheets and fields
    sample_warnings = {
        "NA Products": {
            "Kontrolle": [
                {
                    "row_id": "1234567",
                    "issue": "Date field is empty",
                    "severity": "warning",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                },
                {
                    "row_id": "1234890",
                    "issue": "Date field is empty",
                    "severity": "warning",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                },
                {
                    "row_id": "1235012",
                    "issue": "Date field is empty",
                    "severity": "warning",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                },
            ],
            "K von": [
                {
                    "row_id": "1234567",
                    "issue": "User field is empty",
                    "severity": "warning",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                },
                {
                    "row_id": "1234890",
                    "issue": "User field is empty",
                    "severity": "warning",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                },
            ],
            "BE am": [
                {
                    "row_id": "2345678",
                    "issue": "Date format invalid",
                    "severity": "error",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                },
            ],
        },
        "NF Products": {
            "C am": [
                {
                    "row_id": "3456789",
                    "issue": "Date field is empty",
                    "severity": "warning",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                },
                {
                    "row_id": "3456901",
                    "issue": "Date field is empty",
                    "severity": "warning",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                },
            ],
            "C von": [
                {
                    "row_id": "3456789",
                    "issue": "User field is empty",
                    "severity": "warning",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                },
            ],
        },
        "NH Products": {
            "K am": [
                {
                    "row_id": "4567890",
                    "issue": "Date value is null",
                    "severity": "warning",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                },
            ],
            "Amazon": [
                {
                    "row_id": "4567890",
                    "issue": "Marketplace indicator missing",
                    "severity": "info",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                },
                {
                    "row_id": "4568012",
                    "issue": "Marketplace indicator missing",
                    "severity": "info",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                },
            ],
        },
        "NM Products": {
            "BE von": [
                {
                    "row_id": "5678901",
                    "issue": "User field is empty",
                    "severity": "warning",
                    "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M"),
                },
            ],
        },
    }

    return sample_warnings


def create_missing_data_summary_table(warnings_by_sheet: Dict[str, Dict[str, List[Dict[str, Any]]]]) -> "Table":
    """Create a summary table showing counts of missing data by sheet and field.

    Args:
        warnings_by_sheet: Dictionary of {sheet_name: {field_name: [warnings]}}

    Returns:
        Table: ReportLab table with summary statistics
    """
    styles = getSampleStyleSheet()

    # Calculate summary statistics
    sheet_stats = []
    total_warnings = 0
    total_errors = 0
    total_info = 0

    for sheet_name, fields in warnings_by_sheet.items():
        sheet_warnings = 0
        sheet_errors = 0
        sheet_info = 0
        field_count = len(fields)

        for field_name, warnings in fields.items():
            for w in warnings:
                severity = w.get("severity", "warning")
                if severity == "error":
                    sheet_errors += 1
                    total_errors += 1
                elif severity == "info":
                    sheet_info += 1
                    total_info += 1
                else:
                    sheet_warnings += 1
                    total_warnings += 1

        sheet_stats.append({
            "sheet": sheet_name,
            "fields_affected": field_count,
            "errors": sheet_errors,
            "warnings": sheet_warnings,
            "info": sheet_info,
            "total": sheet_errors + sheet_warnings + sheet_info,
        })

    # Sort by total issues (descending)
    sheet_stats.sort(key=lambda x: x["total"], reverse=True)

    # Create table header
    table_data = [[
        Paragraph("<b>Sheet</b>", styles['Normal']),
        Paragraph("<b>Fields</b>", styles['Normal']),
        Paragraph("<b>Errors</b>", styles['Normal']),
        Paragraph("<b>Warnings</b>", styles['Normal']),
        Paragraph("<b>Info</b>", styles['Normal']),
        Paragraph("<b>Total</b>", styles['Normal']),
    ]]

    # Add data rows
    for stat in sheet_stats:
        error_text = f'<font color="#DC3545">{stat["errors"]}</font>' if stat["errors"] > 0 else "0"
        warning_text = f'<font color="#FFC107">{stat["warnings"]}</font>' if stat["warnings"] > 0 else "0"
        info_text = f'<font color="#17A2B8">{stat["info"]}</font>' if stat["info"] > 0 else "0"

        table_data.append([
            Paragraph(stat["sheet"][:20], styles['Normal']),
            str(stat["fields_affected"]),
            Paragraph(error_text, styles['Normal']),
            Paragraph(warning_text, styles['Normal']),
            Paragraph(info_text, styles['Normal']),
            str(stat["total"]),
        ])

    # Add totals row
    grand_total = total_errors + total_warnings + total_info
    table_data.append([
        Paragraph("<b>TOTAL</b>", styles['Normal']),
        "",
        Paragraph(f'<font color="#DC3545"><b>{total_errors}</b></font>', styles['Normal']),
        Paragraph(f'<font color="#FFC107"><b>{total_warnings}</b></font>', styles['Normal']),
        Paragraph(f'<font color="#17A2B8"><b>{total_info}</b></font>', styles['Normal']),
        Paragraph(f"<b>{grand_total}</b>", styles['Normal']),
    ])

    # Create table with column widths
    col_widths = [45*mm, 20*mm, 20*mm, 25*mm, 15*mm, 20*mm]

    style_commands = [
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#E8E8E8")),
        ('BACKGROUND', (0, -1), (-1, -1), colors.HexColor("#F0F0F0")),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('ALIGN', (1, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]

    # Use paginated table for header repetition on multi-page tables
    table = create_paginated_table(
        data=table_data,
        col_widths=col_widths,
        style_commands=style_commands,
        config=PaginationConfig(
            repeat_header_rows=1,
            min_rows_per_page=3,
            avoid_orphans=True
        )
    )
    return table


def create_missing_data_details_table(warnings_by_sheet: Dict[str, Dict[str, List[Dict[str, Any]]]], max_rows_per_sheet: int = 5) -> List["Flowable"]:
    """Create a detailed table showing missing data grouped by sheet and field.

    Args:
        warnings_by_sheet: Dictionary of {sheet_name: {field_name: [warnings]}}
        max_rows_per_sheet: Maximum warning rows to show per sheet

    Returns:
        list: List of ReportLab elements (tables and paragraphs)
    """
    styles = getSampleStyleSheet()
    elements = []

    for sheet_name, fields in warnings_by_sheet.items():
        # Sheet header
        sheet_header_data = [[Paragraph(f"<b>{sheet_name}</b>", styles['Normal'])]]
        sheet_header_table = Table(sheet_header_data, colWidths=[160*mm])
        sheet_header_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#6C757D")),
            ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
            ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
            ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 10),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ]))
        elements.append(sheet_header_table)

        # Field details table
        table_data = [[
            Paragraph("<b>Field</b>", styles['Normal']),
            Paragraph("<b>Count</b>", styles['Normal']),
            Paragraph("<b>Severity</b>", styles['Normal']),
            Paragraph("<b>Sample Issues</b>", styles['Normal']),
        ]]

        for field_name, warnings in fields.items():
            # Count by severity
            error_count = sum(1 for w in warnings if w.get("severity") == "error")
            warning_count = sum(1 for w in warnings if w.get("severity") == "warning")
            info_count = sum(1 for w in warnings if w.get("severity") == "info")

            # Determine primary severity
            if error_count > 0:
                primary_severity = "error"
                sev_color = "#DC3545"
                sev_label = "Error"
            elif warning_count > 0:
                primary_severity = "warning"
                sev_color = "#FFC107"
                sev_label = "Warning"
            else:
                primary_severity = "info"
                sev_color = "#17A2B8"
                sev_label = "Info"

            # Get sample row IDs (up to 3)
            sample_rows = [w.get("row_id", "N/A") for w in warnings[:3]]
            sample_text = ", ".join(sample_rows)
            if len(warnings) > 3:
                sample_text += f" (+{len(warnings) - 3} more)"

            table_data.append([
                Paragraph(field_name[:20], styles['Normal']),
                str(len(warnings)),
                Paragraph(f'<font color="{sev_color}">\u25cf</font> {sev_label}', styles['Normal']),
                Paragraph(f"Rows: {sample_text}", styles['Normal']),
            ])

        # Create field table with column widths
        col_widths = [35*mm, 20*mm, 30*mm, 75*mm]

        style_commands = [
            ('GRID', (0, 0), (-1, -1), 0.5, colors.HexColor("#DEE2E6")),
            ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#F8F9FA")),
            ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
            ('FONTSIZE', (0, 0), (-1, -1), 8),
            ('ALIGN', (1, 0), (1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('LEFTPADDING', (0, 0), (-1, -1), 4),
        ]

        # Add alternating row colors
        for i in range(1, len(table_data)):
            if i % 2 == 0:
                style_commands.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor("#F8F8F8")))

        # Use paginated table for header repetition on multi-page tables
        field_table = create_paginated_table(
            data=table_data,
            col_widths=col_widths,
            style_commands=style_commands,
            config=PaginationConfig(
                repeat_header_rows=1,
                min_rows_per_page=3,
                avoid_orphans=True
            )
        )
        elements.append(field_table)
        elements.append(Spacer(1, 3*mm))

    return elements


# ============================================================================
# CLEANUP SUGGESTIONS SECTION
# ============================================================================

# Priority colors for cleanup suggestions - accessible color palette
SUGGESTION_PRIORITY_COLORS = {
    "CRITICAL": colors.HexColor("#DC3545"),  # Red
    "HIGH": colors.HexColor("#FD7E14"),      # Orange
    "MEDIUM": colors.HexColor("#FFC107"),    # Yellow/Amber
    "LOW": colors.HexColor("#17A2B8"),       # Cyan/Info
}

# Effort colors for cleanup suggestions
SUGGESTION_EFFORT_COLORS = {
    "quick_fix": colors.HexColor("#28A745"),     # Green - quick wins
    "moderate": colors.HexColor("#17A2B8"),      # Cyan - moderate effort
    "significant": colors.HexColor("#FFC107"),   # Yellow - significant effort
    "major": colors.HexColor("#DC3545"),         # Red - major effort
}

# Effort labels for display
SUGGESTION_EFFORT_LABELS = {
    "quick_fix": "Quick Fix (<15 min)",
    "moderate": "Moderate (15-60 min)",
    "significant": "Significant (1-4 hrs)",
    "major": "Major (4+ hrs)",
}


def generate_sample_cleanup_suggestions() -> Dict[str, Any]:
    """Generate sample cleanup suggestions for demonstration.

    Returns:
        dict: A dictionary containing suggestions, summary, top_suggestions, and quick_wins
    """
    from datetime import datetime

    # Sample suggestions mimicking the structure from CleanupSuggestionGenerator.export_for_report()
    return {
        "suggestions": [
            {
                "id": "SUG-20240115-001",
                "title": "Address Missing Data Issues",
                "description": "Detected 45 instances of missing data across 3 groups. Missing data can cause processing failures and incomplete reports.",
                "priority": "CRITICAL",
                "priority_value": 1,
                "effort": "moderate",
                "impact_score": 85.5,
                "frequency": 45,
                "affected_groups": ["NA", "NF", "NH"],
                "affected_fields": ["Kontrolle", "BE am"],
                "can_batch_fix": True,
                "estimated_time": "1-2 hours",
                "actionable_steps": [
                    {"action": "Identify rows with missing required fields", "priority": 1, "details": "Run a report to find all rows with null or empty values in critical columns", "action_type": "automatic", "estimated_time": "5 minutes"},
                    {"action": "Categorize missing data by type", "priority": 2, "details": "Group missing data by field name to prioritize cleanup", "action_type": "manual", "estimated_time": "15 minutes"},
                    {"action": "Populate missing values or mark as intentionally blank", "priority": 3, "details": "Update affected rows with correct data", "action_type": "manual", "estimated_time": "30-60 minutes"},
                ],
            },
            {
                "id": "SUG-20240115-002",
                "title": "Fix Data Format Issues",
                "description": "Detected 28 format-related errors. Inconsistent formatting can cause parsing failures and data quality issues.",
                "priority": "HIGH",
                "priority_value": 2,
                "effort": "moderate",
                "impact_score": 72.3,
                "frequency": 28,
                "affected_groups": ["NA", "NP"],
                "affected_fields": ["date fields", "formatted fields"],
                "can_batch_fix": True,
                "estimated_time": "30 minutes - 1 hour",
                "actionable_steps": [
                    {"action": "Identify non-standard date formats", "priority": 1, "details": "Dates should be in YYYY-MM-DD format for consistent parsing", "action_type": "automatic", "estimated_time": "5 minutes"},
                    {"action": "Standardize date entries", "priority": 2, "details": "Convert all dates to the standard format", "action_type": "manual", "estimated_time": "20-30 minutes"},
                ],
            },
            {
                "id": "SUG-20240115-003",
                "title": "Clean Up Null/Empty Values",
                "description": "Found 18 null or empty values in data fields. These may indicate incomplete data entry or processing issues.",
                "priority": "MEDIUM",
                "priority_value": 3,
                "effort": "quick_fix",
                "impact_score": 55.0,
                "frequency": 18,
                "affected_groups": ["NF", "NM"],
                "affected_fields": ["data fields"],
                "can_batch_fix": True,
                "estimated_time": "15-30 minutes",
                "actionable_steps": [
                    {"action": "Export affected rows to spreadsheet", "priority": 1, "details": "Create a list of rows with null values for review", "action_type": "automatic", "estimated_time": "2 minutes"},
                    {"action": "Review and correct null values", "priority": 2, "details": "Update cells with appropriate values", "action_type": "manual", "estimated_time": "10-20 minutes"},
                ],
            },
            {
                "id": "SUG-20240115-004",
                "title": "Clean Up Group-Specific Issues",
                "description": "Group 'NP' has 12 errors, which is 2.3x the average. This group may need focused attention.",
                "priority": "MEDIUM",
                "priority_value": 3,
                "effort": "moderate",
                "impact_score": 48.5,
                "frequency": 12,
                "affected_groups": ["NP"],
                "affected_fields": [],
                "can_batch_fix": True,
                "estimated_time": "1-2 hours",
                "actionable_steps": [
                    {"action": "Review group configuration", "priority": 1, "details": "Check settings for group NP", "action_type": "manual", "estimated_time": "15 minutes"},
                    {"action": "Audit data quality in affected group", "priority": 2, "details": "Review sample of rows for common issues", "action_type": "manual", "estimated_time": "30 minutes"},
                ],
            },
            {
                "id": "SUG-20240115-005",
                "title": "Address Repeated Errors",
                "description": "Found 8 repeated errors. These recurring issues should be investigated for root cause.",
                "priority": "LOW",
                "priority_value": 4,
                "effort": "moderate",
                "impact_score": 32.0,
                "frequency": 8,
                "affected_groups": ["NH"],
                "affected_fields": [],
                "can_batch_fix": False,
                "estimated_time": "30 minutes - 1 hour",
                "actionable_steps": [
                    {"action": "Analyze error patterns", "priority": 1, "details": "Review sample errors to identify commonalities", "action_type": "manual", "estimated_time": "15 minutes"},
                    {"action": "Identify root cause", "priority": 2, "details": "Determine why errors are recurring", "action_type": "manual", "estimated_time": "15-30 minutes"},
                ],
            },
        ],
        "summary": {
            "total_count": 5,
            "by_priority": {
                "CRITICAL": 1,
                "HIGH": 1,
                "MEDIUM": 2,
                "LOW": 1,
            },
            "by_effort": {
                "quick_fix": 1,
                "moderate": 4,
                "significant": 0,
                "major": 0,
            },
            "total_affected_errors": 111,
        },
        "top_suggestions": None,  # Will use suggestions list
        "quick_wins": None,  # Will be filtered from suggestions
        "generated_at": datetime.now().isoformat(),
    }


def create_suggestions_priority_legend() -> "Table":
    """Create a legend explaining priority levels and effort indicators."""
    styles = getSampleStyleSheet()

    legend_data = [[
        Paragraph('<font color="#DC3545">\u25cf</font> <b>Critical</b>', styles['Normal']),
        Paragraph('<font color="#FD7E14">\u25cf</font> <b>High</b>', styles['Normal']),
        Paragraph('<font color="#FFC107">\u25cf</font> <b>Medium</b>', styles['Normal']),
        Paragraph('<font color="#17A2B8">\u25cf</font> <b>Low</b>', styles['Normal']),
    ]]

    table = Table(legend_data, colWidths=[40*mm, 40*mm, 40*mm, 40*mm])
    table.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
    ]))

    return table


def create_suggestions_summary_table(summary_data: Dict[str, Dict[str, int]]) -> "Table":
    """Create a summary table showing counts by priority and effort.

    Args:
        summary_data: Dictionary with by_priority and by_effort counts

    Returns:
        ReportLab Table object
    """
    styles = getSampleStyleSheet()

    # Build the summary table data
    table_data = [
        [
            Paragraph("<b>Priority</b>", styles['Normal']),
            Paragraph("<b>Count</b>", styles['Normal']),
            Paragraph("<b>Effort Level</b>", styles['Normal']),
            Paragraph("<b>Count</b>", styles['Normal']),
        ]
    ]

    # Get priority and effort data
    priority_data = summary_data.get("by_priority", {})
    effort_data = summary_data.get("by_effort", {})

    # Order priorities
    priority_order = ["CRITICAL", "HIGH", "MEDIUM", "LOW"]
    effort_order = ["quick_fix", "moderate", "significant", "major"]

    # Fill in rows
    max_rows = max(len(priority_order), len(effort_order))

    for i in range(max_rows):
        row = []

        # Priority column
        if i < len(priority_order):
            priority = priority_order[i]
            count = priority_data.get(priority, 0)
            priority_color = SUGGESTION_PRIORITY_COLORS.get(priority, colors.grey)
            row.append(Paragraph(f'<font color="{priority_color.hexval()}">\u25cf</font> {priority.title()}', styles['Normal']))
            row.append(str(count))
        else:
            row.extend(["", ""])

        # Effort column
        if i < len(effort_order):
            effort = effort_order[i]
            count = effort_data.get(effort, 0)
            effort_color = SUGGESTION_EFFORT_COLORS.get(effort, colors.grey)
            effort_label = SUGGESTION_EFFORT_LABELS.get(effort, effort.replace("_", " ").title())
            row.append(Paragraph(f'<font color="{effort_color.hexval()}">\u25cf</font> {effort_label}', styles['Normal']))
            row.append(str(count))
        else:
            row.extend(["", ""])

        table_data.append(row)

    # Create table
    table = Table(table_data, colWidths=[35*mm, 15*mm, 60*mm, 15*mm])
    table.setStyle(TableStyle([
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#E8E8E8")),
        ('ALIGN', (1, 0), (1, -1), 'CENTER'),
        ('ALIGN', (3, 0), (3, -1), 'CENTER'),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 9),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
    ]))

    return table


def create_impact_indicator(impact_score: int, width: int = 40, height: int = 10) -> "Drawing":
    """Create a visual impact score bar indicator.

    Args:
        impact_score: Score from 0-100
        width: Width of the bar in points
        height: Height of the bar in points

    Returns:
        ReportLab Drawing object
    """
    drawing = Drawing(width + 30, height + 4)

    # Background bar (gray)
    drawing.add(Rect(0, 2, width, height, fillColor=colors.HexColor("#E8E8E8"), strokeColor=None))

    # Determine color based on score
    if impact_score >= 70:
        bar_color = colors.HexColor("#DC3545")  # Red for high impact
    elif impact_score >= 40:
        bar_color = colors.HexColor("#FFC107")  # Yellow for medium impact
    else:
        bar_color = colors.HexColor("#28A745")  # Green for low impact

    # Filled portion
    fill_width = (impact_score / 100) * width
    drawing.add(Rect(0, 2, fill_width, height, fillColor=bar_color, strokeColor=None))

    # Score text
    drawing.add(String(width + 3, 3, f"{impact_score:.0f}", fontSize=8, fillColor=colors.black))

    return drawing


def create_suggestions_details_table(suggestions: List[Dict[str, Any]], max_rows: int = 10) -> Union["Table", "Paragraph"]:
    """Create a detailed table showing individual cleanup suggestions.

    Args:
        suggestions: List of suggestion dictionaries
        max_rows: Maximum number of rows to display

    Returns:
        ReportLab Table object or Paragraph if no data
    """
    styles = getSampleStyleSheet()

    if not suggestions:
        return Paragraph("<i>No cleanup suggestions available.</i>", styles['Normal'])

    # Sort by priority value (lower = higher priority), then by impact score
    sorted_suggestions = sorted(
        suggestions,
        key=lambda x: (x.get("priority_value", 4), -x.get("impact_score", 0))
    )

    # Limit to max_rows
    display_suggestions = sorted_suggestions[:max_rows]
    truncated = len(sorted_suggestions) > max_rows

    # Create header row
    table_data = [[
        Paragraph("<b>Pri.</b>", styles['Normal']),
        Paragraph("<b>Suggestion</b>", styles['Normal']),
        Paragraph("<b>Impact</b>", styles['Normal']),
        Paragraph("<b>Effort</b>", styles['Normal']),
        Paragraph("<b>Groups</b>", styles['Normal']),
        Paragraph("<b>Batch</b>", styles['Normal']),
    ]]

    # Add data rows
    for suggestion in display_suggestions:
        priority = suggestion.get("priority", "MEDIUM")
        priority_color = SUGGESTION_PRIORITY_COLORS.get(priority, colors.grey)
        priority_initial = priority[0]  # C, H, M, or L

        # Title (truncated if needed)
        title = suggestion.get("title", "")[:35]
        if len(suggestion.get("title", "")) > 35:
            title += "..."

        # Impact score
        impact_score = suggestion.get("impact_score", 0)

        # Effort
        effort = suggestion.get("effort", "moderate")
        effort_color = SUGGESTION_EFFORT_COLORS.get(effort, colors.grey)
        effort_short = effort.replace("_", " ").title()[:10]

        # Groups
        groups = suggestion.get("affected_groups", [])
        groups_str = ", ".join(groups[:3])
        if len(groups) > 3:
            groups_str += f" +{len(groups) - 3}"

        # Batch fix indicator
        can_batch = suggestion.get("can_batch_fix", False)
        batch_str = "\u2714" if can_batch else "-"
        batch_color = "#28A745" if can_batch else "#6C757D"

        row = [
            Paragraph(f'<font color="{priority_color.hexval()}"><b>{priority_initial}</b></font>', styles['Normal']),
            Paragraph(title, styles['Normal']),
            create_impact_indicator(impact_score),
            Paragraph(f'<font color="{effort_color.hexval()}">{effort_short}</font>', styles['Normal']),
            Paragraph(groups_str, styles['Normal']),
            Paragraph(f'<font color="{batch_color}">{batch_str}</font>', styles['Normal']),
        ]
        table_data.append(row)

    # Add truncation notice if needed
    if truncated:
        remaining = len(sorted_suggestions) - max_rows
        table_data.append([
            "",
            Paragraph(f"<i>... and {remaining} more suggestion(s)</i>", styles['Normal']),
            "", "", "", ""
        ])

    # Create table with appropriate column widths
    col_widths = [12*mm, 55*mm, 25*mm, 25*mm, 28*mm, 15*mm]

    # Style the table
    style_commands = [
        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#E8E8E8")),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('ALIGN', (0, 0), (0, -1), 'CENTER'),
        ('ALIGN', (5, 0), (5, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('TOPPADDING', (0, 0), (-1, -1), 4),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
        ('LEFTPADDING', (0, 0), (-1, -1), 3),
        ('RIGHTPADDING', (0, 0), (-1, -1), 3),
    ]

    # Add alternating row colors and priority highlighting
    for i in range(1, len(table_data)):
        if i % 2 == 0:
            style_commands.append(('BACKGROUND', (0, i), (-1, i), colors.HexColor("#F8F8F8")))

        # Highlight critical and high priority rows
        if i <= len(display_suggestions):
            suggestion = display_suggestions[i - 1]
            priority = suggestion.get("priority", "MEDIUM")
            if priority == "CRITICAL":
                style_commands.append(('BACKGROUND', (0, i), (0, i), colors.HexColor("#FFEBEE")))
            elif priority == "HIGH":
                style_commands.append(('BACKGROUND', (0, i), (0, i), colors.HexColor("#FFF3E0")))

    # Use landscape-aware table for automatic landscape handling
    # This table has 6 columns which meets the threshold for landscape mode
    pagination_config = PaginationConfig(
        repeat_header_rows=1,
        min_rows_per_page=3,
        avoid_orphans=True,
        orphan_threshold=2
    )
    flowables = create_landscape_aware_table(
        data=table_data,
        col_widths=col_widths,
        style_commands=style_commands,
        section_name="Cleanup Suggestions",
        pagination_config=pagination_config,
        return_flowables=True
    )

    return flowables


def create_actionable_steps_section(suggestions: List[Dict[str, Any]], max_suggestions: int = 3) -> List["Flowable"]:
    """Create a section showing actionable steps for top suggestions.

    Args:
        suggestions: List of suggestion dictionaries
        max_suggestions: Maximum number of suggestions to show steps for

    Returns:
        List of ReportLab elements
    """
    styles = getSampleStyleSheet()
    elements = []

    if not suggestions:
        return elements

    # Get top suggestions by priority and impact
    sorted_suggestions = sorted(
        suggestions,
        key=lambda x: (x.get("priority_value", 4), -x.get("impact_score", 0))
    )[:max_suggestions]

    for suggestion in sorted_suggestions:
        title = suggestion.get("title", "Unknown")
        priority = suggestion.get("priority", "MEDIUM")
        priority_color = SUGGESTION_PRIORITY_COLORS.get(priority, colors.grey)
        steps = suggestion.get("actionable_steps", [])
        estimated_time = suggestion.get("estimated_time", "Unknown")

        if not steps:
            continue

        # Suggestion header with colored priority indicator
        header_text = f'<font color="{priority_color.hexval()}">\u25cf</font> <b>{title}</b> (Est. time: {estimated_time})'
        elements.append(Paragraph(header_text, styles['Normal']))
        elements.append(Spacer(1, 2*mm))

        # Create steps table
        steps_data = []
        for i, step in enumerate(steps[:5], 1):  # Limit to 5 steps
            action = step.get("action", "")[:60]
            if len(step.get("action", "")) > 60:
                action += "..."
            step_time = step.get("estimated_time", "")
            action_type = step.get("action_type", "manual")

            # Type indicator
            type_icon = "\u2699" if action_type == "automatic" else "\u270B" if action_type == "manual" else "\u26A0"

            steps_data.append([
                f"{i}.",
                Paragraph(f"{type_icon} {action}", styles['Normal']),
                step_time,
            ])

        if steps_data:
            steps_table = Table(steps_data, colWidths=[8*mm, 120*mm, 25*mm])
            steps_table.setStyle(TableStyle([
                ('FONTSIZE', (0, 0), (-1, -1), 8),
                ('VALIGN', (0, 0), (-1, -1), 'TOP'),
                ('TOPPADDING', (0, 0), (-1, -1), 2),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 2),
                ('TEXTCOLOR', (2, 0), (2, -1), colors.HexColor("#6C757D")),
            ]))
            elements.append(steps_table)

        elements.append(Spacer(1, 4*mm))

    return elements


def add_cleanup_suggestions_section(story: List["Flowable"], suggestions_data: Optional[Dict[str, Any]] = None, show_sample: bool = False) -> None:
    """Add the cleanup suggestions section to the PDF story.

    Displays a prioritized list of suggested cleanup actions with impact assessment.
    Suggestions are presented in a clear, actionable format to help users focus
    on the most important data quality improvements.

    Args:
        story: The ReportLab story list to append content to
        suggestions_data: Dictionary containing suggestions from CleanupSuggestionGenerator.export_for_report().
                         If None and show_sample is True, sample data will be generated.
        show_sample: If True and no data provided, generate sample data for demonstration
    """
    styles = getSampleStyleSheet()
    subheading_style = styles['Heading2']
    normal_style = styles['Normal']

    # Add page break before section
    story.append(PageBreak())

    # Add section marker for page header navigation
    story.append(SectionMarker("Cleanup Suggestions"))

    # Section header with colored background (blue/teal to differentiate from error report)
    header_data = [["Cleanup Suggestions"]]
    header_table = Table(header_data, colWidths=[160*mm], rowHeights=[12*mm])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#0066CC")),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 16),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 5*mm))

    # Description text
    story.append(Paragraph(
        "This section provides prioritized cleanup suggestions based on error pattern analysis. "
        "Suggestions are ranked by impact and effort to help you focus on the most valuable improvements first.",
        normal_style
    ))
    story.append(Spacer(1, 5*mm))

    # Get suggestions data (use sample if needed)
    if suggestions_data is None:
        if show_sample:
            suggestions_data = generate_sample_cleanup_suggestions()
            story.append(Paragraph(
                "<i>Sample data shown for demonstration. In production, this section will display actual cleanup suggestions based on detected errors.</i>",
                normal_style
            ))
            story.append(Spacer(1, 5*mm))
        else:
            # Try to get real data from the global suggestion generator
            try:
                from cleanup_suggestions_generator import get_global_suggestion_generator
                generator = get_global_suggestion_generator()
                suggestions_data = generator.export_for_report()
            except Exception as e:
                logger.warning(f"Could not load cleanup suggestions: {e}")
                suggestions_data = {"suggestions": [], "summary": {"total_count": 0, "by_priority": {}, "by_effort": {}, "total_affected_errors": 0}}

    suggestions = suggestions_data.get("suggestions", [])
    summary = suggestions_data.get("summary", {})

    if not suggestions:
        # No suggestions to report - show success message
        success_data = [[
            Paragraph(
                '<font color="#28A745">\u2714</font> <b>No cleanup suggestions needed</b>',
                styles['Normal']
            )
        ]]
        success_table = Table(success_data, colWidths=[160*mm])
        success_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#D4EDDA")),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ]))
        story.append(success_table)
        story.append(Spacer(1, 5*mm))
        story.append(Paragraph(
            "Data quality is good! No significant cleanup actions are recommended at this time.",
            normal_style
        ))
        return

    # Overview box with summary
    total_count = summary.get("total_count", len(suggestions))
    total_affected = summary.get("total_affected_errors", sum(s.get("frequency", 0) for s in suggestions))
    critical_count = summary.get("by_priority", {}).get("CRITICAL", 0)
    high_count = summary.get("by_priority", {}).get("HIGH", 0)

    overview_text = (
        f"Found <b>{total_count}</b> cleanup suggestion(s) addressing <b>{total_affected}</b> total issues. "
    )
    if critical_count > 0:
        overview_text += f'<font color="#DC3545"><b>{critical_count} critical</b></font> '
    if high_count > 0:
        overview_text += f'and <font color="#FD7E14"><b>{high_count} high priority</b></font> items need attention.'

    # Determine box color based on priority distribution
    if critical_count > 0:
        box_color = colors.HexColor("#F8D7DA")  # Light red
        border_color = colors.HexColor("#DC3545")
    elif high_count > 0:
        box_color = colors.HexColor("#FFE5D0")  # Light orange
        border_color = colors.HexColor("#FD7E14")
    else:
        box_color = colors.HexColor("#D1ECF1")  # Light blue
        border_color = colors.HexColor("#17A2B8")

    overview_data = [[Paragraph(overview_text, normal_style)]]
    overview_table = Table(overview_data, colWidths=[158*mm])
    overview_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), box_color),
        ('BOX', (0, 0), (-1, -1), 2, border_color),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(overview_table)
    story.append(Spacer(1, 8*mm))

    # Summary by priority and effort
    story.append(Paragraph("Summary by Priority & Effort", subheading_style))
    story.append(Spacer(1, 3*mm))

    summary_table = create_suggestions_summary_table(summary)
    story.append(summary_table)
    story.append(Spacer(1, 8*mm))

    # Priority legend
    story.append(Paragraph("Priority Levels", subheading_style))
    story.append(Spacer(1, 2*mm))
    legend = create_suggestions_priority_legend()
    story.append(legend)
    story.append(Spacer(1, 5*mm))

    # Detailed suggestions table
    story.append(Paragraph("Prioritized Suggestions", subheading_style))
    story.append(Spacer(1, 3*mm))

    details_table = create_suggestions_details_table(suggestions, max_rows=10)
    story.append(details_table)
    story.append(Spacer(1, 8*mm))

    # Top actionable steps section
    story.append(Paragraph("Recommended Actions", subheading_style))
    story.append(Spacer(1, 3*mm))

    story.append(Paragraph(
        "<i>Key icons: \u2699 = Automatic/Script, \u270B = Manual, \u26A0 = Escalate</i>",
        styles['Normal']
    ))
    story.append(Spacer(1, 3*mm))

    actionable_elements = create_actionable_steps_section(suggestions, max_suggestions=3)
    for element in actionable_elements:
        story.append(element)

    # Quick wins section (if available)
    quick_wins = [s for s in suggestions if s.get("effort") in ("quick_fix", "moderate") and s.get("impact_score", 0) >= 40]
    if quick_wins:
        story.append(Spacer(1, 5*mm))
        story.append(Paragraph("Quick Wins (High Impact, Low Effort)", subheading_style))
        story.append(Spacer(1, 3*mm))

        for qw in quick_wins[:3]:
            title = qw.get("title", "")
            impact = qw.get("impact_score", 0)
            time_est = qw.get("estimated_time", "")
            groups = ", ".join(qw.get("affected_groups", [])[:3])

            qw_text = f'\u2605 <b>{title}</b> - Impact: {impact:.0f}/100, Time: {time_est}'
            if groups:
                qw_text += f' (Groups: {groups})'
            story.append(Paragraph(qw_text, normal_style))
            story.append(Spacer(1, 2*mm))


def add_missing_data_warnings_section(story: List["Flowable"], warnings_data: Optional[Dict[str, Dict[str, List[Dict[str, Any]]]]] = None, show_sample: bool = False) -> None:
    """Add the missing/incomplete data warnings section to the PDF story.

    This section displays warnings about missing or incomplete data fields,
    grouped by sheet and field for easy identification of systematic issues.

    Args:
        story: The ReportLab story list to append content to
        warnings_data: Dictionary of warnings grouped by sheet and field.
                      Format: {sheet_name: {field_name: [warnings]}}
                      If None and show_sample is True, sample data will be generated.
        show_sample: If True and no warnings provided, generate sample data for demonstration
    """
    styles = getSampleStyleSheet()
    subheading_style = styles['Heading2']
    normal_style = styles['Normal']

    # Add page break before section
    story.append(PageBreak())

    # Add section marker for page header navigation
    story.append(SectionMarker("Missing Data Warnings"))

    # Section header with colored background
    header_data = [["Missing Data Warnings"]]
    header_table = Table(header_data, colWidths=[160*mm], rowHeights=[12*mm])
    header_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#856404")),
        ('TEXTCOLOR', (0, 0), (-1, -1), colors.white),
        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('FONTNAME', (0, 0), (-1, -1), 'Helvetica-Bold'),
        ('FONTSIZE', (0, 0), (-1, -1), 16),
    ]))
    story.append(header_table)
    story.append(Spacer(1, 5*mm))

    # Description text
    story.append(Paragraph(
        "This section identifies missing or incomplete data fields across sheets, "
        "grouped for easy identification of systematic issues that may need attention.",
        normal_style
    ))
    story.append(Spacer(1, 5*mm))

    # Get warnings (use sample if needed)
    if warnings_data is None:
        if show_sample:
            warnings_data = generate_sample_missing_data_warnings()
            story.append(Paragraph(
                "<i>Sample data shown for demonstration. In production, this section will display actual missing data warnings.</i>",
                normal_style
            ))
            story.append(Spacer(1, 5*mm))
        else:
            warnings_data = {}

    if not warnings_data:
        # No warnings to report - show success message
        success_data = [[
            Paragraph(
                '<font color="#28A745">\u2714</font> <b>No missing data issues detected</b>',
                styles['Normal']
            )
        ]]
        success_table = Table(success_data, colWidths=[160*mm])
        success_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#D4EDDA")),
            ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
            ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
            ('FONTSIZE', (0, 0), (-1, -1), 12),
            ('TOPPADDING', (0, 0), (-1, -1), 10),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 10),
        ]))
        story.append(success_table)
        story.append(Spacer(1, 5*mm))
        story.append(Paragraph(
            "All required fields have been populated. No missing data warnings to report.",
            normal_style
        ))
        return

    # Count total issues for overview
    total_sheets = len(warnings_data)
    total_fields = sum(len(fields) for fields in warnings_data.values())
    total_issues = sum(
        len(warnings)
        for fields in warnings_data.values()
        for warnings in fields.values()
    )

    # Overview box
    overview_text = (
        f"Found missing data issues in <b>{total_sheets}</b> sheet(s), "
        f"affecting <b>{total_fields}</b> field(s), "
        f"with <b>{total_issues}</b> total issue(s)."
    )

    overview_data = [[Paragraph(overview_text, normal_style)]]
    overview_table = Table(overview_data, colWidths=[158*mm])
    overview_table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor("#FFF3CD")),
        ('BOX', (0, 0), (-1, -1), 2, colors.HexColor("#856404")),
        ('TOPPADDING', (0, 0), (-1, -1), 8),
        ('BOTTOMPADDING', (0, 0), (-1, -1), 8),
        ('LEFTPADDING', (0, 0), (-1, -1), 10),
        ('RIGHTPADDING', (0, 0), (-1, -1), 10),
    ]))
    story.append(overview_table)
    story.append(Spacer(1, 8*mm))

    # Summary by sheet
    story.append(Paragraph("Summary by Sheet", subheading_style))
    story.append(Spacer(1, 3*mm))

    summary_table = create_missing_data_summary_table(warnings_data)
    story.append(summary_table)
    story.append(Spacer(1, 8*mm))

    # Add legend
    legend_data = [[
        Paragraph('<font color="#DC3545">\u25cf</font> Error = Critical issue', styles['Normal']),
        Paragraph('<font color="#FFC107">\u25cf</font> Warning = Needs attention', styles['Normal']),
        Paragraph('<font color="#17A2B8">\u25cf</font> Info = For review', styles['Normal']),
    ]]
    legend_table = Table(legend_data, colWidths=[50*mm, 55*mm, 50*mm])
    legend_table.setStyle(TableStyle([
        ('FONTSIZE', (0, 0), (-1, -1), 8),
        ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
        ('ALIGN', (0, 0), (-1, -1), 'LEFT'),
    ]))
    story.append(legend_table)
    story.append(Spacer(1, 5*mm))

    # Detailed breakdown by sheet and field
    story.append(Paragraph("Detailed Breakdown by Sheet & Field", subheading_style))
    story.append(Spacer(1, 3*mm))

    detail_elements = create_missing_data_details_table(warnings_data)
    for elem in detail_elements:
        story.append(elem)

    story.append(Spacer(1, 8*mm))

    # Recommendations section
    story.append(Paragraph("Recommended Actions", subheading_style))
    story.append(Spacer(1, 3*mm))

    # Group recommendations by severity
    has_errors = any(
        any(w.get("severity") == "error" for w in warnings)
        for fields in warnings_data.values()
        for warnings in fields.values()
    )
    has_warnings = any(
        any(w.get("severity") == "warning" for w in warnings)
        for fields in warnings_data.values()
        for warnings in fields.values()
    )
    has_info = any(
        any(w.get("severity") == "info" for w in warnings)
        for fields in warnings_data.values()
        for warnings in fields.values()
    )

    if has_errors:
        story.append(Paragraph(
            '<font color="#DC3545">\u25cf</font> <b>High Priority:</b> '
            'Address error-level issues immediately as they may indicate data corruption or invalid entries.',
            normal_style
        ))
    if has_warnings:
        story.append(Paragraph(
            '<font color="#FFC107">\u25cf</font> <b>Medium Priority:</b> '
            'Review warning-level issues during regular data maintenance. '
            'Consider adding default values or making fields required.',
            normal_style
        ))
    if has_info:
        story.append(Paragraph(
            '<font color="#17A2B8">\u25cf</font> <b>Low Priority:</b> '
            'Informational items can be reviewed periodically. '
            'These may indicate optional fields that are commonly left empty.',
            normal_style
        ))

    # Systematic issue notice if many issues in same field
    field_issue_counts = {}
    for sheet_name, fields in warnings_data.items():
        for field_name, warnings in fields.items():
            if field_name not in field_issue_counts:
                field_issue_counts[field_name] = 0
            field_issue_counts[field_name] += len(warnings)

    high_issue_fields = [f for f, c in field_issue_counts.items() if c >= 3]
    if high_issue_fields:
        story.append(Spacer(1, 3*mm))
        story.append(Paragraph(
            f'<font color="#856404">\u26A0</font> <b>Systematic Issue Detected:</b> '
            f'Fields with recurring issues across rows: {", ".join(high_issue_fields)}. '
            f'Consider reviewing data entry processes or sheet configuration.',
            normal_style
        ))


def add_special_activities_section(story: List["Flowable"], start_date: date, end_date: date) -> None:
    """Adds the special activities summary and chart to the report story, using the correct date range."""

    # Get all special activities for the report's date range
    user_activity_data, total_activities, total_hours = get_special_activities(start_date, end_date)

    if not user_activity_data:
        logger.info("No special activities found for the period. Skipping section.")
        return

    story.append(PageBreak())

    # Add section marker for page header navigation
    story.append(SectionMarker("Special Activities"))

    story.append(Paragraph("4. Special Activities", styles['h2']))
    story.append(Spacer(1, 5*mm))
    
    # --- NEW: Process the data for the pie chart ---
    # The get_special_activities function returns data per-user. We need to aggregate it by category.
    category_hours_agg = defaultdict(float)
    for user_data in user_activity_data.values():
        for category, hours in user_data.get("categories", {}).items():
            category_hours_agg[category] += hours
    
    # Sort the aggregated categories by hours
    sorted_category_hours = sorted(category_hours_agg.items(), key=lambda x: x[1], reverse=True)
    # --- END NEW ---

    # Add summary text
    story.append(Paragraph(
        f"Overview of special activities from {start_date.strftime('%b %d')} to {end_date.strftime('%b %d')}. Total hours: {total_hours:.1f}",
        normal_style
    ))
    story.append(Spacer(1, 5*mm))
    
    # Create and add pie chart using the correctly aggregated and sorted data
    pie_chart = create_activities_pie_chart(sorted_category_hours, total_hours)
    story.append(pie_chart)

    # --- NEW: Hours Distribution Bar Chart ---
    # Calculate hours distribution metrics and create visualization
    try:
        distribution_summary = calculate_hours_distribution(
            user_activity_data, start_date, end_date
        )
        viz_data = get_distribution_visualization_data(distribution_summary)

        story.append(Spacer(1, 10*mm))
        story.append(Paragraph("Hours per Item by Category", subheading_style))
        story.append(Spacer(1, 3*mm))

        # Create and add the hours distribution bar chart
        hours_chart = make_hours_distribution_chart(
            viz_data,
            title="Hours Distribution by Category",
            width=500,
            height=280,
            show_labels=True,
            show_average_line=True,
            color_by_efficiency=True
        )
        story.append(hours_chart)

        # Add efficiency legend
        story.append(Spacer(1, 3*mm))
        legend_text = (
            '<font size="8" color="#555555">'
            '<b>Color Legend:</b> '
            '<font color="#1B5E20">■</font> Highly Efficient | '
            '<font color="#43A047">■</font> Efficient | '
            '<font color="#8B6914">■</font> Average | '
            '<font color="#C65102">■</font> Inefficient | '
            '<font color="#B71C1C">■</font> Highly Inefficient'
            '</font>'
        )
        story.append(Paragraph(legend_text, normal_style))

    except Exception as e:
        logger.warning(f"Could not create hours distribution chart: {e}")
    # --- END: Hours Distribution Bar Chart ---

    
    # Add table with details
    story.append(Spacer(1, 10*mm))
    story.append(Paragraph("Detailed Breakdown", subheading_style))
    
    # Create and add breakdown table
    breakdown_table = create_special_activities_breakdown(sorted_category_hours, total_hours)
    story.append(breakdown_table)

    # --- Period-over-Period Comparison Section ---
    # Calculate previous period dates (same duration as current period)
    period_duration = (end_date - start_date).days + 1
    previous_end = start_date - timedelta(days=1)
    previous_start = previous_end - timedelta(days=period_duration - 1)

    # Fetch previous period data
    try:
        previous_activity_data, prev_activities, prev_hours = get_special_activities(previous_start, previous_end)

        if previous_activity_data:
            # Create comparison object
            comparison = compare_special_activities_periods(
                current_data=user_activity_data,
                previous_data=previous_activity_data,
                current_start=start_date,
                current_end=end_date,
                previous_start=previous_start,
                previous_end=previous_end,
            )

            # Add comparison section to story
            story.append(Spacer(1, 10*mm))
            comparison_flowables = create_special_activities_comparison_section(comparison)
            story.extend(comparison_flowables)

            logger.info(
                f"Added comparison section: {comparison.hours_change:+.1f}h "
                f"({format_percent_change(comparison.hours_percent_change)})"
            )
        else:
            logger.info("No previous period data available for comparison.")
    except Exception as e:
        logger.warning(f"Could not create comparison section: {e}")


def create_items_per_period_metrics_card(
    user_metrics: UserAverageItems,
    team_avg_per_day: float,
    team_avg_per_week: float,
    width: int = 160,
    height: int = 80,
) -> "Drawing":
    """Create a metrics card showing items per day/week with team comparison.

    Displays prominently:
    - Items per day with comparison to team average
    - Items per week with comparison to team average
    - Performance level indicator
    - Visual comparison (above/below/at team average)

    Args:
        user_metrics: UserAverageItems object with user's metrics
        team_avg_per_day: Team average items per day
        team_avg_per_week: Team average items per week
        width: Card width in points
        height: Card height in points

    Returns:
        Drawing object containing the metrics card
    """
    drawing = Drawing(width * mm, height)

    # Performance level colors
    performance_colors = {
        PerformanceLevel.EXCEPTIONAL: colors.HexColor("#1B5E20"),  # Dark green
        PerformanceLevel.HIGH: colors.HexColor("#388E3C"),  # Green
        PerformanceLevel.AVERAGE: colors.HexColor("#8B6914"),  # Gold
        PerformanceLevel.LOW: colors.HexColor("#E65100"),  # Orange
        PerformanceLevel.MINIMAL: colors.HexColor("#B71C1C"),  # Red
    }

    # Background card
    bg_color = colors.HexColor("#f8f9fa")
    border_color = colors.HexColor("#dee2e6")
    drawing.add(Rect(0, 0, width * mm, height, fillColor=bg_color, strokeColor=border_color, strokeWidth=0.5))

    # Calculate comparison indicators
    day_diff = user_metrics.items_per_day - team_avg_per_day
    week_diff = user_metrics.items_per_week - team_avg_per_week

    day_ratio = (user_metrics.items_per_day / team_avg_per_day * 100) if team_avg_per_day > 0 else 100
    week_ratio = (user_metrics.items_per_week / team_avg_per_week * 100) if team_avg_per_week > 0 else 100

    # Comparison colors
    above_color = colors.HexColor("#2e7d32")  # Green for above average
    below_color = colors.HexColor("#c62828")  # Red for below average
    equal_color = colors.HexColor("#757575")  # Gray for equal

    # Determine comparison color for day
    if day_diff > 0.5:
        day_comparison_color = above_color
        day_indicator = "▲"
    elif day_diff < -0.5:
        day_comparison_color = below_color
        day_indicator = "▼"
    else:
        day_comparison_color = equal_color
        day_indicator = "●"

    # Determine comparison color for week
    if week_diff > 1:
        week_comparison_color = above_color
        week_indicator = "▲"
    elif week_diff < -1:
        week_comparison_color = below_color
        week_indicator = "▼"
    else:
        week_comparison_color = equal_color
        week_indicator = "●"

    # Section title
    title_y = height - 12
    drawing.add(String(5, title_y, "Average Items Per Period", fontName="Helvetica-Bold", fontSize=9, fillColor=colors.HexColor("#333333")))

    # Performance level badge
    perf_level = user_metrics.performance_level
    perf_color = performance_colors.get(perf_level, colors.grey)
    badge_x = width * mm - 65
    badge_y = title_y - 3
    drawing.add(Rect(badge_x, badge_y, 60, 12, fillColor=perf_color, strokeColor=None, rx=3, ry=3))
    drawing.add(String(badge_x + 5, badge_y + 3, perf_level.value.upper(), fontName="Helvetica-Bold", fontSize=7, fillColor=colors.white))

    # Dividing line
    line_y = height - 20
    drawing.add(Line(5, line_y, width * mm - 5, line_y, strokeColor=border_color, strokeWidth=0.5))

    # Items per day section (left side)
    col1_x = 10
    col1_value_y = height - 42
    col1_label_y = height - 52
    col1_compare_y = height - 64

    # Large value for items per day
    drawing.add(String(col1_x, col1_value_y, f"{user_metrics.items_per_day:.1f}", fontName="Helvetica-Bold", fontSize=18, fillColor=colors.HexColor("#1a1a1a")))
    drawing.add(String(col1_x, col1_label_y, "items/day", fontName="Helvetica", fontSize=8, fillColor=colors.HexColor("#666666")))

    # Team comparison for items per day
    day_compare_text = f"{day_indicator} {abs(day_diff):+.1f} vs team avg ({team_avg_per_day:.1f})"
    if day_diff >= 0:
        day_compare_text = f"{day_indicator} +{abs(day_diff):.1f} vs team avg ({team_avg_per_day:.1f})"
    else:
        day_compare_text = f"{day_indicator} -{abs(day_diff):.1f} vs team avg ({team_avg_per_day:.1f})"
    drawing.add(String(col1_x, col1_compare_y, day_compare_text, fontName="Helvetica", fontSize=7, fillColor=day_comparison_color))

    # Percentage badge for day
    day_pct_x = col1_x + 45
    day_pct_y = col1_value_y - 2
    pct_bg_color = above_color if day_ratio >= 100 else below_color
    drawing.add(Rect(day_pct_x, day_pct_y, 35, 14, fillColor=pct_bg_color, strokeColor=None, rx=2, ry=2))
    drawing.add(String(day_pct_x + 3, day_pct_y + 4, f"{day_ratio:.0f}%", fontName="Helvetica-Bold", fontSize=8, fillColor=colors.white))

    # Vertical divider
    divider_x = width * mm / 2
    drawing.add(Line(divider_x, height - 25, divider_x, 5, strokeColor=border_color, strokeWidth=0.5))

    # Items per week section (right side)
    col2_x = divider_x + 10
    col2_value_y = height - 42
    col2_label_y = height - 52
    col2_compare_y = height - 64

    # Large value for items per week
    drawing.add(String(col2_x, col2_value_y, f"{user_metrics.items_per_week:.1f}", fontName="Helvetica-Bold", fontSize=18, fillColor=colors.HexColor("#1a1a1a")))
    drawing.add(String(col2_x, col2_label_y, "items/week", fontName="Helvetica", fontSize=8, fillColor=colors.HexColor("#666666")))

    # Team comparison for items per week
    if week_diff >= 0:
        week_compare_text = f"{week_indicator} +{abs(week_diff):.1f} vs team avg ({team_avg_per_week:.1f})"
    else:
        week_compare_text = f"{week_indicator} -{abs(week_diff):.1f} vs team avg ({team_avg_per_week:.1f})"
    drawing.add(String(col2_x, col2_compare_y, week_compare_text, fontName="Helvetica", fontSize=7, fillColor=week_comparison_color))

    # Percentage badge for week
    week_pct_x = col2_x + 45
    week_pct_y = col2_value_y - 2
    pct_bg_color = above_color if week_ratio >= 100 else below_color
    drawing.add(Rect(week_pct_x, week_pct_y, 35, 14, fillColor=pct_bg_color, strokeColor=None, rx=2, ry=2))
    drawing.add(String(week_pct_x + 3, week_pct_y + 4, f"{week_ratio:.0f}%", fontName="Helvetica-Bold", fontSize=8, fillColor=colors.white))

    # Active days info at bottom
    active_info_y = 5
    active_text = f"Active: {user_metrics.active_days} days, {user_metrics.active_weeks} weeks"
    drawing.add(String(col1_x, active_info_y, active_text, fontName="Helvetica", fontSize=7, fillColor=colors.HexColor("#888888")))

    return drawing


def add_user_details_section(story: List["Flowable"], metrics: Dict[str, Any], changes: Optional[List[Dict[str, Any]]] = None) -> None:
    """Add a section showing details for each active user."""
    styles = getSampleStyleSheet()
    heading_style = styles['Heading1']
    subheading_style = styles['Heading2']
    normal_style = styles['Normal']
    
    # Add page break before user details section
    story.append(PageBreak())

    # Add section marker for page header navigation
    story.append(SectionMarker("User Activity"))

    # Add section header
    story.append(Paragraph("User Activity Analysis", heading_style))
    story.append(Spacer(1, 5*mm))
    
    # Add explanation
    story.append(Paragraph(
        "Detailed breakdown of activity by user across product groups in the last 30 days.",
        normal_style
    ))
    story.append(Spacer(1, 10*mm))

    # Add team contribution overview section with pie chart and percentage bars
    if changes:
        contribution_summary = get_contribution_summary(changes, top_n=5)

        if contribution_summary.total_team_activity > 0:
            # Add contribution pie chart
            story.append(Paragraph("Team Contribution Distribution", subheading_style))
            story.append(Spacer(1, 5*mm))

            pie_chart = create_user_contribution_pie_chart(contribution_summary)
            story.append(pie_chart)
            story.append(Spacer(1, 10*mm))

            # Add contribution percentage bars
            bars_chart = create_user_contribution_bars(contribution_summary)
            story.append(bars_chart)
            story.append(Spacer(1, 8*mm))

            # Add top contributors table with highlighting
            if contribution_summary.top_contributors:
                story.append(Paragraph("Top Contributors", styles['Heading3']))
                story.append(Spacer(1, 3*mm))

                table_data = create_top_contributors_highlight(contribution_summary, top_n=5)
                if table_data:
                    top_table = Table(table_data, colWidths=[30, 60, 50, 55, 40])
                    top_table.setStyle(TableStyle([
                        ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor("#1565C0")),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, -1), 9),
                        ('ALIGN', (0, 0), (0, -1), 'CENTER'),
                        ('ALIGN', (2, 0), (-1, -1), 'CENTER'),
                        # Highlight top 3 rows
                        ('BACKGROUND', (0, 1), (-1, 1), colors.HexColor("#FFF8E1")),  # Gold tint
                        ('BACKGROUND', (0, 2), (-1, 2), colors.HexColor("#ECEFF1")),  # Silver tint
                        ('BACKGROUND', (0, 3), (-1, 3), colors.HexColor("#EFEBE9")),  # Bronze tint
                    ]))
                    story.append(top_table)

            story.append(Spacer(1, 10*mm))
            story.append(PageBreak())

    # Get active users (excluding empty user names)
    active_users = {user: count for user, count in metrics["users"].items() if user and user.strip()}
    
    if not active_users:
        story.append(Paragraph("No active users found in this period.", normal_style))
        return
    
    # Sort users by activity count
    sorted_users = sorted(active_users.items(), key=lambda x: x[1], reverse=True)

    # Calculate team averages for comparison
    team_avg_items_per_day = 0.0
    team_avg_items_per_week = 0.0
    user_avg_metrics: Dict[str, UserAverageItems] = {}

    if changes:
        # Calculate average items per user for the period
        user_avg_metrics = calculate_average_items_all_users(
            changes,
            include_weekly_breakdown=False,
            include_group_breakdown=False
        )
        if user_avg_metrics:
            total_items_per_day = sum(m.items_per_day for m in user_avg_metrics.values())
            total_items_per_week = sum(m.items_per_week for m in user_avg_metrics.values())
            num_users = len(user_avg_metrics)
            team_avg_items_per_day = total_items_per_day / num_users if num_users > 0 else 0.0
            team_avg_items_per_week = total_items_per_week / num_users if num_users > 0 else 0.0

    # Process each user
    for i, (user, count) in enumerate(sorted_users):
        # Add page break between users (but not before the first one)
        if i > 0:
            story.append(PageBreak())

        # Normalize user name for PDF display
        safe_user = prepare_for_pdf(user)

        # Create colored header for user
        user_color = USER_COLORS.get(user, colors.steelblue)
        user_header_data = [[f"User: {safe_user}"]]
        user_header = Table(user_header_data, colWidths=[150*mm], rowHeights=[10*mm])
        user_header.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), user_color),
            ('TEXTCOLOR', (0,0), (-1,-1), colors.white),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 14),
        ]))
        story.append(user_header)
        story.append(Spacer(1, 5*mm))
        
        # Add user activity summary
        story.append(Paragraph(f"Total changes: {count}", normal_style))
        story.append(Spacer(1, 3*mm))

        # Add average items per period metrics (prominently displayed)
        user_metrics = user_avg_metrics.get(user)
        if user_metrics:
            items_per_day = user_metrics.items_per_day
            items_per_week = user_metrics.items_per_week

            # Create a styled table for prominent display of metrics
            metrics_data = [
                ["Metric", "User", "Team Avg", "Comparison"]
            ]

            # Items per day comparison
            day_diff = items_per_day - team_avg_items_per_day
            day_pct = (day_diff / team_avg_items_per_day * 100) if team_avg_items_per_day > 0 else 0.0
            day_indicator = "↑" if day_diff > 0 else "↓" if day_diff < 0 else "="
            day_comparison = f"{day_indicator} {abs(day_pct):.1f}%" if team_avg_items_per_day > 0 else "N/A"

            # Items per week comparison
            week_diff = items_per_week - team_avg_items_per_week
            week_pct = (week_diff / team_avg_items_per_week * 100) if team_avg_items_per_week > 0 else 0.0
            week_indicator = "↑" if week_diff > 0 else "↓" if week_diff < 0 else "="
            week_comparison = f"{week_indicator} {abs(week_pct):.1f}%" if team_avg_items_per_week > 0 else "N/A"

            metrics_data.append([
                "Items/Day",
                f"{items_per_day:.1f}",
                f"{team_avg_items_per_day:.1f}",
                day_comparison
            ])
            metrics_data.append([
                "Items/Week",
                f"{items_per_week:.1f}",
                f"{team_avg_items_per_week:.1f}",
                week_comparison
            ])

            # Create the metrics table
            metrics_table = Table(metrics_data, colWidths=[35*mm, 25*mm, 25*mm, 30*mm])

            # Style the table with colors based on comparison
            table_style = [
                # Header row
                ('BACKGROUND', (0, 0), (-1, 0), colors.Color(0.2, 0.4, 0.6)),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.white),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 10),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('VALIGN', (0, 0), (-1, -1), 'MIDDLE'),
                # Data rows
                ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
                ('FONTSIZE', (0, 1), (-1, -1), 10),
                ('GRID', (0, 0), (-1, -1), 0.5, colors.grey),
                ('ROWHEIGHTS', (0, 0), (-1, -1), 8*mm),
            ]

            # Add color coding for comparison column
            if day_diff > 0:
                table_style.append(('BACKGROUND', (3, 1), (3, 1), colors.Color(0.85, 0.95, 0.85)))  # Light green
            elif day_diff < 0:
                table_style.append(('BACKGROUND', (3, 1), (3, 1), colors.Color(0.95, 0.85, 0.85)))  # Light red

            if week_diff > 0:
                table_style.append(('BACKGROUND', (3, 2), (3, 2), colors.Color(0.85, 0.95, 0.85)))  # Light green
            elif week_diff < 0:
                table_style.append(('BACKGROUND', (3, 2), (3, 2), colors.Color(0.95, 0.85, 0.85)))  # Light red

            metrics_table.setStyle(TableStyle(table_style))
            story.append(metrics_table)
            story.append(Spacer(1, 5*mm))

        # Create user's activity by group stacked bar chart
        user_group_data = collect_user_group_data(metrics, user)
        if user_group_data:
            # Add product activity section
            story.append(Paragraph("Product Activity", subheading_style))
            
            chart, legend_data = make_user_detail_chart(user, user_group_data)
            story.append(chart)
            
            # Add legend for groups
            if legend_data:
                legend = create_horizontal_legend(legend_data, width=400)
                story.append(legend)
                
            story.append(Spacer(1, 10*mm))
        else:
            story.append(Paragraph("No detailed product data available for this user.", normal_style))
            story.append(Spacer(1, 5*mm))

        # Add phase distribution donut chart for this user
        if changes:
            # Calculate phase distribution for this specific user
            user_phase_dist = calculate_user_phase_distribution(changes, user)
            if user_phase_dist.total_items > 0:
                story.append(Paragraph("Phase Distribution", subheading_style))
                story.append(Spacer(1, 3*mm))

                # Create and add the phase distribution donut chart
                phase_dist_chart = create_user_phase_distribution_chart(
                    user, user_phase_dist, width=400, height=250
                )
                story.append(phase_dist_chart)
                story.append(Spacer(1, 10*mm))

        # Get user's special activities (without the section header)
        category_hours, total_hours = get_user_special_activities(user)

        if total_hours > 0:
            # Create and add pie chart for this user's activities - with smaller size
            pie_chart = create_activities_pie_chart(category_hours, total_hours, width=400, height=250)
            story.append(pie_chart)
        else:
            # Only show a small note if no activities
            story.append(Paragraph(
                f"No special activities recorded.",
                normal_style
            ))

# This is the corrected function definition.
def upload_pdf_to_smartsheet(file_path: str, row_id: int) -> None:
    """Upload a PDF report file as an attachment to a Smartsheet row.

    Attaches the generated PDF report to the specified row in the report
    metadata sheet, making it accessible from within Smartsheet. Uses
    retry logic for API resilience.

    Args:
        file_path (str): Absolute or relative path to the PDF file to upload.
        row_id (int): The row ID in REPORT_METADATA_SHEET_ID where the
            file should be attached.

    Returns:
        None

    Side Effects:
        - Uploads file to Smartsheet via API
        - Logs success/failure messages

    Note:
        - Does nothing if REPORT_METADATA_SHEET_ID or row_id is not configured
        - Does nothing if the file doesn't exist
        - Silently handles errors with logging (doesn't raise exceptions)

    Examples:
        >>> upload_pdf_to_smartsheet("reports/weekly_2024-01.pdf", 5089581251235716)
        # Logs: "Successfully uploaded PDF to row 5089581251235716."
    """

    # Check if the upload feature is configured
    if not REPORT_METADATA_SHEET_ID or not row_id:
        logger.warning("Smartsheet upload not configured. Skipping PDF upload.")
        return

    if not os.path.exists(file_path):
        logger.error(f"Cannot upload file: {file_path} does not exist.")
        return

    try:
        base_client = smartsheet.Smartsheet(token)
        base_client.errors_as_exceptions(True)
        client = SmartsheetRetryClient(
            base_client,
            max_retries=3,
            base_delay=1.0,
            max_delay=30.0,
            continue_on_failure=True
        )
        logger.info(f"Uploading {os.path.basename(file_path)} to row {row_id}...")

        # Use the passed row_id with retry logic
        result = client.attach_file_to_row(
            REPORT_METADATA_SHEET_ID,
            row_id,
            (os.path.basename(file_path), open(file_path, 'rb'), 'application/pdf')
        )

        if result is None:
            logger.warning(f"PDF upload to row {row_id} failed after retries")
            return

        logger.info(f"Successfully uploaded PDF to row {row_id}.")

    except Exception as e:
        logger.error(f"Failed to upload PDF to Smartsheet: {e}", exc_info=True)


def update_smartsheet_cells(sheet_id: int, row_id: int, column_map: Dict[str, int], filename: str, date_range_str: str) -> None:
    """Updates cells in a specific Smartsheet row with report metadata."""

    # Check if the feature is configured
    if not all([sheet_id, row_id, column_map]):
        logger.warning("Smartsheet cell update not configured. Skipping.")
        return

    try:
        base_client = smartsheet.Smartsheet(token)
        base_client.errors_as_exceptions(True)
        client = SmartsheetRetryClient(
            base_client,
            max_retries=3,
            base_delay=1.0,
            max_delay=30.0,
            continue_on_failure=True
        )

        # Get the Column IDs from our map
        primary_col_id = column_map.get("Primäre Spalte")
        # --- CORRECTED: Removed the space in "Spalte 2" ---
        secondary_col_id = column_map.get("Spalte2")

        if not primary_col_id or not secondary_col_id:
            logger.error("Could not find 'Primäre Spalte' or 'Spalte2' in the sheet's columns.")
            return

        # Build the cell objects with the new values
        cell_filename = smartsheet.models.Cell({
            'column_id': primary_col_id,
            'value': filename
        })
        cell_date_range = smartsheet.models.Cell({
            'column_id': secondary_col_id,
            'value': date_range_str
        })

        # Build the row object to be updated
        row_to_update = smartsheet.models.Row({
            'id': row_id,
            'cells': [cell_filename, cell_date_range]
        })

        logger.info(f"Updating metadata in row {row_id}...")
        result = client.update_rows(sheet_id, [row_to_update])

        if result is None:
            logger.warning(f"Row update for {row_id} failed after retries")
            return

        logger.info("Successfully updated Smartsheet cells.")

    except Exception as e:
        logger.error(f"Failed to update Smartsheet cells: {e}", exc_info=True)


def create_weekly_report(start_date: date, end_date: date, force: bool = False) -> Optional[str]:
    """Generate a comprehensive weekly PDF report of Smartsheet activity.

    Creates a multi-page PDF report containing executive summary, charts,
    metrics, and detailed breakdowns of changes recorded during the specified
    week. Includes graceful degradation for unavailable sheets and week-over-week
    comparisons.

    Args:
        start_date (date): First day of the reporting week (typically Monday).
        end_date (date): Last day of the reporting week (typically Sunday).
        force (bool, optional): If True, generate report even when no changes
            are found (using sample data). Defaults to False.

    Returns:
        str or None: Path to the generated PDF file, or None if no data found
            and force=False.

    Generated Report Sections:
        - Executive Summary with KPIs and health scores
        - Week-over-week comparison metrics
        - Activity charts by group and phase
        - User activity breakdown
        - Special activities summary
        - Error and data quality reports

    Side Effects:
        - Creates PDF file in reports/weekly/ directory
        - Uploads PDF to Smartsheet (if configured)
        - Updates Smartsheet metadata cells (if configured)
        - Logs progress throughout generation

    Examples:
        >>> from datetime import date
        >>> pdf_path = create_weekly_report(date(2024, 1, 1), date(2024, 1, 7))
        >>> pdf_path
        'reports/weekly/weekly_report_2024-W01.pdf'
        >>> create_weekly_report(date(2024, 1, 1), date(2024, 1, 7), force=True)
        'reports/weekly/weekly_report_2024-W01.pdf'  # Even if no data
    """
    # Start timing for PDF generation
    report_timer = PerformanceTimer("pdf_generation.weekly_report")
    report_timer.start()

    # Generate output filename
    week_str = f"{start_date.isocalendar()[0]}-W{start_date.isocalendar()[1]:02d}"
    out_dir = os.path.join(REPORTS_DIR, "weekly")
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"weekly_report_{week_str}.pdf")

    # Check sheet availability for graceful degradation
    logger.info("Checking sheet availability for weekly report...")
    unavailable_sheets = check_sheet_availability()

    # Load changes for the week
    changes = load_changes(start_date, end_date)

    # Check if we have data
    has_data = len(changes) > 0

    # If no changes and not forcing, return None
    if not changes and not force:
        logger.warning(f"No changes found for week {week_str}")
        return None

    # Try to load all changes if no data for this period
    all_changes = changes if has_data else load_changes()

    # Collect metrics with unavailable sheets info for graceful degradation
    metrics = collect_metrics(changes if has_data else all_changes, unavailable_sheets)

    # Capture generation timestamp for version tracking
    generation_timestamp = format_report_timestamp()
    logger.info(f"Report generation timestamp: {generation_timestamp}")

    # Create PDF document
    doc = SimpleDocTemplate(filename, pagesize=A4,
                          leftMargin=25*mm, rightMargin=25*mm,
                          topMargin=20*mm, bottomMargin=20*mm)

    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading_style = styles['Heading1']
    subheading_style = styles['Heading2']
    normal_style = styles['Normal']

    # Create a colored heading style for group headers
    colored_group_style = ParagraphStyle(
        'ColoredGroupHeading',
        parent=heading_style,
        textColor=colors.white,  # White text
    )

    # Build the PDF content
    story = []

    # Reset section tracker for this new report
    reset_section_tracker(report_type="Weekly")
    # Reset bookmark tracker for fresh bookmark generation
    reset_bookmark_tracker()

    # Add professional cover page with branding
    period_str = f"{start_date.strftime('%d.%m.%Y')} - {end_date.strftime('%d.%m.%Y')}"
    create_cover_page(
        story=story,
        report_title="Smartsheet Activity Report",
        period_str=period_str,
        generation_timestamp=generation_timestamp,
        company_name="Automaker Financial Accounts",
        report_type="Weekly"
    )

    # Add Executive Summary as the second page (after cover)
    add_executive_summary_section(story, metrics, period_str, report_type="Weekly",
                                   start_date=start_date, end_date=end_date,
                                   generation_timestamp=generation_timestamp)

    # Add Period Comparison section (Week-over-Week)
    add_period_comparison_section(story, report_type="Weekly",
                                   start_date=start_date, end_date=end_date)
    story.append(PageBreak())

    # Add section marker for Activity Overview
    story.append(SectionMarker("Activity Overview"))

    # Title
    story.append(Paragraph(f"Weekly Smartsheet Changes Report", title_style))
    
    # Period information
    if not has_data:
        story.append(Paragraph(f"Period: {start_date.strftime('%d.%m.%Y')} - {end_date.strftime('%d.%m.%Y')}", normal_style))
        if all_changes:
            story.append(Paragraph(f"<i>No data for this period. Showing sample with data from all available history.</i>", normal_style))
        else:
            story.append(Paragraph(f"<i>Sample report - no data available yet</i>", normal_style))
    else:
        story.append(Paragraph(f"Period: {start_date.strftime('%d.%m.%Y')} - {end_date.strftime('%d.%m.%Y')}", normal_style))
    
    story.append(Spacer(1, 10*mm))
    
    # Summary
    story.append(Paragraph("Summary", heading_style))
    summary_data = [
        ["Total Changes", str(metrics["total_changes"])],
        ["Groups with Activity", str(len(metrics["groups"]))],
        ["Users Active", str(len(metrics["users"]))],
    ]
    summary_table = Table(summary_data, colWidths=[100*mm, 50*mm])
    summary_table.setStyle(TableStyle([
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('BACKGROUND', (0,0), (-1,0), colors.lightgrey),
        ('ALIGN', (1,0), (1,-1), 'RIGHT')
    ]))
    story.append(summary_table)
    story.append(Spacer(1, 10*mm))
    
    # Main page charts - side by side
    story.append(Paragraph("Activity Overview", heading_style))
    
    # Create both charts
    group_chart = make_group_bar_chart(metrics["groups"], "Changes by Group")
    phase_chart = make_phase_bar_chart(metrics["phases"], "Changes by Phase")
    
    # Put them in a table side by side
    chart_table_data = [[group_chart, phase_chart]]
    chart_table = Table(chart_table_data)
    story.append(chart_table)
    story.append(Spacer(1, 15*mm))
    
    # Group detail pages with grouped bar charts
    # Include both groups with data AND unavailable sheets for graceful degradation
    all_groups = set(metrics["group_phase_user"].keys())
    unavailable_sheets = metrics.get("unavailable_sheets", {})
    # Add unavailable sheets that may not have data in group_phase_user
    for unavailable_group in unavailable_sheets.keys():
        all_groups.add(unavailable_group)

    for group in sorted(all_groups):
        if not group:
            continue

        story.append(PageBreak())

        # Get display name for section marker
        display_name = get_group_display_name(group)

        # Add section marker for this group's details (sub-section under Activity Overview)
        story.append(SectionMarker(f"{display_name} Details", bookmark_level=1))

        # Check if this group's sheet is unavailable
        if group in unavailable_sheets:
            # Display "Data unavailable" section for this group
            reason = unavailable_sheets[group]
            unavailable_elements = create_data_unavailable_section(group, reason, styles)
            for element in unavailable_elements:
                story.append(element)
            continue  # Skip to next group

        # Create colored header for group - use display name for bundle groups
        group_color = GROUP_COLORS.get(group, colors.steelblue)
        group_header_data = [[f"{display_name} Details"]]
        group_header = Table(group_header_data, colWidths=[150*mm], rowHeights=[10*mm])
        group_header.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), group_color),
            ('TEXTCOLOR', (0,0), (-1,-1), colors.white),
            ('ALIGN', (0,0), (-1,-1), 'CENTER'),
            ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
            ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'),
            ('FONTSIZE', (0,0), (-1,-1), 14),
        ]))
        story.append(group_header)
        story.append(Spacer(1, 5*mm))

        story.append(Paragraph(f"Total changes: {metrics['groups'].get(group, 0)}", normal_style))
        
        # Grouped bar chart for this group
        phase_user_data = metrics["group_phase_user"].get(group, {})
        if phase_user_data:
            chart, legend_data = make_group_detail_chart(
                group,
                phase_user_data,
                f"User Activity by Phase for {display_name}"
            )
            story.append(chart)
            
            # Add horizontal legend below
            if legend_data:
                # Split legend into chunks of 5 if there are many users
                chunk_size = 5
                legend_chunks = [legend_data[i:i+chunk_size] for i in range(0, len(legend_data), chunk_size)]

                for chunk in legend_chunks:
                    legend = create_horizontal_legend(chunk, width=400)
                    story.append(legend)

            # Add Phase Progression Funnel Chart for this group
            story.append(Spacer(1, 8*mm))
            story.append(Paragraph("Phase Progression Funnel", subheading_style))

            try:
                # Filter changes for this group
                group_changes_for_funnel = [c for c in (all_changes if not has_data else changes) if c.get("Group") == group]

                if group_changes_for_funnel:
                    # Calculate funnel metrics for this group
                    funnel_metrics = calculate_funnel_metrics(group_changes_for_funnel, group)

                    # Get visualization data
                    funnel_viz_data = get_funnel_visualization_data(funnel_metrics)

                    # Create and add the funnel chart
                    funnel_chart = create_funnel_chart(
                        funnel_viz_data,
                        title=f"Phase Progression for {display_name}",
                        width=500,
                        height=300
                    )
                    story.append(funnel_chart)

                    # Add bottleneck analysis if there are significant bottlenecks
                    critical_bottlenecks = identify_bottlenecks(funnel_metrics, BottleneckSeverity.HIGH)
                    if critical_bottlenecks:
                        story.append(Spacer(1, 4*mm))
                        bottleneck_style = ParagraphStyle(
                            'BottleneckWarning',
                            parent=normal_style,
                            textColor=colors.HexColor("#B71C1C"),
                            fontSize=9
                        )
                        for bn in critical_bottlenecks[:2]:  # Show top 2 bottlenecks
                            bn_text = f"<b>⚠ Bottleneck at {bn.phase_name}:</b> {bn.drop_off_rate:.1f}% drop-off ({bn.items_lost} items)"
                            story.append(Paragraph(bn_text, bottleneck_style))
                else:
                    story.append(Paragraph("No phase progression data available for funnel analysis", normal_style))

            except Exception as funnel_error:
                logger.warning(f"Could not create funnel chart for group {group}: {funnel_error}")
                story.append(Paragraph("Funnel chart data unavailable", normal_style))

            # Add the gauge charts for this group - side by side
            story.append(Spacer(1, 8*mm))  # REDUCED from 15mm to 8mm
            story.append(Paragraph("Activity Metrics", subheading_style))
                
            # Get smartsheet data for gauges filtered by group
            try:
                metrics_data = query_smartsheet_data(group)
                
                # Get color for this group
                group_color = GROUP_COLORS.get(group, colors.HexColor("#2ecc71"))
                
                # Get the fixed total products for this group
                total_products = TOTAL_PRODUCTS.get(group, 0)
                
                # Calculate correct percentage based on fixed product count
                if total_products > 0:
                    correct_percentage = (metrics_data["recent_activity_items"] / total_products) * 100
                else:
                    correct_percentage = 0
                
                # Create both gauge charts with same color
                recent_gauge = draw_half_circle_gauge(
                    correct_percentage,  # Use our recalculated percentage
                    metrics_data["recent_activity_items"],
                    "30-Day Activity",
                    color=group_color
                )
                
                # Use fixed total products value
                total_gauge = draw_full_gauge(
                    total_products,
                    "Total Products",
                    color=group_color
                )
                
                # Put them in a table side by side
                gauge_table_data = [[recent_gauge, total_gauge]]
                gauge_table = Table(gauge_table_data)
                story.append(gauge_table)

                # Add Completion Rate section with gauge and progress bar
                story.append(Spacer(1, 8*mm))
                story.append(Paragraph("Completion Rate", subheading_style))

                # Calculate completion rate for this group
                try:
                    # Filter changes for this group and calculate completion rate
                    group_changes = [c for c in (all_changes if not has_data else changes) if c.get("Group") == group]
                    completion_metrics = calculate_completion_rate(group_changes, group)

                    # Get completion rate and counts
                    completion_rate = completion_metrics.completion_rate
                    completed_count = completion_metrics.items_completed
                    started_count = completion_metrics.items_started

                    # Define target completion rate (80% for excellent performance)
                    target_rate = 80.0

                    # Create completion rate gauge
                    completion_gauge = draw_completion_rate_gauge(
                        completion_rate,
                        completed_count,
                        started_count,
                        label="Completion Rate",
                        target_rate=target_rate,
                        show_target=True
                    )

                    # Create completion progress bar
                    completion_bar = draw_completion_progress_bar(
                        completion_rate,
                        completed_count,
                        started_count,
                        label="Progress to Target",
                        target_rate=target_rate,
                        show_target=True
                    )

                    # Put gauge and progress bar side by side
                    completion_table_data = [[completion_gauge, completion_bar]]
                    completion_table = Table(completion_table_data)
                    story.append(completion_table)

                    # Add performance summary text
                    if completion_rate >= target_rate:
                        status_text = f"<b>Status:</b> Meeting target ({completion_rate:.1f}% vs {target_rate:.0f}% target)"
                        status_color = "#1B5E20"  # Green
                    elif completion_rate >= target_rate * 0.75:
                        status_text = f"<b>Status:</b> Approaching target ({completion_rate:.1f}% vs {target_rate:.0f}% target)"
                        status_color = "#8B6914"  # Gold
                    else:
                        status_text = f"<b>Status:</b> Below target ({completion_rate:.1f}% vs {target_rate:.0f}% target)"
                        status_color = "#B71C1C"  # Red

                    performance_style = ParagraphStyle(
                        'PerformanceStatus',
                        parent=normal_style,
                        textColor=colors.HexColor(status_color),
                        fontSize=9
                    )
                    story.append(Paragraph(status_text, performance_style))

                except Exception as completion_error:
                    logger.warning(f"Could not calculate completion rate for group {group}: {completion_error}")
                    story.append(Paragraph("Completion rate data unavailable", normal_style))

                # Add marketplace activity metrics after the completion rate section
                story.append(Spacer(1, 8*mm))  # REDUCED from 15mm to 8mm
                story.append(Paragraph("Marketplace Activity", subheading_style))
                
                # Get marketplace activity data
                most_active, most_inactive = get_marketplace_activity(group, SHEET_IDS[group], start_date, end_date)
                
                # Create tables for most active and inactive marketplaces - SIDE BY SIDE
                active_table = create_activity_table(most_active, "Most Active")
                inactive_table = create_activity_table(most_inactive, "Most Inactive")
                
                # Place tables side by side with FIXED WIDTHS to prevent overflow
                marketplace_table_data = [
                    [Paragraph("Most Active", subheading_style), 
                     Paragraph("Most Inactive", subheading_style)],
                    [active_table, inactive_table]
                ]
                marketplace_table = Table(marketplace_table_data, colWidths=[75*mm, 75*mm])  # Fixed column widths
                marketplace_table.setStyle(TableStyle([
                    ('VALIGN', (0,0), (-1,-1), 'TOP'),
                    ('ALIGN', (0,0), (-1,0), 'CENTER'),
                ]))
                story.append(marketplace_table)
                    
            except Exception as e:
                logger.error(f"Error creating gauge charts for group {group}: {e}")
                # Add a placeholder if there's an error
                story.append(Paragraph(f"Could not generate gauge charts: {str(e)}", normal_style))
            else:
                story.append(Paragraph("No detailed data available for this group", normal_style))
    
    # Add user details section with contribution visualization
    add_user_details_section(story, metrics, changes=changes)
    # Add special activities section
    add_special_activities_section(story, start_date, end_date)
    # Add error/warning report section (with sample data for demonstration)
    add_error_report_section(story, errors=None, show_sample=True)
    # Add cleanup suggestions section (with sample data for demonstration)
    add_cleanup_suggestions_section(story, suggestions_data=None, show_sample=True)
    # Add missing data warnings section (with sample data for demonstration)
    add_missing_data_warnings_section(story, warnings_data=None, show_sample=True)

    # Build the PDF with page numbering and generation timestamp in footer
    doc.build(story, canvasmaker=create_numbered_canvas_with_timestamp(generation_timestamp))
    logger.info(f"Weekly report created: {filename}")
    
    # --- NEW: Upload PDF and update metadata cells ---
    if REPORT_METADATA_SHEET_ID:
        # 1. Upload the PDF to its attachment row
        upload_pdf_to_smartsheet(filename, WEEKLY_REPORT_ATTACHMENT_ROW_ID)
        
        # 2. Get the column map for updating cells
        column_map = get_column_map(REPORT_METADATA_SHEET_ID)
        
        # 3. Define the date string for the report
        date_range_str = f"Week {start_date.strftime('%W')} ({start_date.strftime('%b %d')} - {end_date.strftime('%b %d, %Y')})"
        
        # 4. Update the metadata cells
        update_smartsheet_cells(
            REPORT_METADATA_SHEET_ID,
            WEEKLY_METADATA_ROW_ID,
            column_map,
            os.path.basename(filename),
            date_range_str
        )
    # --- END NEW ---

    # Stop timing and log PDF generation performance
    report_timer.stop()

    logger.info(f"Report file created at: {os.path.abspath(filename)}")
    return filename

# PASTE THIS AS THE NEW create_monthly_report FUNCTION
def create_monthly_report(year: int, month: int, force: bool = False) -> Optional[str]:
    """Generate a comprehensive monthly PDF report of Smartsheet activity.

    Creates a multi-page PDF report containing executive summary, charts,
    metrics, and detailed breakdowns of changes recorded during the specified
    month. Similar to weekly reports but covers the entire calendar month and
    includes month-over-month comparisons.

    Args:
        year (int): The year of the report (e.g., 2024).
        month (int): The month number (1-12).
        force (bool, optional): If True, generate report even when no changes
            are found (using sample data). Defaults to False.

    Returns:
        str or None: Path to the generated PDF file, or None if no data found
            and force=False.

    Generated Report Sections:
        - Executive Summary with KPIs and health scores
        - Month-over-month comparison metrics
        - Activity charts by group and phase
        - User activity breakdown
        - Special activities summary
        - Error and data quality reports

    Side Effects:
        - Creates PDF file in reports/monthly/ directory
        - Uploads PDF to Smartsheet (if configured)
        - Updates Smartsheet metadata cells (if configured)
        - Logs progress throughout generation

    Examples:
        >>> pdf_path = create_monthly_report(2024, 1)  # January 2024
        >>> pdf_path
        'reports/monthly/monthly_report_2024-01.pdf'
        >>> create_monthly_report(2024, 12, force=True)  # December with force
        'reports/monthly/monthly_report_2024-12.pdf'
    """
    # Start timing for PDF generation
    report_timer = PerformanceTimer("pdf_generation.monthly_report")
    report_timer.start()

    # Determine the month's date range
    start_date = date(year, month, 1)
    if month == 12:
        end_date = date(year + 1, 1, 1) - timedelta(days=1)
    else:
        end_date = date(year, month + 1, 1) - timedelta(days=1)

    # Generate output filename
    month_str = f"{year}-{month:02d}"
    out_dir = os.path.join(REPORTS_DIR, "monthly")
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"monthly_report_{month_str}.pdf")

    # Check sheet availability for graceful degradation
    logger.info("Checking sheet availability for monthly report...")
    unavailable_sheets = check_sheet_availability()

    # Load changes for the month
    changes = load_changes(start_date, end_date)

    has_data = len(changes) > 0
    if not changes and not force:
        logger.warning(f"No changes found for month {month_str}")
        return None


    all_changes = changes if has_data else load_changes()
    # Collect metrics with unavailable sheets info for graceful degradation
    metrics = collect_metrics(changes if has_data else all_changes, unavailable_sheets)

    # Capture generation timestamp for version tracking
    generation_timestamp = format_report_timestamp()
    logger.info(f"Report generation timestamp: {generation_timestamp}")
    
    doc = SimpleDocTemplate(filename, pagesize=A4,
                          leftMargin=25*mm, rightMargin=25*mm,
                          topMargin=20*mm, bottomMargin=20*mm)
    
    styles = getSampleStyleSheet()
    title_style = styles['Title']
    heading_style = styles['Heading1']
    subheading_style = styles['Heading2']
    normal_style = styles['Normal']
    
    story = []

    # Reset section tracker for this new report
    reset_section_tracker(report_type="Monthly")
    # Reset bookmark tracker for fresh bookmark generation
    reset_bookmark_tracker()

    # Add professional cover page with branding
    period_str = f"{start_date.strftime('%B %Y')}"
    # For monthly reports, show full date range
    full_period_str = f"{start_date.strftime('%d.%m.%Y')} - {end_date.strftime('%d.%m.%Y')}"
    create_cover_page(
        story=story,
        report_title="Smartsheet Activity Report",
        period_str=full_period_str,
        generation_timestamp=generation_timestamp,
        company_name="Automaker Financial Accounts",
        report_type="Monthly"
    )

    # Add Executive Summary as the second page (after cover)
    add_executive_summary_section(story, metrics, period_str, report_type="Monthly",
                                   start_date=start_date, end_date=end_date,
                                   generation_timestamp=generation_timestamp)

    # Add Period Comparison section (Month-over-Month)
    add_period_comparison_section(story, report_type="Monthly",
                                   start_date=start_date, end_date=end_date)
    story.append(PageBreak())

    # Add section marker for Activity Overview
    story.append(SectionMarker("Activity Overview"))

    story.append(Paragraph(f"Monthly Smartsheet Changes Report", title_style))
    
    if not has_data:
        story.append(Paragraph(f"Period: {start_date.strftime('%B %Y')}", normal_style))
        story.append(Paragraph(f"<i>No data for this period. Showing sample with data from all available history.</i>", normal_style))
    else:
        story.append(Paragraph(f"Period: {start_date.strftime('%B %Y')}", normal_style))
    
    story.append(Spacer(1, 10*mm))
    
    story.append(Paragraph("Monthly Summary", heading_style))
    summary_data = [
        ["Total Changes", str(metrics["total_changes"])],
        ["Groups with Activity", str(len(metrics["groups"]))],
        ["Users Active", str(len(metrics["users"]))],
    ]
    summary_table = Table(summary_data, colWidths=[100*mm, 50*mm])
    summary_table.setStyle(TableStyle([('GRID', (0,0), (-1,-1), 0.5, colors.grey), ('BACKGROUND', (0,0), (-1,0), colors.lightgrey), ('ALIGN', (1,0), (1,-1), 'RIGHT')]))
    story.append(summary_table)
    story.append(Spacer(1, 10*mm))
    
    story.append(Paragraph("Activity Overview", heading_style))
    group_chart = make_group_bar_chart(metrics["groups"], "Changes by Group")
    phase_chart = make_phase_bar_chart(metrics["phases"], "Changes by Phase")
    chart_table_data = [[group_chart, phase_chart]]
    chart_table = Table(chart_table_data)
    story.append(chart_table)
    story.append(Spacer(1, 15*mm))

    # Group detail pages with grouped bar charts
    # Include both groups with data AND unavailable sheets for graceful degradation
    all_groups = set(metrics["group_phase_user"].keys())
    unavailable_sheets = metrics.get("unavailable_sheets", {})
    # Add unavailable sheets that may not have data in group_phase_user
    for unavailable_group in unavailable_sheets.keys():
        all_groups.add(unavailable_group)

    for group in sorted(all_groups):
        if not group:
            continue

        story.append(PageBreak())

        # Get display name for section marker
        display_name = get_group_display_name(group)

        # Add section marker for this group's details (sub-section under Activity Overview)
        story.append(SectionMarker(f"{display_name} Details", bookmark_level=1))

        # Check if this group's sheet is unavailable
        if group in unavailable_sheets:
            # Display "Data unavailable" section for this group
            reason = unavailable_sheets[group]
            unavailable_elements = create_data_unavailable_section(group, reason, styles)
            for element in unavailable_elements:
                story.append(element)
            continue  # Skip to next group

        # Create colored header for group - use display name for bundle groups
        group_color = GROUP_COLORS.get(group, colors.steelblue)
        group_header_data = [[f"{display_name} Details"]]
        group_header = Table(group_header_data, colWidths=[150*mm], rowHeights=[10*mm])
        group_header.setStyle(TableStyle([('BACKGROUND', (0,0), (-1,-1), group_color), ('TEXTCOLOR', (0,0), (-1,-1), colors.white), ('ALIGN', (0,0), (-1,-1), 'CENTER'), ('VALIGN', (0,0), (-1,-1), 'MIDDLE'), ('FONTNAME', (0,0), (-1,-1), 'Helvetica-Bold'), ('FONTSIZE', (0,0), (-1,-1), 14)]))
        story.append(group_header)
        story.append(Spacer(1, 5*mm))

        story.append(Paragraph(f"Total changes: {metrics['groups'].get(group, 0)}", normal_style))

        phase_user_data = metrics["group_phase_user"].get(group, {})
        if phase_user_data:
            chart, legend_data = make_group_detail_chart(group, phase_user_data, f"User Activity by Phase for {display_name}")
            story.append(chart)
            
            if legend_data:
                legend = create_horizontal_legend(legend_data, width=400)
                story.append(legend)

            # Add Phase Progression Funnel Chart for this group
            story.append(Spacer(1, 8*mm))
            story.append(Paragraph("Phase Progression Funnel", subheading_style))

            try:
                # Filter changes for this group
                group_changes_for_funnel = [c for c in (all_changes if not has_data else changes) if c.get("Group") == group]

                if group_changes_for_funnel:
                    # Calculate funnel metrics for this group
                    funnel_metrics = calculate_funnel_metrics(group_changes_for_funnel, group)

                    # Get visualization data
                    funnel_viz_data = get_funnel_visualization_data(funnel_metrics)

                    # Create and add the funnel chart
                    funnel_chart = create_funnel_chart(
                        funnel_viz_data,
                        title=f"Phase Progression for {display_name}",
                        width=500,
                        height=300
                    )
                    story.append(funnel_chart)

                    # Add bottleneck analysis if there are significant bottlenecks
                    critical_bottlenecks = identify_bottlenecks(funnel_metrics, BottleneckSeverity.HIGH)
                    if critical_bottlenecks:
                        story.append(Spacer(1, 4*mm))
                        bottleneck_style = ParagraphStyle(
                            'BottleneckWarning',
                            parent=normal_style,
                            textColor=colors.HexColor("#B71C1C"),
                            fontSize=9
                        )
                        for bn in critical_bottlenecks[:2]:  # Show top 2 bottlenecks
                            bn_text = f"<b>⚠ Bottleneck at {bn.phase_name}:</b> {bn.drop_off_rate:.1f}% drop-off ({bn.items_lost} items)"
                            story.append(Paragraph(bn_text, bottleneck_style))
                else:
                    story.append(Paragraph("No phase progression data available for funnel analysis", normal_style))

            except Exception as funnel_error:
                logger.warning(f"Could not create funnel chart for group {group}: {funnel_error}")
                story.append(Paragraph("Funnel chart data unavailable", normal_style))

            story.append(Spacer(1, 8*mm))
            story.append(Paragraph("Activity Metrics", subheading_style))

            try:
                sheet_id = SHEET_IDS.get(group)
                if not sheet_id:
                    raise ValueError(f"No sheet ID found for group {group}")
                
                summary_data = get_sheet_summary_data(sheet_id)
                if not summary_data:
                    raise ValueError("Could not fetch sheet summary data.")

                stacked_gauge = create_stacked_gauge_chart(summary_data)
                story.append(stacked_gauge)
                story.append(Spacer(1, 8*mm))

                story.append(Paragraph("Total Product Counts", subheading_style))
                group_color = GROUP_COLORS.get(group, colors.HexColor("#457B9D"))

                anzahl_produkte = int(str(summary_data.get("Anzahl der Produkte", '0') or '0').replace('.', ''))
                gauge_anzahl = draw_full_gauge(anzahl_produkte, "Anzahl der Produkte", color=group_color, width=250, height=120)
                
                summe_artikel = int(str(summary_data.get("Summe aller Marktplatzartikel", '0') or '0').replace('.', ''))
                gauge_summe = draw_full_gauge(summe_artikel, "Summe Marktplatzartikel", color=group_color, width=250, height=120)
                
                total_gauge_table = Table([[gauge_anzahl, gauge_summe]])
                story.append(total_gauge_table)

            except Exception as e:
                logger.error(f"Error creating summary charts for group {group}: {e}", exc_info=True)
                story.append(Paragraph(f"Could not generate summary metrics: {str(e)}", normal_style))

            # Add Completion Rate section with gauge and progress bar
            story.append(Spacer(1, 8*mm))
            story.append(Paragraph("Completion Rate", subheading_style))

            # Calculate completion rate for this group
            try:
                # Filter changes for this group and calculate completion rate
                group_changes = [c for c in (all_changes if not has_data else changes) if c.get("Group") == group]
                completion_metrics = calculate_completion_rate(group_changes, group)

                # Get completion rate and counts
                completion_rate_val = completion_metrics.completion_rate
                completed_count = completion_metrics.items_completed
                started_count = completion_metrics.items_started

                # Define target completion rate (80% for excellent performance)
                target_rate = 80.0

                # Create completion rate gauge
                completion_gauge = draw_completion_rate_gauge(
                    completion_rate_val,
                    completed_count,
                    started_count,
                    label="Completion Rate",
                    target_rate=target_rate,
                    show_target=True
                )

                # Create completion progress bar
                completion_bar = draw_completion_progress_bar(
                    completion_rate_val,
                    completed_count,
                    started_count,
                    label="Progress to Target",
                    target_rate=target_rate,
                    show_target=True
                )

                # Put gauge and progress bar side by side
                completion_table_data = [[completion_gauge, completion_bar]]
                completion_table = Table(completion_table_data)
                story.append(completion_table)

                # Add performance summary text
                if completion_rate_val >= target_rate:
                    status_text = f"<b>Status:</b> Meeting target ({completion_rate_val:.1f}% vs {target_rate:.0f}% target)"
                    status_color = "#1B5E20"  # Green
                elif completion_rate_val >= target_rate * 0.75:
                    status_text = f"<b>Status:</b> Approaching target ({completion_rate_val:.1f}% vs {target_rate:.0f}% target)"
                    status_color = "#8B6914"  # Gold
                else:
                    status_text = f"<b>Status:</b> Below target ({completion_rate_val:.1f}% vs {target_rate:.0f}% target)"
                    status_color = "#B71C1C"  # Red

                performance_style = ParagraphStyle(
                    'PerformanceStatus',
                    parent=normal_style,
                    textColor=colors.HexColor(status_color),
                    fontSize=9
                )
                story.append(Paragraph(status_text, performance_style))

            except Exception as completion_error:
                logger.warning(f"Could not calculate completion rate for group {group}: {completion_error}")
                story.append(Paragraph("Completion rate data unavailable", normal_style))

            story.append(Spacer(1, 8*mm))
            story.append(Paragraph("Marketplace Activity", subheading_style))

            most_active, most_inactive = get_marketplace_activity(group, SHEET_IDS[group], start_date, end_date)
            
            active_table = create_activity_table(most_active, "Most Active")
            inactive_table = create_activity_table(most_inactive, "Most Inactive")
            
            marketplace_table_data = [[Paragraph("Most Active", subheading_style), Paragraph("Most Inactive", subheading_style)], [active_table, inactive_table]]
            marketplace_table = Table(marketplace_table_data, colWidths=[75*mm, 75*mm])
            marketplace_table.setStyle(TableStyle([('VALIGN', (0,0), (-1,-1), 'TOP'), ('ALIGN', (0,0), (-1,0), 'CENTER')]))
            story.append(marketplace_table)
                
        else:
            story.append(Paragraph("No detailed data available for this group", normal_style))
    
    # Add user details section with contribution visualization
    add_user_details_section(story, metrics, changes=changes)
    add_special_activities_section(story, start_date, end_date)
    # Add error/warning report section (with sample data for demonstration)
    add_error_report_section(story, errors=None, show_sample=True)
    # Add cleanup suggestions section (with sample data for demonstration)
    add_cleanup_suggestions_section(story, suggestions_data=None, show_sample=True)
    # Add missing data warnings section (with sample data for demonstration)
    add_missing_data_warnings_section(story, warnings_data=None, show_sample=True)

    # Build the PDF with page numbering and generation timestamp in footer
    doc.build(story, canvasmaker=create_numbered_canvas_with_timestamp(generation_timestamp))
    logger.info(f"Monthly report created: {filename}")

    if REPORT_METADATA_SHEET_ID:
        upload_pdf_to_smartsheet(filename, MONTHLY_REPORT_ATTACHMENT_ROW_ID)
        column_map = get_column_map(REPORT_METADATA_SHEET_ID)
        date_range_str = start_date.strftime('%B %Y')
        update_smartsheet_cells(REPORT_METADATA_SHEET_ID, MONTHLY_METADATA_ROW_ID, column_map, os.path.basename(filename), date_range_str)

    # Stop timing and log PDF generation performance
    report_timer.stop()

    logger.info(f"Report file created at: {os.path.abspath(filename)}")
    return filename

def get_previous_week() -> Tuple[date, date]:
    """Get the start and end dates for the previous complete week.

    Returns the date range for the most recently completed full week,
    running from Monday to Sunday.

    Returns:
        tuple: A tuple of (start_date, end_date) where:
            - start_date (date): Monday of the previous week
            - end_date (date): Sunday of the previous week

    Examples:
        >>> # If today is Wednesday, January 10, 2024
        >>> start, end = get_previous_week()
        >>> start
        datetime.date(2024, 1, 1)  # Monday
        >>> end
        datetime.date(2024, 1, 7)  # Sunday
    """
    today = date.today()
    previous_week_end = today - timedelta(days=today.weekday() + 1)
    previous_week_start = previous_week_end - timedelta(days=6)
    return previous_week_start, previous_week_end


def get_current_week() -> Tuple[date, date]:
    """Get the start and end dates for the current week in progress.

    Returns the date range from the Monday of the current week through today.
    Useful for generating in-progress weekly reports.

    Returns:
        tuple: A tuple of (start_date, end_date) where:
            - start_date (date): Monday of the current week
            - end_date (date): Today's date

    Examples:
        >>> # If today is Wednesday, January 10, 2024
        >>> start, end = get_current_week()
        >>> start
        datetime.date(2024, 1, 8)  # Monday
        >>> end
        datetime.date(2024, 1, 10)  # Today (Wednesday)
    """
    today = date.today()
    start = today - timedelta(days=today.weekday())  # Monday
    end = today
    return start, end


def get_previous_month() -> Tuple[int, int]:
    """Get the year and month for the previous calendar month.

    Returns the year and month number for the most recently completed
    month. Handles year boundary (December -> January).

    Returns:
        tuple: A tuple of (year, month) where:
            - year (int): The year (may be previous year if currently January)
            - month (int): The month number (1-12)

    Examples:
        >>> # If today is in March 2024
        >>> year, month = get_previous_month()
        >>> (year, month)
        (2024, 2)  # February
        >>> # If today is in January 2024
        >>> year, month = get_previous_month()
        >>> (year, month)
        (2023, 12)  # December of previous year
    """
    today = date.today()
    if today.month == 1:
        return today.year - 1, 12
    else:
        return today.year, today.month - 1


def get_current_month() -> Tuple[int, int]:
    """Get the year and month for the current calendar month.

    Returns:
        tuple: A tuple of (year, month) where:
            - year (int): The current year
            - month (int): The current month number (1-12)

    Examples:
        >>> # If today is March 15, 2024
        >>> year, month = get_current_month()
        >>> (year, month)
        (2024, 3)
    """
    today = date.today()
    return today.year, today.month

# Note: parse_date_argument is now imported from date_utilities module

def validate_date_range(start_date: date, end_date: date) -> Tuple[bool, Optional[str]]:
    """Validate that a date range is logically correct.

    Checks that the start date is before or equal to the end date, and
    logs a warning if the range spans more than one year (which may
    result in very large reports).

    Args:
        start_date (date): The start of the date range.
        end_date (date): The end of the date range.

    Returns:
        tuple: A tuple of (is_valid, error_message) where:
            - is_valid (bool): True if the date range is valid
            - error_message (str or None): Error description if invalid, None if valid

    Examples:
        >>> from datetime import date
        >>> validate_date_range(date(2024, 1, 1), date(2024, 1, 31))
        (True, None)
        >>> validate_date_range(date(2024, 2, 1), date(2024, 1, 1))
        (False, 'Start date (2024-02-01) must be before or equal to end date (2024-01-01)')
    """
    if start_date > end_date:
        return False, f"Start date ({start_date}) must be before or equal to end date ({end_date})"

    # Warn if the date range is very large (over 1 year)
    days_diff = (end_date - start_date).days
    if days_diff > 365:
        logger.warning(f"Date range spans {days_diff} days (over 1 year). Report may be very large.")

    return True, None


def create_custom_report(start_date: date, end_date: date, force: bool = False) -> Optional[str]:
    """Generate a PDF report for a custom date range.

    Creates a comprehensive PDF report similar to weekly/monthly reports
    but for an arbitrary date range specified by the user. Useful for
    ad-hoc reporting needs.

    Args:
        start_date (date): Start date for the report (inclusive).
        end_date (date): End date for the report (inclusive).
        force (bool, optional): If True, generate report even when no
            changes are found (using sample data). Defaults to False.

    Returns:
        str or None: Path to the generated PDF file, or None if no data
            found and force=False.

    Note:
        Uses DateRangeFilter internally for consistent date handling.
        Delegates to create_custom_report_with_filter for actual generation.

    Examples:
        >>> from datetime import date
        >>> pdf = create_custom_report(date(2024, 1, 1), date(2024, 3, 31))
        >>> pdf
        'reports/custom/custom_report_2024-01-01_to_2024-03-31.pdf'

    See Also:
        create_custom_report_with_filter: The underlying implementation.
        create_weekly_report: For standard weekly reports.
        create_monthly_report: For standard monthly reports.
    """
    # Create a DateRangeFilter for the custom range
    date_range_filter = create_date_range(start_date, end_date)

    return create_custom_report_with_filter(date_range_filter, force=force)


def create_custom_report_with_filter(date_range_filter: DateRangeFilter, force: bool = False) -> Optional[str]:
    """Generate a custom date range PDF report using a DateRangeFilter.

    This is the preferred method for creating custom reports as it ensures
    all calculations and visualizations respect the custom date range parameters.
    Provides consistent handling of both preset and custom date ranges.

    Filename and title formatting:
        - Preset ranges: Include preset name (e.g., "last_7_days_2026-01-01_to_2026-01-07")
        - Custom ranges: Include readable date range (e.g., "2026-01_10-15" for same month)

    Args:
        date_range_filter (DateRangeFilter): A DateRangeFilter object specifying
            the date range. Contains start_date, end_date, label, preset, and
            filename_label properties.
        force (bool, optional): If True, generate report even when no changes
            are found (using sample data). Defaults to False.

    Returns:
        str or None: Path to the generated PDF file in reports/custom/ directory,
            or None if no data found and force=False.

    Generated Report Sections:
        - Executive Summary with KPIs
        - Activity charts by group and phase
        - User activity breakdown
        - Special activities summary
        - Error and data quality reports

    Side Effects:
        - Creates PDF file in reports/custom/ directory
        - Logs progress throughout generation

    Examples:
        >>> from date_range_filter import DateRangeFilter, DateRangePreset
        >>> filter = DateRangeFilter(preset=DateRangePreset.LAST_30_DAYS)
        >>> pdf = create_custom_report_with_filter(filter)
        >>> pdf
        'reports/custom/custom_report_last_30_days_2024-01-01_to_2024-01-30.pdf'

    See Also:
        create_custom_report: Convenience wrapper with start/end dates.
        DateRangeFilter: The filter class from date_range_filter module.
    """
    # Start timing for PDF generation
    report_timer = PerformanceTimer("pdf_generation.custom_report")
    report_timer.start()

    start_date = date_range_filter.start_date
    end_date = date_range_filter.end_date

    # Generate output filename using the DateRangeFilter's filename_label property
    # This provides a clear, descriptive filename that identifies the date range
    filename_label = date_range_filter.filename_label
    out_dir = os.path.join(REPORTS_DIR, "custom")
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, f"custom_report_{filename_label}.pdf")

    logger.info(f"Generating custom report with date range: {date_range_filter.label}")

    # Check sheet availability for graceful degradation
    logger.info("Checking sheet availability for custom report...")
    unavailable_sheets = check_sheet_availability()

    # Load changes using the DateRangeFilter
    changes = load_changes_with_filter(date_range_filter)

    # Check if we have data
    has_data = len(changes) > 0

    # If no changes and not forcing, return None
    if not changes and not force:
        logger.warning(f"No changes found for date range {start_date} to {end_date}")
        return None

    # Try to load all changes if no data for this period
    all_changes = changes if has_data else load_changes()

    # Collect metrics using the DateRangeFilter-aware function
    if has_data:
        metrics = collect_metrics_for_range(date_range_filter, unavailable_sheets)
    else:
        metrics = collect_metrics(all_changes, unavailable_sheets)
        # Add date range info for the empty case too
        metrics["date_range_info"] = {
            "start_date": start_date.isoformat(),
            "end_date": end_date.isoformat(),
            "label": date_range_filter.label,
            "preset": date_range_filter.preset.value if hasattr(date_range_filter.preset, 'value') else str(date_range_filter.preset),
            "days_in_range": date_range_filter.days_in_range,
            "daily_average": 0,
        }

    # Capture generation timestamp for version tracking
    generation_timestamp = format_report_timestamp()
    logger.info(f"Report generation timestamp: {generation_timestamp}")

    # Create PDF document
    doc = SimpleDocTemplate(
        filename,
        pagesize=landscape(A4),
        rightMargin=15*mm,
        leftMargin=15*mm,
        topMargin=15*mm,
        bottomMargin=15*mm,
    )

    story = []
    styles = getSampleStyleSheet()

    # Reset section tracker for this new report
    reset_section_tracker(report_type="Custom")
    # Reset bookmark tracker for fresh bookmark generation
    reset_bookmark_tracker()

    # Add professional cover page with branding
    period_str = f"{start_date.strftime('%d.%m.%Y')} - {end_date.strftime('%d.%m.%Y')}"
    create_cover_page(
        story=story,
        report_title="Smartsheet Activity Report",
        period_str=period_str,
        generation_timestamp=generation_timestamp,
        company_name="Automaker Financial Accounts",
        report_type="Custom"
    )

    # Add section marker for the title/overview section
    story.append(SectionMarker("Report Overview"))

    # Title with date range - use the report_title property for clear identification
    title_style = ParagraphStyle(
        'CustomTitle',
        parent=styles['Title'],
        fontSize=28,
        leading=34,
        spaceAfter=8*mm,
        alignment=1,  # Center
        textColor=colors.HexColor("#2c3e50"),
    )

    # Use the DateRangeFilter's report_title property for a clear, descriptive title
    report_title = date_range_filter.report_title
    story.append(Paragraph(f"Smartsheet Changes: {report_title}", title_style))

    # Subtitle shows the human-readable date range label prominently
    subtitle_style = ParagraphStyle(
        'CustomSubtitle',
        parent=styles['Normal'],
        fontSize=18,
        leading=22,
        spaceAfter=6*mm,
        alignment=1,  # Center
        textColor=colors.HexColor("#34495e"),
    )
    story.append(Paragraph(date_range_filter.label, subtitle_style))

    # Add detailed date range information using report_subtitle property
    range_details_style = ParagraphStyle(
        'RangeDetails',
        parent=styles['Normal'],
        fontSize=11,
        leading=14,
        spaceAfter=5*mm,
        alignment=1,  # Center
        textColor=colors.HexColor("#7f8c8d"),
    )
    # Show exact dates in ISO format for clarity
    date_range_detail = f"{start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} • {date_range_filter.days_in_range} day{'s' if date_range_filter.days_in_range != 1 else ''}"
    story.append(Paragraph(date_range_detail, range_details_style))

    # Add generation timestamp on cover page
    timestamp_style = ParagraphStyle(
        'GenerationTimestamp',
        parent=styles['Normal'],
        fontSize=9,
        leading=11,
        spaceAfter=3*mm,
        alignment=1,  # Center
        textColor=colors.HexColor("#95a5a6"),
    )
    story.append(Paragraph(f"Report generated: {generation_timestamp}", timestamp_style))

    # Add preset indicator if using a preset range
    if date_range_filter.preset != DateRangePreset.CUSTOM:
        preset_style = ParagraphStyle(
            'PresetIndicator',
            parent=styles['Normal'],
            fontSize=9,
            leading=11,
            spaceAfter=5*mm,
            alignment=1,  # Center
            textColor=colors.HexColor("#95a5a6"),
        )
        story.append(Paragraph(f"Preset: {date_range_filter.preset.value.replace('_', ' ').title()}", preset_style))

    story.append(Spacer(1, 5*mm))

    # Add data availability notice if we're using all data
    if not has_data:
        notice_style = ParagraphStyle(
            'Notice',
            parent=styles['Normal'],
            fontSize=10,
            textColor=colors.gray,
            alignment=1,
        )
        story.append(Paragraph(
            f"Note: No changes found for the selected date range. Showing sample data.",
            notice_style
        ))
        story.append(Spacer(1, 5*mm))

    # Add executive summary section
    add_executive_summary(story, metrics, start_date, end_date, has_data)
    story.append(PageBreak())

    # Add group summary section with 7-day sparkline trends
    add_group_summary(story, metrics, has_data, trend_days=7, end_date=end_date)
    story.append(PageBreak())

    # Add phase distribution section
    add_phase_distribution(story, metrics, has_data)
    story.append(PageBreak())

    # Add user activity section (landscape mode since custom reports use landscape(A4))
    add_user_activity(story, metrics, has_data, use_landscape=True)

    # Add error report section (landscape mode for wider error tables)
    add_error_report_section(story, errors=None, show_sample=True, use_landscape=True)
    # Add cleanup suggestions section (with sample data for demonstration)
    add_cleanup_suggestions_section(story, suggestions_data=None, show_sample=True)
    # Add missing data warnings section (with sample data for demonstration)
    add_missing_data_warnings_section(story, warnings_data=None, show_sample=True)

    # Build the PDF with page numbering and generation timestamp in footer
    doc.build(story, canvasmaker=create_numbered_canvas_with_timestamp(generation_timestamp))
    logger.info(f"Custom report created: {filename}")

    # Stop timing and log PDF generation performance
    report_timer.stop()

    logger.info(f"Report file created at: {os.path.abspath(filename)}")
    return filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate Smartsheet change reports",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Weekly report for previous week:
    python smartsheet_report.py --weekly --previous

  Monthly report for specific month:
    python smartsheet_report.py --monthly --year 2024 --month 6

  Custom date range report:
    python smartsheet_report.py --custom --start-date 2024-01-01 --end-date 2024-03-31

  With debug logging:
    python smartsheet_report.py --weekly --previous --log-level DEBUG

Date formats supported for --start-date and --end-date:
  - YYYY-MM-DD (ISO format, preferred): 2024-01-15
  - DD.MM.YYYY (European format): 15.01.2024
  - DD/MM/YYYY (European format with slashes): 15/01/2024
        """
    )
    report_type = parser.add_mutually_exclusive_group(required=True)
    report_type.add_argument("--weekly", action="store_true", help="Generate weekly report")
    report_type.add_argument("--monthly", action="store_true", help="Generate monthly report")
    report_type.add_argument("--custom", action="store_true", help="Generate custom date range report (requires --start-date and --end-date)")

    parser.add_argument("--year", type=int, help="Year for report (defaults to current year)")
    parser.add_argument("--month", type=int, help="Month number for monthly report")
    parser.add_argument("--week", type=int, help="ISO week number for weekly report")
    parser.add_argument("--previous", action="store_true", help="Generate report for previous week/month")
    parser.add_argument("--current", action="store_true", help="Generate report for current week/month to date")
    parser.add_argument("--start-date", type=parse_date_argument, dest="start_date",
                        help="Start date for custom report (formats: YYYY-MM-DD, DD.MM.YYYY, DD/MM/YYYY)")
    parser.add_argument("--end-date", type=parse_date_argument, dest="end_date",
                        help="End date for custom report (formats: YYYY-MM-DD, DD.MM.YYYY, DD/MM/YYYY)")
    parser.add_argument("--force", action="store_true", help="Force report generation even with no data")
    add_log_level_argument(parser)

    args = parser.parse_args()

    # Configure logging with CLI argument or environment variable
    configure_logging(
        log_file="smartsheet_report.log",
        log_level=args.log_level
    )

    try:
        if args.weekly:
            if args.previous:
                start_date, end_date = get_previous_week()
                filename = create_weekly_report(start_date, end_date, force=args.force)
            elif args.current:
                start_date, end_date = get_current_week()
                filename = create_weekly_report(start_date, end_date, force=args.force)
            elif args.year and args.week:
                # Calculate date from ISO week
                start_date = datetime.fromisocalendar(args.year, args.week, 1).date()
                end_date = start_date + timedelta(days=6)
                filename = create_weekly_report(start_date, end_date, force=args.force)
            else:
                logger.error("For weekly reports, specify --previous OR --current OR (--year and --week)")
                exit(1)

        elif args.monthly:
            if args.previous:
                year, month = get_previous_month()
                filename = create_monthly_report(year, month, force=args.force)
            elif args.current:
                year, month = get_current_month()
                filename = create_monthly_report(year, month, force=args.force)
            elif args.year and args.month:
                filename = create_monthly_report(args.year, args.month, force=args.force)
            else:
                logger.error("For monthly reports, specify --previous OR --current OR (--year and --month)")
                exit(1)

        elif args.custom:
            # Validate that both start-date and end-date are provided
            if not args.start_date or not args.end_date:
                logger.error("For custom reports, both --start-date and --end-date are required")
                exit(1)

            # Validate date range
            is_valid, error_msg = validate_date_range(args.start_date, args.end_date)
            if not is_valid:
                logger.error(error_msg)
                exit(1)

            filename = create_custom_report(args.start_date, args.end_date, force=args.force)

        if filename:
            logger.info(f"Report successfully generated: {filename}")
            exit(0)
        else:
            logger.warning("Report generation completed but no file was created")
            exit(1)

    except Exception as e:
        logger.error(f"Error generating report: {e}", exc_info=True)
        exit(1)
