"""
Centralized Constants Module for Smartsheet Tracker

This module provides named constants for all magic numbers used throughout the codebase.
Centralizing constants improves code readability, maintainability, and makes it easier
to modify values consistently across the application.

Categories:
- HTTP Status Codes
- Smartsheet API Error Codes
- Retry Configuration
- Performance Thresholds
- WCAG Accessibility Constants
- Chart Layout Constants
- Report Page Layout
"""

from enum import IntEnum


# ============================================================================
# HTTP STATUS CODES
# ============================================================================

class HTTPStatus(IntEnum):
    """Standard HTTP status codes used in API responses."""
    OK = 200
    CREATED = 201
    NO_CONTENT = 204
    BAD_REQUEST = 400
    UNAUTHORIZED = 401
    FORBIDDEN = 403
    NOT_FOUND = 404
    TOO_MANY_REQUESTS = 429
    INTERNAL_SERVER_ERROR = 500
    BAD_GATEWAY = 502
    SERVICE_UNAVAILABLE = 503
    GATEWAY_TIMEOUT = 504


# ============================================================================
# SMARTSHEET API ERROR CODES
# ============================================================================

class SmartsheetErrorCode(IntEnum):
    """Smartsheet API-specific error codes.

    Reference: https://smartsheet-platform.github.io/api-docs/#error-handling
    """
    # Authentication/Authorization Errors
    INVALID_TOKEN = 1002
    TOKEN_EXPIRED = 1003
    INVALID_API_KEY = 1004
    INVALID_ACCESS_TOKEN = 4001

    # Resource Not Found Errors
    NOT_FOUND = 1006
    SHEET_NOT_FOUND = 1020

    # Permission Errors
    SHEET_PRIVATE = 1012
    USER_CANNOT_ACCESS = 1014
    ACCESS_DENIED = 1016
    OPERATION_DENIED = 1019

    # Rate Limiting
    RATE_LIMIT_EXCEEDED = 4003


# Permission-related error codes that indicate sheet-level access issues
PERMISSION_ERROR_CODES = frozenset({
    SmartsheetErrorCode.SHEET_PRIVATE,
    SmartsheetErrorCode.USER_CANNOT_ACCESS,
    SmartsheetErrorCode.ACCESS_DENIED,
    SmartsheetErrorCode.OPERATION_DENIED,
})

# Not found error codes
NOT_FOUND_ERROR_CODES = frozenset({
    SmartsheetErrorCode.NOT_FOUND,
    SmartsheetErrorCode.SHEET_NOT_FOUND,
})

# Token/authentication error codes
TOKEN_ERROR_CODES = frozenset({
    SmartsheetErrorCode.INVALID_TOKEN,
    SmartsheetErrorCode.TOKEN_EXPIRED,
    SmartsheetErrorCode.INVALID_API_KEY,
    SmartsheetErrorCode.INVALID_ACCESS_TOKEN,
})


# ============================================================================
# RETRY CONFIGURATION CONSTANTS
# ============================================================================

# Default retry configuration for API calls
DEFAULT_MAX_RETRIES = 3
DEFAULT_BASE_DELAY = 1.0  # seconds
DEFAULT_MAX_DELAY = 30.0  # seconds
DEFAULT_EXPONENTIAL_BASE = 2

# Rate limiting configuration
DEFAULT_RATE_LIMIT_COOLDOWN = 60.0  # seconds
DEFAULT_QUEUE_SIZE = 100
DEFAULT_QUEUE_COOLDOWN = 60.0  # seconds

# Extended delay for certain operations
EXTENDED_MAX_DELAY = 15.0  # seconds


# ============================================================================
# PERFORMANCE THRESHOLD CONSTANTS
# ============================================================================

class PerformanceThreshold:
    """Thresholds for classifying user performance levels (items per day)."""
    EXCEPTIONAL = 50  # >= 50 items/day
    HIGH = 25         # >= 25 items/day
    AVERAGE = 10      # >= 10 items/day
    LOW = 3           # >= 3 items/day
    # MINIMAL is for anything below LOW threshold


# ============================================================================
# HEALTH SCORE CONFIGURATION
# ============================================================================

class HealthScoreDefaults:
    """Default values for health score calculation."""
    # Weight distribution (must sum to 1.0)
    ACTIVITY_WEIGHT = 0.35
    COMPLETION_WEIGHT = 0.40
    OVERDUE_WEIGHT = 0.25

    # Threshold scores for status classification
    GREEN_THRESHOLD = 70   # >= 70 is GREEN (healthy)
    YELLOW_THRESHOLD = 40  # >= 40 is YELLOW (caution), < 40 is RED (critical)

    # Lookback periods
    ACTIVITY_LOOKBACK_DAYS = 30
    OVERDUE_THRESHOLD_DAYS = 30

    # Validation tolerance for weight sum
    WEIGHT_SUM_TOLERANCE = 0.01


# ============================================================================
# WCAG ACCESSIBILITY CONSTANTS
# ============================================================================

class WCAGConstants:
    """WCAG 2.1 accessibility constants for color contrast calculations.

    Reference: https://www.w3.org/WAI/WCAG21/Understanding/contrast-minimum.html
    """
    # Color normalization threshold (sRGB linear threshold)
    SRGB_LINEAR_THRESHOLD = 0.03928

    # Gamma correction constants for sRGB to linear conversion
    GAMMA_OFFSET = 0.055
    GAMMA_DIVISOR = 1.055
    GAMMA_EXPONENT = 2.4

    # Color normalization divisor for non-linear range
    LINEAR_DIVISOR = 12.92

    # RGB to luminance weights (ITU-R BT.709)
    RED_LUMINANCE_WEIGHT = 0.2126
    GREEN_LUMINANCE_WEIGHT = 0.7152
    BLUE_LUMINANCE_WEIGHT = 0.0722

    # Contrast ratio calculation adjustment
    CONTRAST_ADJUSTMENT = 0.05

    # WCAG AA minimum contrast ratios
    NORMAL_TEXT_CONTRAST_RATIO = 4.5  # For normal text
    LARGE_TEXT_CONTRAST_RATIO = 3.0   # For large text (18pt+) and graphics

    # RGB maximum value
    RGB_MAX = 255


# ============================================================================
# CHART LAYOUT CONSTANTS
# ============================================================================

class PieChartConstants:
    """Constants for pie chart label positioning and layout."""
    # Initial label position (as fraction of radius from center)
    INITIAL_LABEL_RADIUS_RATIO = 0.6

    # Outer label position for repositioning (as fraction of radius)
    OUTER_LABEL_RADIUS_RATIO = 1.15

    # Far outer position for heavily overlapping labels
    FAR_OUTER_RADIUS_RATIO = 1.35

    # Leader line start position (as fraction of radius)
    LEADER_LINE_START_RATIO = 0.95

    # Label offset multiplier for vertical repositioning
    LABEL_OFFSET_MULTIPLIER = 1.5

    # Padding for overlap detection
    LABEL_OVERLAP_PADDING = 2

    # Starting angle for pie charts (top = 90 degrees)
    PIE_START_ANGLE = 90

    # Full circle in degrees
    FULL_CIRCLE_DEGREES = 360

    # Percentage conversion factor
    PERCENTAGE_MULTIPLIER = 100


class ChartDimensions:
    """Common chart dimension constants."""
    # Bar chart defaults
    DEFAULT_BAR_HEIGHT = 25
    DEFAULT_BAR_STROKE_WIDTH = 0.5

    # Chart container defaults
    DEFAULT_CHART_WIDTH = 500
    DEFAULT_CHART_HEIGHT = 80

    # Position offsets
    DEFAULT_X_OFFSET = 50
    DEFAULT_Y_OFFSET = 30


# ============================================================================
# REPORT PAGE LAYOUT CONSTANTS (in millimeters)
# ============================================================================

class PageLayoutMM:
    """Page layout constants in millimeters."""
    # Margins
    LEFT_MARGIN = 15
    RIGHT_MARGIN = 15
    TOP_MARGIN = 25
    BOTTOM_MARGIN = 25

    # Common spacing values
    SMALL_SPACE = 10
    MEDIUM_SPACE = 15
    LARGE_SPACE = 20
    XLARGE_SPACE = 25


class FontSizes:
    """Standard font sizes for report elements."""
    # Heading sizes
    H1 = 32
    H2 = 18
    H3 = 14

    # Body text sizes
    BODY_LARGE = 12
    BODY = 11
    BODY_SMALL = 10

    # Caption/footnote
    CAPTION = 8

    # Line height for paragraphs
    DEFAULT_LEADING = 38


class ParagraphSpacing:
    """Standard paragraph spacing in millimeters."""
    SPACE_BEFORE_SMALL = 15
    SPACE_BEFORE_MEDIUM = 20
    SPACE_BEFORE_LARGE = 25

    SPACE_AFTER_SMALL = 15
    SPACE_AFTER_MEDIUM = 20
    SPACE_AFTER_LARGE = 25


# ============================================================================
# SHEET IDS AND ROW IDS (Smartsheet-specific identifiers)
# ============================================================================

class SmartsheetIDs:
    """Smartsheet sheet and row identifiers for report metadata."""
    REPORT_METADATA_SHEET_ID = 7888169555939204
    MONTHLY_REPORT_ATTACHMENT_ROW_ID = 5089581251235716
    MONTHLY_METADATA_ROW_ID = 5089581251235716
    WEEKLY_REPORT_ATTACHMENT_ROW_ID = 1192484760260484
    WEEKLY_METADATA_ROW_ID = 1192484760260484


# ============================================================================
# MISCELLANEOUS CONSTANTS
# ============================================================================

# Minimum percentage threshold for displaying pie chart labels
DEFAULT_MIN_LABEL_PERCENTAGE = 5.0

# Default colors
DEFAULT_LEADER_LINE_COLOR = "#555555"

# Week start day (0 = Monday in Python's weekday())
WEEK_START_MONDAY = 0

# Days in a week
DAYS_IN_WEEK = 7


# ============================================================================
# LANDSCAPE MODE CONSTANTS
# ============================================================================

class LandscapeMode:
    """Constants for automatic landscape orientation detection for wide tables."""
    # Minimum number of columns to trigger landscape mode
    COLUMN_THRESHOLD = 6

    # Minimum total column width (in mm) to trigger landscape mode
    MIN_TOTAL_WIDTH_MM = 180

    # Scale factor for portrait to landscape column width adjustment
    LANDSCAPE_SCALE_FACTOR = 1.55

    # Landscape page margins (in mm)
    LANDSCAPE_LEFT_MARGIN = 15
    LANDSCAPE_RIGHT_MARGIN = 15
    LANDSCAPE_TOP_MARGIN = 15
    LANDSCAPE_BOTTOM_MARGIN = 15

    # Portrait page margins (in mm)
    PORTRAIT_LEFT_MARGIN = 25
    PORTRAIT_RIGHT_MARGIN = 25
    PORTRAIT_TOP_MARGIN = 20
    PORTRAIT_BOTTOM_MARGIN = 20

    # Available content width in landscape A4 (841pt - 2*15mm = 757pt ≈ 267mm)
    LANDSCAPE_AVAILABLE_WIDTH_MM = 267

    # Available content width in portrait A4 (595pt - 2*25mm = 453pt ≈ 160mm)
    PORTRAIT_AVAILABLE_WIDTH_MM = 160
