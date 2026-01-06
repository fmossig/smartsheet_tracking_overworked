"""
Error Collection Framework for Smartsheet Tracker

Provides a centralized system for aggregating all errors and warnings encountered
during processing. Stores errors with full context for later reporting.

Usage:
    from error_collector import (
        ErrorCollector,
        CollectedError,
        ErrorSeverity,
        ErrorCategory,
        ErrorType,
        get_global_collector,
        get_categories_for_type,
    )

    # Get the global collector instance
    collector = get_global_collector()

    # Collect an error with context
    collector.collect(
        error=some_exception,
        severity=ErrorSeverity.ERROR,
        category=ErrorCategory.API_ERROR,
        context={
            "sheet_id": 12345,
            "group": "NA",
            "row_id": 67890,
            "operation": "get_sheet",
        }
    )

    # Get all errors for reporting
    all_errors = collector.get_all_errors()
    summary = collector.get_summary()
"""

import logging
import threading
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Callable
from functools import wraps

# Get logger for this module
logger = logging.getLogger(__name__)


def _get_stack_trace_from_exception(exception: Optional[Exception]) -> Optional[str]:
    """
    Extract stack trace from an exception.

    Args:
        exception: The exception to extract trace from.

    Returns:
        Formatted stack trace string, or None if no exception.
    """
    if exception is None:
        return None

    try:
        tb_lines = traceback.format_exception(
            type(exception),
            exception,
            exception.__traceback__
        )
        return ''.join(tb_lines).strip()
    except Exception:
        return None


class ErrorSeverity(Enum):
    """Severity levels for collected errors."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


class ErrorType(Enum):
    """
    High-level error types for grouping related error categories.

    These types represent the main classification of errors:
    - DATA_QUALITY_ISSUES: Problems with data integrity, format, or completeness
    - MISSING_DATA: Absence of required data or resources
    - INVALID_FORMAT: Data that doesn't match expected patterns or types
    - API_ERRORS: Issues with external API communication
    - PERMISSION_ISSUES: Authentication and authorization problems
    - OTHER: Errors that don't fit other categories
    """
    DATA_QUALITY_ISSUES = "data_quality_issues"
    MISSING_DATA = "missing_data"
    INVALID_FORMAT = "invalid_format"
    API_ERRORS = "api_errors"
    PERMISSION_ISSUES = "permission_issues"
    OTHER = "other"

    @classmethod
    def from_string(cls, value: str) -> "ErrorType":
        """Create error type from string value."""
        value_lower = value.lower().strip()
        for member in cls:
            if member.value == value_lower:
                return member
        return cls.OTHER


class ErrorCategory(Enum):
    """Categories for classifying errors."""
    # Data quality errors
    DATA_QUALITY = "data_quality"
    MISSING_DATA = "missing_data"
    INVALID_FORMAT = "invalid_format"
    NULL_VALUE = "null_value"

    # API errors
    API_ERROR = "api_error"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    CONNECTION = "connection"

    # Authentication/Authorization errors
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    TOKEN_EXPIRED = "token_expired"
    TOKEN_INVALID = "token_invalid"

    # Resource errors
    SHEET_NOT_FOUND = "sheet_not_found"
    COLUMN_NOT_FOUND = "column_not_found"
    ROW_NOT_FOUND = "row_not_found"

    # Processing errors
    PROCESSING = "processing"
    CELL_ACCESS = "cell_access"
    DATE_PARSING = "date_parsing"
    CSV_WRITE = "csv_write"
    ROW_PROCESSING = "row_processing"
    PHASE_PROCESSING = "phase_processing"

    # Validation errors
    VALIDATION = "validation"
    MISSING_REQUIRED = "missing_required"
    TYPE_MISMATCH = "type_mismatch"
    TYPE_COERCION = "type_coercion"

    # Configuration errors
    CONFIGURATION = "configuration"

    # Other
    OTHER = "other"
    UNKNOWN = "unknown"

    def get_error_type(self) -> ErrorType:
        """
        Get the high-level error type for this category.

        Maps each ErrorCategory to its corresponding ErrorType for
        high-level filtering and grouping.
        """
        return CATEGORY_TO_TYPE_MAP.get(self, ErrorType.OTHER)


# Mapping from ErrorCategory to ErrorType for high-level classification
CATEGORY_TO_TYPE_MAP: Dict[ErrorCategory, ErrorType] = {
    # Data Quality Issues - problems with data integrity
    ErrorCategory.DATA_QUALITY: ErrorType.DATA_QUALITY_ISSUES,
    ErrorCategory.NULL_VALUE: ErrorType.DATA_QUALITY_ISSUES,
    ErrorCategory.VALIDATION: ErrorType.DATA_QUALITY_ISSUES,
    ErrorCategory.CELL_ACCESS: ErrorType.DATA_QUALITY_ISSUES,

    # Missing Data - absence of required data or resources
    ErrorCategory.MISSING_DATA: ErrorType.MISSING_DATA,
    ErrorCategory.MISSING_REQUIRED: ErrorType.MISSING_DATA,
    ErrorCategory.SHEET_NOT_FOUND: ErrorType.MISSING_DATA,
    ErrorCategory.COLUMN_NOT_FOUND: ErrorType.MISSING_DATA,
    ErrorCategory.ROW_NOT_FOUND: ErrorType.MISSING_DATA,

    # Invalid Format - data doesn't match expected patterns
    ErrorCategory.INVALID_FORMAT: ErrorType.INVALID_FORMAT,
    ErrorCategory.DATE_PARSING: ErrorType.INVALID_FORMAT,
    ErrorCategory.CONFIGURATION: ErrorType.INVALID_FORMAT,
    ErrorCategory.TYPE_MISMATCH: ErrorType.INVALID_FORMAT,
    ErrorCategory.TYPE_COERCION: ErrorType.DATA_QUALITY_ISSUES,

    # API Errors - external service communication issues
    ErrorCategory.API_ERROR: ErrorType.API_ERRORS,
    ErrorCategory.RATE_LIMIT: ErrorType.API_ERRORS,
    ErrorCategory.TIMEOUT: ErrorType.API_ERRORS,
    ErrorCategory.CONNECTION: ErrorType.API_ERRORS,

    # Permission Issues - authentication and authorization
    ErrorCategory.AUTHENTICATION: ErrorType.PERMISSION_ISSUES,
    ErrorCategory.PERMISSION: ErrorType.PERMISSION_ISSUES,
    ErrorCategory.TOKEN_EXPIRED: ErrorType.PERMISSION_ISSUES,
    ErrorCategory.TOKEN_INVALID: ErrorType.PERMISSION_ISSUES,

    # Other - processing and uncategorized errors
    ErrorCategory.PROCESSING: ErrorType.OTHER,
    ErrorCategory.CSV_WRITE: ErrorType.OTHER,
    ErrorCategory.ROW_PROCESSING: ErrorType.OTHER,
    ErrorCategory.PHASE_PROCESSING: ErrorType.OTHER,
    ErrorCategory.OTHER: ErrorType.OTHER,
    ErrorCategory.UNKNOWN: ErrorType.OTHER,
}


def get_categories_for_type(error_type: ErrorType) -> List[ErrorCategory]:
    """
    Get all ErrorCategory values that belong to a given ErrorType.

    Args:
        error_type: The high-level error type

    Returns:
        List of ErrorCategory values belonging to this type
    """
    return [
        category for category, mapped_type in CATEGORY_TO_TYPE_MAP.items()
        if mapped_type == error_type
    ]


@dataclass
class CollectedError:
    """
    Represents a single collected error with full context.

    Attributes:
        error_id: Unique identifier for this error
        timestamp: When the error occurred
        severity: Severity level of the error
        category: Category of the error
        message: Human-readable error message
        error_type: The type/class name of the exception
        error_str: String representation of the original exception
        original_exception: Reference to the original exception (if available)
        context: Dictionary of contextual information
        suggested_action: Optional suggested action to resolve the error
        is_recoverable: Whether the error is recoverable
        source_module: The module where the error occurred
        source_function: The function where the error occurred
        stack_trace: Full stack trace for debugging
        sheet_name: Name/group identifier of the sheet
        row_id: Smartsheet row ID
        field_name: Name of the field being processed
        user: User associated with the row/operation
    """
    error_id: str
    timestamp: datetime
    severity: ErrorSeverity
    category: ErrorCategory
    message: str
    error_type: Optional[str] = None
    error_str: Optional[str] = None
    original_exception: Optional[Exception] = None
    context: Dict[str, Any] = field(default_factory=dict)
    suggested_action: Optional[str] = None
    is_recoverable: bool = True
    source_module: Optional[str] = None
    source_function: Optional[str] = None
    stack_trace: Optional[str] = None
    sheet_name: Optional[str] = None
    row_id: Optional[Any] = None
    field_name: Optional[str] = None
    user: Optional[str] = None

    def get_high_level_type(self) -> ErrorType:
        """
        Get the high-level error type for this error.

        Maps the error's category to its corresponding ErrorType for
        filtering and grouping purposes.

        Returns:
            The ErrorType that this error belongs to
        """
        return self.category.get_error_type()

    def to_dict(self) -> Dict[str, Any]:
        """Convert the error to a dictionary for serialization."""
        return {
            "error_id": self.error_id,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "category": self.category.value,
            "error_type_high_level": self.get_high_level_type().value,
            "message": self.message,
            "error_type": self.error_type,
            "error_str": self.error_str,
            "context": self.context,
            "suggested_action": self.suggested_action,
            "is_recoverable": self.is_recoverable,
            "source_module": self.source_module,
            "source_function": self.source_function,
            "stack_trace": self.stack_trace,
            "sheet_name": self.sheet_name,
            "row_id": self.row_id,
            "field_name": self.field_name,
            "user": self.user,
        }

    def __str__(self) -> str:
        """Human-readable representation of the error."""
        location_parts = []

        # Use dedicated fields first, fall back to context
        sheet_name = self.sheet_name or self.context.get("group")
        row_id = self.row_id or self.context.get("row_id")
        field_name = self.field_name or self.context.get("field_name") or self.context.get("date_col")
        user = self.user or self.context.get("user")

        if sheet_name:
            location_parts.append(f"Sheet: {sheet_name}")
        if self.context.get("sheet_id"):
            location_parts.append(f"ID: {self.context['sheet_id']}")
        if row_id:
            location_parts.append(f"Row: {row_id}")
        if field_name:
            location_parts.append(f"Field: {field_name}")
        if self.context.get("phase"):
            location_parts.append(f"Phase: {self.context['phase']}")
        if user:
            location_parts.append(f"User: {user}")

        location = f"[{', '.join(location_parts)}] " if location_parts else ""
        return f"{location}{self.severity.value.upper()}: {self.message}"

    def format_detailed(self) -> str:
        """
        Format error with full context and stack trace for debugging.

        Returns:
            Multi-line string with complete error details.
        """
        lines = [
            "=" * 70,
            "ERROR DETAILS",
            "=" * 70,
            f"Error ID: {self.error_id}",
            f"Timestamp: {self.timestamp.isoformat()}",
            f"Severity: {self.severity.value.upper()}",
            f"Category: {self.category.value}",
            f"Type: {self.get_high_level_type().value}",
        ]

        # Data location
        sheet_name = self.sheet_name or self.context.get("group")
        row_id = self.row_id or self.context.get("row_id")
        field_name = self.field_name or self.context.get("field_name") or self.context.get("date_col")
        user = self.user or self.context.get("user")

        lines.append("\n--- Data Location ---")
        if sheet_name:
            lines.append(f"  Sheet Name: {sheet_name}")
        if self.context.get("sheet_id"):
            lines.append(f"  Sheet ID: {self.context['sheet_id']}")
        if row_id:
            lines.append(f"  Row ID: {row_id}")
        if field_name:
            lines.append(f"  Field Name: {field_name}")
        if self.context.get("phase"):
            lines.append(f"  Phase: {self.context['phase']}")
        if user:
            lines.append(f"  User: {user}")
        if self.context.get("marketplace"):
            lines.append(f"  Marketplace: {self.context['marketplace']}")

        # Error message
        lines.append("\n--- Error Message ---")
        lines.append(f"  {self.message}")
        if self.error_type:
            lines.append(f"  Exception Type: {self.error_type}")

        # Source location
        if self.source_module or self.source_function:
            lines.append("\n--- Source Location ---")
            if self.source_module:
                lines.append(f"  Module: {self.source_module}")
            if self.source_function:
                lines.append(f"  Function: {self.source_function}")

        # Suggested action
        if self.suggested_action:
            lines.append("\n--- Suggested Action ---")
            lines.append(f"  {self.suggested_action}")

        # Stack trace
        if self.stack_trace:
            lines.append("\n--- Stack Trace ---")
            lines.append(self.stack_trace)

        # Additional context
        other_context = {
            k: v for k, v in self.context.items()
            if k not in ["group", "sheet_id", "row_id", "field_name", "date_col",
                         "phase", "user", "marketplace", "stack_trace"]
        }
        if other_context:
            lines.append("\n--- Additional Context ---")
            for key, value in other_context.items():
                lines.append(f"  {key}: {value}")

        lines.append("=" * 70)
        return "\n".join(lines)


class ErrorCollector:
    """
    Centralized error collection system for aggregating errors during processing.

    This class provides thread-safe error collection with:
    - Categorization and severity levels
    - Full context preservation
    - Statistics tracking
    - Export capabilities for reporting

    Usage:
        collector = ErrorCollector()

        # Collect an error
        collector.collect(
            error=my_exception,
            severity=ErrorSeverity.ERROR,
            category=ErrorCategory.API_ERROR,
            context={"sheet_id": 12345}
        )

        # Get summary
        print(collector.get_summary())
    """

    def __init__(self, max_errors: int = 10000, log_on_collect: bool = True):
        """
        Initialize the error collector.

        Args:
            max_errors: Maximum number of errors to store (prevents memory issues)
            log_on_collect: Whether to log errors when they are collected
        """
        self._errors: List[CollectedError] = []
        self._lock = threading.RLock()
        self._max_errors = max_errors
        self._log_on_collect = log_on_collect
        self._error_counter = 0
        self._overflow_count = 0

        # Statistics counters
        self._stats = {
            "total_collected": 0,
            "by_severity": {s.value: 0 for s in ErrorSeverity},
            "by_category": {c.value: 0 for c in ErrorCategory},
            "by_group": {},
            "by_sheet_id": {},
            "first_error_time": None,
            "last_error_time": None,
        }

    def _generate_error_id(self) -> str:
        """Generate a unique error ID."""
        self._error_counter += 1
        return f"ERR-{datetime.now().strftime('%Y%m%d%H%M%S')}-{self._error_counter:06d}"

    def _get_suggested_action(self, category: ErrorCategory, context: Dict[str, Any]) -> Optional[str]:
        """Get suggested action based on error category."""
        suggestions = {
            ErrorCategory.TOKEN_EXPIRED: (
                "Your Smartsheet API token has expired. "
                "Generate a new token at Account > Personal Settings > API Access "
                "and update SMARTSHEET_TOKEN in your .env file."
            ),
            ErrorCategory.TOKEN_INVALID: (
                "Your Smartsheet API token is invalid. "
                "Verify SMARTSHEET_TOKEN in your .env file is correct."
            ),
            ErrorCategory.PERMISSION: (
                "You do not have permission to access this resource. "
                "Contact the sheet owner to request access."
            ),
            ErrorCategory.SHEET_NOT_FOUND: (
                "The sheet could not be found. "
                "Verify the sheet ID is correct and the sheet has not been deleted."
            ),
            ErrorCategory.RATE_LIMIT: (
                "API rate limit exceeded. "
                "The system will automatically retry after the cooldown period."
            ),
            ErrorCategory.MISSING_DATA: (
                "Required data is missing. "
                "Review the data source and ensure all required fields are populated."
            ),
            ErrorCategory.INVALID_FORMAT: (
                "Data format is invalid. "
                "Check the data format matches expected patterns."
            ),
            ErrorCategory.DATE_PARSING: (
                "Could not parse date value. "
                "Ensure date is in a recognized format (e.g., YYYY-MM-DD)."
            ),
        }
        return suggestions.get(category)

    def collect(
        self,
        error: Optional[Exception] = None,
        severity: ErrorSeverity = ErrorSeverity.ERROR,
        category: ErrorCategory = ErrorCategory.UNKNOWN,
        message: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        suggested_action: Optional[str] = None,
        is_recoverable: bool = True,
        source_module: Optional[str] = None,
        source_function: Optional[str] = None,
        include_stack_trace: bool = True,
        sheet_name: Optional[str] = None,
        row_id: Optional[Any] = None,
        field_name: Optional[str] = None,
        user: Optional[str] = None,
    ) -> CollectedError:
        """
        Collect an error with full context including stack trace.

        Args:
            error: The exception that occurred (optional)
            severity: Severity level of the error
            category: Category of the error
            message: Human-readable message (defaults to str(error))
            context: Dictionary of contextual information
            suggested_action: Optional suggested action to resolve
            is_recoverable: Whether processing can continue
            source_module: The module where the error occurred
            source_function: The function where the error occurred
            include_stack_trace: Whether to capture and store stack trace
            sheet_name: Name/group identifier of the sheet (e.g., "NA", "NF")
            row_id: Smartsheet row ID
            field_name: Name of the field being processed
            user: User associated with the row/operation

        Returns:
            The CollectedError that was created
        """
        with self._lock:
            # Check capacity
            if len(self._errors) >= self._max_errors:
                self._overflow_count += 1
                if self._overflow_count == 1:
                    logger.warning(
                        f"Error collector reached max capacity ({self._max_errors}). "
                        f"Oldest errors will be removed to make room for new ones."
                    )
                # Remove oldest error
                self._errors.pop(0)

            # Build the collected error
            error_id = self._generate_error_id()
            timestamp = datetime.now()

            # Determine message
            if message is None:
                if error is not None:
                    message = str(error)
                else:
                    message = f"Unknown {category.value} error"

            # Get error details
            error_type = type(error).__name__ if error else None
            error_str = str(error) if error else None

            # Capture stack trace if enabled and error is provided
            stack_trace = None
            if include_stack_trace and error is not None:
                stack_trace = _get_stack_trace_from_exception(error)

            # Get suggested action if not provided
            if suggested_action is None:
                suggested_action = self._get_suggested_action(category, context or {})

            # Extract context fields if not explicitly provided
            ctx = context or {}
            if sheet_name is None:
                sheet_name = ctx.get("group")
            if row_id is None:
                row_id = ctx.get("row_id")
            if field_name is None:
                field_name = ctx.get("field_name") or ctx.get("date_col")
            if user is None:
                user = ctx.get("user")

            collected_error = CollectedError(
                error_id=error_id,
                timestamp=timestamp,
                severity=severity,
                category=category,
                message=message,
                error_type=error_type,
                error_str=error_str,
                original_exception=error,
                context=ctx,
                suggested_action=suggested_action,
                is_recoverable=is_recoverable,
                source_module=source_module,
                source_function=source_function,
                stack_trace=stack_trace,
                sheet_name=sheet_name,
                row_id=row_id,
                field_name=field_name,
                user=user,
            )

            # Store the error
            self._errors.append(collected_error)

            # Update statistics
            self._stats["total_collected"] += 1
            self._stats["by_severity"][severity.value] += 1
            self._stats["by_category"][category.value] += 1

            if context:
                if "group" in context:
                    group = context["group"]
                    self._stats["by_group"][group] = self._stats["by_group"].get(group, 0) + 1
                if "sheet_id" in context:
                    sheet_id = str(context["sheet_id"])
                    self._stats["by_sheet_id"][sheet_id] = self._stats["by_sheet_id"].get(sheet_id, 0) + 1

            if self._stats["first_error_time"] is None:
                self._stats["first_error_time"] = timestamp
            self._stats["last_error_time"] = timestamp

            # Log if configured
            if self._log_on_collect:
                log_level = {
                    ErrorSeverity.INFO: logging.INFO,
                    ErrorSeverity.WARNING: logging.WARNING,
                    ErrorSeverity.ERROR: logging.ERROR,
                    ErrorSeverity.CRITICAL: logging.CRITICAL,
                }.get(severity, logging.ERROR)

                # Use detailed format for ERROR and CRITICAL levels to aid debugging
                if severity in (ErrorSeverity.ERROR, ErrorSeverity.CRITICAL):
                    logger.log(log_level, f"Collected error:\n{collected_error.format_detailed()}")
                else:
                    logger.log(log_level, f"Collected error: {collected_error}")

            return collected_error

    def collect_from_exception(
        self,
        error: Exception,
        context: Optional[Dict[str, Any]] = None,
        source_module: Optional[str] = None,
        source_function: Optional[str] = None,
    ) -> CollectedError:
        """
        Collect an error by automatically detecting category and severity from exception type.

        Args:
            error: The exception that occurred
            context: Dictionary of contextual information
            source_module: The module where the error occurred
            source_function: The function where the error occurred

        Returns:
            The CollectedError that was created
        """
        # Import here to avoid circular imports
        from smartsheet_retry import (
            SmartsheetTimeoutError,
            RateLimitError,
            SheetNotFoundError,
            TokenAuthenticationError,
            PermissionDeniedError,
        )

        # Determine category and severity based on exception type
        severity = ErrorSeverity.ERROR
        category = ErrorCategory.UNKNOWN
        is_recoverable = True
        suggested_action = None

        if isinstance(error, TokenAuthenticationError):
            category = ErrorCategory.TOKEN_EXPIRED if error.is_expired else ErrorCategory.TOKEN_INVALID
            is_recoverable = False
            suggested_action = error.get_actionable_message()
            severity = ErrorSeverity.CRITICAL
        elif isinstance(error, PermissionDeniedError):
            category = ErrorCategory.PERMISSION
            suggested_action = error.get_actionable_message()
            severity = ErrorSeverity.WARNING
        elif isinstance(error, SheetNotFoundError):
            category = ErrorCategory.SHEET_NOT_FOUND
            severity = ErrorSeverity.WARNING
        elif isinstance(error, RateLimitError):
            category = ErrorCategory.RATE_LIMIT
            severity = ErrorSeverity.WARNING
        elif isinstance(error, SmartsheetTimeoutError):
            category = ErrorCategory.TIMEOUT
            severity = ErrorSeverity.ERROR
        elif isinstance(error, (TimeoutError, ConnectionError)):
            category = ErrorCategory.CONNECTION
        elif "validation" in type(error).__name__.lower():
            category = ErrorCategory.VALIDATION
            severity = ErrorSeverity.WARNING
        elif isinstance(error, ValueError):
            category = ErrorCategory.INVALID_FORMAT
            severity = ErrorSeverity.WARNING
        elif isinstance(error, KeyError):
            category = ErrorCategory.MISSING_DATA
            severity = ErrorSeverity.WARNING

        return self.collect(
            error=error,
            severity=severity,
            category=category,
            context=context,
            suggested_action=suggested_action,
            is_recoverable=is_recoverable,
            source_module=source_module,
            source_function=source_function,
        )

    def get_all_errors(self) -> List[CollectedError]:
        """Get all collected errors."""
        with self._lock:
            return list(self._errors)

    def get_errors_by_severity(self, severity: ErrorSeverity) -> List[CollectedError]:
        """Get errors filtered by severity level."""
        with self._lock:
            return [e for e in self._errors if e.severity == severity]

    def get_errors_by_category(self, category: ErrorCategory) -> List[CollectedError]:
        """Get errors filtered by category."""
        with self._lock:
            return [e for e in self._errors if e.category == category]

    def get_errors_by_group(self, group: str) -> List[CollectedError]:
        """Get errors filtered by group."""
        with self._lock:
            return [e for e in self._errors if e.context.get("group") == group]

    def get_errors_by_sheet(self, sheet_id: Any) -> List[CollectedError]:
        """Get errors filtered by sheet ID."""
        sheet_id_str = str(sheet_id)
        with self._lock:
            return [e for e in self._errors if str(e.context.get("sheet_id")) == sheet_id_str]

    def get_critical_errors(self) -> List[CollectedError]:
        """Get all critical and non-recoverable errors."""
        with self._lock:
            return [
                e for e in self._errors
                if e.severity == ErrorSeverity.CRITICAL or not e.is_recoverable
            ]

    def get_errors_by_type(self, error_type: ErrorType) -> List[CollectedError]:
        """
        Get errors filtered by high-level error type.

        Args:
            error_type: The ErrorType to filter by

        Returns:
            List of errors belonging to the specified type
        """
        with self._lock:
            return [e for e in self._errors if e.get_high_level_type() == error_type]

    def get_data_quality_issues(self) -> List[CollectedError]:
        """Get all data quality issue errors."""
        return self.get_errors_by_type(ErrorType.DATA_QUALITY_ISSUES)

    def get_missing_data_errors(self) -> List[CollectedError]:
        """Get all missing data errors."""
        return self.get_errors_by_type(ErrorType.MISSING_DATA)

    def get_invalid_format_errors(self) -> List[CollectedError]:
        """Get all invalid format errors."""
        return self.get_errors_by_type(ErrorType.INVALID_FORMAT)

    def get_api_errors(self) -> List[CollectedError]:
        """Get all API-related errors."""
        return self.get_errors_by_type(ErrorType.API_ERRORS)

    def get_permission_issues(self) -> List[CollectedError]:
        """Get all permission and authentication errors."""
        return self.get_errors_by_type(ErrorType.PERMISSION_ISSUES)

    def filter_errors(
        self,
        error_type: Optional[ErrorType] = None,
        category: Optional[ErrorCategory] = None,
        severity: Optional[ErrorSeverity] = None,
        group: Optional[str] = None,
        sheet_id: Optional[Any] = None,
        is_recoverable: Optional[bool] = None,
    ) -> List[CollectedError]:
        """
        Filter errors by multiple criteria.

        All specified criteria must match (AND logic). Unspecified criteria
        are ignored.

        Args:
            error_type: Filter by high-level error type
            category: Filter by error category
            severity: Filter by severity level
            group: Filter by group context
            sheet_id: Filter by sheet ID context
            is_recoverable: Filter by recoverability

        Returns:
            List of errors matching all specified criteria
        """
        with self._lock:
            result = list(self._errors)

            if error_type is not None:
                result = [e for e in result if e.get_high_level_type() == error_type]

            if category is not None:
                result = [e for e in result if e.category == category]

            if severity is not None:
                result = [e for e in result if e.severity == severity]

            if group is not None:
                result = [e for e in result if e.context.get("group") == group]

            if sheet_id is not None:
                sheet_id_str = str(sheet_id)
                result = [e for e in result if str(e.context.get("sheet_id")) == sheet_id_str]

            if is_recoverable is not None:
                result = [e for e in result if e.is_recoverable == is_recoverable]

            return result

    def group_by_type(self) -> Dict[ErrorType, List[CollectedError]]:
        """
        Group all errors by their high-level error type.

        Returns:
            Dictionary mapping ErrorType to list of errors
        """
        with self._lock:
            result: Dict[ErrorType, List[CollectedError]] = {
                error_type: [] for error_type in ErrorType
            }
            for error in self._errors:
                error_type = error.get_high_level_type()
                result[error_type].append(error)
            return result

    def group_by_category(self) -> Dict[ErrorCategory, List[CollectedError]]:
        """
        Group all errors by their category.

        Returns:
            Dictionary mapping ErrorCategory to list of errors
        """
        with self._lock:
            result: Dict[ErrorCategory, List[CollectedError]] = {
                category: [] for category in ErrorCategory
            }
            for error in self._errors:
                result[error.category].append(error)
            return result

    def group_by_severity(self) -> Dict[ErrorSeverity, List[CollectedError]]:
        """
        Group all errors by their severity level.

        Returns:
            Dictionary mapping ErrorSeverity to list of errors
        """
        with self._lock:
            result: Dict[ErrorSeverity, List[CollectedError]] = {
                severity: [] for severity in ErrorSeverity
            }
            for error in self._errors:
                result[error.severity].append(error)
            return result

    def get_type_statistics(self) -> Dict[str, int]:
        """
        Get count of errors by high-level error type.

        Returns:
            Dictionary mapping error type names to counts
        """
        with self._lock:
            result = {error_type.value: 0 for error_type in ErrorType}
            for error in self._errors:
                error_type = error.get_high_level_type()
                result[error_type.value] += 1
            return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about collected errors."""
        with self._lock:
            stats = self._stats.copy()
            stats["current_count"] = len(self._errors)
            stats["overflow_count"] = self._overflow_count
            stats["max_capacity"] = self._max_errors
            # Add type statistics
            stats["by_type"] = self.get_type_statistics()
            return stats

    def get_summary(self) -> str:
        """Get a human-readable summary of collected errors."""
        with self._lock:
            lines = [
                "=" * 60,
                "ERROR COLLECTION SUMMARY",
                "=" * 60,
                f"Total errors collected: {self._stats['total_collected']}",
                f"Currently stored: {len(self._errors)} (max: {self._max_errors})",
            ]

            if self._overflow_count > 0:
                lines.append(f"Overflow count (oldest removed): {self._overflow_count}")

            if self._stats["first_error_time"]:
                lines.append(f"First error: {self._stats['first_error_time'].strftime('%Y-%m-%d %H:%M:%S')}")
            if self._stats["last_error_time"]:
                lines.append(f"Last error: {self._stats['last_error_time'].strftime('%Y-%m-%d %H:%M:%S')}")

            # By severity
            lines.append("\nErrors by Severity:")
            for severity_val, count in sorted(self._stats["by_severity"].items(), key=lambda x: -x[1]):
                if count > 0:
                    lines.append(f"  {severity_val.upper()}: {count}")

            # By high-level type (only non-zero)
            type_stats = self.get_type_statistics()
            type_counts = {k: v for k, v in type_stats.items() if v > 0}
            if type_counts:
                lines.append("\nErrors by Type:")
                for type_val, count in sorted(type_counts.items(), key=lambda x: -x[1]):
                    # Format type name for display (e.g., "data_quality_issues" -> "Data Quality Issues")
                    display_name = type_val.replace("_", " ").title()
                    lines.append(f"  {display_name}: {count}")

            # By category (only non-zero)
            category_counts = {k: v for k, v in self._stats["by_category"].items() if v > 0}
            if category_counts:
                lines.append("\nErrors by Category:")
                for category_val, count in sorted(category_counts.items(), key=lambda x: -x[1]):
                    lines.append(f"  {category_val}: {count}")

            # By group (only non-zero)
            if self._stats["by_group"]:
                lines.append("\nErrors by Group:")
                for group, count in sorted(self._stats["by_group"].items(), key=lambda x: -x[1]):
                    lines.append(f"  {group}: {count}")

            lines.append("=" * 60)

            return "\n".join(lines)

    def export_for_report(self) -> List[Dict[str, Any]]:
        """Export all errors as dictionaries for report generation."""
        with self._lock:
            return [e.to_dict() for e in self._errors]

    def clear(self) -> int:
        """Clear all collected errors. Returns the number of errors cleared."""
        with self._lock:
            count = len(self._errors)
            self._errors.clear()
            self._overflow_count = 0
            # Reset stats
            self._stats = {
                "total_collected": 0,
                "by_severity": {s.value: 0 for s in ErrorSeverity},
                "by_category": {c.value: 0 for c in ErrorCategory},
                "by_group": {},
                "by_sheet_id": {},
                "first_error_time": None,
                "last_error_time": None,
            }
            return count

    def has_critical_errors(self) -> bool:
        """Check if any critical or non-recoverable errors were collected."""
        with self._lock:
            return any(
                e.severity == ErrorSeverity.CRITICAL or not e.is_recoverable
                for e in self._errors
            )

    def has_errors(self) -> bool:
        """Check if any errors (not warnings/info) were collected."""
        with self._lock:
            return any(
                e.severity in (ErrorSeverity.ERROR, ErrorSeverity.CRITICAL)
                for e in self._errors
            )

    def error_count(self) -> int:
        """Get the total number of collected errors."""
        with self._lock:
            return len(self._errors)

    def log_summary(self, log_level: int = logging.INFO) -> None:
        """Log the error collection summary."""
        logger.log(log_level, self.get_summary())


# Global collector instance
_global_collector: Optional[ErrorCollector] = None
_global_collector_lock = threading.Lock()


def get_global_collector() -> ErrorCollector:
    """Get the global error collector instance (creates one if it doesn't exist)."""
    global _global_collector
    with _global_collector_lock:
        if _global_collector is None:
            _global_collector = ErrorCollector()
        return _global_collector


def reset_global_collector() -> None:
    """Reset the global error collector (creates a new instance)."""
    global _global_collector
    with _global_collector_lock:
        _global_collector = ErrorCollector()


def collect_error(
    error: Optional[Exception] = None,
    severity: ErrorSeverity = ErrorSeverity.ERROR,
    category: ErrorCategory = ErrorCategory.UNKNOWN,
    message: Optional[str] = None,
    context: Optional[Dict[str, Any]] = None,
    **kwargs
) -> CollectedError:
    """
    Convenience function to collect an error using the global collector.

    Args:
        error: The exception that occurred (optional)
        severity: Severity level of the error
        category: Category of the error
        message: Human-readable message
        context: Dictionary of contextual information
        **kwargs: Additional arguments passed to ErrorCollector.collect()

    Returns:
        The CollectedError that was created
    """
    return get_global_collector().collect(
        error=error,
        severity=severity,
        category=category,
        message=message,
        context=context,
        **kwargs
    )


def collect_errors(func: Callable = None, *,
                   category: ErrorCategory = ErrorCategory.PROCESSING,
                   severity: ErrorSeverity = ErrorSeverity.ERROR,
                   reraise: bool = True):
    """
    Decorator to automatically collect errors from a function.

    Usage:
        @collect_errors(category=ErrorCategory.API_ERROR)
        def my_function():
            ...

        # Or with default settings:
        @collect_errors
        def my_function():
            ...
    """
    def decorator(fn):
        @wraps(fn)
        def wrapper(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except Exception as e:
                get_global_collector().collect(
                    error=e,
                    severity=severity,
                    category=category,
                    source_module=fn.__module__,
                    source_function=fn.__name__,
                )
                if reraise:
                    raise
                return None
        return wrapper

    if func is not None:
        # Decorator used without parentheses
        return decorator(func)
    return decorator
