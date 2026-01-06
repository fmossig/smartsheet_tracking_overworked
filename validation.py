"""
Validation Module for Smartsheet Tracker

Provides utilities for validating null, None, and empty string values in critical fields.
Implements graceful degradation by logging warnings and skipping rows with missing required data
instead of crashing.

Also handles unicode normalization for text fields to ensure consistent processing
of user names, comments, and other text data containing unicode characters.

Usage:
    from validation import (
        validate_row_data,
        validate_csv_row,
        is_empty_value,
        ValidationResult,
        ValidationStats,
        sanitize_value,
    )
"""

import logging
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple
from enum import Enum
from unicode_utilities import normalize_unicode, remove_invisible_chars

# Get logger from parent module or create new one
logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


@dataclass
class ValidationIssue:
    """Represents a single validation issue."""
    field_name: str
    issue_type: str
    message: str
    severity: ValidationSeverity = ValidationSeverity.WARNING
    row_id: Optional[str] = None
    group: Optional[str] = None

    def __str__(self) -> str:
        location = ""
        if self.group and self.row_id:
            location = f"[{self.group}:{self.row_id}] "
        elif self.group:
            location = f"[{self.group}] "
        elif self.row_id:
            location = f"[Row:{self.row_id}] "
        return f"{location}{self.field_name}: {self.message}"


@dataclass
class ValidationResult:
    """Result of validating a single row or record."""
    is_valid: bool
    issues: List[ValidationIssue] = field(default_factory=list)
    skipped: bool = False
    skip_reason: Optional[str] = None

    def add_issue(self, issue: ValidationIssue) -> None:
        """Add a validation issue to the result."""
        self.issues.append(issue)

    def has_errors(self) -> bool:
        """Check if there are any error-level issues."""
        return any(i.severity == ValidationSeverity.ERROR for i in self.issues)

    def has_warnings(self) -> bool:
        """Check if there are any warning-level issues."""
        return any(i.severity == ValidationSeverity.WARNING for i in self.issues)


@dataclass
class ValidationStats:
    """Statistics for validation runs."""
    total_rows: int = 0
    valid_rows: int = 0
    skipped_rows: int = 0
    rows_with_warnings: int = 0
    rows_with_errors: int = 0
    issues_by_field: Dict[str, int] = field(default_factory=dict)
    issues_by_type: Dict[str, int] = field(default_factory=dict)
    skipped_by_reason: Dict[str, int] = field(default_factory=dict)
    # Row-level error tracking for error isolation
    processing_errors: int = 0
    errors_by_category: Dict[str, int] = field(default_factory=dict)
    failed_row_ids: List[str] = field(default_factory=list)
    error_details: List[Dict[str, Any]] = field(default_factory=list)

    def record_result(self, result: ValidationResult) -> None:
        """Record a validation result in the statistics."""
        self.total_rows += 1

        if result.is_valid and not result.skipped:
            self.valid_rows += 1

        if result.skipped:
            self.skipped_rows += 1
            if result.skip_reason:
                self.skipped_by_reason[result.skip_reason] = \
                    self.skipped_by_reason.get(result.skip_reason, 0) + 1

        if result.has_errors():
            self.rows_with_errors += 1
        elif result.has_warnings():
            self.rows_with_warnings += 1

        for issue in result.issues:
            self.issues_by_field[issue.field_name] = \
                self.issues_by_field.get(issue.field_name, 0) + 1
            self.issues_by_type[issue.issue_type] = \
                self.issues_by_type.get(issue.issue_type, 0) + 1

    def record_processing_error(
        self,
        row_id: Optional[str] = None,
        group: Optional[str] = None,
        error: Optional[Exception] = None,
        error_category: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None
    ) -> None:
        """Record a processing error that occurred during row processing.

        Args:
            row_id: Identifier of the row where the error occurred
            group: Group code (e.g., "NA", "NF")
            error: The exception that was caught
            error_category: Category of error (e.g., "cell_access", "date_parsing", "csv_write")
            context: Additional context information about the error
        """
        self.processing_errors += 1

        # Track by category
        category = error_category or type(error).__name__ if error else "unknown"
        self.errors_by_category[category] = self.errors_by_category.get(category, 0) + 1

        # Track failed row IDs
        if row_id:
            row_identifier = f"{group}:{row_id}" if group else str(row_id)
            if row_identifier not in self.failed_row_ids:
                self.failed_row_ids.append(row_identifier)

        # Store detailed error info (limit to prevent memory issues)
        if len(self.error_details) < 100:  # Cap at 100 detailed errors
            error_detail = {
                "row_id": row_id,
                "group": group,
                "error_type": type(error).__name__ if error else None,
                "error_message": str(error) if error else None,
                "category": category,
                "context": context or {},
            }
            self.error_details.append(error_detail)

    def get_summary(self) -> str:
        """Get a summary of validation statistics."""
        lines = [
            f"Validation Summary:",
            f"  Total rows processed: {self.total_rows}",
            f"  Valid rows: {self.valid_rows}",
            f"  Skipped rows: {self.skipped_rows}",
            f"  Rows with warnings: {self.rows_with_warnings}",
            f"  Rows with errors: {self.rows_with_errors}",
        ]

        # Add processing error statistics
        if self.processing_errors > 0:
            lines.append(f"  Processing errors (isolated): {self.processing_errors}")
            lines.append(f"  Failed row IDs: {len(self.failed_row_ids)}")

        if self.skipped_by_reason:
            lines.append("  Skip reasons:")
            for reason, count in sorted(self.skipped_by_reason.items(), key=lambda x: -x[1]):
                lines.append(f"    - {reason}: {count}")

        if self.issues_by_field:
            lines.append("  Issues by field:")
            for field_name, count in sorted(self.issues_by_field.items(), key=lambda x: -x[1]):
                lines.append(f"    - {field_name}: {count}")

        # Add error category breakdown
        if self.errors_by_category:
            lines.append("  Errors by category:")
            for category, count in sorted(self.errors_by_category.items(), key=lambda x: -x[1]):
                lines.append(f"    - {category}: {count}")

        return "\n".join(lines)

    def log_summary(self, log_level: int = logging.INFO) -> None:
        """Log the validation summary."""
        logger.log(log_level, self.get_summary())

    def sync_to_error_collector(self, error_collector) -> int:
        """Sync validation errors to the centralized error collector.

        Args:
            error_collector: An ErrorCollector instance to sync errors to

        Returns:
            Number of errors synced to the collector
        """
        # Import here to avoid circular imports
        from error_collector import ErrorSeverity, ErrorCategory

        synced_count = 0

        # Map validation categories to error collector categories
        category_map = {
            "cell_access": ErrorCategory.CELL_ACCESS,
            "date_parsing": ErrorCategory.DATE_PARSING,
            "csv_write": ErrorCategory.CSV_WRITE,
            "row_processing": ErrorCategory.ROW_PROCESSING,
            "phase_processing": ErrorCategory.PHASE_PROCESSING,
            "row_id_access": ErrorCategory.CELL_ACCESS,
            "missing_required": ErrorCategory.MISSING_REQUIRED,
            "null_like_value": ErrorCategory.NULL_VALUE,
            "type_mismatch": ErrorCategory.TYPE_MISMATCH,
            "type_coercion_warning": ErrorCategory.TYPE_COERCION,
        }

        for error_detail in self.error_details:
            cat_str = error_detail.get("category", "unknown")
            category = category_map.get(cat_str, ErrorCategory.VALIDATION)

            error_collector.collect(
                message=error_detail.get("error_message", "Validation error"),
                severity=ErrorSeverity.WARNING,
                category=category,
                context={
                    "row_id": error_detail.get("row_id"),
                    "group": error_detail.get("group"),
                    **error_detail.get("context", {})
                },
                source_module="validation"
            )
            synced_count += 1

        return synced_count


def is_empty_value(value: Any) -> bool:
    """Check if a value is null, None, or an empty string.

    Args:
        value: The value to check

    Returns:
        True if the value is considered empty (None, empty string, or whitespace-only string)
    """
    if value is None:
        return True
    if isinstance(value, str):
        return value.strip() == ""
    return False


def is_null_like(value: Any) -> bool:
    """Check if a value represents a null-like value.

    This includes None, empty strings, and string representations of null.

    Args:
        value: The value to check

    Returns:
        True if the value is null-like
    """
    if value is None:
        return True
    if isinstance(value, str):
        stripped = value.strip().lower()
        return stripped in ("", "none", "null", "n/a", "na", "-")
    return False


def get_string_value(value: Any, default: str = "") -> str:
    """Safely convert a value to string, handling None and empty values.

    Args:
        value: The value to convert
        default: Default value if the input is empty

    Returns:
        String representation of the value, or default if empty
    """
    if is_empty_value(value):
        return default
    return str(value).strip()


# Define required and optional fields for different contexts
SMARTSHEET_ROW_REQUIRED_FIELDS = {"row_id"}
SMARTSHEET_ROW_OPTIONAL_FIELDS = {"date_value", "user_value", "marketplace", "group"}

CSV_ROW_REQUIRED_FIELDS = {"Timestamp", "Group", "RowID", "Phase", "DateField", "Date"}
CSV_ROW_OPTIONAL_FIELDS = {"User", "Marketplace"}

# Field-specific validation rules
FIELD_VALIDATION_RULES = {
    "row_id": {
        "type": "required",
        "allow_empty": False,
        "description": "Row identifier",
    },
    "group": {
        "type": "required",
        "allow_empty": False,
        "description": "Group code (e.g., NA, NF)",
    },
    "date_value": {
        "type": "required_for_change",
        "allow_empty": True,  # Empty means no change to track
        "description": "Date field value",
    },
    "user_value": {
        "type": "optional",
        "allow_empty": True,
        "default": "",
        "description": "User initials",
    },
    "marketplace": {
        "type": "optional",
        "allow_empty": True,
        "default": "",
        "description": "Marketplace code",
    },
    "Timestamp": {
        "type": "required",
        "allow_empty": False,
        "description": "Change timestamp",
    },
    "Group": {
        "type": "required",
        "allow_empty": False,
        "description": "Group code",
    },
    "RowID": {
        "type": "required",
        "allow_empty": False,
        "description": "Smartsheet row ID",
    },
    "Phase": {
        "type": "required",
        "allow_empty": False,
        "description": "Phase number (1-5)",
    },
    "DateField": {
        "type": "required",
        "allow_empty": False,
        "description": "Name of the date field",
    },
    "Date": {
        "type": "required",
        "allow_empty": False,
        "description": "Date value",
    },
    "User": {
        "type": "optional",
        "allow_empty": True,
        "default": "",
        "description": "User initials",
    },
    "Marketplace": {
        "type": "optional",
        "allow_empty": True,
        "default": "",
        "description": "Marketplace code",
    },
}


def validate_field(
    field_name: str,
    value: Any,
    row_id: Optional[str] = None,
    group: Optional[str] = None,
    custom_rules: Optional[Dict] = None
) -> Tuple[bool, Optional[ValidationIssue]]:
    """Validate a single field value.

    Args:
        field_name: Name of the field to validate
        value: The value to validate
        row_id: Optional row identifier for context
        group: Optional group identifier for context
        custom_rules: Optional custom validation rules

    Returns:
        Tuple of (is_valid, issue) where issue is None if valid
    """
    rules = custom_rules or FIELD_VALIDATION_RULES.get(field_name, {})
    field_type = rules.get("type", "optional")
    allow_empty = rules.get("allow_empty", True)

    # Check for empty values
    if is_empty_value(value):
        if field_type == "required" and not allow_empty:
            issue = ValidationIssue(
                field_name=field_name,
                issue_type="missing_required",
                message=f"Required field '{field_name}' is empty or null",
                severity=ValidationSeverity.ERROR,
                row_id=row_id,
                group=group,
            )
            return False, issue
        elif field_type == "required_for_change":
            # Not an error, just means nothing to process
            return True, None
        else:
            # Optional field is empty - that's OK
            return True, None

    # Check for null-like string values in required fields
    if is_null_like(value) and field_type == "required":
        issue = ValidationIssue(
            field_name=field_name,
            issue_type="null_like_value",
            message=f"Field '{field_name}' contains null-like value: '{value}'",
            severity=ValidationSeverity.WARNING,
            row_id=row_id,
            group=group,
        )
        return True, issue  # Valid but with warning

    return True, None


def validate_row_data(
    row_id: Any,
    group: str,
    date_value: Any = None,
    user_value: Any = None,
    marketplace: Any = None,
    phase: Any = None,
    date_field: str = None,
    log_issues: bool = True
) -> ValidationResult:
    """Validate a Smartsheet row's data for change tracking.

    This function validates the critical fields needed for tracking changes.
    It logs warnings and marks rows for skipping if required data is missing.

    Args:
        row_id: Smartsheet row ID (required)
        group: Group code like "NA", "NF" (required)
        date_value: Date cell value (required for tracking a change)
        user_value: User cell value (optional, defaults to empty string)
        marketplace: Marketplace value (optional, defaults to empty string)
        phase: Phase number (for context in logging)
        date_field: Date field name (for context in logging)
        log_issues: Whether to log validation issues

    Returns:
        ValidationResult with validation status and any issues
    """
    result = ValidationResult(is_valid=True)
    context_str = f"{group}:{row_id}" if group and row_id else str(row_id or "unknown")

    # Validate row_id (absolutely required)
    if is_empty_value(row_id):
        issue = ValidationIssue(
            field_name="row_id",
            issue_type="missing_required",
            message="Row ID is null or empty - cannot track changes",
            severity=ValidationSeverity.ERROR,
            row_id=str(row_id) if row_id else None,
            group=group,
        )
        result.add_issue(issue)
        result.is_valid = False
        result.skipped = True
        result.skip_reason = "missing_row_id"

        if log_issues:
            logger.warning(f"Skipping row: {issue}")
        return result

    # Validate group (required for proper tracking)
    if is_empty_value(group):
        issue = ValidationIssue(
            field_name="group",
            issue_type="missing_required",
            message="Group code is null or empty",
            severity=ValidationSeverity.ERROR,
            row_id=str(row_id),
            group=None,
        )
        result.add_issue(issue)
        result.is_valid = False
        result.skipped = True
        result.skip_reason = "missing_group"

        if log_issues:
            logger.warning(f"Skipping row {row_id}: {issue}")
        return result

    # Validate date_value (required to have something to track)
    if is_empty_value(date_value):
        # This is not an error - it just means there's nothing to track for this phase
        # Don't add an issue, just indicate it should be skipped for this phase
        result.skipped = True
        result.skip_reason = "empty_date_value"
        # Note: This is a normal condition, not an error, so don't log at warning level
        return result

    # Check for null-like values in date_value
    if is_null_like(date_value) and not is_empty_value(date_value):
        issue = ValidationIssue(
            field_name="date_value",
            issue_type="null_like_value",
            message=f"Date value contains null-like string: '{date_value}'",
            severity=ValidationSeverity.WARNING,
            row_id=str(row_id),
            group=group,
        )
        result.add_issue(issue)
        result.skipped = True
        result.skip_reason = "null_like_date_value"

        if log_issues:
            logger.warning(f"Skipping {context_str}: {issue}")
        return result

    # Validate user_value (optional but log if null-like)
    if is_null_like(user_value) and not is_empty_value(user_value):
        issue = ValidationIssue(
            field_name="user_value",
            issue_type="null_like_value",
            message=f"User value contains null-like string: '{user_value}'",
            severity=ValidationSeverity.INFO,
            row_id=str(row_id),
            group=group,
        )
        result.add_issue(issue)
        # Don't skip, just note the issue
        if log_issues:
            logger.info(f"{context_str}: Using empty string for null-like user value")

    # Validate marketplace (optional but log if null-like)
    if is_null_like(marketplace) and not is_empty_value(marketplace):
        issue = ValidationIssue(
            field_name="marketplace",
            issue_type="null_like_value",
            message=f"Marketplace contains null-like string: '{marketplace}'",
            severity=ValidationSeverity.INFO,
            row_id=str(row_id),
            group=group,
        )
        result.add_issue(issue)
        # Don't skip, just note the issue
        if log_issues:
            logger.info(f"{context_str}: Using empty string for null-like marketplace value")

    return result


def validate_csv_row(
    row: Dict[str, Any],
    log_issues: bool = True
) -> ValidationResult:
    """Validate a CSV row from change_history.csv.

    Args:
        row: Dictionary containing CSV row data
        log_issues: Whether to log validation issues

    Returns:
        ValidationResult with validation status and any issues
    """
    result = ValidationResult(is_valid=True)
    row_id = row.get("RowID", "")
    group = row.get("Group", "")
    context_str = f"{group}:{row_id}" if group and row_id else "unknown"

    # Check required fields
    for field_name in CSV_ROW_REQUIRED_FIELDS:
        value = row.get(field_name)
        is_valid, issue = validate_field(
            field_name=field_name,
            value=value,
            row_id=row_id,
            group=group,
        )

        if issue:
            result.add_issue(issue)

        if not is_valid:
            result.is_valid = False
            result.skipped = True
            result.skip_reason = f"missing_{field_name.lower()}"

            if log_issues:
                logger.warning(f"Skipping CSV row {context_str}: {issue}")
            break  # Stop checking if we find a critical missing field

    # Check for null-like values in optional fields
    if result.is_valid:
        for field_name in CSV_ROW_OPTIONAL_FIELDS:
            value = row.get(field_name)
            if is_null_like(value) and not is_empty_value(value):
                issue = ValidationIssue(
                    field_name=field_name,
                    issue_type="null_like_value",
                    message=f"Field '{field_name}' contains null-like value: '{value}'",
                    severity=ValidationSeverity.INFO,
                    row_id=row_id,
                    group=group,
                )
                result.add_issue(issue)

    return result


def sanitize_value(value: Any, field_name: str = None) -> Any:
    """Sanitize a value by replacing null-like strings with appropriate defaults.

    This function also normalizes unicode characters using NFC normalization,
    which ensures consistent representation of characters like accented letters
    (e.g., "é" is always stored as a single precomposed character rather than
    a combining sequence). This is essential for:
    - Consistent string comparisons
    - Proper display in PDFs
    - Reliable CSV storage and retrieval

    Args:
        value: The value to sanitize
        field_name: Optional field name to look up default value

    Returns:
        Sanitized value with unicode normalized to NFC form
    """
    if is_empty_value(value):
        # Get default from rules if available
        if field_name and field_name in FIELD_VALIDATION_RULES:
            return FIELD_VALIDATION_RULES[field_name].get("default", "")
        return ""

    if is_null_like(value):
        # Replace null-like strings with empty string
        if field_name and field_name in FIELD_VALIDATION_RULES:
            return FIELD_VALIDATION_RULES[field_name].get("default", "")
        return ""

    # For string values, normalize unicode and remove invisible characters
    if isinstance(value, str):
        # Remove invisible/zero-width characters that could cause issues
        value = remove_invisible_chars(value)
        # Normalize unicode to NFC form for consistent representation
        value = normalize_unicode(value, form="NFC")

    return value


def create_validation_stats() -> ValidationStats:
    """Create a new ValidationStats instance for tracking validation results."""
    return ValidationStats()


def validate_and_coerce_value(
    value: Any,
    field_name: str,
    row_id: Optional[str] = None,
    group: Optional[str] = None,
    log_issues: bool = True,
) -> Tuple[Any, Optional[ValidationIssue]]:
    """
    Validate and coerce a value to its expected type.

    This function integrates the type validation module with the main validation
    framework. It checks if a field value matches its expected type and attempts
    safe coercion if needed.

    Args:
        value: The value to validate and coerce
        field_name: Name of the field being validated
        row_id: Optional row identifier for context
        group: Optional group identifier for context
        log_issues: Whether to log validation issues

    Returns:
        Tuple of (coerced_value, validation_issue)
        - coerced_value: The value after type coercion (or original if no coercion needed)
        - validation_issue: ValidationIssue if there was a problem, None otherwise
    """
    # Import here to avoid circular imports
    from type_validation import validate_and_coerce_type

    result = validate_and_coerce_type(
        value=value,
        field_name=field_name,
        row_id=row_id,
        group=group,
        log_issues=log_issues,
    )

    return result.coerced_value, result.issue


def validate_row_with_types(
    row_data: Dict[str, Any],
    row_id: Optional[str] = None,
    group: Optional[str] = None,
    log_issues: bool = True,
) -> Tuple[Dict[str, Any], ValidationResult]:
    """
    Validate and coerce all fields in a row to their expected types.

    This function combines standard validation with type validation to provide
    comprehensive row-level validation.

    Args:
        row_data: Dictionary of field names to values
        row_id: Optional row identifier for context
        group: Optional group identifier for context
        log_issues: Whether to log validation issues

    Returns:
        Tuple of (coerced_row_data, ValidationResult)
        - coerced_row_data: Dictionary with values coerced to expected types
        - ValidationResult: Overall validation result with any issues
    """
    # Import here to avoid circular imports
    from type_validation import validate_row_types

    coerced_data, result, _ = validate_row_types(
        row_data=row_data,
        row_id=row_id,
        group=group,
        log_issues=log_issues,
    )

    return coerced_data, result
