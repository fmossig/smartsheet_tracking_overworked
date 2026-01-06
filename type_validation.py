"""
Type Validation Module for Smartsheet Tracker

Provides type mismatch validation for numeric and text fields. Implements safe type
coercion where possible and logs warnings or skips rows with incompatible types.

This module extends the existing validation framework to support:
- Expected data type validation (numeric, text, date, boolean)
- Safe type coercion (e.g., "123" -> 123, "45.67" -> 45.67)
- Warning logging for type mismatches
- Row skipping for incompatible types

Usage:
    from type_validation import (
        validate_and_coerce_type,
        TypeValidationResult,
        ExpectedType,
        FIELD_TYPE_RULES,
    )
"""

import logging
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Union

from validation import ValidationSeverity, ValidationIssue, ValidationResult

# Get logger from parent module or create new one
logger = logging.getLogger(__name__)


class ExpectedType(Enum):
    """Expected data types for field validation."""
    NUMERIC = "numeric"       # int or float
    INTEGER = "integer"       # int only
    FLOAT = "float"           # float only
    TEXT = "text"             # string
    DATE = "date"             # date or datetime
    BOOLEAN = "boolean"       # bool
    ANY = "any"               # any type (no validation)


class CoercionResult(Enum):
    """Result of type coercion attempt."""
    SUCCESS = "success"           # Coercion successful
    NO_COERCION_NEEDED = "no_coercion_needed"  # Already correct type
    COERCED_WITH_WARNING = "coerced_with_warning"  # Coerced but may have data loss
    FAILED = "failed"             # Cannot coerce


@dataclass
class TypeValidationResult:
    """Result of type validation and coercion."""
    is_valid: bool
    coerced_value: Any
    original_value: Any
    expected_type: ExpectedType
    actual_type: str
    coercion_result: CoercionResult
    issue: Optional[ValidationIssue] = None
    should_skip_row: bool = False

    def __str__(self) -> str:
        status = "valid" if self.is_valid else "invalid"
        return (
            f"TypeValidation({status}): {self.actual_type} -> {self.expected_type.value}, "
            f"coercion={self.coercion_result.value}"
        )


@dataclass
class TypeValidationStats:
    """Statistics for type validation runs."""
    total_validations: int = 0
    successful_validations: int = 0
    coerced_values: int = 0
    coerced_with_warnings: int = 0
    failed_coercions: int = 0
    skipped_rows: int = 0
    issues_by_field: Dict[str, int] = field(default_factory=dict)
    issues_by_type: Dict[str, int] = field(default_factory=dict)

    def record_result(self, result: TypeValidationResult, field_name: str = "") -> None:
        """Record a type validation result in the statistics."""
        self.total_validations += 1

        if result.is_valid:
            self.successful_validations += 1

        if result.coercion_result == CoercionResult.SUCCESS:
            self.coerced_values += 1
        elif result.coercion_result == CoercionResult.COERCED_WITH_WARNING:
            self.coerced_values += 1
            self.coerced_with_warnings += 1
        elif result.coercion_result == CoercionResult.FAILED:
            self.failed_coercions += 1

        if result.should_skip_row:
            self.skipped_rows += 1

        if result.issue and field_name:
            self.issues_by_field[field_name] = \
                self.issues_by_field.get(field_name, 0) + 1

        if result.issue:
            issue_type = result.issue.issue_type
            self.issues_by_type[issue_type] = \
                self.issues_by_type.get(issue_type, 0) + 1

    def get_summary(self) -> str:
        """Get a summary of type validation statistics."""
        lines = [
            "Type Validation Summary:",
            f"  Total validations: {self.total_validations}",
            f"  Successful: {self.successful_validations}",
            f"  Coerced values: {self.coerced_values}",
            f"  Coerced with warnings: {self.coerced_with_warnings}",
            f"  Failed coercions: {self.failed_coercions}",
            f"  Skipped rows: {self.skipped_rows}",
        ]

        if self.issues_by_field:
            lines.append("  Issues by field:")
            for field_name, count in sorted(self.issues_by_field.items(), key=lambda x: -x[1]):
                lines.append(f"    - {field_name}: {count}")

        if self.issues_by_type:
            lines.append("  Issues by type:")
            for issue_type, count in sorted(self.issues_by_type.items(), key=lambda x: -x[1]):
                lines.append(f"    - {issue_type}: {count}")

        return "\n".join(lines)


# Field-specific type rules
# Maps field names to their expected types and coercion settings
FIELD_TYPE_RULES: Dict[str, Dict[str, Any]] = {
    # Numeric fields
    "duration": {
        "expected_type": ExpectedType.FLOAT,
        "allow_coercion": True,
        "skip_on_failure": False,
        "default": 0.0,
        "description": "Duration in hours (numeric)",
    },
    "hours": {
        "expected_type": ExpectedType.FLOAT,
        "allow_coercion": True,
        "skip_on_failure": False,
        "default": 0.0,
        "description": "Hours worked (numeric)",
    },
    "count": {
        "expected_type": ExpectedType.INTEGER,
        "allow_coercion": True,
        "skip_on_failure": False,
        "default": 0,
        "description": "Count value (integer)",
    },
    "score": {
        "expected_type": ExpectedType.FLOAT,
        "allow_coercion": True,
        "skip_on_failure": False,
        "default": 0.0,
        "range": (0, 100),
        "description": "Score value (0-100)",
    },
    "percentage": {
        "expected_type": ExpectedType.FLOAT,
        "allow_coercion": True,
        "skip_on_failure": False,
        "default": 0.0,
        "range": (0, 100),
        "description": "Percentage value (0-100)",
    },
    "Phase": {
        "expected_type": ExpectedType.INTEGER,
        "allow_coercion": True,
        "skip_on_failure": True,
        "default": None,
        "range": (1, 5),
        "description": "Phase number (1-5)",
    },
    # Text fields
    "row_id": {
        "expected_type": ExpectedType.TEXT,
        "allow_coercion": True,
        "skip_on_failure": True,
        "description": "Row identifier",
    },
    "group": {
        "expected_type": ExpectedType.TEXT,
        "allow_coercion": True,
        "skip_on_failure": True,
        "description": "Group code",
    },
    "user_value": {
        "expected_type": ExpectedType.TEXT,
        "allow_coercion": True,
        "skip_on_failure": False,
        "default": "",
        "description": "User initials",
    },
    "marketplace": {
        "expected_type": ExpectedType.TEXT,
        "allow_coercion": True,
        "skip_on_failure": False,
        "default": "",
        "description": "Marketplace code",
    },
    # Date fields
    "date_value": {
        "expected_type": ExpectedType.DATE,
        "allow_coercion": True,
        "skip_on_failure": False,
        "description": "Date field value",
    },
    "Date": {
        "expected_type": ExpectedType.DATE,
        "allow_coercion": True,
        "skip_on_failure": True,
        "description": "Date value",
    },
    "Timestamp": {
        "expected_type": ExpectedType.TEXT,
        "allow_coercion": True,
        "skip_on_failure": True,
        "description": "Change timestamp",
    },
}


def _get_actual_type_name(value: Any) -> str:
    """Get a human-readable type name for a value."""
    if value is None:
        return "None"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        return "text"
    if isinstance(value, datetime):
        return "datetime"
    if isinstance(value, date):
        return "date"
    return type(value).__name__


def _is_numeric_string(value: str) -> bool:
    """Check if a string can be safely converted to a number."""
    if not value or not isinstance(value, str):
        return False

    # Remove whitespace
    cleaned = value.strip()

    # Handle empty strings
    if not cleaned:
        return False

    # Pattern for valid numeric strings (supports European decimal comma)
    # Examples: "123", "-123.45", "1,234.56", "1.234,56" (European)
    numeric_pattern = r'^-?\d{1,3}(?:[.,]\d{3})*(?:[.,]\d+)?$|^-?\d+(?:[.,]\d+)?$'

    return bool(re.match(numeric_pattern, cleaned))


def _coerce_to_numeric(value: Any, target_type: ExpectedType) -> Tuple[Any, CoercionResult]:
    """
    Attempt to coerce a value to a numeric type.

    Args:
        value: The value to coerce
        target_type: Expected numeric type (NUMERIC, INTEGER, or FLOAT)

    Returns:
        Tuple of (coerced_value, CoercionResult)
    """
    if value is None:
        return None, CoercionResult.FAILED

    # Already correct type
    if target_type == ExpectedType.INTEGER and isinstance(value, int) and not isinstance(value, bool):
        return value, CoercionResult.NO_COERCION_NEEDED

    if target_type == ExpectedType.FLOAT and isinstance(value, float):
        return value, CoercionResult.NO_COERCION_NEEDED

    if target_type == ExpectedType.NUMERIC:
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            return value, CoercionResult.NO_COERCION_NEEDED

    # Try to coerce string to number
    if isinstance(value, str):
        cleaned = value.strip()

        if not cleaned:
            return None, CoercionResult.FAILED

        # Handle European format (comma as decimal separator)
        # Check if it looks like European format: "1.234,56" or "123,45"
        if ',' in cleaned:
            # If there's both comma and dot, determine which is decimal separator
            if '.' in cleaned:
                # If comma comes after the last dot, comma is decimal separator (European)
                last_comma = cleaned.rfind(',')
                last_dot = cleaned.rfind('.')
                if last_comma > last_dot:
                    # European format: replace dots (thousands) and comma (decimal)
                    cleaned = cleaned.replace('.', '').replace(',', '.')
                else:
                    # US format: just remove commas (thousands separator)
                    cleaned = cleaned.replace(',', '')
            else:
                # Only comma present, assume it's decimal separator
                cleaned = cleaned.replace(',', '.')

        try:
            float_val = float(cleaned)

            if target_type == ExpectedType.INTEGER:
                # Check if we're losing precision
                if float_val != int(float_val):
                    return int(float_val), CoercionResult.COERCED_WITH_WARNING
                return int(float_val), CoercionResult.SUCCESS

            return float_val, CoercionResult.SUCCESS

        except (ValueError, TypeError):
            return None, CoercionResult.FAILED

    # Try to coerce int to float or float to int
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if target_type == ExpectedType.INTEGER:
            int_val = int(value)
            if float(int_val) != float(value):
                return int_val, CoercionResult.COERCED_WITH_WARNING
            return int_val, CoercionResult.SUCCESS

        if target_type in (ExpectedType.FLOAT, ExpectedType.NUMERIC):
            return float(value), CoercionResult.SUCCESS

    # Boolean to int/float (usually 0 or 1)
    if isinstance(value, bool):
        if target_type == ExpectedType.INTEGER:
            return int(value), CoercionResult.COERCED_WITH_WARNING
        if target_type in (ExpectedType.FLOAT, ExpectedType.NUMERIC):
            return float(value), CoercionResult.COERCED_WITH_WARNING

    return None, CoercionResult.FAILED


def _coerce_to_text(value: Any) -> Tuple[str, CoercionResult]:
    """
    Coerce a value to text (string).

    Args:
        value: The value to coerce

    Returns:
        Tuple of (coerced_value, CoercionResult)
    """
    if value is None:
        return "", CoercionResult.SUCCESS

    if isinstance(value, str):
        return value, CoercionResult.NO_COERCION_NEEDED

    # Convert any value to string
    try:
        return str(value), CoercionResult.SUCCESS
    except Exception:
        return "", CoercionResult.FAILED


def _coerce_to_date(value: Any) -> Tuple[Optional[date], CoercionResult]:
    """
    Coerce a value to a date.

    Args:
        value: The value to coerce

    Returns:
        Tuple of (coerced_value, CoercionResult)
    """
    if value is None:
        return None, CoercionResult.FAILED

    if isinstance(value, datetime):
        return value.date(), CoercionResult.NO_COERCION_NEEDED

    if isinstance(value, date):
        return value, CoercionResult.NO_COERCION_NEEDED

    # Try to parse string as date
    if isinstance(value, str):
        # Import here to avoid circular imports
        from date_utilities import parse_date
        parsed = parse_date(value)
        if parsed is not None:
            return parsed, CoercionResult.SUCCESS
        return None, CoercionResult.FAILED

    # Numeric timestamp (Unix epoch)
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(value).date(), CoercionResult.COERCED_WITH_WARNING
        except (ValueError, OSError, OverflowError):
            return None, CoercionResult.FAILED

    return None, CoercionResult.FAILED


def _coerce_to_boolean(value: Any) -> Tuple[Optional[bool], CoercionResult]:
    """
    Coerce a value to boolean.

    Args:
        value: The value to coerce

    Returns:
        Tuple of (coerced_value, CoercionResult)
    """
    if value is None:
        return None, CoercionResult.FAILED

    if isinstance(value, bool):
        return value, CoercionResult.NO_COERCION_NEEDED

    # String representations
    if isinstance(value, str):
        lower_val = value.strip().lower()
        if lower_val in ('true', 'yes', '1', 'on', 'y'):
            return True, CoercionResult.SUCCESS
        if lower_val in ('false', 'no', '0', 'off', 'n', ''):
            return False, CoercionResult.SUCCESS
        return None, CoercionResult.FAILED

    # Numeric to boolean
    if isinstance(value, (int, float)):
        return bool(value), CoercionResult.SUCCESS

    return None, CoercionResult.FAILED


def validate_and_coerce_type(
    value: Any,
    field_name: str,
    expected_type: Optional[ExpectedType] = None,
    allow_coercion: bool = True,
    skip_on_failure: bool = False,
    row_id: Optional[str] = None,
    group: Optional[str] = None,
    custom_rules: Optional[Dict[str, Any]] = None,
    log_issues: bool = True,
) -> TypeValidationResult:
    """
    Validate a value against an expected type and attempt safe coercion if needed.

    This is the main entry point for type validation. It:
    1. Checks if the value matches the expected type
    2. Attempts safe coercion if allowed
    3. Logs warnings for type mismatches
    4. Returns a result indicating success/failure and the coerced value

    Args:
        value: The value to validate
        field_name: Name of the field being validated
        expected_type: The expected data type (defaults to looking up in FIELD_TYPE_RULES)
        allow_coercion: Whether to attempt type coercion (default True)
        skip_on_failure: Whether to skip the row on validation failure (default False)
        row_id: Optional row identifier for context
        group: Optional group identifier for context
        custom_rules: Optional custom validation rules
        log_issues: Whether to log validation issues (default True)

    Returns:
        TypeValidationResult with validation status and coerced value
    """
    # Look up rules for this field
    rules = custom_rules or FIELD_TYPE_RULES.get(field_name, {})

    if expected_type is None:
        expected_type = rules.get("expected_type", ExpectedType.ANY)

    if expected_type == ExpectedType.ANY:
        # No validation needed
        return TypeValidationResult(
            is_valid=True,
            coerced_value=value,
            original_value=value,
            expected_type=expected_type,
            actual_type=_get_actual_type_name(value),
            coercion_result=CoercionResult.NO_COERCION_NEEDED,
        )

    allow_coercion = rules.get("allow_coercion", allow_coercion)
    skip_on_failure = rules.get("skip_on_failure", skip_on_failure)
    default_value = rules.get("default")
    value_range = rules.get("range")

    actual_type = _get_actual_type_name(value)

    # Attempt coercion based on expected type
    coerced_value: Any = None
    coercion_result: CoercionResult = CoercionResult.FAILED

    if expected_type in (ExpectedType.NUMERIC, ExpectedType.INTEGER, ExpectedType.FLOAT):
        coerced_value, coercion_result = _coerce_to_numeric(value, expected_type)
    elif expected_type == ExpectedType.TEXT:
        coerced_value, coercion_result = _coerce_to_text(value)
    elif expected_type == ExpectedType.DATE:
        coerced_value, coercion_result = _coerce_to_date(value)
    elif expected_type == ExpectedType.BOOLEAN:
        coerced_value, coercion_result = _coerce_to_boolean(value)

    # Check if coercion was successful
    is_valid = coercion_result in (
        CoercionResult.SUCCESS,
        CoercionResult.NO_COERCION_NEEDED,
        CoercionResult.COERCED_WITH_WARNING,
    )

    # Apply range validation for numeric types
    if is_valid and value_range and coerced_value is not None:
        min_val, max_val = value_range
        if isinstance(coerced_value, (int, float)):
            if coerced_value < min_val or coerced_value > max_val:
                is_valid = False
                coercion_result = CoercionResult.FAILED

    # Create validation issue if needed
    issue: Optional[ValidationIssue] = None
    should_skip = False

    if not is_valid or coercion_result == CoercionResult.COERCED_WITH_WARNING:
        if not is_valid:
            severity = ValidationSeverity.ERROR if skip_on_failure else ValidationSeverity.WARNING
            should_skip = skip_on_failure

            message = (
                f"Type mismatch for field '{field_name}': "
                f"expected {expected_type.value}, got {actual_type} "
                f"with value '{value}'"
            )

            if value_range:
                message += f" (valid range: {value_range[0]}-{value_range[1]})"

            if default_value is not None and not skip_on_failure:
                coerced_value = default_value
                message += f". Using default value: {default_value}"
                is_valid = True  # Consider it valid if we can use default

            issue = ValidationIssue(
                field_name=field_name,
                issue_type="type_mismatch",
                message=message,
                severity=severity,
                row_id=row_id,
                group=group,
            )

            if log_issues:
                log_fn = logger.warning if severity == ValidationSeverity.WARNING else logger.error
                context = f"{group}:{row_id}" if group and row_id else str(row_id or "")
                if context:
                    log_fn(f"[{context}] {message}")
                else:
                    log_fn(message)

        elif coercion_result == CoercionResult.COERCED_WITH_WARNING:
            message = (
                f"Type coercion for field '{field_name}': "
                f"converted {actual_type} '{value}' to {expected_type.value} '{coerced_value}' "
                f"with potential data loss"
            )

            issue = ValidationIssue(
                field_name=field_name,
                issue_type="type_coercion_warning",
                message=message,
                severity=ValidationSeverity.INFO,
                row_id=row_id,
                group=group,
            )

            if log_issues:
                context = f"{group}:{row_id}" if group and row_id else str(row_id or "")
                if context:
                    logger.info(f"[{context}] {message}")
                else:
                    logger.info(message)

    return TypeValidationResult(
        is_valid=is_valid,
        coerced_value=coerced_value,
        original_value=value,
        expected_type=expected_type,
        actual_type=actual_type,
        coercion_result=coercion_result,
        issue=issue,
        should_skip_row=should_skip,
    )


def validate_row_types(
    row_data: Dict[str, Any],
    field_rules: Optional[Dict[str, Dict[str, Any]]] = None,
    row_id: Optional[str] = None,
    group: Optional[str] = None,
    log_issues: bool = True,
) -> Tuple[Dict[str, Any], ValidationResult, List[TypeValidationResult]]:
    """
    Validate all fields in a row against their expected types.

    Args:
        row_data: Dictionary of field names to values
        field_rules: Optional custom field rules (defaults to FIELD_TYPE_RULES)
        row_id: Optional row identifier for context
        group: Optional group identifier for context
        log_issues: Whether to log validation issues

    Returns:
        Tuple of (coerced_row_data, ValidationResult, list of TypeValidationResult)
    """
    field_rules = field_rules or FIELD_TYPE_RULES
    result = ValidationResult(is_valid=True)
    type_results: List[TypeValidationResult] = []
    coerced_data: Dict[str, Any] = {}

    for field_name, value in row_data.items():
        # Only validate fields that have type rules defined
        if field_name in field_rules:
            type_result = validate_and_coerce_type(
                value=value,
                field_name=field_name,
                row_id=row_id,
                group=group,
                log_issues=log_issues,
            )
            type_results.append(type_result)

            coerced_data[field_name] = type_result.coerced_value

            if type_result.issue:
                result.add_issue(type_result.issue)

            if type_result.should_skip_row:
                result.is_valid = False
                result.skipped = True
                result.skip_reason = f"type_mismatch_{field_name}"
        else:
            # Keep original value for fields without type rules
            coerced_data[field_name] = value

    return coerced_data, result, type_results


def create_type_validation_stats() -> TypeValidationStats:
    """Create a new TypeValidationStats instance for tracking validation results."""
    return TypeValidationStats()


def get_field_expected_type(field_name: str) -> Optional[ExpectedType]:
    """Get the expected type for a field name."""
    rules = FIELD_TYPE_RULES.get(field_name)
    if rules:
        return rules.get("expected_type")
    return None


def is_numeric_field(field_name: str) -> bool:
    """Check if a field is expected to be numeric."""
    expected = get_field_expected_type(field_name)
    return expected in (ExpectedType.NUMERIC, ExpectedType.INTEGER, ExpectedType.FLOAT)


def is_text_field(field_name: str) -> bool:
    """Check if a field is expected to be text."""
    expected = get_field_expected_type(field_name)
    return expected == ExpectedType.TEXT


def is_date_field(field_name: str) -> bool:
    """Check if a field is expected to be a date."""
    expected = get_field_expected_type(field_name)
    return expected == ExpectedType.DATE
