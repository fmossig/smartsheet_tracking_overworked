"""
Startup Configuration Validator for Smartsheet Tracker

This module provides initialization checks at startup to verify all sheet IDs
are accessible and required columns exist. It implements fail-fast behavior
with clear error messages if configuration is invalid.

Usage:
    from startup_validator import (
        validate_startup_config,
        ConfigValidationResult,
        SheetValidationResult,
    )

    # Basic usage
    result = validate_startup_config(client, SHEET_IDS, PHASE_FIELDS)
    if not result.is_valid:
        for error in result.get_fatal_errors():
            logger.error(error)
        sys.exit(1)
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum

# Get logger for this module
logger = logging.getLogger(__name__)


class ValidationSeverity(Enum):
    """Severity levels for configuration validation issues."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"      # Non-fatal: can continue with degraded functionality
    FATAL = "fatal"      # Fatal: must stop execution


@dataclass
class ConfigIssue:
    """Represents a single configuration validation issue."""
    category: str           # e.g., "sheet_access", "missing_column", "token_error"
    message: str            # Human-readable description
    severity: ValidationSeverity
    sheet_id: Optional[int] = None
    sheet_name: Optional[str] = None
    column_name: Optional[str] = None
    actionable_hint: Optional[str] = None

    def __str__(self) -> str:
        location = ""
        if self.sheet_name:
            location = f"[{self.sheet_name}] "
        elif self.sheet_id:
            location = f"[Sheet:{self.sheet_id}] "

        hint = ""
        if self.actionable_hint:
            hint = f" - Hint: {self.actionable_hint}"

        return f"{self.severity.value.upper()}: {location}{self.message}{hint}"


@dataclass
class SheetValidationResult:
    """Result of validating a single sheet's configuration."""
    sheet_name: str
    sheet_id: int
    is_accessible: bool
    issues: List[ConfigIssue] = field(default_factory=list)

    # Column availability details
    columns_found: List[str] = field(default_factory=list)
    columns_missing: List[str] = field(default_factory=list)

    # Phase availability details
    phases_available: List[int] = field(default_factory=list)
    phases_unavailable: List[int] = field(default_factory=list)
    partial_phases: List[Dict[str, Any]] = field(default_factory=list)

    # Optional columns
    has_amazon_column: bool = False

    def is_usable(self) -> bool:
        """Check if sheet is usable (accessible with at least one phase available)."""
        return self.is_accessible and len(self.phases_available) > 0

    def get_summary(self) -> str:
        """Get a human-readable summary of this sheet's validation."""
        if not self.is_accessible:
            return f"INACCESSIBLE - cannot read sheet"

        total_phases = len(self.phases_available) + len(self.phases_unavailable)
        available = len(self.phases_available)

        if available == total_phases:
            status = "READY"
        elif available > 0:
            status = "DEGRADED"
        else:
            status = "NO_PHASES"

        return f"{status} - {available}/{total_phases} phases available"


@dataclass
class ConfigValidationResult:
    """Overall result of startup configuration validation."""
    is_valid: bool          # True if all critical checks pass (can proceed)
    can_proceed: bool       # True if at least some functionality is available
    issues: List[ConfigIssue] = field(default_factory=list)
    sheet_results: Dict[str, SheetValidationResult] = field(default_factory=dict)

    # Counters
    total_sheets: int = 0
    accessible_sheets: int = 0
    usable_sheets: int = 0

    def add_issue(self, issue: ConfigIssue) -> None:
        """Add a validation issue to the result."""
        self.issues.append(issue)

    def get_fatal_errors(self) -> List[str]:
        """Get list of fatal error messages that require stopping execution."""
        return [str(issue) for issue in self.issues
                if issue.severity == ValidationSeverity.FATAL]

    def get_errors(self) -> List[str]:
        """Get list of error messages (non-fatal)."""
        return [str(issue) for issue in self.issues
                if issue.severity == ValidationSeverity.ERROR]

    def get_warnings(self) -> List[str]:
        """Get list of warning messages."""
        return [str(issue) for issue in self.issues
                if issue.severity == ValidationSeverity.WARNING]

    def get_overall_status(self) -> str:
        """Get overall status: READY, DEGRADED, or FAILED."""
        if not self.can_proceed:
            return "FAILED"
        if self.usable_sheets < self.total_sheets:
            return "DEGRADED"
        return "READY"

    def log_summary(self) -> None:
        """Log a comprehensive summary of the validation results."""
        status = self.get_overall_status()

        logger.info("=" * 60)
        logger.info(f"STARTUP CONFIGURATION VALIDATION: {status}")
        logger.info("=" * 60)

        # Overall stats
        logger.info(f"Sheets: {self.accessible_sheets}/{self.total_sheets} accessible, "
                   f"{self.usable_sheets}/{self.total_sheets} usable")

        # Per-sheet details
        for sheet_name, result in self.sheet_results.items():
            summary = result.get_summary()
            if result.is_usable():
                logger.info(f"  {sheet_name}: {summary}")
            else:
                logger.warning(f"  {sheet_name}: {summary}")

        # Log issues by severity
        fatal_errors = self.get_fatal_errors()
        if fatal_errors:
            logger.error(f"FATAL ERRORS ({len(fatal_errors)}):")
            for error in fatal_errors:
                logger.error(f"  {error}")

        errors = self.get_errors()
        if errors:
            logger.warning(f"ERRORS ({len(errors)}) - processing will continue with degraded functionality:")
            for error in errors:
                logger.warning(f"  {error}")

        warnings = self.get_warnings()
        if warnings:
            logger.info(f"WARNINGS ({len(warnings)}):")
            for warning in warnings:
                logger.info(f"  {warning}")

        logger.info("=" * 60)

        if not self.can_proceed:
            logger.error("Configuration validation FAILED - cannot proceed")
        elif status == "DEGRADED":
            logger.warning("Configuration validation passed with warnings - proceeding with degraded functionality")
        else:
            logger.info("Configuration validation PASSED - all systems ready")


def resolve_column_name(col_map: Dict[str, int], column_variations: Tuple[str, ...]) -> Tuple[Optional[str], Optional[int]]:
    """
    Resolve a column name from a list of possible variations.

    Args:
        col_map: Dictionary mapping column titles to column IDs
        column_variations: Tuple of possible column name variations to try

    Returns:
        Tuple of (resolved_column_name, column_id) or (None, None) if not found
    """
    if isinstance(column_variations, str):
        column_variations = (column_variations,)

    for variation in column_variations:
        if variation in col_map:
            return variation, col_map[variation]

    return None, None


def validate_sheet_columns(
    col_map: Dict[str, int],
    sheet_name: str,
    sheet_id: int,
    phase_fields: List[Tuple[str, Tuple[str, ...], int]],
    optional_columns: List[str] = None
) -> SheetValidationResult:
    """
    Validate that required columns exist in a sheet.

    Args:
        col_map: Dictionary mapping column titles to column IDs
        sheet_name: Name of the sheet group
        sheet_id: Smartsheet ID
        phase_fields: List of (date_column, user_column_variations, phase_number) tuples
        optional_columns: List of optional column names to check

    Returns:
        SheetValidationResult with column validation details
    """
    result = SheetValidationResult(
        sheet_name=sheet_name,
        sheet_id=sheet_id,
        is_accessible=True,
        columns_found=list(col_map.keys())
    )

    # Check each phase field
    for date_col, user_col_variations, phase_no in phase_fields:
        date_exists = date_col in col_map
        user_col_name, user_col_id = resolve_column_name(col_map, user_col_variations)
        user_exists = user_col_id is not None

        if date_exists and user_exists:
            result.phases_available.append(phase_no)
        elif date_exists and not user_exists:
            result.phases_unavailable.append(phase_no)
            result.partial_phases.append({
                'phase': phase_no,
                'date_col': date_col,
                'expected_user_cols': user_col_variations
            })
            result.columns_missing.append(f"user column for phase {phase_no} (expected: {user_col_variations})")
            result.issues.append(ConfigIssue(
                category="missing_column",
                message=f"Phase {phase_no} has date column '{date_col}' but missing user column",
                severity=ValidationSeverity.WARNING,
                sheet_id=sheet_id,
                sheet_name=sheet_name,
                column_name=str(user_col_variations),
                actionable_hint=f"Add one of: {user_col_variations} to enable phase {phase_no} tracking"
            ))
        else:
            result.phases_unavailable.append(phase_no)
            result.columns_missing.append(date_col)
            result.issues.append(ConfigIssue(
                category="missing_column",
                message=f"Phase {phase_no} date column '{date_col}' not found",
                severity=ValidationSeverity.WARNING,
                sheet_id=sheet_id,
                sheet_name=sheet_name,
                column_name=date_col
            ))

    # Check optional columns
    if optional_columns:
        for col_name in optional_columns:
            if col_name in col_map:
                if col_name == "Amazon":
                    result.has_amazon_column = True
            else:
                result.issues.append(ConfigIssue(
                    category="missing_optional_column",
                    message=f"Optional column '{col_name}' not found",
                    severity=ValidationSeverity.INFO,
                    sheet_id=sheet_id,
                    sheet_name=sheet_name,
                    column_name=col_name,
                    actionable_hint=f"Add '{col_name}' column to enable additional tracking features"
                ))

    # Check if sheet is completely unusable
    if not result.phases_available:
        result.issues.append(ConfigIssue(
            category="no_phases",
            message=f"No tracking phases available - sheet will be skipped",
            severity=ValidationSeverity.ERROR,
            sheet_id=sheet_id,
            sheet_name=sheet_name,
            actionable_hint="Ensure at least one phase has both date and user columns"
        ))

    return result


def validate_sheet_access(
    client,
    sheet_name: str,
    sheet_id: int
) -> Tuple[bool, Optional[Any], Optional[ConfigIssue]]:
    """
    Validate that a sheet is accessible.

    Args:
        client: Smartsheet client (with or without retry wrapper)
        sheet_name: Name of the sheet group
        sheet_id: Smartsheet ID

    Returns:
        Tuple of (is_accessible, sheet_object, error_issue)
    """
    # Import error checking functions - these may be in the same project
    try:
        from smartsheet_retry import (
            is_sheet_not_found_error,
            is_token_error,
            is_permission_denied_error,
            SheetNotFoundError,
            TokenAuthenticationError,
            PermissionDeniedError,
        )
    except ImportError:
        # Fallback if smartsheet_retry not available
        is_sheet_not_found_error = lambda e, sid=None: "not found" in str(e).lower()
        is_token_error = lambda e: "token" in str(e).lower() or "unauthorized" in str(e).lower()
        is_permission_denied_error = lambda e, sid=None: "permission" in str(e).lower() or "forbidden" in str(e).lower()
        SheetNotFoundError = Exception
        TokenAuthenticationError = Exception
        PermissionDeniedError = Exception

    try:
        # Try to get the sheet
        sheet = client.get_sheet(sheet_id)

        if sheet is None:
            return False, None, ConfigIssue(
                category="sheet_access",
                message=f"Sheet returned None - may be invalid or deleted",
                severity=ValidationSeverity.ERROR,
                sheet_id=sheet_id,
                sheet_name=sheet_name,
                actionable_hint="Verify the sheet ID is correct and the sheet exists"
            )

        return True, sheet, None

    except TokenAuthenticationError as e:
        # Token errors are fatal - can't access any sheets
        return False, None, ConfigIssue(
            category="token_error",
            message=f"Authentication failed: {e}",
            severity=ValidationSeverity.FATAL,
            sheet_id=sheet_id,
            sheet_name=sheet_name,
            actionable_hint=e.get_actionable_message() if hasattr(e, 'get_actionable_message') else "Check your SMARTSHEET_TOKEN"
        )

    except SheetNotFoundError as e:
        return False, None, ConfigIssue(
            category="sheet_not_found",
            message=f"Sheet not found: {e}",
            severity=ValidationSeverity.ERROR,
            sheet_id=sheet_id,
            sheet_name=sheet_name,
            actionable_hint="Verify the sheet ID is correct and the sheet has not been deleted"
        )

    except PermissionDeniedError as e:
        return False, None, ConfigIssue(
            category="permission_denied",
            message=f"Permission denied: {e}",
            severity=ValidationSeverity.ERROR,
            sheet_id=sheet_id,
            sheet_name=sheet_name,
            actionable_hint=e.get_actionable_message() if hasattr(e, 'get_actionable_message') else "Request access from the sheet owner"
        )

    except Exception as e:
        # Check for specific error types in exception message
        if is_token_error(e):
            return False, None, ConfigIssue(
                category="token_error",
                message=f"Authentication error: {e}",
                severity=ValidationSeverity.FATAL,
                sheet_id=sheet_id,
                sheet_name=sheet_name,
                actionable_hint="Check your SMARTSHEET_TOKEN in .env file"
            )

        if is_sheet_not_found_error(e, sheet_id):
            return False, None, ConfigIssue(
                category="sheet_not_found",
                message=f"Sheet not found: {e}",
                severity=ValidationSeverity.ERROR,
                sheet_id=sheet_id,
                sheet_name=sheet_name,
                actionable_hint="Verify the sheet ID is correct"
            )

        if is_permission_denied_error(e, sheet_id):
            return False, None, ConfigIssue(
                category="permission_denied",
                message=f"Permission denied: {e}",
                severity=ValidationSeverity.ERROR,
                sheet_id=sheet_id,
                sheet_name=sheet_name,
                actionable_hint="Request access from the sheet owner"
            )

        # Unknown error
        return False, None, ConfigIssue(
            category="unknown_error",
            message=f"Failed to access sheet: {type(e).__name__}: {e}",
            severity=ValidationSeverity.ERROR,
            sheet_id=sheet_id,
            sheet_name=sheet_name,
            actionable_hint="Check network connectivity and Smartsheet service status"
        )


def validate_startup_config(
    client,
    sheet_ids: Dict[str, int],
    phase_fields: List[Tuple[str, Tuple[str, ...], int]],
    optional_columns: List[str] = None,
    fail_fast_on_token_error: bool = True
) -> ConfigValidationResult:
    """
    Validate startup configuration by checking all sheet IDs are accessible
    and required columns exist.

    This function implements fail-fast behavior:
    - Token errors are immediately fatal (stops validation)
    - Sheet-level errors allow validation to continue
    - Returns comprehensive results for logging and decision-making

    Args:
        client: Smartsheet client (with or without retry wrapper)
        sheet_ids: Dictionary mapping sheet names to sheet IDs
        phase_fields: List of (date_column, user_column_variations, phase_number) tuples
        optional_columns: List of optional column names to check
        fail_fast_on_token_error: If True, stop validation on first token error

    Returns:
        ConfigValidationResult with complete validation status and details
    """
    result = ConfigValidationResult(
        is_valid=True,
        can_proceed=True,
        total_sheets=len(sheet_ids)
    )

    if optional_columns is None:
        optional_columns = ["Amazon"]

    logger.info(f"Starting configuration validation for {len(sheet_ids)} sheets...")

    # Validate each sheet
    for sheet_name, sheet_id in sheet_ids.items():
        logger.info(f"Validating sheet {sheet_name} (ID: {sheet_id})...")

        # First, check if sheet is accessible
        is_accessible, sheet, access_issue = validate_sheet_access(client, sheet_name, sheet_id)

        if access_issue:
            result.add_issue(access_issue)

            # Check for fatal token error
            if access_issue.severity == ValidationSeverity.FATAL:
                result.is_valid = False
                result.can_proceed = False

                # Create a placeholder sheet result
                sheet_result = SheetValidationResult(
                    sheet_name=sheet_name,
                    sheet_id=sheet_id,
                    is_accessible=False
                )
                sheet_result.issues.append(access_issue)
                result.sheet_results[sheet_name] = sheet_result

                if fail_fast_on_token_error:
                    logger.error(f"Fatal token error detected - stopping validation")
                    return result
                continue

        if not is_accessible:
            # Non-fatal access error
            sheet_result = SheetValidationResult(
                sheet_name=sheet_name,
                sheet_id=sheet_id,
                is_accessible=False
            )
            if access_issue:
                sheet_result.issues.append(access_issue)
            result.sheet_results[sheet_name] = sheet_result
            continue

        # Sheet is accessible - validate columns
        result.accessible_sheets += 1

        # Build column map from sheet
        col_map = {col.title: col.id for col in sheet.columns}

        # Validate columns
        sheet_result = validate_sheet_columns(
            col_map=col_map,
            sheet_name=sheet_name,
            sheet_id=sheet_id,
            phase_fields=phase_fields,
            optional_columns=optional_columns
        )

        # Add sheet-level issues to main result
        for issue in sheet_result.issues:
            result.add_issue(issue)

        # Track usable sheets
        if sheet_result.is_usable():
            result.usable_sheets += 1

        result.sheet_results[sheet_name] = sheet_result
        logger.info(f"  {sheet_name}: {sheet_result.get_summary()}")

    # Determine final validity
    if result.usable_sheets == 0:
        result.is_valid = False
        result.can_proceed = False
        result.add_issue(ConfigIssue(
            category="no_usable_sheets",
            message="No usable sheets found - cannot proceed with tracking",
            severity=ValidationSeverity.FATAL,
            actionable_hint="Ensure at least one sheet is accessible and has required columns"
        ))
    elif result.usable_sheets < result.total_sheets:
        # Some sheets unavailable - can proceed but degraded
        result.is_valid = False  # Not fully valid
        result.can_proceed = True  # But can still proceed

    return result


def create_validation_summary(result: ConfigValidationResult) -> str:
    """
    Create a human-readable summary of validation results.

    Args:
        result: ConfigValidationResult from validate_startup_config

    Returns:
        Formatted summary string
    """
    lines = []
    status = result.get_overall_status()

    lines.append("=" * 60)
    lines.append(f"CONFIGURATION VALIDATION: {status}")
    lines.append("=" * 60)
    lines.append("")
    lines.append(f"Sheets: {result.accessible_sheets}/{result.total_sheets} accessible, "
                f"{result.usable_sheets}/{result.total_sheets} usable")
    lines.append("")

    # Per-sheet summary
    lines.append("Sheet Status:")
    for sheet_name, sheet_result in result.sheet_results.items():
        status_icon = "[OK]" if sheet_result.is_usable() else "[!!]"
        lines.append(f"  {status_icon} {sheet_name}: {sheet_result.get_summary()}")
        if sheet_result.phases_available:
            lines.append(f"      Phases available: {sheet_result.phases_available}")
        if sheet_result.phases_unavailable:
            lines.append(f"      Phases unavailable: {sheet_result.phases_unavailable}")

    lines.append("")

    # Issues by severity
    fatal = result.get_fatal_errors()
    if fatal:
        lines.append(f"FATAL ERRORS ({len(fatal)}):")
        for msg in fatal:
            lines.append(f"  {msg}")
        lines.append("")

    errors = result.get_errors()
    if errors:
        lines.append(f"ERRORS ({len(errors)}):")
        for msg in errors:
            lines.append(f"  {msg}")
        lines.append("")

    warnings = result.get_warnings()
    if warnings:
        lines.append(f"WARNINGS ({len(warnings)}):")
        for msg in warnings:
            lines.append(f"  {msg}")
        lines.append("")

    lines.append("=" * 60)

    if result.can_proceed:
        lines.append("Validation complete - proceeding with available sheets")
    else:
        lines.append("Validation FAILED - cannot proceed")

    return "\n".join(lines)
