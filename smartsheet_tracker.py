import os
import csv
import json
from datetime import datetime, timedelta, date
import logging
import argparse
from typing import Any, Dict, List, Optional, Tuple, TypedDict, Union

import smartsheet
from dotenv import load_dotenv
from smartsheet_retry import (
    SmartsheetRetryClient,
    execute_with_retry,
    SheetNotFoundError,
    is_sheet_not_found_error,
    TokenAuthenticationError,
    is_token_error,
    PermissionDeniedError,
    is_permission_denied_error,
)
from smartsheet_client import (
    create_smartsheet_client,
    get_smartsheet_token,
    fetch_sheet,
    create_column_map,
    ClientConfig,
)
from validation import (
    validate_row_data,
    ValidationResult,
    ValidationStats,
    create_validation_stats,
    is_empty_value,
    sanitize_value,
)
from error_collector import (
    ErrorCollector,
    ErrorSeverity,
    ErrorCategory,
    get_global_collector,
    reset_global_collector,
)
from startup_validator import (
    validate_startup_config,
    ConfigValidationResult,
    create_validation_summary,
)
from date_utilities import (
    parse_date,
    normalize_date_for_comparison,
    SUPPORTED_DATE_FORMATS,
)
from phase_field_utilities import (
    PHASE_FIELDS,
    EXPECTED_DATE_COLUMNS,
    OPTIONAL_COLUMNS,
    resolve_column_name,
    get_phase_columns,
    detect_missing_columns,
)
from logging_config import (
    configure_logging,
    add_log_level_argument,
    get_module_logger,
)
from performance_timing import (
    timed_operation,
    PerformanceTimer,
    time_sheet_fetch,
    time_data_processing,
)

# Logger will be configured in main() after parsing args
logger: logging.Logger = logging.getLogger(__name__)


class TrackerState(TypedDict):
    """Type definition for the tracker state dictionary."""
    last_run: Optional[str]
    processed: Dict[str, str]

# Load environment variables and Smartsheet token using shared utility
try:
    token = get_smartsheet_token()
except ValueError as e:
    logger.error(str(e))
    exit(1)

# Smartsheet IDs and columns to track
SHEET_IDS = {
    "NA": 6141179298008964,
    "NF": 615755411312516,
    "NH": 123340632051588,
    "NP": 3009924800925572,
    "NT": 2199739350077316,
    "NV": 8955413669040004,
    "NM": 4275419734822788,
    "BUNDLE_FAN": 7412589630147852,  # Bundle sheet for FAN products (estimated 2,497 products)
    "BUNDLE_COOLER": 3698521470258963,  # Bundle sheet for COOLER products (estimated 121 products)
}

# PHASE_FIELDS, EXPECTED_DATE_COLUMNS, OPTIONAL_COLUMNS, resolve_column_name,
# get_phase_columns, and detect_missing_columns are now imported from
# phase_field_utilities module to eliminate code duplication.

# Directory to store data
DATA_DIR = "tracking_data"
os.makedirs(DATA_DIR, exist_ok=True)

# State file to track what we've already processed
STATE_FILE = os.path.join(DATA_DIR, "tracker_state.json")
CHANGES_FILE = os.path.join(DATA_DIR, "change_history.csv")

def load_state() -> TrackerState:
    """
    Load previously saved tracking state from the state file.

    Reads the JSON state file containing information about previously processed
    Smartsheet rows and their date field values. If the file doesn't exist or
    cannot be read, returns an empty state structure.

    Returns:
        TrackerState: A dictionary containing:
            - 'last_run' (str or None): ISO timestamp of the last successful run
            - 'processed' (dict): Mapping of field keys to their last known values
              Format: {"group:row_id:date_col": "YYYY-MM-DD", ...}

    Raises:
        No exceptions are raised; errors are logged and an empty state is returned.

    Example:
        >>> state = load_state()
        >>> print(state['last_run'])
        '2025-01-06 10:30:00'
        >>> print(state['processed'].get('NA:12345:Kontrolle'))
        '2025-01-05'
    """
    try:
        if os.path.exists(STATE_FILE):
            with open(STATE_FILE, 'r', encoding='utf-8') as f:
                state: TrackerState = json.load(f)
                logger.info(f"Loaded state file with {len(state.get('processed', {}))} processed items")
                return state
        else:
            logger.warning(f"State file not found: {STATE_FILE}")
            return {"last_run": None, "processed": {}}
    except Exception as e:
        logger.error(f"Error loading state: {e}")
        return {"last_run": None, "processed": {}}

def save_state(state: TrackerState) -> None:
    """
    Save the current tracking state to the state file.

    Persists the tracking state dictionary to a JSON file, enabling the tracker
    to resume from where it left off on subsequent runs.

    Args:
        state: The state dictionary to save, containing:
            - 'last_run' (str): ISO timestamp of the current run
            - 'processed' (dict): Mapping of field keys to their current values

    Returns:
        None

    Raises:
        No exceptions are raised; errors are logged and the function returns silently.

    Example:
        >>> state = {
        ...     "last_run": "2025-01-06 10:30:00",
        ...     "processed": {"NA:12345:Kontrolle": "2025-01-05"}
        ... }
        >>> save_state(state)
    """
    try:
        with open(STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False)
            logger.info(f"Saved state with {len(state.get('processed', {}))} processed items")
    except Exception as e:
        logger.error(f"Error saving state: {e}")

def ensure_changes_file() -> None:
    """
    Ensure the change history CSV file exists with proper headers.

    Creates the change history file if it doesn't exist, initializing it with
    the appropriate column headers. If the file already exists, this function
    does nothing, preserving existing data.

    The CSV file tracks all detected changes with the following columns:
    - Timestamp: When the change was detected
    - Group: Sheet group identifier (e.g., "NA", "NF", "BUNDLE_FAN")
    - RowID: Smartsheet row identifier
    - Phase: Phase number (1-5) indicating the workflow stage
    - DateField: Name of the date column that changed
    - Date: The new date value (ISO format)
    - User: User associated with the change
    - Marketplace: Optional marketplace identifier (e.g., Amazon)

    Returns:
        None

    Side Effects:
        Creates CHANGES_FILE if it doesn't exist.
        Logs info message when creating a new file.

    Example:
        >>> ensure_changes_file()
        # Creates tracking_data/change_history.csv with headers if not present
    """
    if not os.path.exists(CHANGES_FILE):
        with open(CHANGES_FILE, 'w', newline='', encoding='utf-8') as f:
            writer: csv.writer = csv.writer(f)
            writer.writerow([
                "Timestamp",
                "Group",
                "RowID",
                "Phase",
                "DateField",
                "Date",
                "User",
                "Marketplace"
            ])
            logger.info(f"Created new changes file: {CHANGES_FILE}")

# Note: parse_date and normalize_date_for_comparison are now imported from date_utilities module

def track_changes() -> bool:
    """
    Main function to track changes in Smartsheet tables.

    Connects to the Smartsheet API and processes all configured sheets, detecting
    changes in date fields compared to the previously saved state. New or modified
    dates are recorded to the change history CSV file.

    The function implements comprehensive error handling with graceful degradation:
    - Sheet-level errors skip the problematic sheet and continue with others
    - Row-level errors skip the problematic row and continue processing
    - Phase-level errors skip the problematic phase and continue with other phases
    - Authentication errors halt processing immediately

    Workflow:
        1. Load previous state from state file
        2. Connect to Smartsheet API with retry support
        3. Validate configuration (sheet access, required columns)
        4. Process each configured sheet:
           - Detect available phase columns
           - Process each row, comparing current values to saved state
           - Record changes to CSV file
           - Update state with new values
        5. Save updated state
        6. Log validation and error statistics

    Returns:
        bool: True if tracking completed successfully (even with non-fatal errors).
              False if a fatal error occurred (e.g., authentication failure).

    Side Effects:
        - Updates STATE_FILE with current processed values
        - Appends detected changes to CHANGES_FILE
        - Logs detailed information to smartsheet_tracker.log
        - Updates global error collector with any errors encountered

    Example:
        >>> success = track_changes()
        >>> if success:
        ...     print("Tracking completed successfully")
        ... else:
        ...     print("Tracking failed - check logs for details")
    """
    # Start overall timing for the entire tracking operation
    overall_timer = PerformanceTimer("track_changes_total")
    overall_timer.start()

    logger.info("Starting Smartsheet change tracking")

    # Initialize
    state: TrackerState = load_state()
    ensure_changes_file()

    # Initialize validation statistics
    validation_stats: ValidationStats = create_validation_stats()

    # Initialize error collector for centralized error aggregation
    reset_global_collector()  # Start with a fresh collector
    error_collector: ErrorCollector = get_global_collector()

    # Connect to Smartsheet with retry-enabled client using shared utility
    try:
        client: SmartsheetRetryClient = create_smartsheet_client(token)
        logger.info("Connected to Smartsheet API with retry support")
    except Exception as e:
        logger.error(f"Failed to connect to Smartsheet: {e}")
        return False

    # Perform startup configuration validation
    logger.info("Performing startup configuration validation...")
    config_validation: ConfigValidationResult = validate_startup_config(
        client=client,
        sheet_ids=SHEET_IDS,
        phase_fields=PHASE_FIELDS,
        optional_columns=OPTIONAL_COLUMNS,
        fail_fast_on_token_error=True
    )

    # Log validation results
    config_validation.log_summary()

    # Check if we can proceed
    if not config_validation.can_proceed:
        logger.error("Startup configuration validation failed - cannot proceed")
        for error in config_validation.get_fatal_errors():
            logger.error(f"  {error}")
        return False

    if config_validation.get_overall_status() == "DEGRADED":
        logger.warning(f"Configuration has issues - proceeding with {config_validation.usable_sheets}/{config_validation.total_sheets} usable sheets")

    # Verify state format and structure
    processed: Dict[str, str] = state.get("processed", {})
    if not processed:
        logger.warning("State file has empty or invalid 'processed' dict - may detect ALL changes as new")

    # Force detect one change for testing
    test_mode: bool = False  # Set to True for testing
    if test_mode:
        logger.info("TEST MODE: Will detect at least one change")
        if processed:
            # Remove one random key from processed
            import random
            key_to_remove: str = random.choice(list(processed.keys()))
            logger.info(f"Removing key {key_to_remove} for test")
            processed.pop(key_to_remove, None)

    # Current timestamp
    now: datetime = datetime.now()

    # Track new changes
    changes_found: int = 0

    # Open file in append mode
    with open(CHANGES_FILE, 'a', newline='', encoding='utf-8') as f:
        writer: csv.writer = csv.writer(f)

        # Process each sheet
        for group, sheet_id in SHEET_IDS.items():
            logger.info(f"Processing sheet {group} (ID: {sheet_id})")

            try:
                # Get sheet with columns and rows (with automatic retry on timeout)
                # Time the sheet fetch operation
                with time_sheet_fetch(sheet_id=sheet_id, group=group):
                    sheet: Any = client.get_sheet(sheet_id)

                # Check if sheet retrieval failed (returns None for sheet not found or after max retries)
                if sheet is None:
                    logger.warning(
                        f"Skipping sheet {group} (ID: {sheet_id}) - Sheet may be invalid, deleted, or API call failed. "
                        f"Continuing with other available sheets."
                    )
                    continue

                logger.info(f"Sheet {group} has {len(sheet.rows)} rows")

                # Map column titles to IDs using shared utility
                col_map: Dict[str, int] = create_column_map(sheet)
                amazon_col_id: Optional[int] = col_map.get("Amazon")

                # Detect missing columns and log warnings (graceful degradation)
                column_status: Dict[str, Any] = detect_missing_columns(col_map, group)

                # Check which phase fields exist in this sheet (using column variation resolution)
                found_fields: List[str] = []
                for date_col, user_col_variations, _ in PHASE_FIELDS:
                    date_col_id, resolved_user_col, user_col_id = get_phase_columns(col_map, date_col, user_col_variations)
                    if date_col_id and user_col_id:
                        found_fields.append(date_col)
                logger.info(f"Found {len(found_fields)} phase fields in {group}: {found_fields}")

                # Time the data processing for this sheet
                import time as time_module
                row_count = len(sheet.rows)
                processing_start_time = time_module.perf_counter()

                # Process each row with error isolation
                for row in sheet.rows:
                    # Row-level error isolation: wrap entire row processing in try-except
                    try:
                        # Get row.id safely for error logging
                        current_row_id: Optional[Any] = None
                        try:
                            current_row_id = row.id
                        except Exception as row_id_error:
                            logger.error(
                                f"Sheet {group}: Failed to access row.id - {type(row_id_error).__name__}: {row_id_error}. "
                                f"Skipping row and continuing with remaining rows."
                            )
                            validation_stats.record_processing_error(
                                row_id=None,
                                group=group,
                                error=row_id_error,
                                error_category="row_id_access",
                                context={"sheet_id": sheet_id}
                            )
                            # Collect error in centralized collector
                            error_collector.collect(
                                error=row_id_error,
                                severity=ErrorSeverity.ERROR,
                                category=ErrorCategory.CELL_ACCESS,
                                context={"group": group, "sheet_id": sheet_id, "field": "row_id"},
                                source_function="track_changes"
                            )
                            continue

                        # Validate row.id is not null/empty
                        if is_empty_value(current_row_id):
                            logger.warning(f"Sheet {group}: Skipping row with null/empty row ID")
                            validation_stats.total_rows += 1
                            validation_stats.skipped_rows += 1
                            validation_stats.skipped_by_reason["missing_row_id"] = \
                                validation_stats.skipped_by_reason.get("missing_row_id", 0) + 1
                            continue

                        # Get marketplace if available - with error isolation
                        marketplace: str = ""
                        if amazon_col_id:
                            try:
                                for cell in row.cells:
                                    if cell.column_id == amazon_col_id:
                                        raw_marketplace: Any = cell.display_value
                                        marketplace = sanitize_value(raw_marketplace, "marketplace")
                                        break
                            except Exception as marketplace_error:
                                logger.warning(
                                    f"Sheet {group}: Row {current_row_id}: Error extracting marketplace - "
                                    f"{type(marketplace_error).__name__}: {marketplace_error}. Using empty marketplace."
                                )
                                marketplace = ""

                        # Check each phase field with error isolation
                        for date_col, user_col_variations, phase_no in PHASE_FIELDS:
                            # Phase-level error isolation
                            try:
                                # Use column variation resolution
                                date_col_id: Optional[int]
                                resolved_user_col: Optional[str]
                                user_col_id: Optional[int]
                                date_col_id, resolved_user_col, user_col_id = get_phase_columns(col_map, date_col, user_col_variations)
                                if not date_col_id or not user_col_id:
                                    continue

                                date_cell: Any = None
                                user_cell: Any = None

                                # Get date and user values - with error isolation for cell access
                                try:
                                    for cell in row.cells:
                                        if cell.column_id == date_col_id:
                                            date_cell = cell
                                        if cell.column_id == user_col_id:
                                            user_cell = cell
                                except Exception as cell_access_error:
                                    logger.warning(
                                        f"Sheet {group}: Row {current_row_id}: Phase {phase_no}: "
                                        f"Error accessing cells - {type(cell_access_error).__name__}: {cell_access_error}. "
                                        f"Skipping phase and continuing."
                                    )
                                    validation_stats.record_processing_error(
                                        row_id=str(current_row_id),
                                        group=group,
                                        error=cell_access_error,
                                        error_category="cell_access",
                                        context={"phase": phase_no, "date_col": date_col}
                                    )
                                    # Collect error in centralized collector
                                    error_collector.collect(
                                        error=cell_access_error,
                                        severity=ErrorSeverity.WARNING,
                                        category=ErrorCategory.CELL_ACCESS,
                                        context={"group": group, "sheet_id": sheet_id, "row_id": current_row_id, "phase": phase_no, "date_col": date_col},
                                        source_function="track_changes"
                                    )
                                    continue

                                # Extract raw values
                                date_val: Any = date_cell.value if date_cell else None
                                raw_user_val: Any = user_cell.display_value if user_cell else None

                                # Validate row data using the validation module
                                validation_result: ValidationResult = validate_row_data(
                                    row_id=current_row_id,
                                    group=group,
                                    date_value=date_val,
                                    user_value=raw_user_val,
                                    marketplace=marketplace,
                                    phase=phase_no,
                                    date_field=date_col,
                                    log_issues=True  # Log validation issues
                                )

                                # Record validation result in statistics
                                validation_stats.record_result(validation_result)

                                # Skip if validation determined row should be skipped
                                if validation_result.skipped:
                                    continue

                                # Sanitize user value (convert null-like to empty string)
                                user_val: Any = sanitize_value(raw_user_val, "user_value")

                                # Create unique key for this field
                                field_key: str = f"{group}:{current_row_id}:{date_col}"

                                # Normalize both values for robust comparison
                                # This handles format differences (datetime vs date vs string)
                                normalized_current: Optional[str] = normalize_date_for_comparison(date_val)
                                prev_val: Optional[str] = state["processed"].get(field_key)
                                normalized_prev: Optional[str] = normalize_date_for_comparison(prev_val)

                                if normalized_prev == normalized_current:
                                    continue

                                # Only log detailed info for changes
                                logger.info(f"Change detected in {field_key}")
                                logger.info(f"  Previous: '{prev_val}' (normalized: '{normalized_prev}')")
                                logger.info(f"  Current:  '{date_val}' (normalized: '{normalized_current}')")

                                # Parse date - with error isolation
                                try:
                                    parsed_date: Optional[date] = parse_date(date_val)
                                    if not parsed_date:
                                        logger.warning(f"Could not parse date: {date_val} for {field_key}")
                                        continue
                                except Exception as date_parse_error:
                                    logger.warning(
                                        f"Sheet {group}: Row {current_row_id}: Phase {phase_no}: "
                                        f"Unexpected error parsing date '{date_val}' - "
                                        f"{type(date_parse_error).__name__}: {date_parse_error}. Skipping this phase."
                                    )
                                    validation_stats.record_processing_error(
                                        row_id=str(current_row_id),
                                        group=group,
                                        error=date_parse_error,
                                        error_category="date_parsing",
                                        context={"phase": phase_no, "date_col": date_col, "date_value": str(date_val)}
                                    )
                                    # Collect error in centralized collector
                                    error_collector.collect(
                                        error=date_parse_error,
                                        severity=ErrorSeverity.WARNING,
                                        category=ErrorCategory.DATE_PARSING,
                                        context={"group": group, "sheet_id": sheet_id, "row_id": current_row_id, "phase": phase_no, "date_col": date_col, "date_value": str(date_val)},
                                        source_function="track_changes"
                                    )
                                    continue

                                # Record the change - with error isolation for CSV write
                                try:
                                    writer.writerow([
                                        now.strftime("%Y-%m-%d %H:%M:%S"),
                                        group,
                                        current_row_id,
                                        phase_no,
                                        date_col,
                                        parsed_date.isoformat(),
                                        user_val,
                                        marketplace
                                    ])
                                except Exception as csv_write_error:
                                    logger.error(
                                        f"Sheet {group}: Row {current_row_id}: Phase {phase_no}: "
                                        f"Failed to write to CSV - {type(csv_write_error).__name__}: {csv_write_error}. "
                                        f"Change may not be recorded. Continuing with remaining rows."
                                    )
                                    validation_stats.record_processing_error(
                                        row_id=str(current_row_id),
                                        group=group,
                                        error=csv_write_error,
                                        error_category="csv_write",
                                        context={"phase": phase_no, "date_col": date_col}
                                    )
                                    # Collect error in centralized collector
                                    error_collector.collect(
                                        error=csv_write_error,
                                        severity=ErrorSeverity.ERROR,
                                        category=ErrorCategory.CSV_WRITE,
                                        context={"group": group, "sheet_id": sheet_id, "row_id": current_row_id, "phase": phase_no, "date_col": date_col},
                                        source_function="track_changes"
                                    )
                                    continue

                                # Update state with normalized date (always YYYY-MM-DD)
                                state["processed"][field_key] = normalized_current

                                changes_found += 1

                            except Exception as phase_error:
                                # Catch any unexpected errors at the phase level
                                logger.error(
                                    f"Sheet {group}: Row {current_row_id}: Phase {phase_no}: "
                                    f"Unexpected error - {type(phase_error).__name__}: {phase_error}. "
                                    f"Skipping this phase and continuing with remaining phases."
                                )
                                validation_stats.record_processing_error(
                                    row_id=str(current_row_id),
                                    group=group,
                                    error=phase_error,
                                    error_category="phase_processing",
                                    context={"phase": phase_no, "date_col": date_col}
                                )
                                # Collect error in centralized collector
                                error_collector.collect(
                                    error=phase_error,
                                    severity=ErrorSeverity.ERROR,
                                    category=ErrorCategory.PHASE_PROCESSING,
                                    context={"group": group, "sheet_id": sheet_id, "row_id": current_row_id, "phase": phase_no, "date_col": date_col},
                                    source_function="track_changes"
                                )
                                continue

                    except Exception as row_error:
                        # Catch any unexpected errors at the row level
                        row_id_str = str(current_row_id) if current_row_id else "unknown"
                        logger.error(
                            f"Sheet {group}: Row {row_id_str}: Unexpected error during row processing - "
                            f"{type(row_error).__name__}: {row_error}. "
                            f"Skipping this row and continuing with remaining rows."
                        )
                        validation_stats.record_processing_error(
                            row_id=row_id_str,
                            group=group,
                            error=row_error,
                            error_category="row_processing",
                            context={"sheet_id": sheet_id}
                        )
                        # Collect error in centralized collector
                        error_collector.collect(
                            error=row_error,
                            severity=ErrorSeverity.ERROR,
                            category=ErrorCategory.ROW_PROCESSING,
                            context={"group": group, "sheet_id": sheet_id, "row_id": row_id_str},
                            source_function="track_changes"
                        )
                        continue

                # Log timing for row processing
                from performance_timing import log_timing
                processing_duration = time_module.perf_counter() - processing_start_time
                log_timing("data_processing.row_processing", processing_duration, group=group, row_count=row_count)

            except SheetNotFoundError as e:
                # Specific handling for sheet not found (invalid ID or deleted)
                logger.error(
                    f"Sheet {group} (ID: {sheet_id}) not found - The sheet ID may be invalid or the sheet has been deleted. "
                    f"Error: {e}. Continuing with other available sheets."
                )
                # Collect error in centralized collector
                error_collector.collect(
                    error=e,
                    severity=ErrorSeverity.WARNING,
                    category=ErrorCategory.SHEET_NOT_FOUND,
                    context={"group": group, "sheet_id": sheet_id},
                    source_function="track_changes"
                )
                continue
            except PermissionDeniedError as e:
                # Specific handling for permission denied errors (user lacks access to this sheet)
                # This is a non-fatal, sheet-level error - continue with other sheets
                logger.warning(
                    f"Permission denied for sheet {group} (ID: {sheet_id}): {e}\n"
                    f"{e.get_actionable_message()}"
                )
                logger.info(f"Skipping inaccessible sheet {group} and continuing with other authorized sheets.")
                # Collect error in centralized collector
                error_collector.collect(
                    error=e,
                    severity=ErrorSeverity.WARNING,
                    category=ErrorCategory.PERMISSION,
                    context={"group": group, "sheet_id": sheet_id},
                    suggested_action=e.get_actionable_message(),
                    source_function="track_changes"
                )
                continue
            except TokenAuthenticationError as e:
                # Specific handling for token/authentication errors
                logger.error(
                    f"Authentication failed for sheet {group} (ID: {sheet_id}): {e}\n"
                    f"{e.get_actionable_message()}"
                )
                # Token errors affect all sheets, so we should stop processing
                logger.error("Stopping processing due to authentication error. Please refresh your token.")
                # Collect error in centralized collector
                error_collector.collect(
                    error=e,
                    severity=ErrorSeverity.CRITICAL,
                    category=ErrorCategory.TOKEN_EXPIRED if e.is_expired else ErrorCategory.TOKEN_INVALID,
                    context={"group": group, "sheet_id": sheet_id},
                    suggested_action=e.get_actionable_message(),
                    is_recoverable=False,
                    source_function="track_changes"
                )
                return False
            except Exception as e:
                # Check if this looks like a token error that wasn't caught
                if is_token_error(e):
                    logger.error(
                        f"Authentication error detected for sheet {group} (ID: {sheet_id}): {e}\n"
                        "Your Smartsheet API token may be expired or invalid.\n"
                        "To resolve this:\n"
                        "  1. Log in to Smartsheet and navigate to Account > Personal Settings > API Access\n"
                        "  2. Generate a new API Access Token\n"
                        "  3. Update the SMARTSHEET_TOKEN value in your .env file\n"
                        "  4. Restart the application"
                    )
                    return False
                # Check if this looks like a permission denied error that wasn't caught
                elif is_permission_denied_error(e):
                    logger.warning(
                        f"Permission denied for sheet {group} (ID: {sheet_id}): {e}\n"
                        "You do not have access to this sheet. Contact the sheet owner to request access.\n"
                        "Continuing with other authorized sheets."
                    )
                # Check if this looks like a sheet-not-found error that wasn't caught
                elif is_sheet_not_found_error(e):
                    logger.error(
                        f"Sheet {group} (ID: {sheet_id}) not found - The sheet ID may be invalid or the sheet has been deleted. "
                        f"Error: {e}. Continuing with other available sheets."
                    )
                else:
                    logger.error(f"Error processing sheet {group} (ID: {sheet_id}): {e}")
                continue

    # Update state
    state["last_run"] = now.strftime("%Y-%m-%d %H:%M:%S")
    save_state(state)

    # Log validation statistics
    if validation_stats.total_rows > 0:
        validation_stats.log_summary()

    # Log error collector summary
    if error_collector.error_count() > 0:
        error_collector.log_summary()
        logger.info(f"Error Collection: {error_collector.error_count()} errors collected")
        if error_collector.has_critical_errors():
            logger.warning("Critical errors were encountered during processing")

    logger.info(f"Change tracking completed. Found {changes_found} changes.")
    logger.info(f"Validation: {validation_stats.valid_rows} valid rows, {validation_stats.skipped_rows} skipped rows")

    # Log overall timing
    overall_timer.stop()
    return True

def reset_tracking_state() -> bool:
    """
    Reset the tracking state to current Smartsheet data.

    Rebuilds the state file from scratch by reading all current values from
    Smartsheet. After reset, subsequent tracking runs will only detect changes
    made after the reset, not existing data.

    This function is useful when:
    - Starting fresh after configuration changes
    - Recovering from a corrupted state file
    - Ignoring historical data and only tracking future changes

    Workflow:
        1. Connect to Smartsheet API with retry support
        2. Create empty state with current timestamp
        3. Process each configured sheet:
           - Map column titles to IDs
           - Extract current date values for all rows and phases
           - Add to processed state (no changes are recorded)
        4. Save new state file
        5. Reset change history file to headers only

    Returns:
        bool: True if reset completed successfully.
              False if a fatal error occurred (e.g., authentication failure).

    Side Effects:
        - Overwrites STATE_FILE with current Smartsheet values
        - Overwrites CHANGES_FILE with headers only (clears history)
        - Logs reset progress and summary

    Example:
        >>> success = reset_tracking_state()
        >>> if success:
        ...     print(f"Reset complete - ready to track future changes")
    """
    logger.info("Resetting tracking state...")

    # Connect to Smartsheet with retry support using shared utility
    client: SmartsheetRetryClient = create_smartsheet_client(token)
    state: TrackerState = {"last_run": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "processed": {}}

    # Process each sheet to build state
    for group, sid in SHEET_IDS.items():
        logger.info(f"Processing sheet {group}...")
        try:
            sheet: Any = client.get_sheet(sid)

            # Check if sheet retrieval failed (returns None for sheet not found or after max retries)
            if sheet is None:
                logger.warning(
                    f"Skipping sheet {group} (ID: {sid}) in reset - Sheet may be invalid, deleted, or API call failed. "
                    f"Continuing with other available sheets."
                )
                continue

            # Map column titles to IDs using shared utility
            col_map: Dict[str, int] = create_column_map(sheet)

            # Detect missing columns and log warnings (graceful degradation)
            column_status: Dict[str, Any] = detect_missing_columns(col_map, group)

            # Process each row with error isolation
            for row in sheet.rows:
                # Row-level error isolation for reset operation
                try:
                    # Get row.id safely
                    current_row_id: Optional[Any] = None
                    try:
                        current_row_id = row.id
                    except Exception as row_id_error:
                        logger.error(
                            f"Sheet {group}: Failed to access row.id during reset - "
                            f"{type(row_id_error).__name__}: {row_id_error}. "
                            f"Skipping row and continuing with remaining rows."
                        )
                        continue

                    if is_empty_value(current_row_id):
                        continue

                    for date_col, _, _ in PHASE_FIELDS:
                        try:
                            col_id: Optional[int] = col_map.get(date_col)
                            if not col_id:
                                continue

                            # Find cell with this column ID - with error isolation
                            try:
                                for cell in row.cells:
                                    if cell.column_id == col_id and cell.value:
                                        # Add to processed state with normalized date
                                        field_key: str = f"{group}:{current_row_id}:{date_col}"
                                        state["processed"][field_key] = normalize_date_for_comparison(cell.value)
                                        break
                            except Exception as cell_error:
                                logger.warning(
                                    f"Sheet {group}: Row {current_row_id}: Error accessing cells during reset - "
                                    f"{type(cell_error).__name__}: {cell_error}. Skipping this date column."
                                )
                                continue
                        except Exception as date_col_error:
                            logger.warning(
                                f"Sheet {group}: Row {current_row_id}: Error processing date column '{date_col}' - "
                                f"{type(date_col_error).__name__}: {date_col_error}. Continuing with other columns."
                            )
                            continue

                except Exception as row_error:
                    row_id_str = str(current_row_id) if current_row_id else "unknown"
                    logger.error(
                        f"Sheet {group}: Row {row_id_str}: Unexpected error during reset - "
                        f"{type(row_error).__name__}: {row_error}. "
                        f"Skipping this row and continuing with remaining rows."
                    )
                    continue

        except SheetNotFoundError as e:
            # Specific handling for sheet not found (invalid ID or deleted)
            logger.error(
                f"Sheet {group} (ID: {sid}) not found during reset - The sheet ID may be invalid or the sheet has been deleted. "
                f"Error: {e}. Continuing with other available sheets."
            )
            continue
        except PermissionDeniedError as e:
            # Specific handling for permission denied errors (user lacks access to this sheet)
            # This is a non-fatal, sheet-level error - continue with other sheets
            logger.warning(
                f"Permission denied for sheet {group} (ID: {sid}) during reset: {e}\n"
                f"{e.get_actionable_message()}"
            )
            logger.info(f"Skipping inaccessible sheet {group} and continuing with other authorized sheets.")
            continue
        except TokenAuthenticationError as e:
            # Specific handling for token/authentication errors
            logger.error(
                f"Authentication failed during reset for sheet {group} (ID: {sid}): {e}\n"
                f"{e.get_actionable_message()}"
            )
            logger.error("Stopping reset due to authentication error. Please refresh your token.")
            return False
        except Exception as e:
            # Check if this looks like a token error that wasn't caught
            if is_token_error(e):
                logger.error(
                    f"Authentication error detected during reset for sheet {group} (ID: {sid}): {e}\n"
                    "Your Smartsheet API token may be expired or invalid.\n"
                    "To resolve this:\n"
                    "  1. Log in to Smartsheet and navigate to Account > Personal Settings > API Access\n"
                    "  2. Generate a new API Access Token\n"
                    "  3. Update the SMARTSHEET_TOKEN value in your .env file\n"
                    "  4. Restart the application"
                )
                return False
            # Check if this looks like a permission denied error that wasn't caught
            elif is_permission_denied_error(e):
                logger.warning(
                    f"Permission denied for sheet {group} (ID: {sid}) during reset: {e}\n"
                    "You do not have access to this sheet. Contact the sheet owner to request access.\n"
                    "Continuing with other authorized sheets."
                )
            # Check if this looks like a sheet-not-found error that wasn't caught
            elif is_sheet_not_found_error(e):
                logger.error(
                    f"Sheet {group} (ID: {sid}) not found during reset - The sheet ID may be invalid or the sheet has been deleted. "
                    f"Error: {e}. Continuing with other available sheets."
                )
            else:
                logger.error(f"Error processing sheet {group} (ID: {sid}) during reset: {e}")
            continue

    # Save state
    save_state(state)

    # Reset change history file
    with open(CHANGES_FILE, "w", newline="", encoding="utf-8") as f:
        writer: csv.writer = csv.writer(f)
        writer.writerow([
            "Timestamp",
            "Group",
            "RowID",
            "Phase",
            "DateField",
            "Date",
            "User",
            "Marketplace"
        ])

    logger.info(f"Reset complete: Marked {len(state['processed'])} items as processed")
    return True

def bootstrap_tracking(days_back: int = 0) -> bool:
    """
    Initialize tracking by clearing state and running change detection.

    Performs a fresh start by clearing the previous state and running
    change tracking. This causes all current data to be detected as new
    changes and recorded to the change history file.

    This function is useful when:
    - Setting up tracking for the first time
    - Capturing all current data as the initial baseline
    - Testing change detection with a known starting point

    Args:
        days_back: Reserved for future use. Currently not implemented.
            Intended to allow bootstrapping with a historical cutoff date.
            Default is 0 (track all current data).

    Returns:
        bool: True if bootstrap and subsequent tracking completed successfully.
              False if tracking failed.

    Side Effects:
        - Clears STATE_FILE to empty state before tracking
        - Records all current date values as changes in CHANGES_FILE
        - Logs bootstrap progress

    Example:
        >>> success = bootstrap_tracking()
        >>> if success:
        ...     print("Bootstrap complete - all current data recorded")
    """
    logger.info(f"Starting bootstrap (tracking new data only)")

    # Reset state to force reprocessing
    state: TrackerState = {"last_run": None, "processed": {}}
    save_state(state)

    # Run tracking
    return track_changes()

def test_changes() -> bool:
    """
    Test change detection by artificially removing one item from state.

    Modifies the state file by removing a randomly selected processed item,
    causing that item to be detected as a change on the next tracking run.
    This is useful for testing and debugging the change detection mechanism.

    Warning:
        This function modifies the state file. The removed item will be
        re-detected as a change on the next run of track_changes().

    Workflow:
        1. Load current state from state file
        2. Validate that processed items exist
        3. Randomly select and remove one processed item
        4. Save modified state

    Returns:
        bool: True if test modification was successful.
              False if no processed items exist to remove.

    Side Effects:
        - Modifies STATE_FILE by removing one processed item
        - Logs which item was removed

    Example:
        >>> test_changes()  # Remove one item from state
        True
        >>> track_changes()  # Will detect the removed item as a change
    """
    logger.info("Testing change detection...")

    # Load state
    state: TrackerState = load_state()
    processed: Dict[str, str] = state.get("processed", {})

    if not processed:
        logger.error("No processed items found in state file. Run reset first.")
        return False

    # Remove one item to force detection
    import random
    key_to_remove: str = random.choice(list(processed.keys()))
    logger.info(f"Removing key {key_to_remove} to force change detection")
    processed.pop(key_to_remove, None)

    # Save modified state
    save_state(state)
    logger.info("Test modification saved. Now run tracking to detect the forced change.")
    return True

def validate_config_only() -> bool:
    """
    Run configuration validation only, without processing any changes.

    Performs a comprehensive validation of the tracker configuration by
    connecting to Smartsheet and verifying that all configured sheets are
    accessible and contain the required columns. This is useful for:
    - Pre-flight checks before scheduling automated runs
    - Debugging access or permission issues
    - Verifying configuration after changes to SHEET_IDS or PHASE_FIELDS

    Validation checks performed:
        1. API token validity and authentication
        2. Sheet accessibility for each configured sheet ID
        3. Presence of required date columns (Kontrolle, BE am, K am, etc.)
        4. Presence of corresponding user columns (K von, BE von, etc.)
        5. Availability of optional columns (Amazon marketplace)

    Returns:
        bool: True if configuration is valid and tracking can proceed.
              False if critical validation errors were found.

    Side Effects:
        - Logs detailed validation results to logger
        - Prints validation summary to stdout
        - Does NOT modify state or change history files

    Example:
        >>> # Verify configuration before scheduling automated runs
        >>> if validate_config_only():
        ...     print("Configuration valid - safe to schedule automated runs")
        ... else:
        ...     print("Configuration issues found - check logs for details")

        >>> # Can also be run from command line:
        >>> # python smartsheet_tracker.py --validate-config
    """
    logger.info("Running configuration validation only...")

    # Connect to Smartsheet using shared utility
    try:
        client: SmartsheetRetryClient = create_smartsheet_client(token)
        logger.info("Connected to Smartsheet API")
    except Exception as e:
        logger.error(f"Failed to connect to Smartsheet: {e}")
        return False

    # Perform validation
    config_validation: ConfigValidationResult = validate_startup_config(
        client=client,
        sheet_ids=SHEET_IDS,
        phase_fields=PHASE_FIELDS,
        optional_columns=OPTIONAL_COLUMNS,
        fail_fast_on_token_error=True
    )

    # Log detailed summary
    config_validation.log_summary()

    # Print summary to stdout as well
    summary: str = create_validation_summary(config_validation)
    print(summary)

    return config_validation.can_proceed


if __name__ == "__main__":
    parser: argparse.ArgumentParser = argparse.ArgumentParser(description="Smartsheet Change Tracker")
    parser.add_argument("--bootstrap", action="store_true", help="Initialize tracking for all data")
    parser.add_argument("--reset", action="store_true", help="Reset tracking state to current data")
    parser.add_argument("--test", action="store_true", help="Test change detection by forcing changes")
    parser.add_argument("--validate-config", action="store_true",
                       help="Validate configuration only (check sheet access and columns)")
    add_log_level_argument(parser)
    args: argparse.Namespace = parser.parse_args()

    # Configure logging with CLI argument or environment variable
    configure_logging(
        log_file="smartsheet_tracker.log",
        log_level=args.log_level
    )

    success: bool
    if args.validate_config:
        success = validate_config_only()
    elif args.reset:
        success = reset_tracking_state()
    elif args.bootstrap:
        success = bootstrap_tracking()
    elif args.test:
        success = test_changes()
    else:
        success = track_changes()

    exit(0 if success else 1)
