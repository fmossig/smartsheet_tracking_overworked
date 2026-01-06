"""
Shared Smartsheet client utilities for consistent API access across all scripts.

This module centralizes Smartsheet client initialization, sheet fetching, and
common operations with built-in error handling. It eliminates code duplication
across scripts by providing a single source of truth for:

1. Client initialization with standard retry configuration
2. Token loading from environment variables
3. Sheet fetching with comprehensive error handling
4. Column mapping utilities
5. Result dataclasses for structured return values

Usage:
    from smartsheet_client import (
        create_smartsheet_client,
        fetch_sheet,
        fetch_sheet_with_columns,
        get_smartsheet_token,
        create_column_map,
        SheetFetchResult,
    )

    # Get a client with standard configuration
    client = create_smartsheet_client()

    # Fetch a sheet with error handling
    result = fetch_sheet(client, sheet_id)
    if result.success:
        sheet = result.sheet
        col_map = result.column_map

Example:
    >>> token = get_smartsheet_token()
    >>> client = create_smartsheet_client(token)
    >>> result = fetch_sheet(client, 6141179298008964)
    >>> if result.success:
    ...     print(f"Sheet has {len(result.sheet.rows)} rows")
"""

import os
import logging
from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List, Tuple, Union

import smartsheet
from dotenv import load_dotenv

from smartsheet_retry import (
    SmartsheetRetryClient,
    SheetNotFoundError,
    TokenAuthenticationError,
    PermissionDeniedError,
    is_sheet_not_found_error,
    is_token_error,
    is_permission_denied_error,
    DEFAULT_MAX_RETRIES,
    DEFAULT_BASE_DELAY,
    DEFAULT_MAX_DELAY,
)

# Get logger for this module
logger = logging.getLogger(__name__)

# Default configuration constants
DEFAULT_CONTINUE_ON_FAILURE = True


@dataclass
class SheetFetchResult:
    """
    Result container for sheet fetch operations.

    Provides a structured way to return sheet data along with metadata
    and error information from fetch operations.

    Attributes:
        success (bool): True if the sheet was fetched successfully.
        sheet: The fetched Smartsheet sheet object, or None if fetch failed.
        column_map (Dict[str, int]): Mapping of column titles to column IDs.
        error (Optional[Exception]): The exception if fetch failed, None otherwise.
        error_type (Optional[str]): Categorized error type for handling decisions.
            Values: 'not_found', 'permission_denied', 'auth_error', 'other'
        message (str): Human-readable status or error message.
        sheet_id (Optional[int]): The sheet ID that was requested.
        group_name (Optional[str]): Optional group name for logging context.

    Example:
        >>> result = fetch_sheet(client, sheet_id)
        >>> if result.success:
        ...     for row in result.sheet.rows:
        ...         process_row(row, result.column_map)
        ... elif result.error_type == 'permission_denied':
        ...     logger.warning(f"Access denied: {result.message}")
    """
    success: bool
    sheet: Any = None
    column_map: Dict[str, int] = field(default_factory=dict)
    error: Optional[Exception] = None
    error_type: Optional[str] = None
    message: str = ""
    sheet_id: Optional[int] = None
    group_name: Optional[str] = None

    def __bool__(self) -> bool:
        """Allow using result directly in boolean context."""
        return self.success


@dataclass
class ClientConfig:
    """
    Configuration for SmartsheetRetryClient initialization.

    Centralizes retry and error handling configuration for consistent
    client behavior across all scripts.

    Attributes:
        max_retries (int): Maximum number of retry attempts for failed API calls.
        base_delay (float): Initial delay in seconds between retries.
        max_delay (float): Maximum delay in seconds between retries.
        continue_on_failure (bool): If True, return None after max retries
            instead of raising an exception.

    Example:
        >>> config = ClientConfig(max_retries=5, base_delay=2.0)
        >>> client = create_smartsheet_client(config=config)
    """
    max_retries: int = DEFAULT_MAX_RETRIES
    base_delay: float = DEFAULT_BASE_DELAY
    max_delay: float = DEFAULT_MAX_DELAY
    continue_on_failure: bool = DEFAULT_CONTINUE_ON_FAILURE


# Default configuration instance
DEFAULT_CONFIG = ClientConfig()


def get_smartsheet_token(
    env_var_name: str = "SMARTSHEET_TOKEN",
    load_env: bool = True,
    required: bool = True
) -> Optional[str]:
    """
    Load the Smartsheet API token from environment variables.

    Centralizes token loading with consistent error handling across all scripts.
    Optionally loads from .env file using python-dotenv.

    Args:
        env_var_name (str): Name of the environment variable containing the token.
            Defaults to "SMARTSHEET_TOKEN".
        load_env (bool): Whether to load variables from .env file.
            Defaults to True.
        required (bool): If True, raises ValueError when token is not found.
            If False, returns None when token is not found.
            Defaults to True.

    Returns:
        Optional[str]: The Smartsheet API token, or None if not found and not required.

    Raises:
        ValueError: If required is True and token is not found.

    Example:
        >>> token = get_smartsheet_token()  # Raises if not found
        >>> token = get_smartsheet_token(required=False)  # Returns None if not found
    """
    if load_env:
        load_dotenv()

    token = os.getenv(env_var_name)

    if not token and required:
        error_msg = (
            f"{env_var_name} not found in environment or .env file. "
            "Please ensure the token is set in your environment or .env file."
        )
        logger.error(error_msg)
        raise ValueError(error_msg)

    if not token:
        logger.warning(f"{env_var_name} not found in environment or .env file")

    return token


def create_smartsheet_client(
    token: Optional[str] = None,
    config: Optional[ClientConfig] = None,
    errors_as_exceptions: bool = True
) -> SmartsheetRetryClient:
    """
    Create a SmartsheetRetryClient with standard configuration.

    Factory function that creates a properly configured SmartsheetRetryClient
    with retry logic and rate limit handling. This is the recommended way to
    create Smartsheet clients across all scripts.

    Args:
        token (Optional[str]): Smartsheet API token. If None, loads from
            environment using get_smartsheet_token().
        config (Optional[ClientConfig]): Client configuration. If None,
            uses DEFAULT_CONFIG.
        errors_as_exceptions (bool): Whether to raise exceptions for API errors.
            Defaults to True (recommended).

    Returns:
        SmartsheetRetryClient: A configured client with retry and rate limit handling.

    Raises:
        ValueError: If token is not provided and cannot be loaded from environment.
        Exception: If client initialization fails.

    Example:
        >>> # Using default configuration
        >>> client = create_smartsheet_client()

        >>> # With custom configuration
        >>> config = ClientConfig(max_retries=5)
        >>> client = create_smartsheet_client(config=config)

        >>> # With explicit token
        >>> client = create_smartsheet_client(token="your-api-token")
    """
    if token is None:
        token = get_smartsheet_token()

    if config is None:
        config = DEFAULT_CONFIG

    try:
        base_client = smartsheet.Smartsheet(token)
        base_client.errors_as_exceptions(errors_as_exceptions)

        client = SmartsheetRetryClient(
            base_client,
            max_retries=config.max_retries,
            base_delay=config.base_delay,
            max_delay=config.max_delay,
            continue_on_failure=config.continue_on_failure
        )

        logger.debug(
            f"Created SmartsheetRetryClient with config: "
            f"max_retries={config.max_retries}, "
            f"base_delay={config.base_delay}s, "
            f"max_delay={config.max_delay}s"
        )

        return client

    except Exception as e:
        logger.error(f"Failed to create Smartsheet client: {e}")
        raise


def create_column_map(sheet) -> Dict[str, int]:
    """
    Create a mapping of column titles to column IDs from a sheet.

    Utility function to standardize column map creation across all scripts.

    Args:
        sheet: A Smartsheet sheet object with a columns attribute.

    Returns:
        Dict[str, int]: Mapping of column titles to column IDs.

    Example:
        >>> sheet = client.get_sheet(sheet_id)
        >>> col_map = create_column_map(sheet)
        >>> kontrolle_col_id = col_map.get("Kontrolle")
    """
    if sheet is None:
        return {}

    try:
        return {col.title: col.id for col in sheet.columns}
    except (AttributeError, TypeError) as e:
        logger.warning(f"Could not create column map from sheet: {e}")
        return {}


def fetch_sheet(
    client: SmartsheetRetryClient,
    sheet_id: int,
    group_name: Optional[str] = None,
    include: Optional[List[str]] = None,
    log_errors: bool = True
) -> SheetFetchResult:
    """
    Fetch a sheet with comprehensive error handling.

    Wraps the sheet fetch operation with standardized error handling,
    returning a structured result that includes the sheet, column map,
    and any error information.

    Args:
        client (SmartsheetRetryClient): The Smartsheet client to use.
        sheet_id (int): The ID of the sheet to fetch.
        group_name (Optional[str]): Optional name for logging context.
        include (Optional[List[str]]): List of elements to include in response.
            Common values: ['columns'], ['attachments'], ['discussions'].
        log_errors (bool): Whether to log errors. Defaults to True.

    Returns:
        SheetFetchResult: A result object containing:
            - success: True if fetch succeeded
            - sheet: The fetched sheet (or None)
            - column_map: Column title to ID mapping
            - error: Exception if failed
            - error_type: Categorized error type
            - message: Human-readable message

    Example:
        >>> result = fetch_sheet(client, 6141179298008964, group_name="NA")
        >>> if result.success:
        ...     print(f"Fetched {len(result.sheet.rows)} rows")
        ...     kontrolle_id = result.column_map.get("Kontrolle")
        ... elif result.error_type == 'not_found':
        ...     print(f"Sheet not found: {result.message}")
    """
    context = f"sheet {group_name} (ID: {sheet_id})" if group_name else f"sheet {sheet_id}"

    try:
        # Fetch the sheet
        if include:
            sheet = client.get_sheet(sheet_id, include=include)
        else:
            sheet = client.get_sheet(sheet_id)

        # Check if sheet retrieval failed (returns None after max retries)
        if sheet is None:
            message = (
                f"Failed to fetch {context} - Sheet may be invalid, deleted, "
                "or API call failed after max retries."
            )
            if log_errors:
                logger.warning(message)

            return SheetFetchResult(
                success=False,
                sheet=None,
                column_map={},
                error=None,
                error_type='fetch_failed',
                message=message,
                sheet_id=sheet_id,
                group_name=group_name
            )

        # Create column map
        col_map = create_column_map(sheet)

        message = f"Successfully fetched {context} with {len(sheet.rows)} rows"
        logger.debug(message)

        return SheetFetchResult(
            success=True,
            sheet=sheet,
            column_map=col_map,
            error=None,
            error_type=None,
            message=message,
            sheet_id=sheet_id,
            group_name=group_name
        )

    except SheetNotFoundError as e:
        message = (
            f"Sheet not found: {context} - The sheet ID may be invalid "
            f"or the sheet has been deleted. Error: {e}"
        )
        if log_errors:
            logger.error(message)

        return SheetFetchResult(
            success=False,
            sheet=None,
            column_map={},
            error=e,
            error_type='not_found',
            message=message,
            sheet_id=sheet_id,
            group_name=group_name
        )

    except PermissionDeniedError as e:
        message = (
            f"Permission denied for {context}: {e}\n"
            f"{e.get_actionable_message()}"
        )
        if log_errors:
            logger.warning(message)

        return SheetFetchResult(
            success=False,
            sheet=None,
            column_map={},
            error=e,
            error_type='permission_denied',
            message=message,
            sheet_id=sheet_id,
            group_name=group_name
        )

    except TokenAuthenticationError as e:
        message = (
            f"Authentication failed for {context}: {e}\n"
            f"{e.get_actionable_message()}"
        )
        if log_errors:
            logger.error(message)

        return SheetFetchResult(
            success=False,
            sheet=None,
            column_map={},
            error=e,
            error_type='auth_error',
            message=message,
            sheet_id=sheet_id,
            group_name=group_name
        )

    except Exception as e:
        # Check for specific error types that may not be caught as custom exceptions
        if is_token_error(e):
            error_type = 'auth_error'
            message = (
                f"Authentication error for {context}: {e}\n"
                "Your Smartsheet API token may be expired or invalid."
            )
            if log_errors:
                logger.error(message)
        elif is_permission_denied_error(e):
            error_type = 'permission_denied'
            message = f"Permission denied for {context}: {e}"
            if log_errors:
                logger.warning(message)
        elif is_sheet_not_found_error(e):
            error_type = 'not_found'
            message = f"Sheet not found: {context} - {e}"
            if log_errors:
                logger.error(message)
        else:
            error_type = 'other'
            message = f"Error fetching {context}: {e}"
            if log_errors:
                logger.error(message)

        return SheetFetchResult(
            success=False,
            sheet=None,
            column_map={},
            error=e,
            error_type=error_type,
            message=message,
            sheet_id=sheet_id,
            group_name=group_name
        )


def fetch_sheet_with_columns(
    client: SmartsheetRetryClient,
    sheet_id: int,
    group_name: Optional[str] = None,
    log_errors: bool = True
) -> SheetFetchResult:
    """
    Fetch a sheet with column information included.

    Convenience function that calls fetch_sheet with include=['columns'].
    Useful when you only need the sheet structure without full row data.

    Args:
        client (SmartsheetRetryClient): The Smartsheet client to use.
        sheet_id (int): The ID of the sheet to fetch.
        group_name (Optional[str]): Optional name for logging context.
        log_errors (bool): Whether to log errors. Defaults to True.

    Returns:
        SheetFetchResult: A result object containing the sheet and column map.

    Example:
        >>> result = fetch_sheet_with_columns(client, sheet_id, "NA")
        >>> if result.success:
        ...     print(f"Columns: {list(result.column_map.keys())}")
    """
    return fetch_sheet(
        client=client,
        sheet_id=sheet_id,
        group_name=group_name,
        include=['columns'],
        log_errors=log_errors
    )


def fetch_multiple_sheets(
    client: SmartsheetRetryClient,
    sheet_ids: Dict[str, int],
    include: Optional[List[str]] = None,
    log_errors: bool = True,
    stop_on_auth_error: bool = True
) -> Dict[str, SheetFetchResult]:
    """
    Fetch multiple sheets with error handling for each.

    Iterates through a dictionary of sheet IDs, fetching each sheet
    and collecting results. Optionally stops on authentication errors.

    Args:
        client (SmartsheetRetryClient): The Smartsheet client to use.
        sheet_ids (Dict[str, int]): Mapping of group names to sheet IDs.
        include (Optional[List[str]]): Elements to include in response.
        log_errors (bool): Whether to log errors. Defaults to True.
        stop_on_auth_error (bool): If True, stops processing on auth errors.
            Defaults to True.

    Returns:
        Dict[str, SheetFetchResult]: Mapping of group names to fetch results.

    Example:
        >>> sheet_ids = {"NA": 123, "NF": 456, "NH": 789}
        >>> results = fetch_multiple_sheets(client, sheet_ids)
        >>> for group, result in results.items():
        ...     if result.success:
        ...         print(f"{group}: {len(result.sheet.rows)} rows")
        ...     else:
        ...         print(f"{group}: {result.error_type}")
    """
    results = {}

    for group_name, sheet_id in sheet_ids.items():
        result = fetch_sheet(
            client=client,
            sheet_id=sheet_id,
            group_name=group_name,
            include=include,
            log_errors=log_errors
        )

        results[group_name] = result

        # Stop on authentication errors if requested
        if stop_on_auth_error and result.error_type == 'auth_error':
            logger.error(
                "Stopping due to authentication error. "
                "Please refresh your API token."
            )
            break

    return results


def check_sheet_availability(
    client: SmartsheetRetryClient,
    sheet_ids: Dict[str, int]
) -> Tuple[Dict[str, bool], Dict[str, str]]:
    """
    Check availability of multiple sheets without fetching full data.

    Performs a lightweight check of sheet accessibility, useful for
    pre-flight validation before starting data processing.

    Args:
        client (SmartsheetRetryClient): The Smartsheet client to use.
        sheet_ids (Dict[str, int]): Mapping of group names to sheet IDs.

    Returns:
        Tuple[Dict[str, bool], Dict[str, str]]: A tuple containing:
            - available: Mapping of group names to availability status
            - messages: Mapping of group names to status messages

    Example:
        >>> available, messages = check_sheet_availability(client, SHEET_IDS)
        >>> unavailable = [g for g, ok in available.items() if not ok]
        >>> if unavailable:
        ...     print(f"Unavailable sheets: {unavailable}")
    """
    available = {}
    messages = {}

    for group_name, sheet_id in sheet_ids.items():
        result = fetch_sheet_with_columns(
            client=client,
            sheet_id=sheet_id,
            group_name=group_name,
            log_errors=False  # Suppress logging for availability check
        )

        available[group_name] = result.success
        messages[group_name] = result.message

    return available, messages


def get_cell_value_by_column(
    row,
    column_id: int,
    default: Any = None,
    use_display_value: bool = False
) -> Any:
    """
    Get a cell value from a row by column ID.

    Utility function to safely extract cell values with error handling.

    Args:
        row: A Smartsheet row object.
        column_id (int): The ID of the column to retrieve.
        default: Default value if cell not found or empty. Defaults to None.
        use_display_value (bool): If True, return display_value instead of value.
            Defaults to False.

    Returns:
        Any: The cell value, display value, or default.

    Example:
        >>> date_val = get_cell_value_by_column(row, col_map["Kontrolle"])
        >>> user_name = get_cell_value_by_column(row, col_map["K von"], use_display_value=True)
    """
    try:
        for cell in row.cells:
            if cell.column_id == column_id:
                if use_display_value:
                    return cell.display_value if cell.display_value else default
                return cell.value if cell.value is not None else default
        return default
    except (AttributeError, TypeError) as e:
        logger.debug(f"Error getting cell value for column {column_id}: {e}")
        return default


def get_cell_values_by_columns(
    row,
    column_ids: Dict[str, int],
    defaults: Optional[Dict[str, Any]] = None,
    use_display_values: Optional[Dict[str, bool]] = None
) -> Dict[str, Any]:
    """
    Get multiple cell values from a row by column IDs.

    Batch version of get_cell_value_by_column for efficiency when
    reading multiple values from the same row.

    Args:
        row: A Smartsheet row object.
        column_ids (Dict[str, int]): Mapping of field names to column IDs.
        defaults (Optional[Dict[str, Any]]): Default values per field.
        use_display_values (Optional[Dict[str, bool]]): Whether to use
            display_value per field.

    Returns:
        Dict[str, Any]: Mapping of field names to cell values.

    Example:
        >>> column_ids = {"date": col_map["Kontrolle"], "user": col_map["K von"]}
        >>> values = get_cell_values_by_columns(row, column_ids,
        ...     use_display_values={"user": True})
        >>> print(values["date"], values["user"])
    """
    defaults = defaults or {}
    use_display_values = use_display_values or {}

    # Build reverse lookup of column_id -> field_name
    id_to_field = {col_id: name for name, col_id in column_ids.items()}

    # Initialize results with defaults
    results = {name: defaults.get(name) for name in column_ids.keys()}

    try:
        for cell in row.cells:
            if cell.column_id in id_to_field:
                field_name = id_to_field[cell.column_id]
                use_display = use_display_values.get(field_name, False)

                if use_display:
                    value = cell.display_value if cell.display_value else defaults.get(field_name)
                else:
                    value = cell.value if cell.value is not None else defaults.get(field_name)

                results[field_name] = value

    except (AttributeError, TypeError) as e:
        logger.debug(f"Error getting cell values: {e}")

    return results


def get_cell_value_with_type_validation(
    row,
    column_id: int,
    field_name: str,
    default: Any = None,
    use_display_value: bool = False,
    row_id: Optional[str] = None,
    group: Optional[str] = None,
    log_issues: bool = True
) -> Tuple[Any, bool]:
    """
    Get a cell value from a row by column ID with type validation and coercion.

    This function extends get_cell_value_by_column with type validation.
    It validates the cell value against expected types and attempts safe
    coercion when needed.

    Args:
        row: A Smartsheet row object.
        column_id (int): The ID of the column to retrieve.
        field_name (str): The field name for type lookup.
        default: Default value if cell not found or empty. Defaults to None.
        use_display_value (bool): If True, return display_value instead of value.
        row_id: Optional row identifier for context in logging.
        group: Optional group code for context in logging.
        log_issues: Whether to log validation issues.

    Returns:
        Tuple[Any, bool]: (coerced_value, is_valid)
        - coerced_value: The value after type validation/coercion
        - is_valid: True if the value is valid (or was successfully coerced)

    Example:
        >>> value, is_valid = get_cell_value_with_type_validation(
        ...     row, col_map["Duration"], "duration", default=0.0
        ... )
        >>> if is_valid:
        ...     process_duration(value)
    """
    # First get the raw cell value
    raw_value = get_cell_value_by_column(row, column_id, default, use_display_value)

    # If the value is the default (meaning cell not found or empty), return it as-is
    if raw_value == default or raw_value is None:
        return default, True

    # Import type validation here to avoid circular imports
    from type_validation import validate_and_coerce_type, FIELD_TYPE_RULES

    # Only validate if we have type rules for this field
    if field_name not in FIELD_TYPE_RULES:
        return raw_value, True

    # Validate and coerce the type
    result = validate_and_coerce_type(
        value=raw_value,
        field_name=field_name,
        row_id=row_id,
        group=group,
        log_issues=log_issues,
    )

    # If validation failed and we should skip, return default
    if result.should_skip_row:
        return default, False

    return result.coerced_value if result.is_valid else default, result.is_valid


def get_cell_values_with_type_validation(
    row,
    column_ids: Dict[str, int],
    defaults: Optional[Dict[str, Any]] = None,
    use_display_values: Optional[Dict[str, bool]] = None,
    row_id: Optional[str] = None,
    group: Optional[str] = None,
    log_issues: bool = True
) -> Tuple[Dict[str, Any], bool, List[str]]:
    """
    Get multiple cell values from a row with type validation and coercion.

    Batch version of get_cell_value_with_type_validation for efficiency
    when reading multiple values from the same row.

    Args:
        row: A Smartsheet row object.
        column_ids (Dict[str, int]): Mapping of field names to column IDs.
        defaults (Optional[Dict[str, Any]]): Default values per field.
        use_display_values (Optional[Dict[str, bool]]): Whether to use
            display_value per field.
        row_id: Optional row identifier for context in logging.
        group: Optional group code for context in logging.
        log_issues: Whether to log validation issues.

    Returns:
        Tuple[Dict[str, Any], bool, List[str]]:
        - values: Dictionary mapping field names to coerced values
        - all_valid: True if all values are valid
        - invalid_fields: List of field names that failed validation

    Example:
        >>> column_ids = {"duration": col_map["Duration"], "count": col_map["Count"]}
        >>> values, all_valid, invalid = get_cell_values_with_type_validation(
        ...     row, column_ids, defaults={"duration": 0.0, "count": 0}
        ... )
        >>> if all_valid:
        ...     process_values(values)
        ... else:
        ...     logger.warning(f"Invalid fields: {invalid}")
    """
    # First get all raw cell values
    raw_values = get_cell_values_by_columns(row, column_ids, defaults, use_display_values)

    # Import type validation here to avoid circular imports
    from type_validation import validate_and_coerce_type, FIELD_TYPE_RULES

    defaults = defaults or {}
    results: Dict[str, Any] = {}
    invalid_fields: List[str] = []
    all_valid = True

    for field_name, value in raw_values.items():
        # If the value is the default (meaning cell not found or empty), keep it as-is
        if value == defaults.get(field_name) or value is None:
            results[field_name] = defaults.get(field_name)
            continue

        # Only validate if we have type rules for this field
        if field_name not in FIELD_TYPE_RULES:
            results[field_name] = value
            continue

        # Validate and coerce the type
        result = validate_and_coerce_type(
            value=value,
            field_name=field_name,
            row_id=row_id,
            group=group,
            log_issues=log_issues,
        )

        if result.should_skip_row or not result.is_valid:
            invalid_fields.append(field_name)
            all_valid = False
            results[field_name] = defaults.get(field_name)
        else:
            results[field_name] = result.coerced_value

    return results, all_valid, invalid_fields


# Convenience aliases for backwards compatibility and cleaner imports
get_token = get_smartsheet_token
create_client = create_smartsheet_client
