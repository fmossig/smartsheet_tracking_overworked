"""
Retry utility module for Smartsheet API calls with exponential backoff.

This module provides a decorator and utility functions for handling timeout errors
and transient failures when making Smartsheet API calls. It implements exponential
backoff with jitter to avoid thundering herd problems.

Includes intelligent HTTP 429 (Rate Limit) handling with Retry-After header support
and request queue management.
"""

import time
import random
import logging
import re
import threading
from functools import wraps
from collections import deque
from datetime import datetime, timedelta

from constants import (
    HTTPStatus,
    SmartsheetErrorCode,
    NOT_FOUND_ERROR_CODES,
    PERMISSION_ERROR_CODES,
    TOKEN_ERROR_CODES,
    DEFAULT_MAX_RETRIES,
    DEFAULT_BASE_DELAY,
    DEFAULT_MAX_DELAY,
    DEFAULT_EXPONENTIAL_BASE,
    DEFAULT_QUEUE_SIZE,
    DEFAULT_QUEUE_COOLDOWN,
)

# Get logger for this module
logger = logging.getLogger(__name__)

# Exceptions that should trigger a retry
RETRYABLE_EXCEPTIONS = (
    TimeoutError,
    ConnectionError,
    ConnectionResetError,
    BrokenPipeError,
)


class SmartsheetTimeoutError(Exception):
    """Custom exception for Smartsheet timeout after max retries."""
    def __init__(self, message, attempts=0, last_error=None):
        super().__init__(message)
        self.attempts = attempts
        self.last_error = last_error


class RateLimitError(Exception):
    """
    Custom exception for HTTP 429 Rate Limit errors.

    Captures rate limit information including the retry-after delay
    and provides structured access to rate limit metadata.
    """
    def __init__(self, message, retry_after=None, error_code=None,
                 status_code=429, original_exception=None):
        super().__init__(message)
        self.retry_after = retry_after  # Delay in seconds from Retry-After header
        self.error_code = error_code    # Smartsheet error code (e.g., 4003)
        self.status_code = status_code  # HTTP status code (429)
        self.original_exception = original_exception
        self.timestamp = datetime.now()

    def __str__(self):
        base_msg = super().__str__()
        if self.retry_after:
            return f"{base_msg} (retry after {self.retry_after}s)"
        return base_msg


class SheetNotFoundError(Exception):
    """
    Custom exception for sheet not found errors (invalid ID or deleted sheet).

    This error should NOT be retried as the sheet genuinely doesn't exist.
    Error codes:
    - 1006: Not Found (general not found)
    - 1020: Sheet not found (specific)
    """
    def __init__(self, message, sheet_id=None, error_code=None,
                 status_code=404, original_exception=None):
        super().__init__(message)
        self.sheet_id = sheet_id
        self.error_code = error_code
        self.status_code = status_code
        self.original_exception = original_exception
        self.timestamp = datetime.now()

    def __str__(self):
        base_msg = super().__str__()
        if self.sheet_id:
            return f"{base_msg} (sheet_id: {self.sheet_id})"
        return base_msg


class TokenAuthenticationError(Exception):
    """
    Custom exception for authentication/authorization errors related to API tokens.

    This error should NOT be retried as the token is invalid or expired.
    Error codes:
    - 1002: Access token is invalid
    - 1003: Access token has expired
    - 1004: Invalid API key
    - 4001: Invalid access token
    - HTTP 401: Unauthorized
    - HTTP 403: Forbidden (may indicate insufficient permissions)
    """
    def __init__(self, message, error_code=None, status_code=None,
                 original_exception=None, is_expired=False):
        super().__init__(message)
        self.error_code = error_code
        self.status_code = status_code
        self.original_exception = original_exception
        self.is_expired = is_expired  # True if token is expired vs invalid
        self.timestamp = datetime.now()

    def __str__(self):
        base_msg = super().__str__()
        if self.is_expired:
            return f"{base_msg} (token expired)"
        if self.status_code:
            return f"{base_msg} (HTTP {self.status_code})"
        return base_msg

    def get_actionable_message(self):
        """
        Get a user-friendly message with actionable steps to resolve the error.

        Returns:
            str: A detailed message with instructions for fixing the token issue.
        """
        if self.is_expired:
            return (
                "Your Smartsheet API token has expired. "
                "To resolve this:\n"
                "  1. Log in to Smartsheet and navigate to Account > Personal Settings > API Access\n"
                "  2. Generate a new API Access Token\n"
                "  3. Update the SMARTSHEET_TOKEN value in your .env file\n"
                "  4. Restart the application"
            )
        else:
            return (
                "Your Smartsheet API token is invalid or unauthorized. "
                "To resolve this:\n"
                "  1. Verify that SMARTSHEET_TOKEN in your .env file is correct\n"
                "  2. Ensure the token has not been revoked in Smartsheet settings\n"
                "  3. Check that the token has the necessary permissions (read/write access)\n"
                "  4. If issues persist, generate a new token at Account > Personal Settings > API Access"
            )


class PermissionDeniedError(Exception):
    """
    Custom exception for permission denied errors (HTTP 403 Forbidden) for specific sheets.

    This error indicates the user's token is valid but lacks permission to access
    a specific sheet or perform a specific operation. Unlike TokenAuthenticationError,
    this is a sheet-level error that should NOT stop processing of other sheets.

    Error codes:
    - 1003: Access denied (specific resource)
    - 1004: Access to the resource is forbidden
    - 1012: Sheet is private (user doesn't have access)
    - 1014: User cannot access sheet
    - 1016: Access to the sheet is denied
    - 1019: User cannot perform operation on this sheet
    - HTTP 403: Forbidden (permission denied for this resource)

    Key differences from TokenAuthenticationError:
    - TokenAuthenticationError: Token-level issue affecting all sheets (stop processing)
    - PermissionDeniedError: Sheet-level issue, continue with other sheets
    """
    def __init__(self, message, sheet_id=None, operation=None, error_code=None,
                 status_code=403, original_exception=None):
        super().__init__(message)
        self.sheet_id = sheet_id
        self.operation = operation  # The operation that was denied (e.g., 'read', 'write')
        self.error_code = error_code
        self.status_code = status_code
        self.original_exception = original_exception
        self.timestamp = datetime.now()

    def __str__(self):
        base_msg = super().__str__()
        parts = [base_msg]
        if self.sheet_id:
            parts.append(f"sheet_id: {self.sheet_id}")
        if self.operation:
            parts.append(f"operation: {self.operation}")
        if self.error_code:
            parts.append(f"error_code: {self.error_code}")
        return f"{parts[0]} ({', '.join(parts[1:])})" if len(parts) > 1 else base_msg

    def get_actionable_message(self):
        """
        Get a user-friendly message with actionable steps to resolve the permission error.

        Returns:
            str: A detailed message with instructions for gaining access to the sheet.
        """
        sheet_info = f" (Sheet ID: {self.sheet_id})" if self.sheet_id else ""
        return (
            f"Permission denied for sheet{sheet_info}. "
            "To resolve this:\n"
            "  1. Contact the sheet owner to request access\n"
            "  2. Ask for at least 'Viewer' permission to read the sheet\n"
            "  3. If you need to write changes, request 'Editor' permission\n"
            "  4. Alternatively, ask the sheet owner to share the sheet with your account\n"
            "  5. Note: Processing will continue with other accessible sheets"
        )


def is_rate_limit_error(exception):
    """
    Determine if an exception represents an HTTP 429 Rate Limit error.

    Args:
        exception: The exception to check

    Returns:
        bool: True if the exception is a rate limit error, False otherwise
    """
    # Check if it's our custom RateLimitError
    if isinstance(exception, RateLimitError):
        return True

    # Check exception message for rate limit indicators
    error_message = str(exception).lower()
    rate_limit_keywords = ['429', 'rate limit', 'too many requests', 'error code: 4003',
                          'errorcode: 4003', 'rate_limit_exceeded', 'ratelimit']

    if any(keyword in error_message for keyword in rate_limit_keywords):
        return True

    # Check for Smartsheet SDK ApiError with error code 4003
    try:
        # Handle smartsheet.exceptions.ApiError structure
        if hasattr(exception, 'error') and hasattr(exception.error, 'result'):
            result = exception.error.result
            if hasattr(result, 'error_code') and result.error_code == SmartsheetErrorCode.RATE_LIMIT_EXCEEDED:
                return True
            if hasattr(result, 'status_code') and result.status_code == HTTPStatus.TOO_MANY_REQUESTS:
                return True
    except (AttributeError, TypeError):
        pass

    return False


def is_sheet_not_found_error(exception, sheet_id=None):
    """
    Determine if an exception represents a sheet not found error.

    This covers cases where:
    - Sheet ID is invalid
    - Sheet has been deleted
    - User doesn't have access to the sheet

    Args:
        exception: The exception to check
        sheet_id: Optional sheet ID for context in error messages

    Returns:
        bool: True if the exception is a sheet not found error, False otherwise
    """
    # Check if it's our custom SheetNotFoundError
    if isinstance(exception, SheetNotFoundError):
        return True

    # Check exception message for not-found indicators
    error_message = str(exception).lower()
    not_found_keywords = [
        'not found',
        'sheet not found',
        'does not exist',
        'invalid sheet',
        'error code: 1006',
        'error code: 1020',
        'errorcode: 1006',
        'errorcode: 1020',
        'resource not found',
        '404',
    ]

    if any(keyword in error_message for keyword in not_found_keywords):
        return True

    # Check for Smartsheet SDK ApiError with specific error codes
    try:
        # Handle smartsheet.exceptions.ApiError structure
        if hasattr(exception, 'error') and hasattr(exception.error, 'result'):
            result = exception.error.result
            # Error code 1006: Not Found
            # Error code 1020: Sheet not found (specific)
            if hasattr(result, 'error_code') and result.error_code in NOT_FOUND_ERROR_CODES:
                return True
            if hasattr(result, 'status_code') and result.status_code == HTTPStatus.NOT_FOUND:
                return True
    except (AttributeError, TypeError):
        pass

    return False


def is_token_error(exception):
    """
    Determine if an exception represents an authentication/token error.

    This covers cases where:
    - Token is invalid or malformed
    - Token has expired
    - Token has been revoked
    - Token lacks required permissions (403)

    Smartsheet error codes:
    - 1002: Access token is invalid
    - 1003: Access token has expired
    - 1004: Invalid API key
    - 4001: Invalid access token

    HTTP status codes:
    - 401: Unauthorized
    - 403: Forbidden (insufficient permissions)

    Args:
        exception: The exception to check

    Returns:
        bool: True if the exception is a token/authentication error, False otherwise
    """
    # Check if it's our custom TokenAuthenticationError
    if isinstance(exception, TokenAuthenticationError):
        return True

    # Check exception message for token/auth indicators
    error_message = str(exception).lower()
    token_keywords = [
        'unauthorized',
        'authentication failed',
        'authentication error',
        'invalid token',
        'token expired',
        'access token is invalid',
        'access token has expired',
        'invalid api key',
        'invalid access token',
        'error code: 1002',
        'error code: 1003',
        'error code: 1004',
        'error code: 4001',
        'errorcode: 1002',
        'errorcode: 1003',
        'errorcode: 1004',
        'errorcode: 4001',
        '401',
        'forbidden',
        '403',
    ]

    if any(keyword in error_message for keyword in token_keywords):
        return True

    # Check for Smartsheet SDK ApiError with specific error codes
    try:
        # Handle smartsheet.exceptions.ApiError structure
        if hasattr(exception, 'error') and hasattr(exception.error, 'result'):
            result = exception.error.result
            # Error code 1002: Access token is invalid
            # Error code 1003: Access token has expired
            # Error code 1004: Invalid API key
            # Error code 4001: Invalid access token
            if hasattr(result, 'error_code') and result.error_code in TOKEN_ERROR_CODES:
                return True
            if hasattr(result, 'status_code') and result.status_code in (HTTPStatus.UNAUTHORIZED, HTTPStatus.FORBIDDEN):
                return True
    except (AttributeError, TypeError):
        pass

    return False


def is_expired_token_error(exception):
    """
    Determine if an exception specifically indicates an expired token.

    This is a more specific check than is_token_error() to differentiate
    between expired tokens and other authentication failures.

    Args:
        exception: The exception to check

    Returns:
        bool: True if the exception specifically indicates an expired token
    """
    # Check if it's our custom TokenAuthenticationError with is_expired flag
    if isinstance(exception, TokenAuthenticationError) and exception.is_expired:
        return True

    # Check exception message for expiration indicators
    error_message = str(exception).lower()
    expiration_keywords = [
        'expired',
        'token has expired',
        'access token has expired',
        'error code: 1003',
        'errorcode: 1003',
    ]

    if any(keyword in error_message for keyword in expiration_keywords):
        return True

    # Check for Smartsheet SDK ApiError with error code 1003 (expired)
    try:
        if hasattr(exception, 'error') and hasattr(exception.error, 'result'):
            result = exception.error.result
            if hasattr(result, 'error_code') and result.error_code == SmartsheetErrorCode.TOKEN_EXPIRED:
                return True
    except (AttributeError, TypeError):
        pass

    return False


def create_token_error(exception):
    """
    Create a TokenAuthenticationError from a generic exception.

    Extracts relevant information from the original exception to create
    a structured TokenAuthenticationError with appropriate metadata.

    Args:
        exception: The original exception to convert

    Returns:
        TokenAuthenticationError: A structured token error with actionable message
    """
    is_expired = is_expired_token_error(exception)
    error_code = None
    status_code = None

    # Try to extract error/status codes from Smartsheet SDK exception
    try:
        if hasattr(exception, 'error') and hasattr(exception.error, 'result'):
            result = exception.error.result
            if hasattr(result, 'error_code'):
                error_code = result.error_code
            if hasattr(result, 'status_code'):
                status_code = result.status_code
    except (AttributeError, TypeError):
        pass

    # Create appropriate message based on error type
    if is_expired:
        message = "Smartsheet API token has expired"
    else:
        message = "Smartsheet API token is invalid or unauthorized"

    return TokenAuthenticationError(
        message=message,
        error_code=error_code,
        status_code=status_code,
        original_exception=exception,
        is_expired=is_expired
    )


def is_permission_denied_error(exception, sheet_id=None):
    """
    Determine if an exception represents a permission denied error for a specific sheet.

    This is distinct from token authentication errors - permission denied means:
    - The token is valid
    - The user is authenticated
    - BUT the user lacks permission to access this specific sheet

    This covers cases where:
    - User doesn't have share access to the sheet
    - Sheet is private and not shared with the user
    - User lacks permission for the specific operation (e.g., write on read-only)

    Smartsheet error codes indicating permission denied:
    - 1012: Sheet is private (user doesn't have access)
    - 1014: User cannot access sheet
    - 1016: Access to the sheet is denied
    - 1019: User cannot perform operation on this sheet

    HTTP status codes:
    - 403: Forbidden (when combined with sheet-specific error codes)

    Args:
        exception: The exception to check
        sheet_id: Optional sheet ID for context in error messages

    Returns:
        bool: True if the exception is a permission denied error, False otherwise
    """
    # Check if it's our custom PermissionDeniedError
    if isinstance(exception, PermissionDeniedError):
        return True

    # Don't classify as permission denied if it's clearly a token error
    # (Token errors are global, permission denied is sheet-specific)
    if isinstance(exception, TokenAuthenticationError):
        return False

    # Check exception message for permission denied indicators
    error_message = str(exception).lower()
    permission_denied_keywords = [
        'permission denied',
        'access denied',
        'access to the resource is forbidden',
        'sheet is private',
        'user cannot access',
        'you do not have access',
        'you don\'t have access',
        'not shared with you',
        'access to the sheet is denied',
        'cannot perform operation',
        'insufficient permissions',
        'requires sharing access',
        'error code: 1012',
        'error code: 1014',
        'error code: 1016',
        'error code: 1019',
        'errorcode: 1012',
        'errorcode: 1014',
        'errorcode: 1016',
        'errorcode: 1019',
    ]

    if any(keyword in error_message for keyword in permission_denied_keywords):
        return True

    # Check for Smartsheet SDK ApiError with specific permission error codes
    try:
        # Handle smartsheet.exceptions.ApiError structure
        if hasattr(exception, 'error') and hasattr(exception.error, 'result'):
            result = exception.error.result
            # Permission-specific error codes
            # 1012: Sheet is private
            # 1014: User cannot access sheet
            # 1016: Access to the sheet is denied
            # 1019: User cannot perform operation on this sheet
            if hasattr(result, 'error_code') and result.error_code in PERMISSION_ERROR_CODES:
                return True

            # Check for 403 with sheet-specific context
            # Note: 403 alone could be token error, but with certain error codes it's permission
            if hasattr(result, 'status_code') and result.status_code == HTTPStatus.FORBIDDEN:
                # Check if error message mentions sheet access specifically
                if hasattr(result, 'message'):
                    msg = str(result.message).lower()
                    sheet_keywords = ['sheet', 'access', 'permission', 'share', 'private']
                    if any(kw in msg for kw in sheet_keywords):
                        return True
    except (AttributeError, TypeError):
        pass

    return False


def create_permission_error(exception, sheet_id=None, operation=None):
    """
    Create a PermissionDeniedError from a generic exception.

    Extracts relevant information from the original exception to create
    a structured PermissionDeniedError with appropriate metadata.

    Args:
        exception: The original exception to convert
        sheet_id: Optional sheet ID for context
        operation: Optional operation name (e.g., 'read', 'write', 'get_sheet')

    Returns:
        PermissionDeniedError: A structured permission error with actionable message
    """
    error_code = None
    status_code = HTTPStatus.FORBIDDEN  # Default for permission denied

    # Try to extract error/status codes from Smartsheet SDK exception
    try:
        if hasattr(exception, 'error') and hasattr(exception.error, 'result'):
            result = exception.error.result
            if hasattr(result, 'error_code'):
                error_code = result.error_code
            if hasattr(result, 'status_code'):
                status_code = result.status_code
    except (AttributeError, TypeError):
        pass

    # Create descriptive message
    if sheet_id:
        message = f"Permission denied: Cannot access sheet {sheet_id}"
    else:
        message = "Permission denied: Cannot access the requested sheet"

    if operation:
        message += f" (operation: {operation})"

    return PermissionDeniedError(
        message=message,
        sheet_id=sheet_id,
        operation=operation,
        error_code=error_code,
        status_code=status_code,
        original_exception=exception
    )


def extract_retry_after_delay(exception, default_delay=60.0):
    """
    Extract the Retry-After delay from an exception or response.

    The Retry-After header can be specified as:
    1. Number of seconds: "60"
    2. HTTP date: "Wed, 21 Oct 2015 07:28:00 GMT"

    Args:
        exception: The exception that may contain retry-after information
        default_delay: Default delay in seconds if Retry-After cannot be extracted

    Returns:
        float: The delay in seconds before retrying
    """
    # First, try to extract from custom RateLimitError
    if isinstance(exception, RateLimitError) and exception.retry_after:
        return float(exception.retry_after)

    # Try to extract from Smartsheet SDK exception attributes
    try:
        # Check for response headers in the exception
        if hasattr(exception, 'response'):
            response = exception.response
            if hasattr(response, 'headers'):
                headers = response.headers
                retry_after = headers.get('Retry-After') or headers.get('retry-after')
                if retry_after:
                    return _parse_retry_after_value(retry_after)

        # Check for headers in error.result
        if hasattr(exception, 'error') and hasattr(exception.error, 'result'):
            result = exception.error.result
            if hasattr(result, 'headers'):
                retry_after = result.headers.get('Retry-After') or result.headers.get('retry-after')
                if retry_after:
                    return _parse_retry_after_value(retry_after)

        # Try to extract from exception message
        error_message = str(exception)
        # Look for patterns like "Retry-After: 60" or "retry after 60 seconds"
        retry_patterns = [
            r'[Rr]etry[-_]?[Aa]fter[:\s]+(\d+)',
            r'retry after (\d+)\s*(?:seconds|s)',
            r'wait (\d+)\s*(?:seconds|s)',
        ]
        for pattern in retry_patterns:
            match = re.search(pattern, error_message)
            if match:
                return float(match.group(1))

    except (AttributeError, TypeError, ValueError) as e:
        logger.debug(f"Could not extract Retry-After from exception: {e}")

    # Return default delay (Smartsheet recommends 60 seconds for rate limits)
    return default_delay


def _parse_retry_after_value(value):
    """
    Parse a Retry-After header value.

    Args:
        value: The Retry-After header value (seconds or HTTP date)

    Returns:
        float: The delay in seconds
    """
    # Try to parse as integer/float (seconds)
    try:
        return float(value)
    except (ValueError, TypeError):
        pass

    # Try to parse as HTTP date
    try:
        from email.utils import parsedate_to_datetime
        retry_time = parsedate_to_datetime(value)
        delay = (retry_time - datetime.now(retry_time.tzinfo)).total_seconds()
        return max(0, delay)  # Don't return negative delays
    except (ValueError, TypeError, ImportError):
        pass

    # Default fallback
    return 60.0


def is_retryable_error(exception):
    """
    Determine if an exception is retryable.

    Args:
        exception: The exception to check

    Returns:
        bool: True if the exception is retryable, False otherwise
    """
    # Sheet not found errors are NEVER retryable (invalid ID or deleted sheet)
    if is_sheet_not_found_error(exception):
        return False

    # Token/authentication errors are NEVER retryable (need token refresh)
    if is_token_error(exception):
        return False

    # Permission denied errors are NEVER retryable (need access grant from sheet owner)
    if is_permission_denied_error(exception):
        return False

    # Rate limit errors are always retryable
    if is_rate_limit_error(exception):
        return True

    # Check for known retryable exception types
    if isinstance(exception, RETRYABLE_EXCEPTIONS):
        return True

    # Check exception message for timeout-related keywords
    error_message = str(exception).lower()
    timeout_keywords = ['timeout', 'timed out', 'connection reset', 'connection refused',
                       'temporary failure', 'service unavailable', '503', '504', '429',
                       'rate limit', 'too many requests']

    return any(keyword in error_message for keyword in timeout_keywords)


class RateLimitQueue:
    """
    Thread-safe queue manager for handling rate-limited requests.

    When the Smartsheet API returns a 429 error, this queue helps manage
    pending requests by holding them until the rate limit window resets.
    """

    def __init__(self, max_queue_size=100, default_cooldown=60.0):
        """
        Initialize the rate limit queue.

        Args:
            max_queue_size: Maximum number of requests to queue
            default_cooldown: Default cooldown period in seconds
        """
        self._queue = deque(maxlen=max_queue_size)
        self._lock = threading.RLock()
        self._rate_limited_until = None
        self._default_cooldown = default_cooldown
        self._consecutive_rate_limits = 0
        self._stats = {
            'total_rate_limits': 0,
            'total_requests_queued': 0,
            'total_requests_processed': 0,
            'max_wait_time': 0,
        }

    @property
    def is_rate_limited(self):
        """Check if we're currently in a rate-limited state."""
        with self._lock:
            if self._rate_limited_until is None:
                return False
            if datetime.now() >= self._rate_limited_until:
                self._rate_limited_until = None
                self._consecutive_rate_limits = 0
                return False
            return True

    @property
    def time_until_reset(self):
        """Get seconds until rate limit resets (0 if not rate limited)."""
        with self._lock:
            if self._rate_limited_until is None:
                return 0
            remaining = (self._rate_limited_until - datetime.now()).total_seconds()
            return max(0, remaining)

    @property
    def queue_size(self):
        """Get the current queue size."""
        with self._lock:
            return len(self._queue)

    @property
    def statistics(self):
        """Get queue statistics."""
        with self._lock:
            return self._stats.copy()

    def set_rate_limited(self, retry_after=None):
        """
        Mark the queue as rate-limited.

        Args:
            retry_after: Seconds to wait (from Retry-After header or default)
        """
        with self._lock:
            delay = retry_after or self._default_cooldown

            # Apply progressive backoff for consecutive rate limits
            if self._consecutive_rate_limits > 0:
                multiplier = min(2 ** self._consecutive_rate_limits, 4)
                delay = delay * multiplier
                logger.warning(
                    f"Consecutive rate limit #{self._consecutive_rate_limits + 1}, "
                    f"applying {multiplier}x multiplier: {delay:.1f}s delay"
                )

            self._rate_limited_until = datetime.now() + timedelta(seconds=delay)
            self._consecutive_rate_limits += 1
            self._stats['total_rate_limits'] += 1
            self._stats['max_wait_time'] = max(self._stats['max_wait_time'], delay)

            logger.info(
                f"Rate limit activated. Waiting {delay:.1f}s until "
                f"{self._rate_limited_until.strftime('%H:%M:%S')}"
            )

    def clear_rate_limit(self):
        """Clear the rate-limited state after successful request."""
        with self._lock:
            self._rate_limited_until = None
            self._consecutive_rate_limits = 0

    def wait_if_rate_limited(self):
        """
        Block until the rate limit period expires.

        Returns:
            float: The number of seconds waited (0 if not rate limited)
        """
        wait_time = self.time_until_reset
        if wait_time > 0:
            logger.info(f"Waiting {wait_time:.1f}s for rate limit to reset...")
            time.sleep(wait_time)
        return wait_time

    def enqueue(self, func, args, kwargs, callback=None):
        """
        Add a request to the queue.

        Args:
            func: The function to call
            args: Positional arguments
            kwargs: Keyword arguments
            callback: Optional callback for when request completes

        Returns:
            bool: True if enqueued, False if queue is full
        """
        with self._lock:
            if len(self._queue) >= self._queue.maxlen:
                logger.warning("Rate limit queue is full, cannot enqueue request")
                return False

            self._queue.append({
                'func': func,
                'args': args,
                'kwargs': kwargs,
                'callback': callback,
                'enqueued_at': datetime.now(),
            })
            self._stats['total_requests_queued'] += 1
            logger.debug(f"Request enqueued. Queue size: {len(self._queue)}")
            return True

    def process_queue(self, delay_between_requests=0.1):
        """
        Process all queued requests.

        Args:
            delay_between_requests: Delay in seconds between processing requests

        Returns:
            list: Results from processed requests
        """
        results = []
        with self._lock:
            while self._queue:
                # Wait if still rate limited
                self.wait_if_rate_limited()

                request = self._queue.popleft()
                try:
                    result = request['func'](*request['args'], **request['kwargs'])
                    results.append(('success', result))
                    self._stats['total_requests_processed'] += 1

                    if request['callback']:
                        request['callback'](result, None)

                except Exception as e:
                    results.append(('error', e))
                    if request['callback']:
                        request['callback'](None, e)

                    # If we hit another rate limit, re-enqueue and wait
                    if is_rate_limit_error(e):
                        retry_after = extract_retry_after_delay(e)
                        self.set_rate_limited(retry_after)
                        # Re-add to front of queue
                        self._queue.appendleft(request)

                if delay_between_requests > 0:
                    time.sleep(delay_between_requests)

        return results


# Global rate limit queue instance for shared rate limit state
_global_rate_limit_queue = RateLimitQueue()


def get_rate_limit_queue():
    """Get the global rate limit queue instance."""
    return _global_rate_limit_queue


def calculate_backoff_delay(attempt, base_delay=DEFAULT_BASE_DELAY,
                           max_delay=DEFAULT_MAX_DELAY,
                           exponential_base=DEFAULT_EXPONENTIAL_BASE):
    """
    Calculate the delay for the next retry attempt using exponential backoff with jitter.

    Args:
        attempt: The current attempt number (0-based)
        base_delay: The base delay in seconds
        max_delay: The maximum delay in seconds
        exponential_base: The base for exponential calculation

    Returns:
        float: The delay in seconds before the next retry
    """
    # Calculate exponential delay
    delay = base_delay * (exponential_base ** attempt)

    # Add jitter (random factor between 0.5 and 1.5)
    jitter = random.uniform(0.5, 1.5)
    delay = delay * jitter

    # Cap at max delay
    return min(delay, max_delay)


def retry_with_backoff(max_retries=DEFAULT_MAX_RETRIES,
                       base_delay=DEFAULT_BASE_DELAY,
                       max_delay=DEFAULT_MAX_DELAY,
                       continue_on_failure=True,
                       rate_limit_queue=None):
    """
    Decorator that adds retry logic with exponential backoff to a function.

    Includes intelligent HTTP 429 rate limit handling with Retry-After header support.

    Args:
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds between retries
        max_delay: Maximum delay in seconds between retries
        continue_on_failure: If True, return None after max retries instead of raising
        rate_limit_queue: Optional RateLimitQueue for managing rate limit state

    Returns:
        Decorated function with retry logic

    Usage:
        @retry_with_backoff(max_retries=3, base_delay=1.0)
        def fetch_sheet(client, sheet_id):
            return client.Sheets.get_sheet(sheet_id)
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            last_exception = None
            queue = rate_limit_queue or _global_rate_limit_queue

            # Check if we're currently rate limited and wait if necessary
            queue.wait_if_rate_limited()

            for attempt in range(max_retries + 1):
                try:
                    result = func(*args, **kwargs)
                    # Success! Clear rate limit state
                    queue.clear_rate_limit()
                    return result
                except Exception as e:
                    last_exception = e

                    # Check if this is a sheet not found error (invalid ID or deleted sheet)
                    if is_sheet_not_found_error(e):
                        # Extract sheet_id from args if available for clear error message
                        sheet_id_for_log = args[0] if args else 'unknown'
                        logger.error(
                            f"Sheet not found in {func.__name__}: Sheet ID '{sheet_id_for_log}' is invalid or has been deleted. "
                            f"Error: {e}. Skipping this sheet and continuing with other available sheets."
                        )
                        if continue_on_failure:
                            return None
                        else:
                            raise SheetNotFoundError(
                                f"Sheet not found: {e}",
                                sheet_id=sheet_id_for_log,
                                original_exception=e
                            )

                    # Check if this is a permission denied error (user lacks access to this sheet)
                    if is_permission_denied_error(e):
                        sheet_id_for_log = args[0] if args else 'unknown'
                        permission_error = create_permission_error(e, sheet_id=sheet_id_for_log, operation=func.__name__)
                        logger.warning(
                            f"Permission denied in {func.__name__}: {permission_error}\n"
                            f"{permission_error.get_actionable_message()}"
                        )
                        # Permission errors are non-fatal for sheet-level operations
                        # Continue with other sheets
                        if continue_on_failure:
                            return None
                        else:
                            raise permission_error

                    # Check if this is a token/authentication error (expired or invalid token)
                    if is_token_error(e):
                        token_error = create_token_error(e)
                        logger.error(
                            f"Authentication error in {func.__name__}: {token_error}\n"
                            f"{token_error.get_actionable_message()}"
                        )
                        # Token errors should not be retried - raise immediately
                        if continue_on_failure:
                            return None
                        else:
                            raise token_error

                    if not is_retryable_error(e):
                        # Non-retryable error, raise immediately
                        logger.error(f"Non-retryable error in {func.__name__}: {e}")
                        raise

                    # Check if this is specifically a rate limit error
                    if is_rate_limit_error(e):
                        retry_after = extract_retry_after_delay(e)
                        queue.set_rate_limited(retry_after)

                        logger.warning(
                            f"Rate limit (HTTP 429) in {func.__name__} "
                            f"(attempt {attempt + 1}/{max_retries + 1}): {e}. "
                            f"Waiting {retry_after:.1f}s (from Retry-After header)..."
                        )

                        # Wait for the rate limit to clear
                        queue.wait_if_rate_limited()
                    else:
                        # Regular retryable error - use exponential backoff
                        if attempt < max_retries:
                            delay = calculate_backoff_delay(attempt, base_delay, max_delay)
                            logger.warning(
                                f"Timeout/connection error in {func.__name__} "
                                f"(attempt {attempt + 1}/{max_retries + 1}): {e}. "
                                f"Retrying in {delay:.2f} seconds..."
                            )
                            time.sleep(delay)
                        else:
                            # Max retries reached
                            logger.error(
                                f"Max retries ({max_retries}) exceeded for {func.__name__}. "
                                f"Last error: {e}"
                            )

                            if continue_on_failure:
                                logger.warning(
                                    f"Continuing with available data after {func.__name__} failed"
                                )
                                return None
                            else:
                                raise SmartsheetTimeoutError(
                                    f"Failed after {max_retries + 1} attempts",
                                    attempts=max_retries + 1,
                                    last_error=e
                                )

            return None

        return wrapper
    return decorator


def execute_with_retry(func, *args, max_retries=DEFAULT_MAX_RETRIES,
                       base_delay=DEFAULT_BASE_DELAY,
                       max_delay=DEFAULT_MAX_DELAY,
                       continue_on_failure=True,
                       operation_name=None,
                       rate_limit_queue=None,
                       **kwargs):
    """
    Execute a function with retry logic and exponential backoff.

    This is a non-decorator version for cases where you can't use decorators.
    Includes intelligent HTTP 429 rate limit handling with Retry-After header support.

    Args:
        func: The function to execute
        *args: Positional arguments to pass to the function
        max_retries: Maximum number of retry attempts
        base_delay: Base delay in seconds between retries
        max_delay: Maximum delay in seconds between retries
        continue_on_failure: If True, return None after max retries instead of raising
        operation_name: Optional name for logging (defaults to function name)
        rate_limit_queue: Optional RateLimitQueue for managing rate limit state
        **kwargs: Keyword arguments to pass to the function

    Returns:
        The result of the function, or None if all retries failed and continue_on_failure is True

    Usage:
        sheet = execute_with_retry(
            client.Sheets.get_sheet,
            sheet_id,
            max_retries=3,
            operation_name="get_sheet"
        )
    """
    op_name = operation_name or getattr(func, '__name__', 'unknown_operation')
    last_exception = None
    queue = rate_limit_queue or _global_rate_limit_queue

    # Check if we're currently rate limited and wait if necessary
    queue.wait_if_rate_limited()

    for attempt in range(max_retries + 1):
        try:
            result = func(*args, **kwargs)
            # Success! Clear rate limit state
            queue.clear_rate_limit()
            return result
        except Exception as e:
            last_exception = e

            # Check if this is a sheet not found error (invalid ID or deleted sheet)
            if is_sheet_not_found_error(e):
                # Extract sheet_id from args if available for clear error message
                sheet_id_for_log = args[0] if args else 'unknown'
                logger.error(
                    f"Sheet not found in {op_name}: Sheet ID '{sheet_id_for_log}' is invalid or has been deleted. "
                    f"Error: {e}. Skipping this sheet and continuing with other available sheets."
                )
                if continue_on_failure:
                    return None
                else:
                    raise SheetNotFoundError(
                        f"Sheet not found: {e}",
                        sheet_id=sheet_id_for_log,
                        original_exception=e
                    )

            # Check if this is a permission denied error (user lacks access to this sheet)
            if is_permission_denied_error(e):
                sheet_id_for_log = args[0] if args else 'unknown'
                permission_error = create_permission_error(e, sheet_id=sheet_id_for_log, operation=op_name)
                logger.warning(
                    f"Permission denied in {op_name}: {permission_error}\n"
                    f"{permission_error.get_actionable_message()}"
                )
                # Permission errors are non-fatal for sheet-level operations
                # Continue with other sheets
                if continue_on_failure:
                    return None
                else:
                    raise permission_error

            # Check if this is a token/authentication error (expired or invalid token)
            if is_token_error(e):
                token_error = create_token_error(e)
                logger.error(
                    f"Authentication error in {op_name}: {token_error}\n"
                    f"{token_error.get_actionable_message()}"
                )
                # Token errors should not be retried - raise immediately
                if continue_on_failure:
                    return None
                else:
                    raise token_error

            if not is_retryable_error(e):
                # Non-retryable error, raise immediately
                logger.error(f"Non-retryable error in {op_name}: {e}")
                raise

            # Check if this is specifically a rate limit error
            if is_rate_limit_error(e):
                retry_after = extract_retry_after_delay(e)
                queue.set_rate_limited(retry_after)

                logger.warning(
                    f"Rate limit (HTTP 429) in {op_name} "
                    f"(attempt {attempt + 1}/{max_retries + 1}): {e}. "
                    f"Waiting {retry_after:.1f}s (from Retry-After header)..."
                )

                # Wait for the rate limit to clear
                queue.wait_if_rate_limited()
            else:
                # Regular retryable error - use exponential backoff
                if attempt < max_retries:
                    delay = calculate_backoff_delay(attempt, base_delay, max_delay)
                    logger.warning(
                        f"Timeout/connection error in {op_name} "
                        f"(attempt {attempt + 1}/{max_retries + 1}): {e}. "
                        f"Retrying in {delay:.2f} seconds..."
                    )
                    time.sleep(delay)
                else:
                    # Max retries reached
                    logger.error(
                        f"Max retries ({max_retries}) exceeded for {op_name}. "
                        f"Last error: {e}"
                    )

                    if continue_on_failure:
                        logger.warning(
                            f"Continuing with available data after {op_name} failed"
                        )
                        return None
                    else:
                        raise SmartsheetTimeoutError(
                            f"Failed after {max_retries + 1} attempts",
                            attempts=max_retries + 1,
                            last_error=e
                        )

    return None


class SmartsheetRetryClient:
    """
    A wrapper around the Smartsheet client that adds retry logic to API calls.

    Includes intelligent HTTP 429 rate limit handling with Retry-After header support
    and queue management for graceful handling of rate limit situations.

    Usage:
        import smartsheet
        base_client = smartsheet.Smartsheet(token)
        client = SmartsheetRetryClient(base_client)

        # Now all API calls have automatic retry and rate limit handling
        sheet = client.get_sheet(sheet_id)

        # Check rate limit statistics
        stats = client.rate_limit_statistics
    """

    def __init__(self, client, max_retries=DEFAULT_MAX_RETRIES,
                 base_delay=DEFAULT_BASE_DELAY, max_delay=DEFAULT_MAX_DELAY,
                 continue_on_failure=True, rate_limit_queue=None):
        """
        Initialize the retry client wrapper.

        Args:
            client: The base Smartsheet client
            max_retries: Maximum number of retry attempts
            base_delay: Base delay in seconds between retries
            max_delay: Maximum delay in seconds between retries
            continue_on_failure: If True, return None after max retries
            rate_limit_queue: Optional custom RateLimitQueue (uses global queue if not provided)
        """
        self._client = client
        self._max_retries = max_retries
        self._base_delay = base_delay
        self._max_delay = max_delay
        self._continue_on_failure = continue_on_failure
        self._rate_limit_queue = rate_limit_queue or _global_rate_limit_queue

    @property
    def client(self):
        """Access the underlying Smartsheet client."""
        return self._client

    @property
    def rate_limit_queue(self):
        """Access the rate limit queue."""
        return self._rate_limit_queue

    @property
    def is_rate_limited(self):
        """Check if we're currently rate limited."""
        return self._rate_limit_queue.is_rate_limited

    @property
    def rate_limit_statistics(self):
        """Get rate limit queue statistics."""
        return self._rate_limit_queue.statistics

    def get_sheet(self, sheet_id, **kwargs):
        """Get a sheet with retry logic and rate limit handling."""
        return execute_with_retry(
            self._client.Sheets.get_sheet,
            sheet_id,
            max_retries=self._max_retries,
            base_delay=self._base_delay,
            max_delay=self._max_delay,
            continue_on_failure=self._continue_on_failure,
            operation_name=f"get_sheet({sheet_id})",
            rate_limit_queue=self._rate_limit_queue,
            **kwargs
        )

    def get_sheet_summary(self, sheet_id, **kwargs):
        """Get sheet summary with retry logic and rate limit handling."""
        return execute_with_retry(
            self._client.Sheets.get_sheet_summary,
            sheet_id,
            max_retries=self._max_retries,
            base_delay=self._base_delay,
            max_delay=self._max_delay,
            continue_on_failure=self._continue_on_failure,
            operation_name=f"get_sheet_summary({sheet_id})",
            rate_limit_queue=self._rate_limit_queue,
            **kwargs
        )

    def update_rows(self, sheet_id, rows, **kwargs):
        """Update rows with retry logic and rate limit handling."""
        return execute_with_retry(
            self._client.Sheets.update_rows,
            sheet_id,
            rows,
            max_retries=self._max_retries,
            base_delay=self._base_delay,
            max_delay=self._max_delay,
            continue_on_failure=self._continue_on_failure,
            operation_name=f"update_rows({sheet_id})",
            rate_limit_queue=self._rate_limit_queue,
            **kwargs
        )

    def attach_file_to_row(self, sheet_id, row_id, file_tuple, **kwargs):
        """Attach file to row with retry logic and rate limit handling."""
        return execute_with_retry(
            self._client.Attachments.attach_file_to_row,
            sheet_id,
            row_id,
            file_tuple,
            max_retries=self._max_retries,
            base_delay=self._base_delay,
            max_delay=self._max_delay,
            continue_on_failure=self._continue_on_failure,
            operation_name=f"attach_file_to_row({sheet_id}, {row_id})",
            rate_limit_queue=self._rate_limit_queue,
            **kwargs
        )

    def errors_as_exceptions(self, value):
        """Pass through to the underlying client."""
        return self._client.errors_as_exceptions(value)

    def wait_for_rate_limit_reset(self):
        """
        Manually wait for the rate limit to reset.

        Returns:
            float: The number of seconds waited (0 if not rate limited)
        """
        return self._rate_limit_queue.wait_if_rate_limited()

    def get_time_until_rate_limit_reset(self):
        """
        Get the time in seconds until the rate limit resets.

        Returns:
            float: Seconds until reset (0 if not rate limited)
        """
        return self._rate_limit_queue.time_until_reset
