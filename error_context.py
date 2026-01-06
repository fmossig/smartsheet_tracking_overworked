"""
Enhanced Error Context Module for Smartsheet Tracker

Provides utilities for capturing and formatting comprehensive error context
including sheet name, row ID, field name, user, and stack trace information.

This module enhances debugging capabilities by ensuring all relevant context
is captured and logged when errors occur during processing.

Usage:
    from error_context import (
        ErrorContext,
        capture_error_context,
        format_error_for_logging,
        get_stack_trace,
    )

    # Capture context when an error occurs
    try:
        process_row(row)
    except Exception as e:
        context = capture_error_context(
            error=e,
            sheet_name="NA",
            sheet_id=12345,
            row_id=67890,
            field_name="Kontrolle",
            user="john.doe@example.com",
            phase=1,
            operation="process_row"
        )
        logger.error(format_error_for_logging(context))

    # Or use the ErrorContext dataclass directly
    context = ErrorContext(
        sheet_name="NA",
        row_id=67890,
        field_name="Kontrolle",
        user="john.doe@example.com"
    )
    context.capture_exception(e)
    logger.error(context.format_full())
"""

import logging
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Dict, List, Optional, Union
import inspect

# Get logger for this module
logger = logging.getLogger(__name__)


def get_stack_trace(
    exception: Optional[Exception] = None,
    limit: Optional[int] = None,
    skip_frames: int = 0
) -> str:
    """
    Capture the current stack trace or the stack trace from an exception.

    Args:
        exception: Optional exception to get trace from. If None, captures current stack.
        limit: Optional limit on number of frames to capture.
        skip_frames: Number of frames to skip from the top of the stack.

    Returns:
        Formatted stack trace as a string.

    Example:
        >>> try:
        ...     raise ValueError("test error")
        ... except Exception as e:
        ...     trace = get_stack_trace(e)
        ...     print(trace)
    """
    if exception is not None:
        # Get traceback from exception
        tb_lines = traceback.format_exception(
            type(exception),
            exception,
            exception.__traceback__,
            limit=limit
        )
        return ''.join(tb_lines).strip()
    else:
        # Get current stack trace
        stack = traceback.extract_stack()
        if skip_frames > 0:
            stack = stack[:-skip_frames]
        if limit is not None:
            stack = stack[-limit:]
        return ''.join(traceback.format_list(stack)).strip()


def get_caller_info(skip_frames: int = 1) -> Dict[str, Any]:
    """
    Get information about the calling function.

    Args:
        skip_frames: Number of frames to skip (1 = immediate caller).

    Returns:
        Dictionary with caller information:
        - module: Module name
        - function: Function name
        - file_path: File path
        - line_number: Line number
        - class_name: Class name (if applicable)
    """
    # Get the stack frame of the caller
    frame = inspect.currentframe()
    try:
        # Skip the specified number of frames plus this function
        for _ in range(skip_frames + 1):
            if frame is not None:
                frame = frame.f_back

        if frame is None:
            return {}

        # Extract information from the frame
        info = inspect.getframeinfo(frame)

        # Try to get class name if in a method
        class_name = None
        if 'self' in frame.f_locals:
            class_name = type(frame.f_locals['self']).__name__
        elif 'cls' in frame.f_locals:
            class_name = frame.f_locals['cls'].__name__

        return {
            "module": frame.f_globals.get('__name__', 'unknown'),
            "function": info.function,
            "file_path": info.filename,
            "line_number": info.lineno,
            "class_name": class_name,
        }
    finally:
        del frame  # Avoid reference cycles


@dataclass
class ErrorContext:
    """
    Comprehensive error context for debugging.

    Captures all relevant information about an error including:
    - Sheet context (name, ID)
    - Row context (ID, index)
    - Field context (name, column ID)
    - User context (who was affected/involved)
    - Operation context (what was being done)
    - Stack trace (where the error occurred)
    - Timing information (when)

    Attributes:
        sheet_name: Name/group identifier of the sheet (e.g., "NA", "NF")
        sheet_id: Smartsheet sheet ID
        row_id: Smartsheet row ID
        row_index: Index of the row in the sheet (if available)
        field_name: Name of the field/column being processed
        column_id: Smartsheet column ID
        user: User associated with the row/operation
        marketplace: Marketplace identifier (if applicable)
        phase: Phase number (1-5) for phase-based operations
        operation: Description of the operation being performed
        error_type: Type/class name of the exception
        error_message: Error message string
        stack_trace: Full stack trace
        timestamp: When the error occurred
        source_module: Module where error occurred
        source_function: Function where error occurred
        source_file: File path where error occurred
        source_line: Line number where error occurred
        additional_context: Any additional context information
    """
    # Sheet context
    sheet_name: Optional[str] = None
    sheet_id: Optional[int] = None

    # Row context
    row_id: Optional[Union[int, str]] = None
    row_index: Optional[int] = None

    # Field context
    field_name: Optional[str] = None
    column_id: Optional[int] = None

    # User context
    user: Optional[str] = None
    marketplace: Optional[str] = None

    # Operation context
    phase: Optional[int] = None
    operation: Optional[str] = None

    # Error details
    error_type: Optional[str] = None
    error_message: Optional[str] = None
    stack_trace: Optional[str] = None
    original_exception: Optional[Exception] = None

    # Timing
    timestamp: datetime = field(default_factory=datetime.now)

    # Source location
    source_module: Optional[str] = None
    source_function: Optional[str] = None
    source_file: Optional[str] = None
    source_line: Optional[int] = None

    # Additional context
    additional_context: Dict[str, Any] = field(default_factory=dict)

    def capture_exception(self, exception: Exception, include_trace: bool = True) -> "ErrorContext":
        """
        Capture exception information into this context.

        Args:
            exception: The exception to capture.
            include_trace: Whether to capture the full stack trace.

        Returns:
            Self for method chaining.
        """
        self.error_type = type(exception).__name__
        self.error_message = str(exception)
        self.original_exception = exception

        if include_trace:
            self.stack_trace = get_stack_trace(exception)

        return self

    def capture_caller(self, skip_frames: int = 1) -> "ErrorContext":
        """
        Capture information about the calling context.

        Args:
            skip_frames: Number of frames to skip.

        Returns:
            Self for method chaining.
        """
        caller_info = get_caller_info(skip_frames + 1)
        self.source_module = caller_info.get("module")
        self.source_function = caller_info.get("function")
        self.source_file = caller_info.get("file_path")
        self.source_line = caller_info.get("line_number")
        return self

    def add_context(self, key: str, value: Any) -> "ErrorContext":
        """
        Add additional context information.

        Args:
            key: Context key.
            value: Context value.

        Returns:
            Self for method chaining.
        """
        self.additional_context[key] = value
        return self

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization or logging.

        Returns:
            Dictionary representation of the error context.
        """
        return {
            "timestamp": self.timestamp.isoformat(),
            "sheet_name": self.sheet_name,
            "sheet_id": self.sheet_id,
            "row_id": self.row_id,
            "row_index": self.row_index,
            "field_name": self.field_name,
            "column_id": self.column_id,
            "user": self.user,
            "marketplace": self.marketplace,
            "phase": self.phase,
            "operation": self.operation,
            "error_type": self.error_type,
            "error_message": self.error_message,
            "stack_trace": self.stack_trace,
            "source_module": self.source_module,
            "source_function": self.source_function,
            "source_file": self.source_file,
            "source_line": self.source_line,
            "additional_context": self.additional_context,
        }

    def to_flat_dict(self) -> Dict[str, Any]:
        """
        Convert to flattened dictionary for error collector context.

        This format is compatible with the existing error_collector.py
        context parameter format.

        Returns:
            Flattened dictionary with non-None values only.
        """
        result = {}

        # Add all non-None fields
        if self.sheet_name:
            result["group"] = self.sheet_name
        if self.sheet_id:
            result["sheet_id"] = self.sheet_id
        if self.row_id:
            result["row_id"] = self.row_id
        if self.row_index is not None:
            result["row_index"] = self.row_index
        if self.field_name:
            result["field_name"] = self.field_name
        if self.column_id:
            result["column_id"] = self.column_id
        if self.user:
            result["user"] = self.user
        if self.marketplace:
            result["marketplace"] = self.marketplace
        if self.phase:
            result["phase"] = self.phase
        if self.operation:
            result["operation"] = self.operation
        if self.stack_trace:
            result["stack_trace"] = self.stack_trace

        # Merge additional context
        result.update(self.additional_context)

        return result

    def format_location(self) -> str:
        """
        Format the data location (sheet, row, field) for logging.

        Returns:
            Formatted location string.
        """
        parts = []
        if self.sheet_name:
            parts.append(f"Sheet: {self.sheet_name}")
        if self.sheet_id:
            parts.append(f"ID: {self.sheet_id}")
        if self.row_id:
            parts.append(f"Row: {self.row_id}")
        if self.field_name:
            parts.append(f"Field: {self.field_name}")
        if self.phase:
            parts.append(f"Phase: {self.phase}")
        if self.user:
            parts.append(f"User: {self.user}")
        if self.marketplace:
            parts.append(f"Marketplace: {self.marketplace}")

        return " | ".join(parts) if parts else "Unknown location"

    def format_source(self) -> str:
        """
        Format the source code location for logging.

        Returns:
            Formatted source location string.
        """
        parts = []
        if self.source_module:
            parts.append(self.source_module)
        if self.source_function:
            parts.append(f"{self.source_function}()")
        if self.source_file and self.source_line:
            parts.append(f"at {self.source_file}:{self.source_line}")
        elif self.source_line:
            parts.append(f"line {self.source_line}")

        return " ".join(parts) if parts else "Unknown source"

    def format_error(self) -> str:
        """
        Format the error information for logging.

        Returns:
            Formatted error string.
        """
        if self.error_type and self.error_message:
            return f"{self.error_type}: {self.error_message}"
        elif self.error_message:
            return self.error_message
        elif self.error_type:
            return f"{self.error_type} (no message)"
        return "Unknown error"

    def format_compact(self) -> str:
        """
        Format as a compact single-line log message.

        Returns:
            Compact formatted string suitable for log messages.
        """
        parts = []

        # Location context
        location = self.format_location()
        if location != "Unknown location":
            parts.append(f"[{location}]")

        # Operation
        if self.operation:
            parts.append(f"during '{self.operation}'")

        # Error
        parts.append(self.format_error())

        return " ".join(parts)

    def format_full(self) -> str:
        """
        Format as a detailed multi-line log message for debugging.

        Returns:
            Detailed formatted string with full context and stack trace.
        """
        lines = [
            "=" * 70,
            "ERROR CONTEXT DETAILS",
            "=" * 70,
            f"Timestamp: {self.timestamp.isoformat()}",
        ]

        # Data location
        lines.append("\n--- Data Location ---")
        if self.sheet_name:
            lines.append(f"  Sheet Name: {self.sheet_name}")
        if self.sheet_id:
            lines.append(f"  Sheet ID: {self.sheet_id}")
        if self.row_id:
            lines.append(f"  Row ID: {self.row_id}")
        if self.row_index is not None:
            lines.append(f"  Row Index: {self.row_index}")
        if self.field_name:
            lines.append(f"  Field Name: {self.field_name}")
        if self.column_id:
            lines.append(f"  Column ID: {self.column_id}")

        # User context
        if self.user or self.marketplace:
            lines.append("\n--- User Context ---")
            if self.user:
                lines.append(f"  User: {self.user}")
            if self.marketplace:
                lines.append(f"  Marketplace: {self.marketplace}")

        # Operation context
        if self.operation or self.phase:
            lines.append("\n--- Operation Context ---")
            if self.operation:
                lines.append(f"  Operation: {self.operation}")
            if self.phase:
                lines.append(f"  Phase: {self.phase}")

        # Error details
        lines.append("\n--- Error Details ---")
        if self.error_type:
            lines.append(f"  Error Type: {self.error_type}")
        if self.error_message:
            lines.append(f"  Error Message: {self.error_message}")

        # Source location
        if any([self.source_module, self.source_function, self.source_file]):
            lines.append("\n--- Source Location ---")
            if self.source_module:
                lines.append(f"  Module: {self.source_module}")
            if self.source_function:
                lines.append(f"  Function: {self.source_function}")
            if self.source_file:
                lines.append(f"  File: {self.source_file}")
            if self.source_line:
                lines.append(f"  Line: {self.source_line}")

        # Additional context
        if self.additional_context:
            lines.append("\n--- Additional Context ---")
            for key, value in self.additional_context.items():
                lines.append(f"  {key}: {value}")

        # Stack trace
        if self.stack_trace:
            lines.append("\n--- Stack Trace ---")
            lines.append(self.stack_trace)

        lines.append("=" * 70)

        return "\n".join(lines)

    def __str__(self) -> str:
        """Return compact format by default."""
        return self.format_compact()


def capture_error_context(
    error: Exception,
    sheet_name: Optional[str] = None,
    sheet_id: Optional[int] = None,
    row_id: Optional[Union[int, str]] = None,
    field_name: Optional[str] = None,
    user: Optional[str] = None,
    marketplace: Optional[str] = None,
    phase: Optional[int] = None,
    operation: Optional[str] = None,
    include_trace: bool = True,
    include_caller: bool = True,
    **additional_context
) -> ErrorContext:
    """
    Convenience function to capture comprehensive error context.

    This function creates an ErrorContext object and populates it with
    all available information about the error and its context.

    Args:
        error: The exception that occurred.
        sheet_name: Name/group identifier of the sheet.
        sheet_id: Smartsheet sheet ID.
        row_id: Smartsheet row ID.
        field_name: Name of the field being processed.
        user: User associated with the operation.
        marketplace: Marketplace identifier.
        phase: Phase number.
        operation: Description of the operation.
        include_trace: Whether to capture stack trace.
        include_caller: Whether to capture caller information.
        **additional_context: Any additional context key-value pairs.

    Returns:
        Populated ErrorContext object.

    Example:
        >>> try:
        ...     process_cell(cell)
        ... except Exception as e:
        ...     ctx = capture_error_context(
        ...         e,
        ...         sheet_name="NA",
        ...         row_id=12345,
        ...         field_name="Kontrolle",
        ...         user="john@example.com"
        ...     )
        ...     logger.error(ctx.format_full())
    """
    context = ErrorContext(
        sheet_name=sheet_name,
        sheet_id=sheet_id,
        row_id=row_id,
        field_name=field_name,
        user=user,
        marketplace=marketplace,
        phase=phase,
        operation=operation,
        additional_context=dict(additional_context),
    )

    context.capture_exception(error, include_trace=include_trace)

    if include_caller:
        context.capture_caller(skip_frames=1)

    return context


def format_error_for_logging(
    context: ErrorContext,
    detailed: bool = False
) -> str:
    """
    Format an ErrorContext for logging.

    Args:
        context: The ErrorContext to format.
        detailed: If True, returns full multi-line format. Otherwise, compact.

    Returns:
        Formatted string for logging.
    """
    if detailed:
        return context.format_full()
    return context.format_compact()


def log_error_with_context(
    error: Exception,
    logger_instance: Optional[logging.Logger] = None,
    level: int = logging.ERROR,
    detailed: bool = True,
    **context_kwargs
) -> ErrorContext:
    """
    Log an error with full context in a single call.

    This is a convenience function that captures context and logs it
    immediately.

    Args:
        error: The exception that occurred.
        logger_instance: Logger to use (defaults to module logger).
        level: Logging level to use.
        detailed: Whether to use detailed format.
        **context_kwargs: Arguments passed to capture_error_context.

    Returns:
        The captured ErrorContext.

    Example:
        >>> try:
        ...     risky_operation()
        ... except Exception as e:
        ...     log_error_with_context(
        ...         e,
        ...         sheet_name="NA",
        ...         row_id=12345,
        ...         operation="risky_operation"
        ...     )
    """
    if logger_instance is None:
        logger_instance = logger

    context = capture_error_context(
        error,
        include_caller=True,
        **context_kwargs
    )

    message = format_error_for_logging(context, detailed=detailed)
    logger_instance.log(level, message)

    return context


class ErrorContextManager:
    """
    Context manager for capturing error context in a block.

    This provides a clean way to capture context for any errors
    that occur within a block of code.

    Usage:
        with ErrorContextManager(sheet_name="NA", row_id=12345) as ctx:
            # Do risky operations
            process_row(row)

        # If an error occurred, ctx.error_context will be populated
        if ctx.error_context:
            logger.error(ctx.error_context.format_full())

    Attributes:
        error_context: ErrorContext populated if an error occurred.
        exception: The original exception if one occurred.
    """

    def __init__(
        self,
        sheet_name: Optional[str] = None,
        sheet_id: Optional[int] = None,
        row_id: Optional[Union[int, str]] = None,
        field_name: Optional[str] = None,
        user: Optional[str] = None,
        marketplace: Optional[str] = None,
        phase: Optional[int] = None,
        operation: Optional[str] = None,
        auto_log: bool = True,
        log_level: int = logging.ERROR,
        detailed_log: bool = True,
        logger_instance: Optional[logging.Logger] = None,
        reraise: bool = True,
        **additional_context
    ):
        """
        Initialize the context manager.

        Args:
            sheet_name: Sheet name/group for context.
            sheet_id: Sheet ID for context.
            row_id: Row ID for context.
            field_name: Field name for context.
            user: User for context.
            marketplace: Marketplace for context.
            phase: Phase number for context.
            operation: Operation description for context.
            auto_log: Whether to automatically log errors.
            log_level: Logging level to use.
            detailed_log: Whether to use detailed log format.
            logger_instance: Logger to use.
            reraise: Whether to reraise exceptions after logging.
            **additional_context: Additional context key-value pairs.
        """
        self.sheet_name = sheet_name
        self.sheet_id = sheet_id
        self.row_id = row_id
        self.field_name = field_name
        self.user = user
        self.marketplace = marketplace
        self.phase = phase
        self.operation = operation
        self.auto_log = auto_log
        self.log_level = log_level
        self.detailed_log = detailed_log
        self.logger_instance = logger_instance or logger
        self.reraise = reraise
        self.additional_context = additional_context

        # Will be populated if an error occurs
        self.error_context: Optional[ErrorContext] = None
        self.exception: Optional[Exception] = None

    def __enter__(self) -> "ErrorContextManager":
        """Enter the context manager."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> bool:
        """
        Exit the context manager, capturing any exception.

        Returns:
            True to suppress the exception (if reraise=False), False otherwise.
        """
        if exc_val is not None:
            self.exception = exc_val
            self.error_context = capture_error_context(
                exc_val,
                sheet_name=self.sheet_name,
                sheet_id=self.sheet_id,
                row_id=self.row_id,
                field_name=self.field_name,
                user=self.user,
                marketplace=self.marketplace,
                phase=self.phase,
                operation=self.operation,
                include_trace=True,
                include_caller=False,  # Already captured in exception
                **self.additional_context
            )

            if self.auto_log:
                message = format_error_for_logging(
                    self.error_context,
                    detailed=self.detailed_log
                )
                self.logger_instance.log(self.log_level, message)

            # Return True to suppress exception, False to reraise
            return not self.reraise

        return False

    def update_context(self, **kwargs) -> None:
        """
        Update context values during the managed block.

        This allows updating context information as more becomes available.

        Args:
            **kwargs: Context fields to update.
        """
        for key, value in kwargs.items():
            if hasattr(self, key):
                setattr(self, key, value)
            else:
                self.additional_context[key] = value
