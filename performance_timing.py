"""
Performance timing utilities for Smartsheet tracker application.

This module provides a consistent way to measure and log the execution time
of major operations, helping identify performance bottlenecks.

Key Features:
1. Context manager for easy timing of code blocks
2. Decorator for timing function execution
3. Timing log entries with operation name, duration, and optional metadata
4. Configurable logging level and format

Usage:
    # Context manager approach:
    from performance_timing import timed_operation, PerformanceTimer

    with timed_operation("sheet_fetch", sheet_id="123"):
        sheet = client.get_sheet(sheet_id)

    # Decorator approach:
    @timed_function("pdf_generation")
    def create_report():
        ...

    # Manual timer for complex flows:
    timer = PerformanceTimer("data_processing")
    timer.start()
    # ... do work ...
    timer.checkpoint("phase_1_complete")
    # ... more work ...
    timer.stop()

Log Output Format:
    PERF: [operation_name] completed in 1.234s {metadata}
    PERF: [operation_name] checkpoint 'phase_1_complete' at 0.567s
"""

import time
import logging
import functools
from typing import Any, Callable, Dict, Optional, TypeVar, Union
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import datetime

# Logger for performance timing messages
logger = logging.getLogger(__name__)

# Type variable for decorator
F = TypeVar('F', bound=Callable[..., Any])


@dataclass
class TimingResult:
    """Result of a timed operation with metadata."""
    operation: str
    duration_seconds: float
    start_time: datetime
    end_time: datetime
    metadata: Dict[str, Any] = field(default_factory=dict)
    checkpoints: Dict[str, float] = field(default_factory=dict)

    @property
    def duration_ms(self) -> float:
        """Duration in milliseconds."""
        return self.duration_seconds * 1000

    def format_duration(self) -> str:
        """Format duration for display (auto-select units)."""
        if self.duration_seconds < 1:
            return f"{self.duration_ms:.1f}ms"
        elif self.duration_seconds < 60:
            return f"{self.duration_seconds:.2f}s"
        else:
            minutes = int(self.duration_seconds // 60)
            seconds = self.duration_seconds % 60
            return f"{minutes}m {seconds:.1f}s"


class PerformanceTimer:
    """
    Manual timer for complex timing scenarios with checkpoints.

    Use this when you need to track multiple stages within a single
    operation or when the context manager approach doesn't fit.

    Example:
        timer = PerformanceTimer("report_generation")
        timer.start()

        # Phase 1: Load data
        changes = load_changes()
        timer.checkpoint("data_loaded", rows=len(changes))

        # Phase 2: Collect metrics
        metrics = collect_metrics(changes)
        timer.checkpoint("metrics_collected")

        # Phase 3: Build PDF
        build_pdf(metrics)
        timer.stop()

        # Timer automatically logs each checkpoint and final duration
    """

    def __init__(self, operation: str, log_level: int = logging.INFO, **metadata: Any):
        """
        Initialize a performance timer.

        Args:
            operation: Name of the operation being timed.
            log_level: Logging level for timing messages (default: INFO).
            **metadata: Additional key-value pairs to include in log output.
        """
        self.operation = operation
        self.log_level = log_level
        self.metadata = metadata
        self._start_time: Optional[float] = None
        self._start_datetime: Optional[datetime] = None
        self._end_time: Optional[float] = None
        self._end_datetime: Optional[datetime] = None
        self._checkpoints: Dict[str, float] = {}
        self._checkpoint_metadata: Dict[str, Dict[str, Any]] = {}

    def start(self) -> 'PerformanceTimer':
        """Start the timer. Returns self for method chaining."""
        self._start_time = time.perf_counter()
        self._start_datetime = datetime.now()
        logger.log(self.log_level, f"PERF: [{self.operation}] started")
        return self

    def checkpoint(self, name: str, **checkpoint_metadata: Any) -> float:
        """
        Record a checkpoint with elapsed time from start.

        Args:
            name: Name of the checkpoint.
            **checkpoint_metadata: Additional data to log with this checkpoint.

        Returns:
            Elapsed time in seconds since start.
        """
        if self._start_time is None:
            raise RuntimeError(f"Timer '{self.operation}' not started")

        elapsed = time.perf_counter() - self._start_time
        self._checkpoints[name] = elapsed
        self._checkpoint_metadata[name] = checkpoint_metadata

        # Format metadata for logging
        meta_str = _format_metadata(checkpoint_metadata) if checkpoint_metadata else ""
        logger.log(
            self.log_level,
            f"PERF: [{self.operation}] checkpoint '{name}' at {_format_duration(elapsed)}{meta_str}"
        )
        return elapsed

    def stop(self) -> TimingResult:
        """
        Stop the timer and log the total duration.

        Returns:
            TimingResult with full timing information.
        """
        if self._start_time is None:
            raise RuntimeError(f"Timer '{self.operation}' not started")

        self._end_time = time.perf_counter()
        self._end_datetime = datetime.now()
        duration = self._end_time - self._start_time

        # Format metadata for logging
        meta_str = _format_metadata(self.metadata) if self.metadata else ""
        logger.log(
            self.log_level,
            f"PERF: [{self.operation}] completed in {_format_duration(duration)}{meta_str}"
        )

        return TimingResult(
            operation=self.operation,
            duration_seconds=duration,
            start_time=self._start_datetime,
            end_time=self._end_datetime,
            metadata=self.metadata,
            checkpoints=self._checkpoints.copy()
        )

    @property
    def elapsed(self) -> float:
        """Current elapsed time in seconds (timer must be started)."""
        if self._start_time is None:
            return 0.0
        end = self._end_time if self._end_time else time.perf_counter()
        return end - self._start_time


@contextmanager
def timed_operation(operation: str, log_level: int = logging.INFO, **metadata: Any):
    """
    Context manager for timing a block of code.

    This is the simplest way to add timing to an operation.

    Args:
        operation: Name of the operation (used in log messages).
        log_level: Logging level for timing messages (default: INFO).
        **metadata: Additional key-value pairs to include in log output.

    Yields:
        PerformanceTimer instance (can be used for checkpoints).

    Example:
        with timed_operation("sheet_fetch", sheet_id=12345) as timer:
            sheet = client.get_sheet(sheet_id)
            timer.checkpoint("api_call_complete", rows=len(sheet.rows))

        # Logs:
        # PERF: [sheet_fetch] started
        # PERF: [sheet_fetch] checkpoint 'api_call_complete' at 1.23s {rows: 500}
        # PERF: [sheet_fetch] completed in 1.25s {sheet_id: 12345}
    """
    timer = PerformanceTimer(operation, log_level=log_level, **metadata)
    timer.start()
    try:
        yield timer
    finally:
        timer.stop()


def timed_function(operation: Optional[str] = None, log_level: int = logging.INFO) -> Callable[[F], F]:
    """
    Decorator for timing function execution.

    Args:
        operation: Name of the operation. If None, uses function name.
        log_level: Logging level for timing messages.

    Returns:
        Decorated function that logs execution time.

    Example:
        @timed_function("pdf_generation")
        def create_weekly_report(start_date, end_date):
            ...

        @timed_function()  # Uses function name
        def collect_metrics(changes):
            ...
    """
    def decorator(func: F) -> F:
        op_name = operation or func.__name__

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            with timed_operation(op_name, log_level=log_level):
                return func(*args, **kwargs)

        return wrapper  # type: ignore

    return decorator


def log_timing(
    operation: str,
    duration_seconds: float,
    log_level: int = logging.INFO,
    **metadata: Any
) -> None:
    """
    Log a timing measurement directly (for cases where you've measured time yourself).

    Args:
        operation: Name of the operation.
        duration_seconds: Duration in seconds.
        log_level: Logging level.
        **metadata: Additional key-value pairs to include.

    Example:
        start = time.time()
        result = external_api_call()
        log_timing("external_api", time.time() - start, response_size=len(result))
    """
    meta_str = _format_metadata(metadata) if metadata else ""
    logger.log(
        log_level,
        f"PERF: [{operation}] completed in {_format_duration(duration_seconds)}{meta_str}"
    )


def _format_duration(seconds: float) -> str:
    """Format duration for log display."""
    if seconds < 1:
        return f"{seconds * 1000:.1f}ms"
    elif seconds < 60:
        return f"{seconds:.2f}s"
    else:
        minutes = int(seconds // 60)
        secs = seconds % 60
        return f"{minutes}m {secs:.1f}s"


def _format_metadata(metadata: Dict[str, Any]) -> str:
    """Format metadata dictionary for log display."""
    if not metadata:
        return ""
    items = [f"{k}={v}" for k, v in metadata.items()]
    return " {" + ", ".join(items) + "}"


# Convenience aliases for common operations
def time_sheet_fetch(sheet_id: Union[int, str], group: str):
    """Context manager specifically for sheet fetching operations."""
    return timed_operation("sheet_fetch", sheet_id=sheet_id, group=group)


def time_data_processing(operation_type: str, **metadata: Any):
    """Context manager specifically for data processing operations."""
    return timed_operation(f"data_processing.{operation_type}", **metadata)


def time_pdf_generation(report_type: str, **metadata: Any):
    """Context manager specifically for PDF generation operations."""
    return timed_operation(f"pdf_generation.{report_type}", **metadata)
