"""
Error Report Data Structure Module for Smartsheet Tracker

Defines comprehensive data structures for storing error and warning information
with support for:
- Severity levels (INFO, WARNING, ERROR, CRITICAL)
- Structured location information (file, line, function, module, context)
- Human-readable messages with details
- Multiple suggested actions for resolution
- Hierarchical categorization with main category and sub-category

This module extends the error collection framework with enhanced data structures
specifically designed for generating detailed error reports.

Usage:
    from error_report_data import (
        ErrorReportItem,
        ErrorLocation,
        ErrorReportSeverity,
        ErrorReportCategory,
        SuggestedAction,
        ErrorReportCollection,
    )

    # Create an error location
    location = ErrorLocation(
        module="smartsheet_tracker",
        function="process_row",
        file_path="/path/to/smartsheet_tracker.py",
        line_number=150,
        context={"sheet_id": 12345, "row_id": 67890}
    )

    # Create suggested actions
    actions = [
        SuggestedAction(
            action="Check Smartsheet permissions",
            priority=1,
            details="Verify that your API token has read access to the sheet"
        ),
        SuggestedAction(
            action="Contact sheet owner",
            priority=2,
            details="Request access if you don't have permissions"
        )
    ]

    # Create an error report item
    error = ErrorReportItem(
        severity=ErrorReportSeverity.ERROR,
        category=ErrorReportCategory.API_ERROR,
        sub_category="permission_denied",
        message="Failed to access sheet",
        details="Permission denied when attempting to read sheet 12345",
        location=location,
        suggested_actions=actions
    )

    # Create a collection to manage errors
    collection = ErrorReportCollection()
    collection.add(error)

    # Get summary
    print(collection.get_summary())
"""

import logging
import threading
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional, Union
import json

# Get logger for this module
logger = logging.getLogger(__name__)


class ErrorReportSeverity(Enum):
    """
    Severity levels for error and warning reports.

    Levels are ordered from lowest to highest severity:
    - INFO: Informational messages, no action required
    - WARNING: Potential issues that don't prevent operation
    - ERROR: Errors that affect functionality but allow continued operation
    - CRITICAL: Critical errors that prevent operation or require immediate attention
    """
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"

    @classmethod
    def from_string(cls, value: str) -> "ErrorReportSeverity":
        """Create severity from string value."""
        value_lower = value.lower().strip()
        for member in cls:
            if member.value == value_lower:
                return member
        raise ValueError(f"Unknown severity: {value}")

    def __lt__(self, other: "ErrorReportSeverity") -> bool:
        """Compare severity levels."""
        order = [self.INFO, self.WARNING, self.ERROR, self.CRITICAL]
        return order.index(self) < order.index(other)

    def __le__(self, other: "ErrorReportSeverity") -> bool:
        """Compare severity levels."""
        return self == other or self < other


class ErrorReportCategory(Enum):
    """
    Main categories for classifying errors and warnings.

    Categories are organized by domain:
    - Data Quality: Issues with data integrity and format
    - API/Network: External service communication issues
    - Authentication: Credential and permission issues
    - Resource: Missing or inaccessible resources
    - Processing: Runtime processing issues
    - Validation: Data validation failures
    - Configuration: System configuration issues
    - System: System-level errors
    """
    # Data Quality errors
    DATA_QUALITY = "data_quality"
    MISSING_DATA = "missing_data"
    INVALID_FORMAT = "invalid_format"
    DATA_CORRUPTION = "data_corruption"
    DUPLICATE_DATA = "duplicate_data"

    # API/Network errors
    API_ERROR = "api_error"
    RATE_LIMIT = "rate_limit"
    TIMEOUT = "timeout"
    CONNECTION = "connection"
    NETWORK = "network"

    # Authentication/Authorization errors
    AUTHENTICATION = "authentication"
    PERMISSION = "permission"
    TOKEN_EXPIRED = "token_expired"
    TOKEN_INVALID = "token_invalid"
    ACCESS_DENIED = "access_denied"

    # Resource errors
    RESOURCE_NOT_FOUND = "resource_not_found"
    SHEET_NOT_FOUND = "sheet_not_found"
    COLUMN_NOT_FOUND = "column_not_found"
    ROW_NOT_FOUND = "row_not_found"
    FILE_NOT_FOUND = "file_not_found"

    # Processing errors
    PROCESSING = "processing"
    CELL_ACCESS = "cell_access"
    DATE_PARSING = "date_parsing"
    CSV_WRITE = "csv_write"
    ROW_PROCESSING = "row_processing"
    PHASE_PROCESSING = "phase_processing"
    CALCULATION = "calculation"

    # Validation errors
    VALIDATION = "validation"
    SCHEMA_VALIDATION = "schema_validation"
    BUSINESS_RULE = "business_rule"
    CONSTRAINT_VIOLATION = "constraint_violation"

    # Configuration errors
    CONFIGURATION = "configuration"
    MISSING_CONFIG = "missing_config"
    INVALID_CONFIG = "invalid_config"

    # System errors
    SYSTEM = "system"
    MEMORY = "memory"
    DISK = "disk"

    # General/Unknown
    OTHER = "other"
    UNKNOWN = "unknown"

    @classmethod
    def from_string(cls, value: str) -> "ErrorReportCategory":
        """Create category from string value."""
        value_lower = value.lower().strip()
        for member in cls:
            if member.value == value_lower:
                return member
        return cls.UNKNOWN


@dataclass
class SuggestedAction:
    """
    Represents a suggested action for resolving an error or warning.

    Attributes:
        action: Brief description of the action to take
        priority: Priority level (1 = highest, higher numbers = lower priority)
        details: Detailed explanation or steps to perform the action
        action_type: Type of action (manual, automatic, escalate, etc.)
        url: Optional URL for more information or to perform the action
        estimated_time: Optional estimated time to complete the action
    """
    action: str
    priority: int = 1
    details: Optional[str] = None
    action_type: str = "manual"
    url: Optional[str] = None
    estimated_time: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "action": self.action,
            "priority": self.priority,
            "details": self.details,
            "action_type": self.action_type,
            "url": self.url,
            "estimated_time": self.estimated_time,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SuggestedAction":
        """Create from dictionary."""
        return cls(
            action=data.get("action", ""),
            priority=data.get("priority", 1),
            details=data.get("details"),
            action_type=data.get("action_type", "manual"),
            url=data.get("url"),
            estimated_time=data.get("estimated_time"),
        )

    def __str__(self) -> str:
        """Human-readable representation."""
        result = f"[P{self.priority}] {self.action}"
        if self.details:
            result += f" - {self.details}"
        return result


@dataclass
class ErrorLocation:
    """
    Represents the location where an error or warning occurred.

    Provides structured information about where in the codebase and data
    the error originated, enabling precise debugging and reporting.

    Attributes:
        module: Python module name where the error occurred
        function: Function or method name
        file_path: Full path to the source file
        line_number: Line number in the source file
        column_number: Column number (if applicable)
        class_name: Class name (if error occurred in a method)
        context: Additional context information (sheet_id, row_id, etc.)
        stack_trace: Optional stack trace for debugging
    """
    module: Optional[str] = None
    function: Optional[str] = None
    file_path: Optional[str] = None
    line_number: Optional[int] = None
    column_number: Optional[int] = None
    class_name: Optional[str] = None
    context: Dict[str, Any] = field(default_factory=dict)
    stack_trace: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "module": self.module,
            "function": self.function,
            "file_path": self.file_path,
            "line_number": self.line_number,
            "column_number": self.column_number,
            "class_name": self.class_name,
            "context": self.context,
            "stack_trace": self.stack_trace,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ErrorLocation":
        """Create from dictionary."""
        return cls(
            module=data.get("module"),
            function=data.get("function"),
            file_path=data.get("file_path"),
            line_number=data.get("line_number"),
            column_number=data.get("column_number"),
            class_name=data.get("class_name"),
            context=data.get("context", {}),
            stack_trace=data.get("stack_trace"),
        )

    @classmethod
    def from_context(cls, context: Dict[str, Any]) -> "ErrorLocation":
        """
        Create an ErrorLocation from a context dictionary.

        This is useful for converting from the existing error_collector.py
        context format to the structured ErrorLocation format.
        """
        # Extract known location fields
        location_fields = {
            "module": context.get("module") or context.get("source_module"),
            "function": context.get("function") or context.get("source_function"),
            "file_path": context.get("file_path") or context.get("file"),
            "line_number": context.get("line_number") or context.get("line"),
            "column_number": context.get("column_number") or context.get("column"),
            "class_name": context.get("class_name") or context.get("class"),
            "stack_trace": context.get("stack_trace") or context.get("traceback"),
        }

        # Remaining items go into context
        remaining_context = {
            k: v for k, v in context.items()
            if k not in {
                "module", "source_module", "function", "source_function",
                "file_path", "file", "line_number", "line", "column_number",
                "column", "class_name", "class", "stack_trace", "traceback"
            }
        }

        return cls(
            **{k: v for k, v in location_fields.items() if v is not None},
            context=remaining_context
        )

    def get_short_location(self) -> str:
        """Get a short location string for display."""
        parts = []
        if self.module:
            parts.append(self.module)
        if self.class_name:
            parts.append(self.class_name)
        if self.function:
            parts.append(f"{self.function}()")
        if self.line_number:
            parts.append(f"line {self.line_number}")
        return ":".join(parts) if parts else "unknown location"

    def get_context_string(self) -> str:
        """Get a formatted string of context information."""
        if not self.context:
            return ""
        parts = []
        # Prioritize common context fields
        priority_fields = ["group", "sheet_id", "row_id", "phase", "column"]
        for field in priority_fields:
            if field in self.context:
                parts.append(f"{field}={self.context[field]}")
        # Add remaining fields
        for key, value in self.context.items():
            if key not in priority_fields:
                parts.append(f"{key}={value}")
        return ", ".join(parts)

    def __str__(self) -> str:
        """Human-readable representation."""
        location = self.get_short_location()
        context = self.get_context_string()
        if context:
            return f"{location} [{context}]"
        return location


@dataclass
class ErrorReportItem:
    """
    Represents a single error or warning item in an error report.

    This is the main data structure for storing error and warning information.
    It supports:
    - Severity classification (INFO, WARNING, ERROR, CRITICAL)
    - Hierarchical categorization (category + sub_category)
    - Structured location information
    - Multiple suggested actions
    - Rich metadata and context

    Attributes:
        id: Unique identifier for this error
        timestamp: When the error occurred
        severity: Severity level of the error
        category: Main error category
        sub_category: Optional sub-category for finer classification
        message: Human-readable error message
        details: Detailed description of the error
        location: Structured location information
        suggested_actions: List of actions to resolve the error
        error_code: Optional application-specific error code
        original_exception: Reference to the original exception (if any)
        is_recoverable: Whether operation can continue after this error
        tags: Optional tags for additional classification
        metadata: Additional metadata
    """
    severity: ErrorReportSeverity
    category: ErrorReportCategory
    message: str
    id: str = ""
    timestamp: datetime = field(default_factory=datetime.now)
    sub_category: Optional[str] = None
    details: Optional[str] = None
    location: Optional[ErrorLocation] = None
    suggested_actions: List[SuggestedAction] = field(default_factory=list)
    error_code: Optional[str] = None
    original_exception: Optional[Exception] = None
    is_recoverable: bool = True
    tags: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Generate ID if not provided."""
        if not self.id:
            self.id = self._generate_id()

    def _generate_id(self) -> str:
        """Generate a unique error ID."""
        import random
        timestamp_str = self.timestamp.strftime("%Y%m%d%H%M%S")
        random_suffix = random.randint(100000, 999999)
        return f"ERR-{timestamp_str}-{random_suffix}"

    def add_suggested_action(
        self,
        action: str,
        priority: int = 1,
        details: Optional[str] = None,
        **kwargs
    ) -> None:
        """Add a suggested action to this error."""
        suggested = SuggestedAction(
            action=action,
            priority=priority,
            details=details,
            **kwargs
        )
        self.suggested_actions.append(suggested)
        # Sort by priority
        self.suggested_actions.sort(key=lambda x: x.priority)

    def get_sorted_actions(self) -> List[SuggestedAction]:
        """Get suggested actions sorted by priority."""
        return sorted(self.suggested_actions, key=lambda x: x.priority)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "severity": self.severity.value,
            "category": self.category.value,
            "sub_category": self.sub_category,
            "message": self.message,
            "details": self.details,
            "location": self.location.to_dict() if self.location else None,
            "suggested_actions": [a.to_dict() for a in self.suggested_actions],
            "error_code": self.error_code,
            "is_recoverable": self.is_recoverable,
            "tags": self.tags,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ErrorReportItem":
        """Create from dictionary."""
        location = None
        if data.get("location"):
            location = ErrorLocation.from_dict(data["location"])

        actions = []
        for action_data in data.get("suggested_actions", []):
            actions.append(SuggestedAction.from_dict(action_data))

        timestamp = data.get("timestamp")
        if isinstance(timestamp, str):
            timestamp = datetime.fromisoformat(timestamp)
        elif timestamp is None:
            timestamp = datetime.now()

        return cls(
            id=data.get("id", ""),
            timestamp=timestamp,
            severity=ErrorReportSeverity.from_string(data.get("severity", "error")),
            category=ErrorReportCategory.from_string(data.get("category", "unknown")),
            sub_category=data.get("sub_category"),
            message=data.get("message", ""),
            details=data.get("details"),
            location=location,
            suggested_actions=actions,
            error_code=data.get("error_code"),
            is_recoverable=data.get("is_recoverable", True),
            tags=data.get("tags", []),
            metadata=data.get("metadata", {}),
        )

    def to_json(self) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=2, default=str)

    @classmethod
    def from_json(cls, json_str: str) -> "ErrorReportItem":
        """Create from JSON string."""
        return cls.from_dict(json.loads(json_str))

    def __str__(self) -> str:
        """Human-readable representation."""
        parts = [f"[{self.severity.value.upper()}]"]

        if self.location:
            loc_str = self.location.get_context_string()
            if loc_str:
                parts.append(f"[{loc_str}]")

        parts.append(self.message)

        if self.suggested_actions:
            actions_str = "; ".join(a.action for a in self.get_sorted_actions()[:2])
            parts.append(f"(Suggested: {actions_str})")

        return " ".join(parts)


class ErrorReportCollection:
    """
    Thread-safe collection for managing error report items.

    Provides:
    - Thread-safe add/remove operations
    - Filtering by severity, category, tags
    - Statistics and summaries
    - Export capabilities (dict, JSON)
    - Capacity management

    Usage:
        collection = ErrorReportCollection()
        collection.add(error_item)

        # Filter errors
        critical_errors = collection.get_by_severity(ErrorReportSeverity.CRITICAL)
        api_errors = collection.get_by_category(ErrorReportCategory.API_ERROR)

        # Get statistics
        stats = collection.get_statistics()
        print(collection.get_summary())

        # Export
        data = collection.export_to_dict()
    """

    def __init__(self, max_items: int = 10000, log_on_add: bool = True):
        """
        Initialize the collection.

        Args:
            max_items: Maximum number of items to store (oldest removed when exceeded)
            log_on_add: Whether to log items when they are added
        """
        self._items: List[ErrorReportItem] = []
        self._lock = threading.RLock()
        self._max_items = max_items
        self._log_on_add = log_on_add
        self._item_counter = 0
        self._overflow_count = 0

        # Statistics
        self._stats = {
            "total_added": 0,
            "by_severity": {s.value: 0 for s in ErrorReportSeverity},
            "by_category": {c.value: 0 for c in ErrorReportCategory},
            "by_sub_category": {},
            "by_tag": {},
            "first_item_time": None,
            "last_item_time": None,
        }

    def add(self, item: ErrorReportItem) -> ErrorReportItem:
        """
        Add an error report item to the collection.

        Args:
            item: The ErrorReportItem to add

        Returns:
            The added item (with generated ID if not set)
        """
        with self._lock:
            # Handle capacity
            if len(self._items) >= self._max_items:
                self._overflow_count += 1
                if self._overflow_count == 1:
                    logger.warning(
                        f"ErrorReportCollection reached max capacity ({self._max_items}). "
                        f"Oldest items will be removed."
                    )
                self._items.pop(0)

            # Ensure ID is set
            if not item.id:
                self._item_counter += 1
                item.id = f"ERR-{item.timestamp.strftime('%Y%m%d%H%M%S')}-{self._item_counter:06d}"

            self._items.append(item)

            # Update statistics
            self._stats["total_added"] += 1
            self._stats["by_severity"][item.severity.value] += 1
            self._stats["by_category"][item.category.value] += 1

            if item.sub_category:
                self._stats["by_sub_category"][item.sub_category] = \
                    self._stats["by_sub_category"].get(item.sub_category, 0) + 1

            for tag in item.tags:
                self._stats["by_tag"][tag] = self._stats["by_tag"].get(tag, 0) + 1

            if self._stats["first_item_time"] is None:
                self._stats["first_item_time"] = item.timestamp
            self._stats["last_item_time"] = item.timestamp

            # Log if configured
            if self._log_on_add:
                log_level = {
                    ErrorReportSeverity.INFO: logging.INFO,
                    ErrorReportSeverity.WARNING: logging.WARNING,
                    ErrorReportSeverity.ERROR: logging.ERROR,
                    ErrorReportSeverity.CRITICAL: logging.CRITICAL,
                }.get(item.severity, logging.ERROR)
                logger.log(log_level, f"Added error report: {item}")

            return item

    def create_and_add(
        self,
        severity: ErrorReportSeverity,
        category: ErrorReportCategory,
        message: str,
        **kwargs
    ) -> ErrorReportItem:
        """
        Create and add a new error report item.

        Convenience method that creates an ErrorReportItem and adds it in one call.

        Args:
            severity: Severity level
            category: Error category
            message: Error message
            **kwargs: Additional arguments for ErrorReportItem

        Returns:
            The created and added ErrorReportItem
        """
        item = ErrorReportItem(
            severity=severity,
            category=category,
            message=message,
            **kwargs
        )
        return self.add(item)

    def get_all(self) -> List[ErrorReportItem]:
        """Get all items in the collection."""
        with self._lock:
            return list(self._items)

    def get_by_severity(self, severity: ErrorReportSeverity) -> List[ErrorReportItem]:
        """Get items filtered by severity."""
        with self._lock:
            return [item for item in self._items if item.severity == severity]

    def get_by_category(self, category: ErrorReportCategory) -> List[ErrorReportItem]:
        """Get items filtered by category."""
        with self._lock:
            return [item for item in self._items if item.category == category]

    def get_by_sub_category(self, sub_category: str) -> List[ErrorReportItem]:
        """Get items filtered by sub-category."""
        with self._lock:
            return [item for item in self._items if item.sub_category == sub_category]

    def get_by_tag(self, tag: str) -> List[ErrorReportItem]:
        """Get items that have a specific tag."""
        with self._lock:
            return [item for item in self._items if tag in item.tags]

    def get_by_id(self, item_id: str) -> Optional[ErrorReportItem]:
        """Get a specific item by ID."""
        with self._lock:
            for item in self._items:
                if item.id == item_id:
                    return item
            return None

    def get_critical(self) -> List[ErrorReportItem]:
        """Get all critical and non-recoverable items."""
        with self._lock:
            return [
                item for item in self._items
                if item.severity == ErrorReportSeverity.CRITICAL or not item.is_recoverable
            ]

    def get_errors_and_above(self) -> List[ErrorReportItem]:
        """Get all ERROR and CRITICAL severity items."""
        with self._lock:
            return [
                item for item in self._items
                if item.severity in (ErrorReportSeverity.ERROR, ErrorReportSeverity.CRITICAL)
            ]

    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the collection."""
        with self._lock:
            stats = self._stats.copy()
            stats["current_count"] = len(self._items)
            stats["overflow_count"] = self._overflow_count
            stats["max_capacity"] = self._max_items
            return stats

    def get_summary(self) -> str:
        """Get a human-readable summary."""
        with self._lock:
            lines = [
                "=" * 60,
                "ERROR REPORT COLLECTION SUMMARY",
                "=" * 60,
                f"Total items added: {self._stats['total_added']}",
                f"Currently stored: {len(self._items)} (max: {self._max_items})",
            ]

            if self._overflow_count > 0:
                lines.append(f"Overflow count (oldest removed): {self._overflow_count}")

            if self._stats["first_item_time"]:
                lines.append(f"First item: {self._stats['first_item_time'].strftime('%Y-%m-%d %H:%M:%S')}")
            if self._stats["last_item_time"]:
                lines.append(f"Last item: {self._stats['last_item_time'].strftime('%Y-%m-%d %H:%M:%S')}")

            # By severity
            lines.append("\nBy Severity:")
            for severity_val, count in sorted(self._stats["by_severity"].items(), key=lambda x: -x[1]):
                if count > 0:
                    lines.append(f"  {severity_val.upper()}: {count}")

            # By category (non-zero only)
            category_counts = {k: v for k, v in self._stats["by_category"].items() if v > 0}
            if category_counts:
                lines.append("\nBy Category:")
                for category_val, count in sorted(category_counts.items(), key=lambda x: -x[1]):
                    lines.append(f"  {category_val}: {count}")

            # By sub-category (non-zero only)
            if self._stats["by_sub_category"]:
                lines.append("\nBy Sub-Category:")
                for sub_cat, count in sorted(self._stats["by_sub_category"].items(), key=lambda x: -x[1]):
                    lines.append(f"  {sub_cat}: {count}")

            # By tag (non-zero only)
            if self._stats["by_tag"]:
                lines.append("\nBy Tag:")
                for tag, count in sorted(self._stats["by_tag"].items(), key=lambda x: -x[1]):
                    lines.append(f"  {tag}: {count}")

            lines.append("=" * 60)
            return "\n".join(lines)

    def export_to_dict(self) -> Dict[str, Any]:
        """Export the collection to a dictionary."""
        with self._lock:
            return {
                "items": [item.to_dict() for item in self._items],
                "statistics": self.get_statistics(),
                "export_timestamp": datetime.now().isoformat(),
            }

    def export_to_json(self, indent: int = 2) -> str:
        """Export the collection to a JSON string."""
        return json.dumps(self.export_to_dict(), indent=indent, default=str)

    def clear(self) -> int:
        """Clear all items. Returns the number of items cleared."""
        with self._lock:
            count = len(self._items)
            self._items.clear()
            self._overflow_count = 0
            self._stats = {
                "total_added": 0,
                "by_severity": {s.value: 0 for s in ErrorReportSeverity},
                "by_category": {c.value: 0 for c in ErrorReportCategory},
                "by_sub_category": {},
                "by_tag": {},
                "first_item_time": None,
                "last_item_time": None,
            }
            return count

    def has_critical(self) -> bool:
        """Check if there are any critical or non-recoverable items."""
        with self._lock:
            return any(
                item.severity == ErrorReportSeverity.CRITICAL or not item.is_recoverable
                for item in self._items
            )

    def has_errors(self) -> bool:
        """Check if there are any ERROR or CRITICAL items."""
        with self._lock:
            return any(
                item.severity in (ErrorReportSeverity.ERROR, ErrorReportSeverity.CRITICAL)
                for item in self._items
            )

    def count(self) -> int:
        """Get the current number of items."""
        with self._lock:
            return len(self._items)

    def __len__(self) -> int:
        """Get the current number of items."""
        return self.count()

    def __iter__(self):
        """Iterate over items."""
        with self._lock:
            return iter(list(self._items))


# Global collection instance
_global_collection: Optional[ErrorReportCollection] = None
_global_collection_lock = threading.Lock()


def get_global_collection() -> ErrorReportCollection:
    """Get the global error report collection (creates one if needed)."""
    global _global_collection
    with _global_collection_lock:
        if _global_collection is None:
            _global_collection = ErrorReportCollection()
        return _global_collection


def reset_global_collection() -> None:
    """Reset the global error report collection."""
    global _global_collection
    with _global_collection_lock:
        _global_collection = ErrorReportCollection()


def create_error_report(
    severity: Union[ErrorReportSeverity, str],
    category: Union[ErrorReportCategory, str],
    message: str,
    details: Optional[str] = None,
    location: Optional[Union[ErrorLocation, Dict[str, Any]]] = None,
    suggested_actions: Optional[List[Union[SuggestedAction, Dict[str, Any], str]]] = None,
    sub_category: Optional[str] = None,
    **kwargs
) -> ErrorReportItem:
    """
    Convenience function to create an error report item.

    This function provides a simpler interface for creating ErrorReportItem
    instances with automatic type conversion.

    Args:
        severity: Severity level (enum or string)
        category: Error category (enum or string)
        message: Error message
        details: Detailed description
        location: Location information (ErrorLocation or dict)
        suggested_actions: List of suggested actions (various formats accepted)
        sub_category: Sub-category for finer classification
        **kwargs: Additional arguments for ErrorReportItem

    Returns:
        Created ErrorReportItem
    """
    # Convert severity if string
    if isinstance(severity, str):
        severity = ErrorReportSeverity.from_string(severity)

    # Convert category if string
    if isinstance(category, str):
        category = ErrorReportCategory.from_string(category)

    # Convert location if dict
    if isinstance(location, dict):
        location = ErrorLocation.from_dict(location)

    # Convert suggested actions
    actions = []
    if suggested_actions:
        for action in suggested_actions:
            if isinstance(action, str):
                actions.append(SuggestedAction(action=action))
            elif isinstance(action, dict):
                actions.append(SuggestedAction.from_dict(action))
            elif isinstance(action, SuggestedAction):
                actions.append(action)

    return ErrorReportItem(
        severity=severity,
        category=category,
        message=message,
        details=details,
        location=location,
        suggested_actions=actions,
        sub_category=sub_category,
        **kwargs
    )


def add_to_global_collection(
    severity: Union[ErrorReportSeverity, str],
    category: Union[ErrorReportCategory, str],
    message: str,
    **kwargs
) -> ErrorReportItem:
    """
    Create an error report and add it to the global collection.

    Convenience function that creates an ErrorReportItem and adds it
    to the global collection in one call.

    Args:
        severity: Severity level
        category: Error category
        message: Error message
        **kwargs: Additional arguments for create_error_report

    Returns:
        The created and added ErrorReportItem
    """
    item = create_error_report(severity, category, message, **kwargs)
    return get_global_collection().add(item)
