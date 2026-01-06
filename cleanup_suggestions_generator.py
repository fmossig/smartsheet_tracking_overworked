"""
Cleanup Suggestions Generator for Smartsheet Tracker

Analyzes error patterns and generates actionable suggestions for data cleanup.
Prioritizes suggestions by impact and frequency to help users focus on the
most important data quality issues first.

Usage:
    from cleanup_suggestions_generator import (
        CleanupSuggestionGenerator,
        CleanupSuggestion,
        get_global_suggestion_generator,
        SuggestionPriority,
    )

    # Get the global generator instance
    generator = get_global_suggestion_generator()

    # Generate suggestions from the error collector
    suggestions = generator.generate_suggestions()

    # Get top suggestions by impact
    top_suggestions = generator.get_top_suggestions(limit=10, sort_by='impact')

    # Get suggestions for a specific group
    group_suggestions = generator.get_suggestions_by_group("NA")

    # Export suggestions for reporting
    report_data = generator.export_for_report()
"""

import logging
import threading
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Dict, List, Optional, Tuple, Set

from error_collector import (
    ErrorCollector,
    CollectedError,
    ErrorSeverity,
    ErrorCategory,
    ErrorType,
    get_global_collector,
    get_categories_for_type,
    CATEGORY_TO_TYPE_MAP,
)
from error_report_data import (
    SuggestedAction,
    ErrorReportCategory,
)

# Get logger for this module
logger = logging.getLogger(__name__)


class SuggestionPriority(Enum):
    """Priority levels for cleanup suggestions."""
    CRITICAL = 1  # Blocking issues, immediate action required
    HIGH = 2      # High impact, should be addressed soon
    MEDIUM = 3    # Moderate impact, address when possible
    LOW = 4       # Low impact, address as time permits


class SuggestionEffort(Enum):
    """Effort levels for implementing cleanup suggestions."""
    QUICK_FIX = "quick_fix"       # Less than 15 minutes
    MODERATE = "moderate"          # 15 minutes to 1 hour
    SIGNIFICANT = "significant"    # 1-4 hours
    MAJOR = "major"               # More than 4 hours


class PatternType(Enum):
    """Types of error patterns that can be detected."""
    MISSING_DATA = "missing_data"
    NULL_VALUES = "null_values"
    FORMAT_ERRORS = "format_errors"
    API_FAILURES = "api_failures"
    PERMISSION_ISSUES = "permission_issues"
    REPEATED_ERRORS = "repeated_errors"
    GROUP_SPECIFIC = "group_specific"
    SYSTEMIC = "systemic"


@dataclass
class ErrorPattern:
    """
    Represents a detected pattern in errors.

    Attributes:
        pattern_type: The type of error pattern
        frequency: How many times this pattern occurred
        affected_groups: Groups where this pattern was detected
        affected_categories: Error categories involved
        sample_errors: Sample error IDs for reference
        first_occurrence: When the pattern was first seen
        last_occurrence: When the pattern was last seen
        severity_distribution: Count by severity level
    """
    pattern_type: PatternType
    frequency: int
    affected_groups: Set[str] = field(default_factory=set)
    affected_categories: Set[ErrorCategory] = field(default_factory=set)
    sample_errors: List[str] = field(default_factory=list)
    first_occurrence: Optional[datetime] = None
    last_occurrence: Optional[datetime] = None
    severity_distribution: Dict[str, int] = field(default_factory=dict)
    context_data: Dict[str, Any] = field(default_factory=dict)

    def get_impact_score(self) -> float:
        """
        Calculate an impact score based on frequency and severity.

        Returns:
            Impact score between 0 and 100
        """
        # Base score from frequency (logarithmic scaling)
        import math
        freq_score = min(50, math.log10(max(1, self.frequency)) * 20)

        # Severity weight
        severity_weights = {
            'critical': 4.0,
            'error': 3.0,
            'warning': 2.0,
            'info': 1.0
        }

        total_weight = 0
        total_count = 0
        for sev, count in self.severity_distribution.items():
            weight = severity_weights.get(sev, 1.0)
            total_weight += weight * count
            total_count += count

        avg_severity = (total_weight / max(1, total_count)) / 4.0  # Normalize to 0-1
        severity_score = avg_severity * 40

        # Groups affected bonus (systemic issues score higher)
        group_score = min(10, len(self.affected_groups) * 2)

        return min(100, freq_score + severity_score + group_score)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "pattern_type": self.pattern_type.value,
            "frequency": self.frequency,
            "affected_groups": list(self.affected_groups),
            "affected_categories": [c.value for c in self.affected_categories],
            "sample_errors": self.sample_errors[:5],  # Limit samples
            "first_occurrence": self.first_occurrence.isoformat() if self.first_occurrence else None,
            "last_occurrence": self.last_occurrence.isoformat() if self.last_occurrence else None,
            "severity_distribution": self.severity_distribution,
            "impact_score": self.get_impact_score(),
            "context_data": self.context_data,
        }


@dataclass
class CleanupSuggestion:
    """
    Represents a cleanup suggestion with actionable steps.

    Attributes:
        id: Unique identifier for this suggestion
        title: Brief title describing the suggestion
        description: Detailed description of the issue and suggestion
        priority: Priority level for this suggestion
        effort: Estimated effort to implement
        impact_score: Calculated impact score (0-100)
        frequency: How many errors this addresses
        pattern: The underlying error pattern
        affected_groups: Groups that would benefit from this cleanup
        affected_fields: Fields that need attention
        actionable_steps: List of specific actions to take
        estimated_time: Human-readable time estimate
        can_batch_fix: Whether this can be fixed in bulk
        related_suggestion_ids: IDs of related suggestions
        metadata: Additional metadata
    """
    id: str
    title: str
    description: str
    priority: SuggestionPriority
    effort: SuggestionEffort
    impact_score: float
    frequency: int
    pattern: ErrorPattern
    affected_groups: List[str] = field(default_factory=list)
    affected_fields: List[str] = field(default_factory=list)
    actionable_steps: List[SuggestedAction] = field(default_factory=list)
    estimated_time: str = ""
    can_batch_fix: bool = False
    related_suggestion_ids: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=datetime.now)

    def add_action(
        self,
        action: str,
        priority: int = 1,
        details: Optional[str] = None,
        action_type: str = "manual",
        estimated_time: Optional[str] = None,
    ) -> None:
        """Add an actionable step to this suggestion."""
        self.actionable_steps.append(SuggestedAction(
            action=action,
            priority=priority,
            details=details,
            action_type=action_type,
            estimated_time=estimated_time,
        ))
        # Sort by priority
        self.actionable_steps.sort(key=lambda x: x.priority)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "id": self.id,
            "title": self.title,
            "description": self.description,
            "priority": self.priority.name,
            "priority_value": self.priority.value,
            "effort": self.effort.value,
            "impact_score": round(self.impact_score, 2),
            "frequency": self.frequency,
            "pattern": self.pattern.to_dict(),
            "affected_groups": self.affected_groups,
            "affected_fields": self.affected_fields,
            "actionable_steps": [a.to_dict() for a in self.actionable_steps],
            "estimated_time": self.estimated_time,
            "can_batch_fix": self.can_batch_fix,
            "related_suggestion_ids": self.related_suggestion_ids,
            "metadata": self.metadata,
            "created_at": self.created_at.isoformat(),
        }

    def __str__(self) -> str:
        """Human-readable representation."""
        return (
            f"[{self.priority.name}] {self.title}\n"
            f"  Impact: {self.impact_score:.1f}/100, Frequency: {self.frequency}\n"
            f"  Effort: {self.effort.value}, Time: {self.estimated_time}\n"
            f"  Groups: {', '.join(self.affected_groups) or 'All'}"
        )


class ErrorPatternAnalyzer:
    """
    Analyzes collected errors to identify patterns.

    Detects patterns such as:
    - Repeated errors in specific groups
    - Missing data patterns
    - Format/validation issues
    - API-related issues
    - Systemic problems affecting multiple groups
    """

    def __init__(self, min_frequency: int = 3):
        """
        Initialize the pattern analyzer.

        Args:
            min_frequency: Minimum occurrence count to consider a pattern
        """
        self.min_frequency = min_frequency
        self._patterns: List[ErrorPattern] = []

    def analyze(self, errors: List[CollectedError]) -> List[ErrorPattern]:
        """
        Analyze errors and identify patterns.

        Args:
            errors: List of collected errors to analyze

        Returns:
            List of detected error patterns
        """
        self._patterns = []

        if not errors:
            return self._patterns

        # Analyze by error type
        self._analyze_by_type(errors)

        # Analyze by category
        self._analyze_by_category(errors)

        # Analyze by group
        self._analyze_by_group(errors)

        # Analyze systemic patterns
        self._analyze_systemic(errors)

        # Analyze temporal patterns
        self._analyze_temporal(errors)

        return self._patterns

    def _analyze_by_type(self, errors: List[CollectedError]) -> None:
        """Analyze errors grouped by high-level error type."""
        type_groups: Dict[ErrorType, List[CollectedError]] = defaultdict(list)

        for error in errors:
            error_type = error.get_high_level_type()
            type_groups[error_type].append(error)

        # Map ErrorType to PatternType
        type_mapping = {
            ErrorType.MISSING_DATA: PatternType.MISSING_DATA,
            ErrorType.INVALID_FORMAT: PatternType.FORMAT_ERRORS,
            ErrorType.API_ERRORS: PatternType.API_FAILURES,
            ErrorType.PERMISSION_ISSUES: PatternType.PERMISSION_ISSUES,
            ErrorType.DATA_QUALITY_ISSUES: PatternType.NULL_VALUES,
        }

        for error_type, error_list in type_groups.items():
            if len(error_list) >= self.min_frequency:
                pattern_type = type_mapping.get(error_type, PatternType.REPEATED_ERRORS)
                pattern = self._create_pattern(pattern_type, error_list)
                pattern.context_data["error_type"] = error_type.value
                self._patterns.append(pattern)

    def _analyze_by_category(self, errors: List[CollectedError]) -> None:
        """Analyze errors grouped by category."""
        category_groups: Dict[ErrorCategory, List[CollectedError]] = defaultdict(list)

        for error in errors:
            category_groups[error.category].append(error)

        for category, error_list in category_groups.items():
            if len(error_list) >= self.min_frequency:
                # Determine pattern type from category
                pattern_type = self._category_to_pattern_type(category)
                pattern = self._create_pattern(pattern_type, error_list)
                pattern.context_data["category"] = category.value

                # Avoid duplicates with type-based patterns
                if not self._is_duplicate_pattern(pattern):
                    self._patterns.append(pattern)

    def _analyze_by_group(self, errors: List[CollectedError]) -> None:
        """Analyze errors to find group-specific patterns."""
        group_errors: Dict[str, List[CollectedError]] = defaultdict(list)

        for error in errors:
            group = error.context.get("group")
            if group:
                group_errors[group].append(error)

        # Check for groups with disproportionate error counts
        if not group_errors:
            return

        avg_errors = len(errors) / len(group_errors)

        for group, error_list in group_errors.items():
            # If a group has 2x the average errors, flag it
            if len(error_list) >= max(self.min_frequency, avg_errors * 2):
                pattern = self._create_pattern(PatternType.GROUP_SPECIFIC, error_list)
                pattern.context_data["primary_group"] = group
                pattern.context_data["error_ratio"] = len(error_list) / avg_errors
                self._patterns.append(pattern)

    def _analyze_systemic(self, errors: List[CollectedError]) -> None:
        """Analyze errors to find systemic issues affecting multiple groups."""
        # Group by message similarity (simplified - using exact message)
        message_groups: Dict[str, List[CollectedError]] = defaultdict(list)

        for error in errors:
            # Normalize message for grouping
            normalized_msg = error.message.lower()[:100]  # First 100 chars
            message_groups[normalized_msg].append(error)

        for msg_key, error_list in message_groups.items():
            if len(error_list) >= self.min_frequency:
                # Check if affects multiple groups
                groups = {e.context.get("group") for e in error_list if e.context.get("group")}
                if len(groups) >= 2:  # Systemic if affects 2+ groups
                    pattern = self._create_pattern(PatternType.SYSTEMIC, error_list)
                    pattern.context_data["message_pattern"] = msg_key[:50]
                    if not self._is_duplicate_pattern(pattern):
                        self._patterns.append(pattern)

    def _analyze_temporal(self, errors: List[CollectedError]) -> None:
        """Analyze temporal patterns in errors."""
        if not errors:
            return

        # Sort by timestamp
        sorted_errors = sorted(errors, key=lambda e: e.timestamp)

        # Check for burst patterns (many errors in short time)
        window_size = timedelta(minutes=5)
        burst_threshold = 10

        i = 0
        while i < len(sorted_errors):
            window_start = sorted_errors[i].timestamp
            window_end = window_start + window_size

            window_errors = []
            j = i
            while j < len(sorted_errors) and sorted_errors[j].timestamp <= window_end:
                window_errors.append(sorted_errors[j])
                j += 1

            if len(window_errors) >= burst_threshold:
                pattern = self._create_pattern(PatternType.REPEATED_ERRORS, window_errors)
                pattern.context_data["burst_detected"] = True
                pattern.context_data["burst_window_minutes"] = 5
                if not self._is_duplicate_pattern(pattern):
                    self._patterns.append(pattern)
                i = j  # Skip past the burst
            else:
                i += 1

    def _create_pattern(
        self,
        pattern_type: PatternType,
        errors: List[CollectedError]
    ) -> ErrorPattern:
        """Create an ErrorPattern from a list of errors."""
        groups = {e.context.get("group") for e in errors if e.context.get("group")}
        categories = {e.category for e in errors}

        severity_dist: Dict[str, int] = defaultdict(int)
        for error in errors:
            severity_dist[error.severity.value] += 1

        timestamps = [e.timestamp for e in errors]

        return ErrorPattern(
            pattern_type=pattern_type,
            frequency=len(errors),
            affected_groups=groups,
            affected_categories=categories,
            sample_errors=[e.error_id for e in errors[:5]],
            first_occurrence=min(timestamps) if timestamps else None,
            last_occurrence=max(timestamps) if timestamps else None,
            severity_distribution=dict(severity_dist),
        )

    def _category_to_pattern_type(self, category: ErrorCategory) -> PatternType:
        """Map an error category to a pattern type."""
        mapping = {
            ErrorCategory.MISSING_DATA: PatternType.MISSING_DATA,
            ErrorCategory.MISSING_REQUIRED: PatternType.MISSING_DATA,
            ErrorCategory.SHEET_NOT_FOUND: PatternType.MISSING_DATA,
            ErrorCategory.COLUMN_NOT_FOUND: PatternType.MISSING_DATA,
            ErrorCategory.ROW_NOT_FOUND: PatternType.MISSING_DATA,
            ErrorCategory.NULL_VALUE: PatternType.NULL_VALUES,
            ErrorCategory.INVALID_FORMAT: PatternType.FORMAT_ERRORS,
            ErrorCategory.DATE_PARSING: PatternType.FORMAT_ERRORS,
            ErrorCategory.API_ERROR: PatternType.API_FAILURES,
            ErrorCategory.RATE_LIMIT: PatternType.API_FAILURES,
            ErrorCategory.TIMEOUT: PatternType.API_FAILURES,
            ErrorCategory.CONNECTION: PatternType.API_FAILURES,
            ErrorCategory.PERMISSION: PatternType.PERMISSION_ISSUES,
            ErrorCategory.AUTHENTICATION: PatternType.PERMISSION_ISSUES,
            ErrorCategory.TOKEN_EXPIRED: PatternType.PERMISSION_ISSUES,
            ErrorCategory.TOKEN_INVALID: PatternType.PERMISSION_ISSUES,
        }
        return mapping.get(category, PatternType.REPEATED_ERRORS)

    def _is_duplicate_pattern(self, new_pattern: ErrorPattern) -> bool:
        """Check if a similar pattern already exists."""
        for existing in self._patterns:
            if (existing.pattern_type == new_pattern.pattern_type and
                existing.affected_groups == new_pattern.affected_groups and
                abs(existing.frequency - new_pattern.frequency) < 2):
                return True
        return False


class SuggestionPrioritizer:
    """
    Prioritizes cleanup suggestions based on impact, frequency, and effort.

    Uses a weighted scoring system to rank suggestions, considering:
    - Error frequency and severity
    - Number of affected groups
    - Estimated effort to fix
    - Whether batch fixing is possible
    """

    # Weights for scoring
    WEIGHT_IMPACT = 0.4
    WEIGHT_FREQUENCY = 0.3
    WEIGHT_EFFORT_INVERSE = 0.2
    WEIGHT_BATCH_BONUS = 0.1

    def prioritize(
        self,
        suggestions: List[CleanupSuggestion]
    ) -> List[CleanupSuggestion]:
        """
        Prioritize suggestions by impact and frequency.

        Args:
            suggestions: List of suggestions to prioritize

        Returns:
            Sorted list with highest priority first
        """
        scored_suggestions = []

        for suggestion in suggestions:
            score = self._calculate_priority_score(suggestion)
            scored_suggestions.append((score, suggestion))

        # Sort by score descending
        scored_suggestions.sort(key=lambda x: x[0], reverse=True)

        # Update priority levels based on position
        result = []
        for i, (score, suggestion) in enumerate(scored_suggestions):
            # Assign priority based on rank
            if i < len(suggestions) * 0.1:  # Top 10%
                suggestion.priority = SuggestionPriority.CRITICAL
            elif i < len(suggestions) * 0.3:  # Top 30%
                suggestion.priority = SuggestionPriority.HIGH
            elif i < len(suggestions) * 0.6:  # Top 60%
                suggestion.priority = SuggestionPriority.MEDIUM
            else:
                suggestion.priority = SuggestionPriority.LOW

            result.append(suggestion)

        return result

    def _calculate_priority_score(self, suggestion: CleanupSuggestion) -> float:
        """Calculate a priority score for a suggestion."""
        # Impact score (0-100)
        impact = suggestion.impact_score * self.WEIGHT_IMPACT

        # Frequency score (logarithmic scaling, 0-100)
        import math
        freq_score = min(100, math.log10(max(1, suggestion.frequency)) * 40)
        frequency = freq_score * self.WEIGHT_FREQUENCY

        # Inverse effort score (quick fixes score higher)
        effort_values = {
            SuggestionEffort.QUICK_FIX: 100,
            SuggestionEffort.MODERATE: 70,
            SuggestionEffort.SIGNIFICANT: 40,
            SuggestionEffort.MAJOR: 20,
        }
        effort_inverse = effort_values.get(suggestion.effort, 50) * self.WEIGHT_EFFORT_INVERSE

        # Batch fix bonus
        batch_bonus = (100 if suggestion.can_batch_fix else 0) * self.WEIGHT_BATCH_BONUS

        return impact + frequency + effort_inverse + batch_bonus


class CleanupSuggestionGenerator:
    """
    Main generator for cleanup suggestions.

    Coordinates error pattern analysis and suggestion generation to provide
    actionable recommendations for data cleanup.

    Usage:
        generator = CleanupSuggestionGenerator()
        suggestions = generator.generate_suggestions()
        top_suggestions = generator.get_top_suggestions(10)
    """

    def __init__(
        self,
        error_collector: Optional[ErrorCollector] = None,
        min_pattern_frequency: int = 3,
    ):
        """
        Initialize the suggestion generator.

        Args:
            error_collector: ErrorCollector to analyze (uses global if not provided)
            min_pattern_frequency: Minimum error count to detect a pattern
        """
        self._collector = error_collector
        self._analyzer = ErrorPatternAnalyzer(min_frequency=min_pattern_frequency)
        self._prioritizer = SuggestionPrioritizer()
        self._suggestions: List[CleanupSuggestion] = []
        self._lock = threading.RLock()
        self._suggestion_counter = 0
        self._last_generated: Optional[datetime] = None

    def _get_collector(self) -> ErrorCollector:
        """Get the error collector to use."""
        if self._collector is not None:
            return self._collector
        return get_global_collector()

    def _generate_suggestion_id(self) -> str:
        """Generate a unique suggestion ID."""
        self._suggestion_counter += 1
        return f"SUG-{datetime.now().strftime('%Y%m%d%H%M%S')}-{self._suggestion_counter:04d}"

    def generate_suggestions(
        self,
        force_refresh: bool = False
    ) -> List[CleanupSuggestion]:
        """
        Generate cleanup suggestions from collected errors.

        Args:
            force_refresh: If True, regenerate even if recently generated

        Returns:
            List of prioritized cleanup suggestions
        """
        with self._lock:
            # Check if we need to regenerate
            if (not force_refresh and
                self._suggestions and
                self._last_generated and
                datetime.now() - self._last_generated < timedelta(minutes=5)):
                return self._suggestions

            collector = self._get_collector()
            errors = collector.get_all_errors()

            if not errors:
                self._suggestions = []
                self._last_generated = datetime.now()
                return self._suggestions

            # Analyze patterns
            patterns = self._analyzer.analyze(errors)

            # Generate suggestions from patterns
            suggestions = []
            for pattern in patterns:
                suggestion = self._create_suggestion_from_pattern(pattern)
                if suggestion:
                    suggestions.append(suggestion)

            # Add category-specific suggestions
            category_suggestions = self._generate_category_suggestions(errors)
            suggestions.extend(category_suggestions)

            # Prioritize all suggestions
            self._suggestions = self._prioritizer.prioritize(suggestions)
            self._last_generated = datetime.now()

            logger.info(f"Generated {len(self._suggestions)} cleanup suggestions")
            return self._suggestions

    def _create_suggestion_from_pattern(
        self,
        pattern: ErrorPattern
    ) -> Optional[CleanupSuggestion]:
        """Create a cleanup suggestion from an error pattern."""
        # Get suggestion details based on pattern type
        details = self._get_pattern_suggestion_details(pattern)
        if not details:
            return None

        suggestion = CleanupSuggestion(
            id=self._generate_suggestion_id(),
            title=details["title"],
            description=details["description"],
            priority=SuggestionPriority.MEDIUM,  # Will be adjusted by prioritizer
            effort=details.get("effort", SuggestionEffort.MODERATE),
            impact_score=pattern.get_impact_score(),
            frequency=pattern.frequency,
            pattern=pattern,
            affected_groups=list(pattern.affected_groups),
            affected_fields=details.get("fields", []),
            estimated_time=details.get("time", "30 minutes - 1 hour"),
            can_batch_fix=details.get("batch_fix", False),
        )

        # Add actionable steps
        for i, step in enumerate(details.get("steps", []), 1):
            suggestion.add_action(
                action=step["action"],
                priority=i,
                details=step.get("details"),
                action_type=step.get("type", "manual"),
                estimated_time=step.get("time"),
            )

        return suggestion

    def _get_pattern_suggestion_details(
        self,
        pattern: ErrorPattern
    ) -> Optional[Dict[str, Any]]:
        """Get suggestion details for a pattern type."""
        suggestions = {
            PatternType.MISSING_DATA: {
                "title": "Address Missing Data Issues",
                "description": (
                    f"Detected {pattern.frequency} instances of missing data across "
                    f"{len(pattern.affected_groups)} groups. Missing data can cause "
                    "processing failures and incomplete reports."
                ),
                "effort": SuggestionEffort.MODERATE,
                "time": "1-2 hours",
                "batch_fix": True,
                "fields": ["various"],
                "steps": [
                    {
                        "action": "Identify rows with missing required fields",
                        "details": "Run a report to find all rows with null or empty values in critical columns",
                        "type": "automatic",
                        "time": "5 minutes"
                    },
                    {
                        "action": "Categorize missing data by type",
                        "details": "Group missing data by field name to prioritize cleanup",
                        "type": "manual",
                        "time": "15 minutes"
                    },
                    {
                        "action": "Populate missing values or mark as intentionally blank",
                        "details": "Update affected rows with correct data or add notes explaining why data is missing",
                        "type": "manual",
                        "time": "30-60 minutes"
                    },
                ],
            },
            PatternType.NULL_VALUES: {
                "title": "Clean Up Null/Empty Values",
                "description": (
                    f"Found {pattern.frequency} null or empty values in data fields. "
                    "These may indicate incomplete data entry or processing issues."
                ),
                "effort": SuggestionEffort.QUICK_FIX,
                "time": "15-30 minutes",
                "batch_fix": True,
                "fields": ["data fields"],
                "steps": [
                    {
                        "action": "Export affected rows to spreadsheet",
                        "details": "Create a list of rows with null values for review",
                        "type": "automatic",
                        "time": "2 minutes"
                    },
                    {
                        "action": "Review and correct null values",
                        "details": "Update cells with appropriate values or confirm they should remain empty",
                        "type": "manual",
                        "time": "10-20 minutes"
                    },
                ],
            },
            PatternType.FORMAT_ERRORS: {
                "title": "Fix Data Format Issues",
                "description": (
                    f"Detected {pattern.frequency} format-related errors. "
                    "Inconsistent formatting can cause parsing failures and data quality issues."
                ),
                "effort": SuggestionEffort.MODERATE,
                "time": "30 minutes - 1 hour",
                "batch_fix": True,
                "fields": ["date fields", "formatted fields"],
                "steps": [
                    {
                        "action": "Identify non-standard date formats",
                        "details": "Dates should be in YYYY-MM-DD format for consistent parsing",
                        "type": "automatic",
                        "time": "5 minutes"
                    },
                    {
                        "action": "Standardize date entries",
                        "details": "Convert all dates to the standard format",
                        "type": "manual",
                        "time": "20-30 minutes"
                    },
                    {
                        "action": "Validate formatting rules are documented",
                        "details": "Ensure team knows the expected formats for each field",
                        "type": "manual",
                        "time": "10 minutes"
                    },
                ],
            },
            PatternType.API_FAILURES: {
                "title": "Address API/Connection Issues",
                "description": (
                    f"Recorded {pattern.frequency} API-related errors. "
                    "These may indicate connectivity problems or rate limiting."
                ),
                "effort": SuggestionEffort.SIGNIFICANT,
                "time": "1-2 hours",
                "batch_fix": False,
                "fields": [],
                "steps": [
                    {
                        "action": "Check API token validity",
                        "details": "Verify the Smartsheet API token hasn't expired",
                        "type": "manual",
                        "time": "5 minutes"
                    },
                    {
                        "action": "Review rate limit status",
                        "details": "Check if requests are being throttled due to rate limits",
                        "type": "automatic",
                        "time": "2 minutes"
                    },
                    {
                        "action": "Implement retry logic if not present",
                        "details": "Ensure failed API calls are retried with exponential backoff",
                        "type": "escalate",
                        "time": "1-2 hours"
                    },
                ],
            },
            PatternType.PERMISSION_ISSUES: {
                "title": "Resolve Permission/Access Issues",
                "description": (
                    f"Found {pattern.frequency} permission-related errors. "
                    "Users may lack necessary access to certain sheets or resources."
                ),
                "effort": SuggestionEffort.MODERATE,
                "time": "30 minutes - 1 hour",
                "batch_fix": False,
                "fields": [],
                "steps": [
                    {
                        "action": "Verify API token permissions",
                        "details": "Ensure the token has adequate permissions for all required operations",
                        "type": "manual",
                        "time": "10 minutes"
                    },
                    {
                        "action": "Check sheet sharing settings",
                        "details": "Verify the service account has access to all required sheets",
                        "type": "manual",
                        "time": "15 minutes"
                    },
                    {
                        "action": "Contact sheet owners for missing access",
                        "details": "Request access to sheets that are not currently accessible",
                        "type": "escalate",
                        "time": "Variable"
                    },
                ],
            },
            PatternType.GROUP_SPECIFIC: {
                "title": f"Clean Up Group-Specific Issues",
                "description": (
                    f"Group '{pattern.context_data.get('primary_group', 'Unknown')}' has "
                    f"{pattern.frequency} errors, which is {pattern.context_data.get('error_ratio', 1):.1f}x "
                    "the average. This group may need focused attention."
                ),
                "effort": SuggestionEffort.MODERATE,
                "time": "1-2 hours",
                "batch_fix": True,
                "fields": [],
                "steps": [
                    {
                        "action": "Review group configuration",
                        "details": f"Check settings for group {pattern.context_data.get('primary_group', '')}",
                        "type": "manual",
                        "time": "15 minutes"
                    },
                    {
                        "action": "Audit data quality in affected group",
                        "details": "Review sample of rows for common issues",
                        "type": "manual",
                        "time": "30 minutes"
                    },
                    {
                        "action": "Apply targeted fixes",
                        "details": "Address the specific issues identified",
                        "type": "manual",
                        "time": "30-60 minutes"
                    },
                ],
            },
            PatternType.SYSTEMIC: {
                "title": "Address Systemic Data Issues",
                "description": (
                    f"Detected a systemic issue affecting {len(pattern.affected_groups)} groups "
                    f"with {pattern.frequency} occurrences. This may indicate a process or configuration problem."
                ),
                "effort": SuggestionEffort.SIGNIFICANT,
                "time": "2-4 hours",
                "batch_fix": True,
                "fields": [],
                "steps": [
                    {
                        "action": "Identify root cause",
                        "details": "Analyze error patterns to find common denominator",
                        "type": "manual",
                        "time": "30 minutes"
                    },
                    {
                        "action": "Review data entry process",
                        "details": "Check if the issue stems from data entry procedures",
                        "type": "manual",
                        "time": "30 minutes"
                    },
                    {
                        "action": "Implement systemic fix",
                        "details": "Apply fix to all affected groups",
                        "type": "manual",
                        "time": "1-2 hours"
                    },
                    {
                        "action": "Add validation to prevent recurrence",
                        "details": "Implement checks to catch this issue in the future",
                        "type": "escalate",
                        "time": "1 hour"
                    },
                ],
            },
            PatternType.REPEATED_ERRORS: {
                "title": "Address Repeated Errors",
                "description": (
                    f"Found {pattern.frequency} repeated errors. "
                    "These recurring issues should be investigated for root cause."
                ),
                "effort": SuggestionEffort.MODERATE,
                "time": "30 minutes - 1 hour",
                "batch_fix": False,
                "fields": [],
                "steps": [
                    {
                        "action": "Analyze error patterns",
                        "details": "Review sample errors to identify commonalities",
                        "type": "manual",
                        "time": "15 minutes"
                    },
                    {
                        "action": "Identify root cause",
                        "details": "Determine why errors are recurring",
                        "type": "manual",
                        "time": "15-30 minutes"
                    },
                    {
                        "action": "Implement fix and monitor",
                        "details": "Apply fix and verify errors stop recurring",
                        "type": "manual",
                        "time": "15-30 minutes"
                    },
                ],
            },
        }

        return suggestions.get(pattern.pattern_type)

    def _generate_category_suggestions(
        self,
        errors: List[CollectedError]
    ) -> List[CleanupSuggestion]:
        """Generate suggestions based on specific error categories."""
        suggestions = []

        # Group errors by category
        category_counts: Dict[ErrorCategory, int] = defaultdict(int)
        for error in errors:
            category_counts[error.category] += 1

        # Create specific suggestions for high-frequency categories
        for category, count in category_counts.items():
            if count >= 5:  # Only for significant counts
                suggestion = self._create_category_suggestion(category, count, errors)
                if suggestion:
                    suggestions.append(suggestion)

        return suggestions

    def _create_category_suggestion(
        self,
        category: ErrorCategory,
        count: int,
        errors: List[CollectedError]
    ) -> Optional[CleanupSuggestion]:
        """Create a suggestion for a specific error category."""
        # Category-specific suggestion templates
        templates = {
            ErrorCategory.DATE_PARSING: {
                "title": "Fix Date Parsing Errors",
                "description": f"Found {count} date parsing errors. Ensure dates are in a consistent format.",
                "effort": SuggestionEffort.QUICK_FIX,
                "time": "15-30 minutes",
                "batch_fix": True,
            },
            ErrorCategory.MISSING_REQUIRED: {
                "title": "Add Missing Required Fields",
                "description": f"Found {count} instances of missing required fields that need attention.",
                "effort": SuggestionEffort.MODERATE,
                "time": "30 minutes - 1 hour",
                "batch_fix": True,
            },
            ErrorCategory.CELL_ACCESS: {
                "title": "Resolve Cell Access Issues",
                "description": f"Detected {count} cell access errors. Review column mappings and data structure.",
                "effort": SuggestionEffort.MODERATE,
                "time": "30 minutes - 1 hour",
                "batch_fix": False,
            },
            ErrorCategory.RATE_LIMIT: {
                "title": "Optimize API Usage",
                "description": f"Hit rate limits {count} times. Consider spacing out API calls.",
                "effort": SuggestionEffort.SIGNIFICANT,
                "time": "1-2 hours",
                "batch_fix": False,
            },
        }

        template = templates.get(category)
        if not template:
            return None

        # Get affected errors for this category
        affected_errors = [e for e in errors if e.category == category]
        groups = {e.context.get("group") for e in affected_errors if e.context.get("group")}

        # Create a pattern for this category
        pattern = ErrorPattern(
            pattern_type=PatternType.REPEATED_ERRORS,
            frequency=count,
            affected_groups=groups,
            affected_categories={category},
            severity_distribution=defaultdict(int),
        )
        for e in affected_errors:
            pattern.severity_distribution[e.severity.value] += 1

        return CleanupSuggestion(
            id=self._generate_suggestion_id(),
            title=template["title"],
            description=template["description"],
            priority=SuggestionPriority.MEDIUM,
            effort=template["effort"],
            impact_score=pattern.get_impact_score(),
            frequency=count,
            pattern=pattern,
            affected_groups=list(groups),
            estimated_time=template["time"],
            can_batch_fix=template["batch_fix"],
        )

    def get_top_suggestions(
        self,
        limit: int = 10,
        sort_by: str = "impact"
    ) -> List[CleanupSuggestion]:
        """
        Get the top suggestions, optionally sorted by a specific criterion.

        Args:
            limit: Maximum number of suggestions to return
            sort_by: Sorting criterion ('impact', 'frequency', 'effort', 'priority')

        Returns:
            List of top suggestions
        """
        suggestions = self.generate_suggestions()

        if sort_by == "impact":
            sorted_suggestions = sorted(
                suggestions, key=lambda s: s.impact_score, reverse=True
            )
        elif sort_by == "frequency":
            sorted_suggestions = sorted(
                suggestions, key=lambda s: s.frequency, reverse=True
            )
        elif sort_by == "effort":
            # Lower effort first (quick wins)
            effort_order = {
                SuggestionEffort.QUICK_FIX: 0,
                SuggestionEffort.MODERATE: 1,
                SuggestionEffort.SIGNIFICANT: 2,
                SuggestionEffort.MAJOR: 3,
            }
            sorted_suggestions = sorted(
                suggestions, key=lambda s: effort_order.get(s.effort, 4)
            )
        elif sort_by == "priority":
            sorted_suggestions = sorted(
                suggestions, key=lambda s: s.priority.value
            )
        else:
            sorted_suggestions = suggestions

        return sorted_suggestions[:limit]

    def get_suggestions_by_group(self, group: str) -> List[CleanupSuggestion]:
        """
        Get suggestions that affect a specific group.

        Args:
            group: Group code to filter by

        Returns:
            List of suggestions affecting the specified group
        """
        suggestions = self.generate_suggestions()
        return [s for s in suggestions if group in s.affected_groups]

    def get_suggestions_by_priority(
        self,
        priority: SuggestionPriority
    ) -> List[CleanupSuggestion]:
        """
        Get suggestions of a specific priority level.

        Args:
            priority: Priority level to filter by

        Returns:
            List of suggestions with the specified priority
        """
        suggestions = self.generate_suggestions()
        return [s for s in suggestions if s.priority == priority]

    def get_quick_wins(self, limit: int = 5) -> List[CleanupSuggestion]:
        """
        Get quick-win suggestions (high impact, low effort).

        Args:
            limit: Maximum number of suggestions to return

        Returns:
            List of quick-win suggestions
        """
        suggestions = self.generate_suggestions()
        quick_fixes = [
            s for s in suggestions
            if s.effort in (SuggestionEffort.QUICK_FIX, SuggestionEffort.MODERATE)
        ]
        # Sort by impact
        quick_fixes.sort(key=lambda s: s.impact_score, reverse=True)
        return quick_fixes[:limit]

    def get_summary(self) -> str:
        """Get a human-readable summary of suggestions."""
        suggestions = self.generate_suggestions()

        if not suggestions:
            return "No cleanup suggestions available. The data appears clean!"

        lines = [
            "=" * 60,
            "CLEANUP SUGGESTIONS SUMMARY",
            "=" * 60,
            f"Total suggestions: {len(suggestions)}",
            "",
        ]

        # By priority
        priority_counts = defaultdict(int)
        for s in suggestions:
            priority_counts[s.priority.name] += 1

        lines.append("By Priority:")
        for priority in SuggestionPriority:
            if priority_counts[priority.name] > 0:
                lines.append(f"  {priority.name}: {priority_counts[priority.name]}")

        # By effort
        effort_counts = defaultdict(int)
        for s in suggestions:
            effort_counts[s.effort.value] += 1

        lines.append("\nBy Effort:")
        for effort in SuggestionEffort:
            if effort_counts[effort.value] > 0:
                lines.append(f"  {effort.value}: {effort_counts[effort.value]}")

        # Top suggestions
        lines.append("\nTop 5 Suggestions by Impact:")
        for i, s in enumerate(self.get_top_suggestions(5), 1):
            lines.append(f"  {i}. [{s.priority.name}] {s.title}")
            lines.append(f"     Impact: {s.impact_score:.1f}, Frequency: {s.frequency}")

        lines.append("=" * 60)
        return "\n".join(lines)

    def export_for_report(self) -> Dict[str, Any]:
        """Export suggestions for report generation."""
        suggestions = self.generate_suggestions()

        return {
            "suggestions": [s.to_dict() for s in suggestions],
            "summary": {
                "total_count": len(suggestions),
                "by_priority": {
                    p.name: len([s for s in suggestions if s.priority == p])
                    for p in SuggestionPriority
                },
                "by_effort": {
                    e.value: len([s for s in suggestions if s.effort == e])
                    for e in SuggestionEffort
                },
                "total_affected_errors": sum(s.frequency for s in suggestions),
            },
            "top_suggestions": [s.to_dict() for s in self.get_top_suggestions(10)],
            "quick_wins": [s.to_dict() for s in self.get_quick_wins(5)],
            "generated_at": datetime.now().isoformat(),
        }

    def clear_cache(self) -> None:
        """Clear cached suggestions to force regeneration."""
        with self._lock:
            self._suggestions = []
            self._last_generated = None


# Global generator instance
_global_generator: Optional[CleanupSuggestionGenerator] = None
_global_generator_lock = threading.Lock()


def get_global_suggestion_generator() -> CleanupSuggestionGenerator:
    """Get the global cleanup suggestion generator (creates one if needed)."""
    global _global_generator
    with _global_generator_lock:
        if _global_generator is None:
            _global_generator = CleanupSuggestionGenerator()
        return _global_generator


def reset_global_suggestion_generator() -> None:
    """Reset the global suggestion generator."""
    global _global_generator
    with _global_generator_lock:
        _global_generator = CleanupSuggestionGenerator()


def generate_cleanup_suggestions(
    limit: int = 10,
    sort_by: str = "impact"
) -> List[CleanupSuggestion]:
    """
    Convenience function to generate cleanup suggestions.

    Args:
        limit: Maximum number of suggestions to return
        sort_by: Sorting criterion ('impact', 'frequency', 'effort', 'priority')

    Returns:
        List of cleanup suggestions
    """
    return get_global_suggestion_generator().get_top_suggestions(limit, sort_by)


def get_cleanup_summary() -> str:
    """
    Convenience function to get a cleanup summary.

    Returns:
        Human-readable summary of cleanup suggestions
    """
    return get_global_suggestion_generator().get_summary()
