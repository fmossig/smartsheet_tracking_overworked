"""
Unicode Utilities Module for Smartsheet Tracker

Provides utilities for handling unicode characters in text fields, including:
- Unicode normalization (NFC/NFKC) for consistent text processing
- Safe text truncation that respects grapheme clusters
- Character encoding normalization for PDF generation
- Zero-width and invisible character removal

This module ensures consistent handling of unicode characters in user names,
comments, and other text fields throughout the application.

Usage:
    from unicode_utilities import (
        normalize_unicode,
        safe_truncate,
        prepare_for_pdf,
        remove_invisible_chars,
    )

    # Normalize unicode text
    normalized = normalize_unicode("Müller")  # Consistent representation

    # Safe truncation for display
    truncated = safe_truncate("José García", max_length=10)  # "José Garcí..."

    # Prepare text for PDF rendering
    pdf_safe = prepare_for_pdf("Schröder")  # Safe for ReportLab
"""

import logging
import unicodedata
import re
from typing import Optional

# Get logger for this module
logger = logging.getLogger(__name__)


def normalize_unicode(text: Optional[str], form: str = "NFC") -> str:
    """
    Normalize unicode text to a consistent form.

    This function normalizes unicode text using Unicode Normalization Forms,
    which ensures that characters like "é" are represented consistently whether
    they're a single precomposed character or a combining sequence (e + ´).

    Args:
        text: The text to normalize. If None or not a string, returns empty string.
        form: Unicode normalization form. Options:
            - "NFC" (default): Canonical Decomposition, followed by Canonical Composition
                Best for general text - composes characters where possible.
            - "NFD": Canonical Decomposition
                Decomposes characters into base + combining marks.
            - "NFKC": Compatibility Decomposition, followed by Canonical Composition
                Like NFC but also normalizes compatibility characters (e.g., ﬁ → fi).
            - "NFKD": Compatibility Decomposition
                Like NFD but also normalizes compatibility characters.

    Returns:
        Normalized unicode string, or empty string if input is None/invalid.

    Example:
        >>> normalize_unicode("café")  # e + combining acute
        'café'  # Single precomposed é
        >>> normalize_unicode("Müller")
        'Müller'
        >>> normalize_unicode(None)
        ''
    """
    if text is None:
        return ""

    if not isinstance(text, str):
        try:
            text = str(text)
        except (ValueError, TypeError):
            logger.warning(f"Could not convert value to string for unicode normalization: {type(text)}")
            return ""

    try:
        normalized = unicodedata.normalize(form, text)
        return normalized
    except (ValueError, TypeError) as e:
        logger.warning(f"Unicode normalization failed for text: {e}")
        return text


def remove_invisible_chars(text: Optional[str]) -> str:
    """
    Remove invisible and zero-width characters from text.

    This removes characters that can cause display issues or invisible differences
    between otherwise identical-looking strings. Preserves normal whitespace like
    spaces and newlines.

    Characters removed include:
    - Zero-width space (U+200B)
    - Zero-width non-joiner (U+200C)
    - Zero-width joiner (U+200D)
    - Left-to-right mark (U+200E)
    - Right-to-left mark (U+200F)
    - Word joiner (U+2060)
    - Zero-width no-break space / BOM (U+FEFF)
    - Soft hyphen (U+00AD)
    - Various invisible format characters

    Args:
        text: The text to clean. If None, returns empty string.

    Returns:
        Text with invisible characters removed.

    Example:
        >>> remove_invisible_chars("hello\\u200bworld")  # Zero-width space
        'helloworld'
    """
    if text is None:
        return ""

    if not isinstance(text, str):
        return str(text) if text else ""

    # Pattern to match invisible/zero-width characters
    # Excludes normal whitespace (space, tab, newline, etc.)
    invisible_pattern = re.compile(
        r'[\u200b-\u200f\u2060-\u206f\ufeff\u00ad\u034f\u061c\u180e]'
    )

    return invisible_pattern.sub('', text)


def safe_truncate(text: Optional[str], max_length: int, suffix: str = "…") -> str:
    """
    Safely truncate text while respecting unicode grapheme clusters.

    Unlike simple string slicing, this function ensures we don't break in the
    middle of a multi-codepoint grapheme cluster (like an emoji with skin tone
    or a character with combining marks).

    Args:
        text: The text to truncate. If None, returns empty string.
        max_length: Maximum length of the result (including suffix if added).
        suffix: String to append if truncation occurs. Default is "…".

    Returns:
        Truncated text, or original text if shorter than max_length.

    Example:
        >>> safe_truncate("Müller", 5)
        'Müll…'
        >>> safe_truncate("abc", 10)
        'abc'
        >>> safe_truncate("José García", 10)
        'José Garc…'
    """
    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    # Normalize first for consistent length calculation
    text = normalize_unicode(text)

    if len(text) <= max_length:
        return text

    if max_length <= len(suffix):
        return suffix[:max_length] if max_length > 0 else ""

    # Calculate available length for actual content
    target_length = max_length - len(suffix)

    # Use grapheme segmentation to find safe truncation point
    # We iterate through grapheme clusters and build the result
    result = []
    current_length = 0

    # Iterate through grapheme clusters using unicodedata
    # A grapheme cluster is a user-perceived character
    i = 0
    while i < len(text):
        # Find the end of the current grapheme cluster
        # Most grapheme clusters are 1 codepoint, but some are multiple
        # (combining marks, emoji sequences, etc.)
        cluster_end = i + 1
        while cluster_end < len(text):
            # Check if next character is a combining character or variation selector
            cat = unicodedata.category(text[cluster_end])
            # M* = Mark (combining), Cf = Format, includes variation selectors
            if cat.startswith('M') or cat == 'Cf':
                cluster_end += 1
            else:
                break

        cluster = text[i:cluster_end]
        new_length = current_length + len(cluster)

        if new_length > target_length:
            break

        result.append(cluster)
        current_length = new_length
        i = cluster_end

    return ''.join(result) + suffix


def prepare_for_pdf(text: Optional[str]) -> str:
    """
    Prepare text for safe PDF rendering with ReportLab.

    This function:
    1. Normalizes unicode to NFC form
    2. Removes invisible characters that could cause issues
    3. Handles special characters that might cause ReportLab issues

    Note: This assumes the PDF is using a font that supports the unicode
    characters in the text. Standard fonts like Helvetica only support
    basic Latin characters. For full unicode support, use a unicode font
    like DejaVu or register appropriate CID fonts.

    Args:
        text: The text to prepare. If None, returns empty string.

    Returns:
        Text prepared for PDF rendering.

    Example:
        >>> prepare_for_pdf("Müller")
        'Müller'
        >>> prepare_for_pdf("hello\\u200bworld")
        'helloworld'
    """
    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    # First normalize unicode
    text = normalize_unicode(text, form="NFC")

    # Remove invisible characters
    text = remove_invisible_chars(text)

    # Replace problematic characters for PDF rendering
    # These are characters that might not render correctly in standard fonts
    replacements = {
        '\u2028': '\n',  # Line separator → newline
        '\u2029': '\n\n',  # Paragraph separator → double newline
    }

    for old, new in replacements.items():
        text = text.replace(old, new)

    return text


def normalize_for_comparison(text: Optional[str]) -> str:
    """
    Normalize text for consistent comparison operations.

    This function normalizes text to enable accurate string comparisons
    that aren't affected by different unicode representations of the
    same visual character.

    Uses NFKC normalization which is more aggressive than NFC - it also
    normalizes compatibility characters like ligatures and width variants.

    Args:
        text: The text to normalize. If None, returns empty string.

    Returns:
        Text normalized for comparison (lowercase, NFKC).

    Example:
        >>> normalize_for_comparison("Müller")
        'müller'
        >>> normalize_for_comparison("MÜLLER")
        'müller'
        >>> text1 = "café"  # Single character é
        >>> text2 = "café"  # e + combining acute
        >>> normalize_for_comparison(text1) == normalize_for_comparison(text2)
        True
    """
    if text is None:
        return ""

    if not isinstance(text, str):
        text = str(text)

    # Use NFKC for compatibility normalization
    normalized = normalize_unicode(text, form="NFKC")

    # Convert to lowercase for case-insensitive comparison
    return normalized.casefold()


def is_valid_unicode(text: Optional[str]) -> bool:
    """
    Check if text contains only valid unicode characters.

    This checks for common encoding issues like replacement characters
    that indicate failed encoding/decoding operations.

    Args:
        text: The text to check. None is considered valid (empty).

    Returns:
        True if text is valid unicode without replacement characters.

    Example:
        >>> is_valid_unicode("Hello")
        True
        >>> is_valid_unicode("Hello\\ufffd")  # Contains replacement character
        False
    """
    if text is None:
        return True

    if not isinstance(text, str):
        return False

    # Check for replacement character indicating encoding issues
    if '\ufffd' in text:
        return False

    # Check for other common encoding error indicators
    try:
        # Try to encode as UTF-8 and decode back
        text.encode('utf-8').decode('utf-8')
        return True
    except (UnicodeEncodeError, UnicodeDecodeError):
        return False


def get_text_display_width(text: Optional[str]) -> int:
    """
    Calculate the approximate display width of text.

    This accounts for the fact that East Asian characters typically
    display as "double-width" in fixed-width fonts, while combining
    characters have zero display width.

    Args:
        text: The text to measure. If None, returns 0.

    Returns:
        Approximate display width in character cells.

    Example:
        >>> get_text_display_width("Hello")
        5
        >>> get_text_display_width("日本語")  # 3 double-width chars
        6
    """
    if text is None or not isinstance(text, str):
        return 0

    width = 0
    for char in text:
        # Get East Asian Width property
        ea_width = unicodedata.east_asian_width(char)
        if ea_width in ('F', 'W'):
            # Fullwidth or Wide - counts as 2
            width += 2
        elif ea_width == 'N' or unicodedata.category(char).startswith('M'):
            # Narrow or combining mark
            category = unicodedata.category(char)
            if category.startswith('M'):
                # Combining character - no width
                width += 0
            else:
                width += 1
        else:
            # Default single width
            width += 1

    return width
