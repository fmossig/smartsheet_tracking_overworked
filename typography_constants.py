"""
Typography Constants for Smartsheet Tracker Reports

This module provides consistent typography constants for font families, sizes,
and weights across all report sections. It ensures a clear readability hierarchy
and consistent styling throughout the PDF reports.

Typography System Overview:
- Font Families: Helvetica family (standard PDF fonts)
- Font Sizes: 8-level scale from caption to display
- Font Weights: Regular, Bold, Oblique (Italic)
- Line Heights: Proportional leading values
- Spacing: Consistent paragraph spacing

This module integrates with:
- constants.py: Uses existing FontSizes as base
- accessibility_colors.py: Uses text color palettes
- ReportLab: Provides ParagraphStyle factory functions

Usage:
    from typography_constants import (
        FontFamily, FontWeight, Typography,
        get_heading_style, get_body_style, get_caption_style
    )

Reference: Based on material design typography scale principles
"""

from typing import Dict, Optional, Any
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib import colors
from reportlab.lib.units import mm

from accessibility_colors import (
    ACCESSIBLE_HEADER_COLORS,
    ACCESSIBLE_TEXT_COLORS,
)


# ============================================================================
# FONT FAMILY CONSTANTS
# ============================================================================

class FontFamily:
    """Standard font families for ReportLab PDF generation.

    These are the built-in PDF fonts available in ReportLab without
    requiring font file embedding.
    """
    # Primary font family (sans-serif)
    SANS_SERIF = "Helvetica"
    SANS_SERIF_BOLD = "Helvetica-Bold"
    SANS_SERIF_ITALIC = "Helvetica-Oblique"
    SANS_SERIF_BOLD_ITALIC = "Helvetica-BoldOblique"

    # Monospace font family (for code/data)
    MONOSPACE = "Courier"
    MONOSPACE_BOLD = "Courier-Bold"
    MONOSPACE_ITALIC = "Courier-Oblique"
    MONOSPACE_BOLD_ITALIC = "Courier-BoldOblique"

    # Serif font family (optional, for formal documents)
    SERIF = "Times-Roman"
    SERIF_BOLD = "Times-Bold"
    SERIF_ITALIC = "Times-Italic"
    SERIF_BOLD_ITALIC = "Times-BoldItalic"

    # Default font family for the report
    DEFAULT = SANS_SERIF
    DEFAULT_BOLD = SANS_SERIF_BOLD
    DEFAULT_ITALIC = SANS_SERIF_ITALIC


class FontWeight:
    """Font weight constants for typography hierarchy.

    Maps semantic weight names to actual font names for ReportLab.
    """
    REGULAR = FontFamily.SANS_SERIF
    BOLD = FontFamily.SANS_SERIF_BOLD
    ITALIC = FontFamily.SANS_SERIF_ITALIC
    BOLD_ITALIC = FontFamily.SANS_SERIF_BOLD_ITALIC


# ============================================================================
# FONT SIZE SCALE
# ============================================================================

class FontSize:
    """Font size scale for consistent typography hierarchy.

    Based on a modular scale (approximately 1.2 ratio) for visual harmony.
    All sizes are in points (pt).

    Hierarchy:
    - DISPLAY: Large promotional/hero text (32pt)
    - H1: Main page/section titles (24pt)
    - H2: Major section headers (18pt)
    - H3: Subsection headers (14pt)
    - H4: Minor section headers (12pt)
    - BODY_LARGE: Emphasized body text (12pt)
    - BODY: Standard body text (11pt)
    - BODY_SMALL: Secondary body text (10pt)
    - CAPTION: Captions, footnotes (9pt)
    - SMALL: Fine print, annotations (8pt)
    - TINY: Minimal text (7pt)
    """
    # Display/Hero text (cover page title)
    DISPLAY = 32

    # Heading hierarchy
    H1 = 24
    H2 = 18
    H3 = 14
    H4 = 12

    # Body text hierarchy
    BODY_LARGE = 12
    BODY = 11
    BODY_SMALL = 10

    # Caption and small text
    CAPTION = 9
    SMALL = 8
    TINY = 7

    # Table-specific sizes
    TABLE_HEADER = 10
    TABLE_BODY = 9
    TABLE_FOOTER = 8


# ============================================================================
# LINE HEIGHT (LEADING) CONSTANTS
# ============================================================================

class LineHeight:
    """Line height (leading) values for proper text spacing.

    Leading is the distance from one baseline to the next.
    Values are in points. As a rule of thumb, leading should be
    120-150% of the font size for optimal readability.
    """
    # Tight leading (120% of font size) - for headers
    TIGHT_RATIO = 1.2

    # Normal leading (140% of font size) - for body text
    NORMAL_RATIO = 1.4

    # Relaxed leading (160% of font size) - for improved readability
    RELAXED_RATIO = 1.6

    # Pre-calculated leading values for common sizes
    DISPLAY = 38      # 32pt * 1.2
    H1 = 29           # 24pt * 1.2
    H2 = 22           # 18pt * 1.2
    H3 = 17           # 14pt * 1.2
    H4 = 14           # 12pt * 1.2
    BODY_LARGE = 17   # 12pt * 1.4
    BODY = 15         # 11pt * 1.4
    BODY_SMALL = 14   # 10pt * 1.4
    CAPTION = 13      # 9pt * 1.4
    SMALL = 11        # 8pt * 1.4
    TINY = 10         # 7pt * 1.4

    # Table-specific leading (tighter for compact tables)
    TABLE_HEADER = 12
    TABLE_BODY = 11
    TABLE_FOOTER = 10

    @staticmethod
    def calculate(font_size: float, ratio: float = 1.4) -> float:
        """Calculate leading for a given font size and ratio.

        Args:
            font_size: Font size in points
            ratio: Leading ratio (default 1.4 for body text)

        Returns:
            Calculated leading value in points
        """
        return round(font_size * ratio)


# ============================================================================
# PARAGRAPH SPACING CONSTANTS
# ============================================================================

class ParagraphSpacing:
    """Paragraph spacing constants in millimeters.

    Defines consistent vertical spacing before and after paragraphs
    to create visual rhythm and clear separation between content blocks.
    """
    # Space before headings (pull away from previous content)
    BEFORE_DISPLAY = 30 * mm
    BEFORE_H1 = 20 * mm
    BEFORE_H2 = 15 * mm
    BEFORE_H3 = 12 * mm
    BEFORE_H4 = 10 * mm

    # Space after headings (stay close to related content)
    AFTER_DISPLAY = 15 * mm
    AFTER_H1 = 10 * mm
    AFTER_H2 = 8 * mm
    AFTER_H3 = 6 * mm
    AFTER_H4 = 4 * mm

    # Space around body paragraphs
    BEFORE_BODY = 4 * mm
    AFTER_BODY = 4 * mm

    # Space around captions and small text
    BEFORE_CAPTION = 2 * mm
    AFTER_CAPTION = 2 * mm

    # Compact spacing for lists and dense content
    COMPACT = 2 * mm

    # Section divider spacing
    SECTION_BREAK = 25 * mm


# ============================================================================
# TEXT ALIGNMENT CONSTANTS
# ============================================================================

class TextAlignment:
    """Text alignment constants for ReportLab.

    ReportLab uses numeric values for alignment:
    - 0: Left
    - 1: Center
    - 2: Right
    - 4: Justify
    """
    LEFT = 0
    CENTER = 1
    RIGHT = 2
    JUSTIFY = 4


# ============================================================================
# TYPOGRAPHY PRESETS
# ============================================================================

class Typography:
    """Complete typography presets combining all settings.

    Each preset includes font name, size, leading, and color information.
    Use these presets to ensure consistent styling across the report.
    """

    # Display/Hero styles (cover page)
    DISPLAY = {
        "fontName": FontFamily.SANS_SERIF_BOLD,
        "fontSize": FontSize.DISPLAY,
        "leading": LineHeight.DISPLAY,
        "textColor": ACCESSIBLE_HEADER_COLORS["title"],
    }

    # Heading styles
    H1 = {
        "fontName": FontFamily.SANS_SERIF_BOLD,
        "fontSize": FontSize.H1,
        "leading": LineHeight.H1,
        "textColor": ACCESSIBLE_HEADER_COLORS["section"],
    }

    H2 = {
        "fontName": FontFamily.SANS_SERIF_BOLD,
        "fontSize": FontSize.H2,
        "leading": LineHeight.H2,
        "textColor": ACCESSIBLE_HEADER_COLORS["subsection"],
    }

    H3 = {
        "fontName": FontFamily.SANS_SERIF_BOLD,
        "fontSize": FontSize.H3,
        "leading": LineHeight.H3,
        "textColor": ACCESSIBLE_HEADER_COLORS["minor_heading"],
    }

    H4 = {
        "fontName": FontFamily.SANS_SERIF_BOLD,
        "fontSize": FontSize.H4,
        "leading": LineHeight.H4,
        "textColor": ACCESSIBLE_TEXT_COLORS["primary"],
    }

    # Body text styles
    BODY_LARGE = {
        "fontName": FontFamily.SANS_SERIF,
        "fontSize": FontSize.BODY_LARGE,
        "leading": LineHeight.BODY_LARGE,
        "textColor": ACCESSIBLE_TEXT_COLORS["primary"],
    }

    BODY = {
        "fontName": FontFamily.SANS_SERIF,
        "fontSize": FontSize.BODY,
        "leading": LineHeight.BODY,
        "textColor": ACCESSIBLE_TEXT_COLORS["primary"],
    }

    BODY_SMALL = {
        "fontName": FontFamily.SANS_SERIF,
        "fontSize": FontSize.BODY_SMALL,
        "leading": LineHeight.BODY_SMALL,
        "textColor": ACCESSIBLE_TEXT_COLORS["primary"],
    }

    BODY_EMPHASIS = {
        "fontName": FontFamily.SANS_SERIF_BOLD,
        "fontSize": FontSize.BODY,
        "leading": LineHeight.BODY,
        "textColor": ACCESSIBLE_TEXT_COLORS["emphasis"],
    }

    BODY_SECONDARY = {
        "fontName": FontFamily.SANS_SERIF,
        "fontSize": FontSize.BODY_SMALL,
        "leading": LineHeight.BODY_SMALL,
        "textColor": ACCESSIBLE_TEXT_COLORS["secondary"],
    }

    # Caption and small text styles
    CAPTION = {
        "fontName": FontFamily.SANS_SERIF,
        "fontSize": FontSize.CAPTION,
        "leading": LineHeight.CAPTION,
        "textColor": ACCESSIBLE_TEXT_COLORS["secondary"],
    }

    CAPTION_EMPHASIS = {
        "fontName": FontFamily.SANS_SERIF_BOLD,
        "fontSize": FontSize.CAPTION,
        "leading": LineHeight.CAPTION,
        "textColor": ACCESSIBLE_TEXT_COLORS["primary"],
    }

    FOOTNOTE = {
        "fontName": FontFamily.SANS_SERIF_ITALIC,
        "fontSize": FontSize.SMALL,
        "leading": LineHeight.SMALL,
        "textColor": ACCESSIBLE_TEXT_COLORS["muted"],
    }

    # Table styles
    TABLE_HEADER = {
        "fontName": FontFamily.SANS_SERIF_BOLD,
        "fontSize": FontSize.TABLE_HEADER,
        "leading": LineHeight.TABLE_HEADER,
        "textColor": ACCESSIBLE_TEXT_COLORS["white"],
    }

    TABLE_BODY = {
        "fontName": FontFamily.SANS_SERIF,
        "fontSize": FontSize.TABLE_BODY,
        "leading": LineHeight.TABLE_BODY,
        "textColor": ACCESSIBLE_TEXT_COLORS["primary"],
    }

    TABLE_FOOTER = {
        "fontName": FontFamily.SANS_SERIF_ITALIC,
        "fontSize": FontSize.TABLE_FOOTER,
        "leading": LineHeight.TABLE_FOOTER,
        "textColor": ACCESSIBLE_TEXT_COLORS["secondary"],
    }

    # Special styles
    MONOSPACE = {
        "fontName": FontFamily.MONOSPACE,
        "fontSize": FontSize.BODY_SMALL,
        "leading": LineHeight.BODY_SMALL,
        "textColor": ACCESSIBLE_TEXT_COLORS["primary"],
    }

    WARNING = {
        "fontName": FontFamily.SANS_SERIF_BOLD,
        "fontSize": FontSize.BODY,
        "leading": LineHeight.BODY,
        "textColor": ACCESSIBLE_TEXT_COLORS["warning"],
    }

    ERROR = {
        "fontName": FontFamily.SANS_SERIF_BOLD,
        "fontSize": FontSize.BODY,
        "leading": LineHeight.BODY,
        "textColor": ACCESSIBLE_TEXT_COLORS["error"],
    }

    SUCCESS = {
        "fontName": FontFamily.SANS_SERIF_BOLD,
        "fontSize": FontSize.BODY,
        "leading": LineHeight.BODY,
        "textColor": ACCESSIBLE_TEXT_COLORS["success"],
    }

    INFO = {
        "fontName": FontFamily.SANS_SERIF,
        "fontSize": FontSize.BODY,
        "leading": LineHeight.BODY,
        "textColor": ACCESSIBLE_TEXT_COLORS["info"],
    }


# ============================================================================
# PARAGRAPH STYLE FACTORY FUNCTIONS
# ============================================================================

# Cache for base styles to avoid repeated getSampleStyleSheet calls
_cached_styles = None


def _get_base_styles():
    """Get cached base styles from ReportLab."""
    global _cached_styles
    if _cached_styles is None:
        _cached_styles = getSampleStyleSheet()
    return _cached_styles


def create_paragraph_style(
    name: str,
    preset: Dict[str, Any],
    alignment: int = TextAlignment.LEFT,
    space_before: float = 0,
    space_after: float = 0,
    first_line_indent: float = 0,
    left_indent: float = 0,
    right_indent: float = 0,
    parent_style: str = "Normal",
    **kwargs
) -> ParagraphStyle:
    """Create a ParagraphStyle from a Typography preset.

    Args:
        name: Unique name for the style
        preset: Typography preset dictionary (e.g., Typography.H1)
        alignment: Text alignment (use TextAlignment constants)
        space_before: Space before paragraph in points
        space_after: Space after paragraph in points
        first_line_indent: First line indentation in points
        left_indent: Left indentation in points
        right_indent: Right indentation in points
        parent_style: Name of parent style from getSampleStyleSheet
        **kwargs: Additional ParagraphStyle parameters

    Returns:
        Configured ParagraphStyle object
    """
    styles = _get_base_styles()

    # Convert hex color to ReportLab color if needed
    text_color = preset.get("textColor")
    if isinstance(text_color, str):
        text_color = colors.HexColor(text_color)

    return ParagraphStyle(
        name,
        parent=styles[parent_style],
        fontName=preset.get("fontName", FontFamily.DEFAULT),
        fontSize=preset.get("fontSize", FontSize.BODY),
        leading=preset.get("leading", LineHeight.BODY),
        textColor=text_color,
        alignment=alignment,
        spaceBefore=space_before,
        spaceAfter=space_after,
        firstLineIndent=first_line_indent,
        leftIndent=left_indent,
        rightIndent=right_indent,
        **kwargs
    )


# ============================================================================
# CONVENIENCE STYLE FACTORY FUNCTIONS
# ============================================================================

def get_display_style(
    name: str = "DisplayTitle",
    alignment: int = TextAlignment.CENTER,
    space_before: float = ParagraphSpacing.BEFORE_DISPLAY,
    space_after: float = ParagraphSpacing.AFTER_DISPLAY,
    **kwargs
) -> ParagraphStyle:
    """Create a display/hero title style (for cover pages)."""
    return create_paragraph_style(
        name=name,
        preset=Typography.DISPLAY,
        alignment=alignment,
        space_before=space_before,
        space_after=space_after,
        parent_style="Title",
        **kwargs
    )


def get_h1_style(
    name: str = "Heading1",
    alignment: int = TextAlignment.LEFT,
    space_before: float = ParagraphSpacing.BEFORE_H1,
    space_after: float = ParagraphSpacing.AFTER_H1,
    **kwargs
) -> ParagraphStyle:
    """Create an H1 heading style (main section headers)."""
    return create_paragraph_style(
        name=name,
        preset=Typography.H1,
        alignment=alignment,
        space_before=space_before,
        space_after=space_after,
        parent_style="Heading1",
        **kwargs
    )


def get_h2_style(
    name: str = "Heading2",
    alignment: int = TextAlignment.LEFT,
    space_before: float = ParagraphSpacing.BEFORE_H2,
    space_after: float = ParagraphSpacing.AFTER_H2,
    **kwargs
) -> ParagraphStyle:
    """Create an H2 heading style (major subsections)."""
    return create_paragraph_style(
        name=name,
        preset=Typography.H2,
        alignment=alignment,
        space_before=space_before,
        space_after=space_after,
        parent_style="Heading2",
        **kwargs
    )


def get_h3_style(
    name: str = "Heading3",
    alignment: int = TextAlignment.LEFT,
    space_before: float = ParagraphSpacing.BEFORE_H3,
    space_after: float = ParagraphSpacing.AFTER_H3,
    **kwargs
) -> ParagraphStyle:
    """Create an H3 heading style (minor subsections)."""
    return create_paragraph_style(
        name=name,
        preset=Typography.H3,
        alignment=alignment,
        space_before=space_before,
        space_after=space_after,
        parent_style="Heading3",
        **kwargs
    )


def get_h4_style(
    name: str = "Heading4",
    alignment: int = TextAlignment.LEFT,
    space_before: float = ParagraphSpacing.BEFORE_H4,
    space_after: float = ParagraphSpacing.AFTER_H4,
    **kwargs
) -> ParagraphStyle:
    """Create an H4 heading style (minor headings)."""
    return create_paragraph_style(
        name=name,
        preset=Typography.H4,
        alignment=alignment,
        space_before=space_before,
        space_after=space_after,
        parent_style="Normal",
        **kwargs
    )


def get_body_style(
    name: str = "BodyText",
    alignment: int = TextAlignment.LEFT,
    space_before: float = ParagraphSpacing.BEFORE_BODY,
    space_after: float = ParagraphSpacing.AFTER_BODY,
    **kwargs
) -> ParagraphStyle:
    """Create a standard body text style."""
    return create_paragraph_style(
        name=name,
        preset=Typography.BODY,
        alignment=alignment,
        space_before=space_before,
        space_after=space_after,
        **kwargs
    )


def get_body_large_style(
    name: str = "BodyLarge",
    alignment: int = TextAlignment.LEFT,
    space_before: float = ParagraphSpacing.BEFORE_BODY,
    space_after: float = ParagraphSpacing.AFTER_BODY,
    **kwargs
) -> ParagraphStyle:
    """Create a large body text style (for emphasis)."""
    return create_paragraph_style(
        name=name,
        preset=Typography.BODY_LARGE,
        alignment=alignment,
        space_before=space_before,
        space_after=space_after,
        **kwargs
    )


def get_body_small_style(
    name: str = "BodySmall",
    alignment: int = TextAlignment.LEFT,
    space_before: float = ParagraphSpacing.BEFORE_BODY,
    space_after: float = ParagraphSpacing.AFTER_BODY,
    **kwargs
) -> ParagraphStyle:
    """Create a small body text style (for secondary content)."""
    return create_paragraph_style(
        name=name,
        preset=Typography.BODY_SMALL,
        alignment=alignment,
        space_before=space_before,
        space_after=space_after,
        **kwargs
    )


def get_caption_style(
    name: str = "Caption",
    alignment: int = TextAlignment.LEFT,
    space_before: float = ParagraphSpacing.BEFORE_CAPTION,
    space_after: float = ParagraphSpacing.AFTER_CAPTION,
    **kwargs
) -> ParagraphStyle:
    """Create a caption/footnote style."""
    return create_paragraph_style(
        name=name,
        preset=Typography.CAPTION,
        alignment=alignment,
        space_before=space_before,
        space_after=space_after,
        **kwargs
    )


def get_table_header_style(
    name: str = "TableHeader",
    alignment: int = TextAlignment.CENTER,
    **kwargs
) -> ParagraphStyle:
    """Create a table header cell style."""
    return create_paragraph_style(
        name=name,
        preset=Typography.TABLE_HEADER,
        alignment=alignment,
        space_before=0,
        space_after=0,
        **kwargs
    )


def get_table_body_style(
    name: str = "TableBody",
    alignment: int = TextAlignment.LEFT,
    **kwargs
) -> ParagraphStyle:
    """Create a table body cell style."""
    return create_paragraph_style(
        name=name,
        preset=Typography.TABLE_BODY,
        alignment=alignment,
        space_before=0,
        space_after=0,
        **kwargs
    )


def get_warning_style(
    name: str = "Warning",
    alignment: int = TextAlignment.LEFT,
    space_before: float = ParagraphSpacing.BEFORE_BODY,
    space_after: float = ParagraphSpacing.AFTER_BODY,
    **kwargs
) -> ParagraphStyle:
    """Create a warning text style."""
    return create_paragraph_style(
        name=name,
        preset=Typography.WARNING,
        alignment=alignment,
        space_before=space_before,
        space_after=space_after,
        **kwargs
    )


def get_error_style(
    name: str = "Error",
    alignment: int = TextAlignment.LEFT,
    space_before: float = ParagraphSpacing.BEFORE_BODY,
    space_after: float = ParagraphSpacing.AFTER_BODY,
    **kwargs
) -> ParagraphStyle:
    """Create an error text style."""
    return create_paragraph_style(
        name=name,
        preset=Typography.ERROR,
        alignment=alignment,
        space_before=space_before,
        space_after=space_after,
        **kwargs
    )


def get_success_style(
    name: str = "Success",
    alignment: int = TextAlignment.LEFT,
    space_before: float = ParagraphSpacing.BEFORE_BODY,
    space_after: float = ParagraphSpacing.AFTER_BODY,
    **kwargs
) -> ParagraphStyle:
    """Create a success text style."""
    return create_paragraph_style(
        name=name,
        preset=Typography.SUCCESS,
        alignment=alignment,
        space_before=space_before,
        space_after=space_after,
        **kwargs
    )


# ============================================================================
# TABLE STYLE HELPERS
# ============================================================================

def get_table_font_commands(
    header_rows: int = 1,
    use_bold_header: bool = True
) -> list:
    """Get standard table font style commands for ReportLab TableStyle.

    Args:
        header_rows: Number of header rows in the table
        use_bold_header: Whether to use bold font for headers

    Returns:
        List of TableStyle commands for font styling
    """
    header_font = FontFamily.SANS_SERIF_BOLD if use_bold_header else FontFamily.SANS_SERIF

    commands = [
        # Header rows
        ('FONTNAME', (0, 0), (-1, header_rows - 1), header_font),
        ('FONTSIZE', (0, 0), (-1, header_rows - 1), FontSize.TABLE_HEADER),
        ('LEADING', (0, 0), (-1, header_rows - 1), LineHeight.TABLE_HEADER),

        # Body rows
        ('FONTNAME', (0, header_rows), (-1, -1), FontFamily.SANS_SERIF),
        ('FONTSIZE', (0, header_rows), (-1, -1), FontSize.TABLE_BODY),
        ('LEADING', (0, header_rows), (-1, -1), LineHeight.TABLE_BODY),
    ]

    return commands


# ============================================================================
# CHART/DRAWING FONT HELPERS
# ============================================================================

def get_chart_title_font() -> Dict[str, Any]:
    """Get font settings for chart titles.

    Returns:
        Dictionary with fontName, fontSize, and fillColor for ReportLab Drawing objects
    """
    return {
        "fontName": FontFamily.SANS_SERIF_BOLD,
        "fontSize": FontSize.H3,
        "fillColor": colors.HexColor(ACCESSIBLE_HEADER_COLORS["subsection"]),
    }


def get_chart_label_font() -> Dict[str, Any]:
    """Get font settings for chart axis labels.

    Returns:
        Dictionary with fontName, fontSize, and fillColor for ReportLab Drawing objects
    """
    return {
        "fontName": FontFamily.SANS_SERIF,
        "fontSize": FontSize.SMALL,
        "fillColor": colors.HexColor(ACCESSIBLE_TEXT_COLORS["secondary"]),
    }


def get_chart_value_font() -> Dict[str, Any]:
    """Get font settings for chart data values.

    Returns:
        Dictionary with fontName, fontSize, and fillColor for ReportLab Drawing objects
    """
    return {
        "fontName": FontFamily.SANS_SERIF_BOLD,
        "fontSize": FontSize.CAPTION,
        "fillColor": colors.HexColor(ACCESSIBLE_TEXT_COLORS["primary"]),
    }


def get_chart_legend_font() -> Dict[str, Any]:
    """Get font settings for chart legends.

    Returns:
        Dictionary with fontName, fontSize, and fillColor for ReportLab Drawing objects
    """
    return {
        "fontName": FontFamily.SANS_SERIF,
        "fontSize": FontSize.SMALL,
        "fillColor": colors.HexColor(ACCESSIBLE_TEXT_COLORS["secondary"]),
    }


# ============================================================================
# TYPOGRAPHY REFERENCE PRINT FUNCTION
# ============================================================================

def print_typography_reference():
    """Print a reference guide for all typography settings."""
    print("=" * 70)
    print("TYPOGRAPHY CONSTANTS REFERENCE GUIDE")
    print("=" * 70)

    print("\nFONT FAMILIES:")
    print("-" * 40)
    print(f"  Sans-Serif (Default): {FontFamily.SANS_SERIF}")
    print(f"  Sans-Serif Bold:      {FontFamily.SANS_SERIF_BOLD}")
    print(f"  Sans-Serif Italic:    {FontFamily.SANS_SERIF_ITALIC}")
    print(f"  Monospace:            {FontFamily.MONOSPACE}")
    print(f"  Serif:                {FontFamily.SERIF}")

    print("\nFONT SIZE SCALE:")
    print("-" * 40)
    print(f"  DISPLAY:    {FontSize.DISPLAY}pt (Cover page title)")
    print(f"  H1:         {FontSize.H1}pt (Main section headers)")
    print(f"  H2:         {FontSize.H2}pt (Subsection headers)")
    print(f"  H3:         {FontSize.H3}pt (Minor headers)")
    print(f"  H4:         {FontSize.H4}pt (Small headers)")
    print(f"  BODY_LARGE: {FontSize.BODY_LARGE}pt (Emphasized body)")
    print(f"  BODY:       {FontSize.BODY}pt (Standard body)")
    print(f"  BODY_SMALL: {FontSize.BODY_SMALL}pt (Secondary body)")
    print(f"  CAPTION:    {FontSize.CAPTION}pt (Captions)")
    print(f"  SMALL:      {FontSize.SMALL}pt (Fine print)")
    print(f"  TINY:       {FontSize.TINY}pt (Minimal text)")

    print("\nLINE HEIGHT (LEADING):")
    print("-" * 40)
    print(f"  Tight ratio:   {LineHeight.TIGHT_RATIO} (for headers)")
    print(f"  Normal ratio:  {LineHeight.NORMAL_RATIO} (for body text)")
    print(f"  Relaxed ratio: {LineHeight.RELAXED_RATIO} (for readability)")

    print("\nTYPOGRAPHY PRESETS AVAILABLE:")
    print("-" * 40)
    presets = [
        "DISPLAY", "H1", "H2", "H3", "H4",
        "BODY_LARGE", "BODY", "BODY_SMALL", "BODY_EMPHASIS", "BODY_SECONDARY",
        "CAPTION", "CAPTION_EMPHASIS", "FOOTNOTE",
        "TABLE_HEADER", "TABLE_BODY", "TABLE_FOOTER",
        "MONOSPACE", "WARNING", "ERROR", "SUCCESS", "INFO"
    ]
    for preset in presets:
        print(f"  Typography.{preset}")

    print("\nSTYLE FACTORY FUNCTIONS:")
    print("-" * 40)
    functions = [
        "get_display_style()", "get_h1_style()", "get_h2_style()",
        "get_h3_style()", "get_h4_style()", "get_body_style()",
        "get_body_large_style()", "get_body_small_style()",
        "get_caption_style()", "get_table_header_style()",
        "get_table_body_style()", "get_warning_style()",
        "get_error_style()", "get_success_style()"
    ]
    for func in functions:
        print(f"  {func}")


if __name__ == "__main__":
    print_typography_reference()
