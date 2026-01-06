"""Table Pagination Utilities for ReportLab PDF Generation.

This module provides enhanced table pagination capabilities to:
- Prevent orphaned rows (single rows on a page)
- Ensure minimum rows per page
- Keep related data together across page breaks
- Support header row repetition on each page
- Enable intelligent table splitting

Usage:
    from table_pagination import create_paginated_table, PaginationConfig

    # Create a paginated table with header repetition
    table = create_paginated_table(
        data=table_data,
        col_widths=[30*mm, 50*mm, 80*mm],
        style_commands=style_commands,
        config=PaginationConfig(
            repeat_header_rows=1,
            min_rows_per_page=3,
            avoid_orphans=True
        )
    )
"""

from dataclasses import dataclass, field
from typing import List, Optional, Tuple, Any, Union
import math

from reportlab.platypus import Table, TableStyle, Flowable, KeepTogether
from reportlab.lib import colors
from reportlab.lib.units import mm


@dataclass
class PaginationConfig:
    """Configuration for table pagination behavior.

    Attributes:
        repeat_header_rows: Number of header rows to repeat on each page (default: 1)
        min_rows_per_page: Minimum data rows to keep together on a page (default: 3)
        avoid_orphans: If True, prevents single orphaned rows at page breaks (default: True)
        orphan_threshold: Minimum rows that must appear together (default: 2)
        max_rows_before_split: Maximum rows before forcing a table split for truncation display
        enable_row_splitting: Allow splitting within tall rows (default: False)
        keep_with_next: Keep this table with the next flowable (default: False)
        group_by_column: Column index to group related rows together (default: None)
    """
    repeat_header_rows: int = 1
    min_rows_per_page: int = 3
    avoid_orphans: bool = True
    orphan_threshold: int = 2
    max_rows_before_split: Optional[int] = None
    enable_row_splitting: bool = False
    keep_with_next: bool = False
    group_by_column: Optional[int] = None


def create_paginated_table(
    data: List[List[Any]],
    col_widths: Optional[List[float]] = None,
    row_heights: Optional[List[float]] = None,
    style_commands: Optional[List[Tuple]] = None,
    config: Optional[PaginationConfig] = None,
    **table_kwargs
) -> Table:
    """Create a table with enhanced pagination support.

    This function creates a ReportLab Table with proper settings to handle
    page breaks gracefully, including header repetition and orphan prevention.

    Args:
        data: 2D list of table data (rows x columns)
        col_widths: List of column widths (None for auto-sizing)
        row_heights: List of row heights (None for auto-sizing)
        style_commands: List of TableStyle commands to apply
        config: PaginationConfig with pagination settings
        **table_kwargs: Additional keyword arguments passed to Table constructor

    Returns:
        Table: A ReportLab Table configured for proper pagination
    """
    if config is None:
        config = PaginationConfig()

    if not data:
        raise ValueError("Table data cannot be empty")

    # Calculate effective number of data rows (excluding headers)
    num_header_rows = config.repeat_header_rows
    num_data_rows = len(data) - num_header_rows

    # Handle grouping if specified
    if config.group_by_column is not None and num_data_rows > 0:
        data = _reorder_by_groups(data, num_header_rows, config.group_by_column)

    # Create the table with pagination settings
    table = Table(
        data,
        colWidths=col_widths,
        rowHeights=row_heights,
        repeatRows=config.repeat_header_rows,
        splitByRow=True,  # Enable row-based splitting
        splitInRow=config.enable_row_splitting,
        **table_kwargs
    )

    # Apply style commands
    if style_commands:
        table.setStyle(TableStyle(style_commands))

    return table


def create_paginated_table_with_orphan_control(
    data: List[List[Any]],
    col_widths: Optional[List[float]] = None,
    row_heights: Optional[List[float]] = None,
    style_commands: Optional[List[Tuple]] = None,
    config: Optional[PaginationConfig] = None,
    available_height: Optional[float] = None,
    row_height_estimate: float = 20,
    **table_kwargs
) -> Union[Table, List[Flowable]]:
    """Create a table with advanced orphan control.

    When orphan control is enabled and the table might leave orphaned rows
    at a page break, this function can return multiple tables that are
    designed to split cleanly across pages.

    Args:
        data: 2D list of table data
        col_widths: List of column widths
        row_heights: List of row heights
        style_commands: List of TableStyle commands
        config: PaginationConfig with pagination settings
        available_height: Available height on current page (for calculation)
        row_height_estimate: Estimated height per row (default: 20 points)
        **table_kwargs: Additional kwargs for Table constructor

    Returns:
        Either a single Table or a list of Flowables for better page control
    """
    if config is None:
        config = PaginationConfig()

    if not data:
        raise ValueError("Table data cannot be empty")

    num_header_rows = config.repeat_header_rows
    num_data_rows = len(data) - num_header_rows

    # If no orphan control needed or small table, return simple table
    if not config.avoid_orphans or num_data_rows <= config.min_rows_per_page:
        return create_paginated_table(
            data, col_widths, row_heights, style_commands, config, **table_kwargs
        )

    # If we have available height info, we can be smarter about splitting
    if available_height is not None:
        rows_per_page = int(available_height / row_height_estimate)

        # Check if we'd have orphaned rows
        if rows_per_page > num_header_rows:
            data_rows_on_first_page = rows_per_page - num_header_rows
            remaining_rows = num_data_rows - data_rows_on_first_page

            if 0 < remaining_rows < config.orphan_threshold:
                # Would create orphans - adjust first page to leave more rows
                adjusted_split = data_rows_on_first_page - config.orphan_threshold
                if adjusted_split >= config.min_rows_per_page:
                    # We can adjust the split point
                    # Use KeepTogether for better control
                    return _create_controlled_split_tables(
                        data, col_widths, row_heights, style_commands,
                        config, adjusted_split, **table_kwargs
                    )

    # Default: use standard pagination features
    return create_paginated_table(
        data, col_widths, row_heights, style_commands, config, **table_kwargs
    )


def _get_cell_text(cell: Any) -> str:
    """Extract text content from a table cell.

    Handles both plain strings and ReportLab Paragraph objects.

    Args:
        cell: A table cell value (string, Paragraph, or other)

    Returns:
        String representation of the cell content
    """
    if cell is None:
        return ""
    if isinstance(cell, str):
        return cell
    # Handle ReportLab Paragraph objects
    if hasattr(cell, 'text'):
        return str(cell.text)
    # Handle objects with getPlainText method
    if hasattr(cell, 'getPlainText'):
        return cell.getPlainText()
    # Fallback to string conversion
    return str(cell)


def _reorder_by_groups(
    data: List[List[Any]],
    num_header_rows: int,
    group_column: int
) -> List[List[Any]]:
    """Reorder data rows to keep same-group items together.

    Args:
        data: The table data
        num_header_rows: Number of header rows to preserve at top
        group_column: Column index to group by

    Returns:
        Reordered data with header rows preserved and data rows grouped
    """
    if num_header_rows >= len(data):
        return data

    headers = data[:num_header_rows]
    data_rows = data[num_header_rows:]

    if not data_rows:
        return data

    # Check if group_column is valid for data rows
    if data_rows and group_column >= len(data_rows[0]):
        return data

    # Group rows by the specified column value
    groups = {}
    for row in data_rows:
        if group_column < len(row):
            key = _get_cell_text(row[group_column])
        else:
            key = ""
        if key not in groups:
            groups[key] = []
        groups[key].append(row)

    # Flatten groups back to a list, preserving group order
    sorted_rows = []
    for key in sorted(groups.keys()):
        sorted_rows.extend(groups[key])

    return headers + sorted_rows


def _create_controlled_split_tables(
    data: List[List[Any]],
    col_widths: Optional[List[float]],
    row_heights: Optional[List[float]],
    style_commands: Optional[List[Tuple]],
    config: PaginationConfig,
    split_at_row: int,
    **table_kwargs
) -> List[Flowable]:
    """Create multiple tables with controlled split points.

    This function splits a large table into multiple smaller tables
    at controlled points to prevent orphaned rows.

    Args:
        data: The table data
        col_widths: Column widths
        row_heights: Row heights
        style_commands: Style commands to apply
        config: Pagination configuration
        split_at_row: Data row index to split at
        **table_kwargs: Additional table kwargs

    Returns:
        List of Flowable objects (tables)
    """
    num_header_rows = config.repeat_header_rows
    headers = data[:num_header_rows]
    data_rows = data[num_header_rows:]

    flowables = []

    # First table: headers + first batch of data rows
    first_table_data = headers + data_rows[:split_at_row]
    first_table = Table(
        first_table_data,
        colWidths=col_widths,
        rowHeights=row_heights[:num_header_rows + split_at_row] if row_heights else None,
        repeatRows=config.repeat_header_rows,
        splitByRow=True,
        **table_kwargs
    )
    if style_commands:
        first_table.setStyle(TableStyle(style_commands))
    flowables.append(first_table)

    # Second table: headers (repeated) + remaining data rows
    if split_at_row < len(data_rows):
        second_table_data = headers + data_rows[split_at_row:]
        second_row_heights = None
        if row_heights:
            header_heights = row_heights[:num_header_rows]
            remaining_heights = row_heights[num_header_rows + split_at_row:]
            second_row_heights = header_heights + remaining_heights

        second_table = Table(
            second_table_data,
            colWidths=col_widths,
            rowHeights=second_row_heights,
            repeatRows=config.repeat_header_rows,
            splitByRow=True,
            **table_kwargs
        )
        if style_commands:
            # Adjust style commands for the new table
            adjusted_commands = _adjust_style_commands_for_split(
                style_commands, num_header_rows, split_at_row, len(data_rows) - split_at_row
            )
            second_table.setStyle(TableStyle(adjusted_commands))
        flowables.append(second_table)

    return flowables


def _adjust_style_commands_for_split(
    style_commands: List[Tuple],
    num_header_rows: int,
    split_at_row: int,
    remaining_rows: int
) -> List[Tuple]:
    """Adjust style commands for a split table portion.

    When a table is split, row-specific style commands need to be adjusted
    to reference the correct rows in the new table.

    Args:
        style_commands: Original style commands
        num_header_rows: Number of header rows
        split_at_row: Data row where split occurred
        remaining_rows: Number of remaining data rows

    Returns:
        Adjusted style commands for the second table portion
    """
    adjusted = []
    total_rows_in_new_table = num_header_rows + remaining_rows

    for cmd in style_commands:
        if len(cmd) < 3:
            adjusted.append(cmd)
            continue

        command_type = cmd[0]
        start_cell = cmd[1]
        end_cell = cmd[2]
        rest = cmd[3:]

        # Handle row coordinates
        start_col, start_row = start_cell
        end_col, end_row = end_cell

        # Normalize negative indices
        if end_row == -1:
            end_row = total_rows_in_new_table - 1

        # Commands for header rows pass through unchanged
        if start_row < num_header_rows and end_row < num_header_rows:
            adjusted.append(cmd)
            continue

        # Commands that span all rows
        if start_row == 0 and (end_row == -1 or end_row >= num_header_rows + split_at_row):
            new_cmd = (command_type, (start_col, 0), (end_col, -1)) + rest
            adjusted.append(new_cmd)
            continue

        # Commands for data rows need adjustment
        if start_row >= num_header_rows:
            orig_data_row_start = start_row - num_header_rows
            if orig_data_row_start >= split_at_row:
                # This row is in the second table
                new_start_row = num_header_rows + (orig_data_row_start - split_at_row)
                new_end_row = end_row
                if end_row >= num_header_rows:
                    orig_data_row_end = end_row - num_header_rows
                    if orig_data_row_end >= split_at_row:
                        new_end_row = num_header_rows + (orig_data_row_end - split_at_row)
                    else:
                        # This command doesn't apply to second table
                        continue

                new_cmd = (command_type, (start_col, new_start_row), (end_col, new_end_row)) + rest
                adjusted.append(new_cmd)
        else:
            # General commands that apply to whole table
            adjusted.append(cmd)

    return adjusted


def calculate_optimal_rows_per_page(
    available_height: float,
    row_height: float = 18,
    header_height: Optional[float] = None,
    num_header_rows: int = 1,
    padding: float = 10
) -> int:
    """Calculate the optimal number of data rows per page.

    Args:
        available_height: Total available height on the page
        row_height: Height per data row (default: 18 points)
        header_height: Height of header rows (defaults to row_height if None)
        num_header_rows: Number of header rows
        padding: Additional padding to account for (default: 10 points)

    Returns:
        Optimal number of data rows that fit on a page
    """
    if header_height is None:
        header_height = row_height

    usable_height = available_height - padding - (num_header_rows * header_height)

    if usable_height <= 0:
        return 1  # At minimum, try to fit one row

    return max(1, int(usable_height / row_height))


def ensure_minimum_rows(
    data: List[List[Any]],
    min_rows: int,
    num_header_rows: int = 1,
    empty_row_template: Optional[List[Any]] = None
) -> List[List[Any]]:
    """Ensure a table has at least a minimum number of data rows.

    This is useful to prevent very small tables that look odd on a page.

    Args:
        data: The table data
        min_rows: Minimum number of data rows required
        num_header_rows: Number of header rows
        empty_row_template: Template for empty rows (defaults to empty strings)

    Returns:
        Data with enough rows to meet minimum requirement
    """
    current_data_rows = len(data) - num_header_rows

    if current_data_rows >= min_rows:
        return data

    rows_needed = min_rows - current_data_rows

    if empty_row_template is None:
        # Create empty row based on first data row or header
        if data:
            num_cols = len(data[0])
            empty_row_template = [""] * num_cols
        else:
            return data

    result = list(data)
    for _ in range(rows_needed):
        result.append(list(empty_row_template))

    return result


class PaginatedTableWrapper(Flowable):
    """A Flowable wrapper that provides additional pagination control.

    This wrapper can be used when you need more control over how a table
    interacts with page breaks and other flowables.
    """

    def __init__(
        self,
        table: Table,
        config: PaginationConfig,
        min_first_page_rows: int = 3
    ):
        """Initialize the wrapper.

        Args:
            table: The Table to wrap
            config: Pagination configuration
            min_first_page_rows: Minimum rows that must appear on first page
        """
        Flowable.__init__(self)
        self.table = table
        self.config = config
        self.min_first_page_rows = min_first_page_rows
        self._calculated_height = None

    def wrap(self, available_width: float, available_height: float) -> Tuple[float, float]:
        """Calculate the space needed for this flowable."""
        width, height = self.table.wrap(available_width, available_height)
        self._calculated_height = height
        return width, height

    def split(self, available_width: float, available_height: float) -> List[Flowable]:
        """Split the flowable if it doesn't fit."""
        # Use the table's built-in split with our configuration
        result = self.table.split(available_width, available_height)

        if result and self.config.avoid_orphans:
            # Check if the split would create orphans
            for i, part in enumerate(result):
                if isinstance(part, Table) and hasattr(part, '_nrows'):
                    data_rows = part._nrows - self.config.repeat_header_rows
                    if data_rows < self.config.orphan_threshold:
                        # This would create an orphan - return empty to force page break
                        if i == len(result) - 1:
                            # It's the last part - we need different handling
                            return result  # Accept the orphan in final part
                        else:
                            return []  # Force page break before

        return result

    def draw(self) -> None:
        """Draw the table."""
        self.table.drawOn(self.canv, 0, 0)


def get_table_row_heights(table: Table) -> List[float]:
    """Extract calculated row heights from a table.

    This is useful for understanding how a table will paginate.

    Args:
        table: A ReportLab Table that has been wrapped

    Returns:
        List of row heights
    """
    if hasattr(table, '_rowHeights') and table._rowHeights:
        return list(table._rowHeights)
    return []


def estimate_table_height(
    num_rows: int,
    row_height: float = 18,
    header_rows: int = 1,
    header_height: Optional[float] = None
) -> float:
    """Estimate the total height of a table.

    Args:
        num_rows: Total number of rows (including headers)
        row_height: Height per data row
        header_rows: Number of header rows
        header_height: Height of header rows (defaults to row_height)

    Returns:
        Estimated total height in points
    """
    if header_height is None:
        header_height = row_height

    data_rows = num_rows - header_rows
    return (header_rows * header_height) + (data_rows * row_height)
