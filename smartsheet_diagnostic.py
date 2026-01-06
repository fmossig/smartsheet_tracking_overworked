import os
import json
import csv
import logging
import argparse
from datetime import datetime
from phase_field_utilities import PHASE_FIELDS, resolve_column_name
from smartsheet_client import (
    get_smartsheet_token,
    create_smartsheet_client,
    create_column_map,
    fetch_sheet,
)
from logging_config import (
    configure_logging,
    add_log_level_argument,
)

# Logger will be configured in main() after parsing args
logger = logging.getLogger(__name__)

# Load environment variables and Smartsheet token using shared utility
try:
    token = get_smartsheet_token()
except ValueError as e:
    print(f"ERROR: {e}")
    exit(1)

# Constants from your tracking script
SHEET_IDS = {
    "NA": 6141179298008964,
    "NF": 615755411312516,
    "NH": 123340632051588,
    "NP": 3009924800925572,
    "NT": 2199739350077316,
    "NV": 8955413669040004,
    "NM": 4275419734822788,
}

# PHASE_FIELDS and resolve_column_name are now imported from
# phase_field_utilities module to eliminate code duplication.

DATA_DIR = "tracking_data"
STATE_FILE = os.path.join(DATA_DIR, "tracker_state.json")
CHANGES_FILE = os.path.join(DATA_DIR, "change_history.csv")

def load_state():
    """Load state file and return the data."""
    if not os.path.exists(STATE_FILE):
        print(f"State file not found: {STATE_FILE}")
        return {"last_run": None, "processed": {}}

    try:
        with open(STATE_FILE, 'r', encoding='utf-8') as f:
            state = json.load(f)
            print(f"Loaded state file with {len(state.get('processed', {}))} processed items")
            return state
    except Exception as e:
        print(f"Error loading state: {e}")
        return {"last_run": None, "processed": {}}

def save_state(state):
    """Save state to file."""
    try:
        with open(STATE_FILE, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False)
            print(f"Saved state with {len(state.get('processed', {}))} processed items")
    except Exception as e:
        print(f"Error saving state: {e}")

def find_differences():
    """Find differences between stored state and current Smartsheet values."""
    state = load_state()
    processed = state.get("processed", {})

    # Connect to Smartsheet using shared client utility
    client = create_smartsheet_client(token)
    print("Connected to Smartsheet API with retry support")

    # Track differences
    differences = []
    current_values = {}

    # Process each sheet
    for group, sheet_id in SHEET_IDS.items():
        print(f"\nProcessing sheet {group} (ID: {sheet_id})")

        # Use shared fetch_sheet utility for error handling
        result = fetch_sheet(client, sheet_id, group_name=group)

        if not result.success:
            print(f"Error processing sheet {group}: {result.message}")
            continue

        sheet = result.sheet
        col_map = result.column_map

        print(f"Sheet {group} has {len(sheet.rows)} rows")

        # Check which tracked fields exist in this sheet
        found_fields = [f for f, _, _ in PHASE_FIELDS if f in col_map]
        print(f"Found tracked fields: {found_fields}")

        # Process each row
        for row in sheet.rows:
            for date_col, user_col_variations, phase_no in PHASE_FIELDS:
                if date_col not in col_map:
                    continue

                # Resolve user column from variations
                resolved_user_col, user_col_id = resolve_column_name(col_map, user_col_variations)

                # Get current value from Smartsheet
                date_val = None
                user_val = ""

                for cell in row.cells:
                    if cell.column_id == col_map.get(date_col):
                        date_val = cell.value
                    if user_col_id and cell.column_id == user_col_id:
                        user_val = cell.display_value or ""

                if not date_val:
                    continue

                # Create field key
                field_key = f"{group}:{row.id}:{date_col}"

                # Store current value
                current_values[field_key] = date_val

                # Compare with stored state
                prev_val = processed.get(field_key)

                if prev_val != date_val:
                    differences.append({
                        "field_key": field_key,
                        "prev_value": prev_val,
                        "current_value": date_val,
                        "user": user_val
                    })

    # Print results
    print("\n" + "="*50)
    print(f"FOUND {len(differences)} DIFFERENCES")
    print("="*50)

    if differences:
        print("\nTop 10 differences:")
        for i, diff in enumerate(differences[:10]):
            print(f"{i+1}. {diff['field_key']}")
            print(f"   Old: {diff['prev_value']}")
            print(f"   New: {diff['current_value']}")
            print(f"   User: {diff['user']}")

    return differences, current_values

def force_track_changes(differences):
    """Force tracking of detected differences."""
    if not differences:
        print("No differences to track.")
        return False

    print(f"\nForcing tracking of {len(differences)} changes...")

    # Load current state
    state = load_state()

    # Get timestamp
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d %H:%M:%S")

    # Ensure tracking_data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)

    # Append to changes file
    try:
        # Create file with headers if it doesn't exist
        if not os.path.exists(CHANGES_FILE):
            with open(CHANGES_FILE, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
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
                print(f"Created new changes file: {CHANGES_FILE}")

        with open(CHANGES_FILE, 'a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)

            for diff in differences:
                # Parse field key
                parts = diff['field_key'].split(":")
                if len(parts) != 3:
                    print(f"Invalid field key format: {diff['field_key']}")
                    continue

                group, row_id, date_col = parts

                # Find phase number
                phase_no = 0
                for dc, _, p in PHASE_FIELDS:
                    if dc == date_col:
                        phase_no = p
                        break

                # Parse date
                date_val = diff['current_value']
                try:
                    # Try ISO format first
                    dt = datetime.fromisoformat(date_val).date()
                except ValueError:
                    try:
                        # Try other formats
                        for fmt in ('%Y-%m-%d', '%d.%m.%Y'):
                            try:
                                dt = datetime.strptime(date_val, fmt).date()
                                break
                            except ValueError:
                                continue
                    except Exception:
                        print(f"Could not parse date: {date_val}")
                        continue

                # Write change record
                writer.writerow([
                    timestamp,
                    group,
                    row_id,
                    phase_no,
                    date_col,
                    dt.isoformat(),
                    diff['user'],
                    ""  # Marketplace (empty)
                ])

            print(f"Added {len(differences)} changes to {CHANGES_FILE}")

            # Update state
            for diff in differences:
                state["processed"][diff['field_key']] = diff['current_value']

            state["last_run"] = timestamp
            save_state(state)

            return True
    except Exception as e:
        print(f"Error writing changes: {e}")
        return False

def update_state_from_current(current_values):
    """Update state file with current Smartsheet values."""
    print("\nUpdating state file with current Smartsheet values...")

    # Load current state
    state = load_state()

    # Update with current values
    state["processed"] = current_values
    state["last_run"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Save updated state
    save_state(state)

    print("State file updated with current values from Smartsheet")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diagnose Smartsheet tracking issues")
    parser.add_argument("--check", action="store_true", help="Check for differences between state and current values")
    parser.add_argument("--force-track", action="store_true", help="Force tracking of any differences found")
    parser.add_argument("--update-state", action="store_true", help="Update state file with current Smartsheet values")
    add_log_level_argument(parser)

    args = parser.parse_args()

    # Configure logging with CLI argument or environment variable
    configure_logging(
        log_file="smartsheet_diagnostic.log",
        log_level=args.log_level
    )

    if args.check or args.force_track:
        differences, current_values = find_differences()

        if args.force_track and differences:
            force_track_changes(differences)

    if args.update_state:
        _, current_values = find_differences()
        update_state_from_current(current_values)
