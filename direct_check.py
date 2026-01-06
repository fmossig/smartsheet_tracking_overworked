import json
import os
from phase_field_utilities import get_phase_date_columns
from smartsheet_client import (
    get_smartsheet_token,
    create_smartsheet_client,
    create_column_map,
    get_cell_value_by_column,
)

# Load environment variables and Smartsheet token using shared utility
try:
    token = get_smartsheet_token()
except ValueError as e:
    print(f"ERROR: {e}")
    exit(1)

# State file
STATE_FILE = "tracking_data/tracker_state.json"

# Check if state file exists
if not os.path.exists(STATE_FILE):
    print(f"ERROR: State file not found: {STATE_FILE}")
    exit(1)

# Load state with UTF-8 encoding for proper unicode support
with open(STATE_FILE, 'r', encoding='utf-8') as f:
    state = json.load(f)
    processed = state.get("processed", {})
    print(f"Loaded state file with {len(processed)} entries")

# Connect to Smartsheet using shared client utility
client = create_smartsheet_client(token)
print("Connected to Smartsheet with retry support")

# Example fields we're checking
SHEET_IDS = {
    "NA": 6141179298008964,
    "NF": 615755411312516,
    "NH": 123340632051588,
    "NP": 3009924800925572,
    "NT": 2199739350077316,
    "NV": 8955413669040004,
    "NM": 4275419734822788,
}

# Phase date columns are now imported from phase_field_utilities module
PHASE_DATE_COLUMNS = get_phase_date_columns()

# Collect current values for fields in state
print("Checking for differences...")
print("=" * 50)

differences = []
for key, stored_value in processed.items():
    try:
        # Parse the key (format: "GROUP:ROW_ID:FIELD")
        parts = key.split(":")
        if len(parts) != 3:
            continue

        group, row_id, field = parts

        if group not in SHEET_IDS:
            continue

        if field not in PHASE_DATE_COLUMNS:
            continue

        # Get current value from Smartsheet using retry client
        sheet = client.get_sheet(SHEET_IDS[group])

        if sheet is None:
            print(f"Sheet {group} not available")
            continue

        # Map column titles to IDs using shared utility
        col_map = create_column_map(sheet)
        if field not in col_map:
            continue

        # Find the row
        row = next((r for r in sheet.rows if str(r.id) == row_id), None)
        if not row:
            print(f"Row not found: {row_id} in {group}")
            continue

        # Get the cell value using shared utility
        current_value = get_cell_value_by_column(row, col_map[field])

        # Compare with stored value
        if stored_value != current_value:
            differences.append({
                "key": key,
                "stored": stored_value,
                "current": current_value
            })
            print(f"DIFFERENCE FOUND: {key}")
            print(f"  Stored: {stored_value}")
            print(f"  Current: {current_value}")

    except Exception as e:
        print(f"Error checking {key}: {e}")

print("=" * 50)
print(f"Total differences found: {len(differences)}")

if len(differences) == 0:
    # Check a sample of entries
    print("\nChecking random sample of 5 entries:")
    sample_count = 0
    for key, stored_value in list(processed.items())[:5]:
        print(f"Sample {sample_count+1}: {key} = {stored_value}")
        sample_count += 1
