import json
import os
import subprocess
from phase_field_utilities import PHASE_FIELDS
from smartsheet_client import (
    get_smartsheet_token,
    create_smartsheet_client,
    create_column_map,
)

# Load environment variables and Smartsheet token using shared utility
try:
    token = get_smartsheet_token()
except ValueError as e:
    print(f"Error: {e}")
    exit(1)

# Sheets to process
SHEET_IDS = {
    "NA": 6141179298008964,
    "NF": 615755411312516,
    "NH": 123340632051588,
    "NP": 3009924800925572,
    "NT": 2199739350077316,
    "NV": 8955413669040004,
    "NM": 4275419734822788,
}

# PHASE_FIELDS is now imported from phase_field_utilities module
# to eliminate code duplication.

# First, reset or create the change history file
history_file = "tracking_data/change_history.csv"
os.makedirs("tracking_data", exist_ok=True)

# Try to reset using git if the file is tracked
try:
    print("Attempting to reset change history file using git...")
    subprocess.run(["git", "checkout", "--", history_file], check=False)
except Exception as e:
    print(f"Note: Git command failed: {e}")

# Create a fresh change history file with headers
print("Creating fresh change history file...")
with open(history_file, "w", newline="", encoding="utf-8") as f:
    f.write("Timestamp,Group,RowID,Phase,DateField,Date,User,Marketplace\n")

print("Connecting to Smartsheet...")
# Create a fresh state file that marks all current data as processed
# Use shared client utility for retry support
client = create_smartsheet_client(token)
print("Connected to Smartsheet with retry support")

state = {"last_run": "2025-10-18 16:39:14", "processed": {}}

# Process each sheet
for group, sid in SHEET_IDS.items():
    print(f"Processing sheet {group}...")
    try:
        sheet = client.get_sheet(sid)

        if sheet is None:
            print(f"Sheet {group} not available, skipping...")
            continue

        # Map column titles to IDs using shared utility
        col_map = create_column_map(sheet)

        # Process each row
        for row in sheet.rows:
            for date_col, _, _ in PHASE_FIELDS:
                col_id = col_map.get(date_col)
                if not col_id:
                    continue

                # Find cell with this column ID
                for cell in row.cells:
                    if cell.column_id == col_id and cell.value:
                        # Add to processed state with normalized date (YYYY-MM-DD)
                        field_key = f"{group}:{row.id}:{date_col}"
                        # Normalize to YYYY-MM-DD format
                        val = cell.value
                        if hasattr(val, 'date'):
                            val = val.date().isoformat()
                        elif hasattr(val, 'isoformat'):
                            val = val.isoformat()
                        else:
                            val = str(val).strip()[:10]  # Take just YYYY-MM-DD part
                        state["processed"][field_key] = val
                        break
    except Exception as e:
        print(f"Error processing sheet {group}: {e}")
        continue

# Save state with UTF-8 encoding for proper unicode support
with open("tracking_data/tracker_state.json", "w", encoding='utf-8') as f:
    json.dump(state, f, ensure_ascii=False)

print(f"Created state file with {len(state['processed'])} processed items")
print("System reset complete - tracking will now only capture new changes")
