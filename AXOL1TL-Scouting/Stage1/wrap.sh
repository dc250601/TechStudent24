#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MAX_RETRIES=3
COUNT=0

while (( COUNT < MAX_RETRIES )); do
    
    PYTHON_GIL=0 /eos/user/d/diptarko/ScoutingAnalyser/scouting13ft/bin/python3 -u "$SCRIPT_DIR/process.py" && break

    echo "Attempt $((COUNT + 1)) failed."
    (( COUNT += 1 ))
    sleep 2
done

if (( COUNT == MAX_RETRIES )); then
    echo "All $MAX_RETRIES attempts failed. Exiting with error."
    exit 1
fi
