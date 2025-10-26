#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

MAX_RETRIES=3
COUNT=0

# The following line sets the CUDA device to be used.
export CUDA_VISIBLE_DEVICES=0

while (( COUNT < MAX_RETRIES )); do
    echo "Attempt $((COUNT + 1)) initiated"

    if /eos/user/d/diptarko/ScoutingAnalyser/scoutingeval/bin/python3 -u \
        "$SCRIPT_DIR/process.py"; then
        break
    fi

    echo "Attempt $((COUNT + 1)) failed."
    (( COUNT += 1 ))
    sleep 2
done

if (( COUNT == MAX_RETRIES )); then
    echo "All $MAX_RETRIES attempts failed. Exiting with error."
    exit 1
fi
