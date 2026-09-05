#!/bin/sh
set -eu
MODE="${1:-local}"
SEED="${2:-42}"
printf '\nReconPilot submission demo\n'
printf 'Generating benchmark (seed %s)...\n' "$SEED"
python -m reconpilot benchmark --seed "$SEED" --mode "$MODE"
printf '\nStarting Streamlit UI...\n'
streamlit run app.py
