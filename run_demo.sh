#!/usr/bin/env bash

set -e

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

echo "======================================================================"
echo "SIH V6.1 — BITCOIN FORENSICS PLATFORM"
echo "======================================================================"
echo

echo "[1/3] Checking Python environment..."
python src/check_environment.py

echo
echo "[2/3] Validating supplied project artifacts..."
python src/validate_project.py

echo
echo "[3/3] Starting investigation dashboard..."
echo
echo "Press Ctrl+C to stop the dashboard."
echo

python -m streamlit run app/dashboard.py
