#!/usr/bin/env bash

set -Eeuo pipefail

# ============================================================
# SIH V6.1 — COMPLETE PROJECT RUNNER
#
# Generates the synthetic dataset, validates it, executes the
# complete forensic pipeline, validates the resulting project,
# and prints instructions for launching the dashboard.
# ============================================================

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"

BOLD='\033[1m'
GREEN='\033[0;32m'
RED='\033[0;31m'
CYAN='\033[0;36m'
RESET='\033[0m'

header() {
    echo
    echo "======================================================================"
    echo -e "${BOLD}${CYAN}$1${RESET}"
    echo "======================================================================"
}

success() {
    echo -e "${GREEN}✓ $1${RESET}"
}

fail() {
    echo -e "${RED}✗ $1${RESET}" >&2
    exit 1
}

run_python() {
    local description="$1"
    shift

    echo
    echo "→ $description"
    "$PYTHON" "$@"
    success "$description"
}


# ============================================================
# START
# ============================================================

clear 2>/dev/null || true

header "SIH V6.1 — FULL PROJECT SETUP"

echo "Project root : $ROOT"
echo "Started      : $(date)"
echo


# ============================================================
# ENVIRONMENT
# ============================================================

header "1/5 — ENVIRONMENT CHECK"

command -v "$PYTHON" >/dev/null 2>&1 \
    || fail "Python executable '$PYTHON' was not found."

echo "Python: $($PYTHON --version 2>&1)"

if [[ ! -f requirements.txt ]]; then
    fail "requirements.txt not found."
fi

echo
echo "Checking project dependencies..."

"$PYTHON" src/check_environment.py

success "Environment check passed"


# ============================================================
# DATASET GENERATION
# ============================================================

header "2/5 — DATASET GENERATION"

echo "Generating the complete SIH V6.1 synthetic dataset..."
echo

run_python \
    "Synthetic dataset generation" \
    generate_dataset.py


# ============================================================
# INPUT VALIDATION
# ============================================================

header "3/5 — INPUT VALIDATION"

run_python \
    "Operational dataset validation" \
    src/ingest_data.py \
    --validate-existing


# ============================================================
# FORENSIC PIPELINE
# ============================================================

header "4/5 — FORENSIC PIPELINE"

if [[ ! -f run_pipeline.sh ]]; then
    fail "run_pipeline.sh not found."
fi

bash run_pipeline.sh

success "Forensic pipeline completed"


# ============================================================
# FINAL VALIDATION
# ============================================================

header "5/5 — PROJECT VALIDATION"

run_python \
    "Final project validation" \
    src/validate_project.py


# ============================================================
# COMPLETE
# ============================================================

header "SIH V6.1 — READY"

echo "The complete project has been generated and validated."
echo
echo "Important outputs:"
echo "  outputs/final_entity_alerts_v61.csv"
echo "  models/final_detector_v61.joblib"
echo "  models/final_detector_config_v61.json"
echo
echo "Launch the investigation dashboard with:"
echo
echo "  streamlit run app/dashboard.py"
echo

success "SIH V6.1 is ready."

