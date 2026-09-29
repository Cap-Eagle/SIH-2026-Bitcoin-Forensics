#!/usr/bin/env bash

set -Eeuo pipefail

# ============================================================
# SIH V6.1 — END-TO-END BITCOIN FORENSICS PIPELINE
# ============================================================

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

PYTHON="${PYTHON:-python}"

START_TIME=$(date +%s)

# ------------------------------------------------------------
# Terminal formatting
# ------------------------------------------------------------

BOLD='\033[1m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
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

warn() {
    echo -e "${YELLOW}⚠ $1${RESET}"
}

fail() {
    echo -e "${RED}✗ $1${RESET}" >&2
    exit 1
}

run_step() {
    local number="$1"
    local name="$2"
    local script="$3"

    header "STEP ${number} — ${name}"

    if [[ ! -f "$script" ]]; then
        fail "Required script not found: $script"
    fi

    local step_start
    step_start=$(date +%s)

    "$PYTHON" "$script"

    local step_end
    step_end=$(date +%s)

    success "${name} completed in $((step_end - step_start))s"
}

require_file() {
    local file="$1"

    if [[ ! -s "$file" ]]; then
        fail "Required input missing or empty: $file"
    fi

    success "Found $file"
}


# ============================================================
# START
# ============================================================

clear 2>/dev/null || true

header "SIH V6.1 — BITCOIN FORENSICS PIPELINE"

echo "Project root : $ROOT"
echo "Python       : $($PYTHON --version 2>&1)"
echo "Started      : $(date)"
echo


# ============================================================
# PYTHON CHECK
# ============================================================

header "ENVIRONMENT CHECK"

command -v "$PYTHON" >/dev/null 2>&1 \
    || fail "Python executable '$PYTHON' was not found."

"$PYTHON" - <<'PY'
import sys

required = {
    "pandas": "pandas",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "networkx": "networkx",
    "joblib": "joblib",
}

missing = []

for module, package in required.items():
    try:
        __import__(module)
        print(f"✓ {package}")
    except ImportError:
        print(f"✗ {package}")
        missing.append(package)

if missing:
    print()
    print(
        "Missing Python packages: "
        + ", ".join(missing)
    )
    print(
        "Install dependencies with: "
        "python -m pip install -r requirements.txt"
    )
    sys.exit(1)
PY

success "Python environment ready"


# ============================================================
# DIRECTORY SETUP
# ============================================================

header "DIRECTORY CHECK"

mkdir -p \
    data/raw \
    data/correlated \
    data/ground_truth \
    outputs \
    models \
    logs

success "Project directories ready"


# ============================================================
# RAW INPUT VALIDATION
# ============================================================

header "INPUT VALIDATION"

require_file "data/raw/blockchain_transactions.csv"
require_file "data/raw/network_events.csv"
require_file "data/raw/ip_enrichment.csv"

echo
echo "Ground-truth files are used for benchmark/model training."
echo "They are not used to construct behavioral features."

require_file "data/ground_truth/ground_truth.csv"
require_file "data/ground_truth/entity_wallet_map.csv"


# ============================================================
# PIPELINE
# ============================================================

run_step \
    "01/10" \
    "Blockchain ↔ P2P Correlation" \
    "src/correlate_csv.py"

require_file "data/correlated/correlations.csv"


run_step \
    "02/10" \
    "Group B Correlation Handoff" \
    "src/build_group_b_handoff.py"

require_file "data/correlated/group_b_features.csv"


run_step \
    "03/10" \
    "Blockchain Graph Construction" \
    "src/build_graph.py"


run_step \
    "04/10" \
    "Entity Resolution" \
    "src/resolve_entities.py"

require_file "outputs/address_to_entity.csv"
require_file "outputs/entity_edges.csv"


run_step \
    "05/10" \
    "V3 Base Entity Feature Engineering" \
    "src/build_features_v3.py"

require_file "outputs/entity_features_v3.csv"


run_step \
    "06/10" \
    "Ground-Truth Entity Resolution" \
    "src/resolve_ground_truth_v3.py"

require_file "outputs/ground_truth_resolved_v3.csv"


run_step \
    "07/10" \
    "V6 Behavioral Feature Engineering" \
    "src/build_behavior_features_v6.py"

require_file "outputs/entity_behavior_features_v6.csv"


run_step \
    "08/10" \
    "V6.1 Temporal / Cross-Entity Feature Engineering" \
    "src/build_behavior_features_v61.py"

require_file "outputs/entity_behavior_features_v61.csv"


run_step \
    "09/10" \
    "Isolation Forest Novelty Model" \
    "src/train_anomaly_model_v5.py"

require_file "outputs/anomaly_model_v5.csv"


run_step \
    "10/10" \
    "Final V6.1 Forensic Detector" \
    "src/final_detector_v61.py"

require_file "outputs/final_entity_alerts_v61.csv"
require_file "models/final_detector_v61.joblib"
require_file "models/final_detector_config_v61.json"


# ============================================================
# FINAL SANITY CHECK
# ============================================================

header "FINAL OUTPUT SANITY CHECK"

"$PYTHON" - <<'PY'
from pathlib import Path

import pandas as pd

root = Path.cwd()

files = {
    "Entity features":
        root / "outputs/entity_features_v3.csv",

    "V6 behavior":
        root / "outputs/entity_behavior_features_v6.csv",

    "V6.1 behavior":
        root / "outputs/entity_behavior_features_v61.csv",

    "Novelty scores":
        root / "outputs/anomaly_model_v5.csv",

    "Final alerts":
        root / "outputs/final_entity_alerts_v61.csv",
}

counts = {}

for name, path in files.items():
    if not path.exists():
        raise SystemExit(
            f"ERROR: missing output {path}"
        )

    df = pd.read_csv(
        path,
        usecols=["entity_id"]
    )

    counts[name] = len(df)

    print(
        f"{name:<20}: "
        f"{len(df):>8,} entities"
    )

unique_counts = set(
    counts.values()
)

if len(unique_counts) != 1:
    print()
    print("WARNING: entity counts differ between outputs.")
    for name, count in counts.items():
        print(f"  {name}: {count:,}")
else:
    print()
    print(
        "✓ Entity counts are consistent "
        f"({next(iter(unique_counts)):,})"
    )


alerts = pd.read_csv(
    root / "outputs/final_entity_alerts_v61.csv"
)

if "risk_priority" in alerts.columns:

    print()
    print("Final risk distribution:")

    distribution = (
        alerts["risk_priority"]
        .value_counts()
    )

    for priority in [
        "CRITICAL",
        "HIGH",
        "MEDIUM",
        "LOW",
    ]:
        print(
            f"  {priority:<8}: "
            f"{int(distribution.get(priority, 0)):>8,}"
        )

print()
print("✓ Final output sanity check passed.")
PY


# ============================================================
# COMPLETE
# ============================================================

END_TIME=$(date +%s)
ELAPSED=$((END_TIME - START_TIME))

MINUTES=$((ELAPSED / 60))
SECONDS=$((ELAPSED % 60))

header "PIPELINE COMPLETE"

echo "Runtime       : ${MINUTES}m ${SECONDS}s"
echo
echo "Final alerts  : outputs/final_entity_alerts_v61.csv"
echo "Model         : models/final_detector_v61.joblib"
echo "Configuration : models/final_detector_config_v61.json"
echo

success "SIH V6.1 pipeline completed successfully."

echo
echo "Launch dashboard with:"
echo
echo "    streamlit run app/dashboard.py"
echo
