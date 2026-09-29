#!/usr/bin/env python3

"""
SIH V6.1 Project Validator
==========================

Performs lightweight integrity checks on the final Bitcoin
forensics project.

This does NOT retrain models.

Checks:
    - required project files
    - required datasets
    - output schemas
    - entity consistency
    - duplicate entity IDs
    - missing critical values
    - model artifacts
    - final alert distribution
"""

from pathlib import Path
import json
import sys

import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

ERRORS = []
WARNINGS = []


def section(title):
    print()
    print("=" * 70)
    print(title)
    print("=" * 70)


def ok(message):
    print(f"✓ {message}")


def error(message):
    print(f"✗ {message}")
    ERRORS.append(message)


def warning(message):
    print(f"⚠ {message}")
    WARNINGS.append(message)


def check_file(path, required=True):

    full = ROOT / path

    if not full.exists():
        if required:
            error(f"Missing: {path}")
        else:
            warning(f"Missing optional file: {path}")
        return False

    if full.is_file() and full.stat().st_size == 0:
        error(f"Empty file: {path}")
        return False

    ok(f"{path}")
    return True


section("SIH V6.1 PROJECT VALIDATION")

print(f"Project root: {ROOT}")


# ============================================================
# PROJECT STRUCTURE
# ============================================================

section("1. PROJECT STRUCTURE")

required_scripts = [
    "src/ingest_data.py",
    "src/correlate_csv.py",
    "src/build_group_b_handoff.py",
    "src/build_graph.py",
    "src/resolve_entities.py",
    "src/build_features_v3.py",
    "src/resolve_ground_truth_v3.py",
    "src/build_behavior_features_v6.py",
    "src/build_behavior_features_v61.py",
    "src/train_anomaly_model_v5.py",
    "src/final_detector_v61.py",
    "app/dashboard.py",
    "run_pipeline.sh",
]

for path in required_scripts:
    check_file(path)


# ============================================================
# INPUT DATA
# ============================================================

section("2. INPUT DATA")

input_files = [
    "data/raw/blockchain_transactions.csv",
    "data/raw/network_events.csv",
    "data/raw/ip_enrichment.csv",
    "data/ground_truth/ground_truth.csv",
    "data/ground_truth/entity_wallet_map.csv",
]

for path in input_files:
    check_file(path)


# ============================================================
# PIPELINE OUTPUTS
# ============================================================

section("3. PIPELINE OUTPUTS")

output_files = [
    "data/correlated/correlations.csv",
    "data/correlated/group_b_features.csv",
    "outputs/address_to_entity.csv",
    "outputs/entity_edges.csv",
    "outputs/entity_features_v3.csv",
    "outputs/ground_truth_resolved_v3.csv",
    "outputs/entity_behavior_features_v6.csv",
    "outputs/entity_behavior_features_v61.csv",
    "outputs/anomaly_model_v5.csv",
    "outputs/final_entity_alerts_v61.csv",
]

for path in output_files:
    check_file(path)


# ============================================================
# MODEL ARTIFACTS
# ============================================================

section("4. MODEL ARTIFACTS")

check_file(
    "models/final_detector_v61.joblib"
)

config_exists = check_file(
    "models/final_detector_config_v61.json"
)

if config_exists:
    try:
        with open(
            ROOT / "models/final_detector_config_v61.json",
            "r",
            encoding="utf-8",
        ) as f:
            config = json.load(f)

        ok("Model configuration JSON is valid")

    except Exception as exc:
        error(
            f"Invalid model configuration JSON: {exc}"
        )


# ============================================================
# ENTITY TABLE CONSISTENCY
# ============================================================

section("5. ENTITY TABLE CONSISTENCY")

entity_tables = {
    "V3 features":
        "outputs/entity_features_v3.csv",

    "V6 behavior":
        "outputs/entity_behavior_features_v6.csv",

    "V6.1 behavior":
        "outputs/entity_behavior_features_v61.csv",

    "Isolation Forest":
        "outputs/anomaly_model_v5.csv",

    "Final alerts":
        "outputs/final_entity_alerts_v61.csv",
}

entity_sets = {}

for name, relative_path in entity_tables.items():

    path = ROOT / relative_path

    if not path.exists():
        continue

    try:
        df = pd.read_csv(
            path,
            usecols=["entity_id"]
        )

    except Exception as exc:
        error(
            f"{name}: unable to read entity_id: {exc}"
        )
        continue

    count = len(df)
    unique = df["entity_id"].nunique()
    duplicates = count - unique

    print(
        f"{name:<20} "
        f"rows={count:>8,} "
        f"unique={unique:>8,} "
        f"duplicates={duplicates:>5,}"
    )

    if duplicates:
        error(
            f"{name} contains "
            f"{duplicates:,} duplicate entity IDs"
        )

    entity_sets[name] = set(
        df["entity_id"].astype(str)
    )


if entity_sets:

    names = list(entity_sets)

    reference_name = names[0]
    reference = entity_sets[reference_name]

    for name in names[1:]:

        current = entity_sets[name]

        missing = reference - current
        extra = current - reference

        if not missing and not extra:
            ok(
                f"{name} entity universe matches "
                f"{reference_name}"
            )

        else:
            error(
                f"{name} entity mismatch: "
                f"{len(missing):,} missing, "
                f"{len(extra):,} extra"
            )


# ============================================================
# V6.1 FEATURE VALIDATION
# ============================================================

section("6. V6.1 FEATURE VALIDATION")

behavior_path = (
    ROOT /
    "outputs/entity_behavior_features_v61.csv"
)

if behavior_path.exists():

    behavior = pd.read_csv(
        behavior_path
    )

    v6 = [
        c for c in behavior.columns
        if c.startswith("v6_")
    ]

    v61 = [
        c for c in behavior.columns
        if c.startswith("v61_")
    ]

    print(
        f"Rows             : {len(behavior):,}"
    )

    print(
        f"V6 features      : {len(v6)}"
    )

    print(
        f"V6.1 features    : {len(v61)}"
    )

    print(
        f"Behavior total   : {len(v6) + len(v61)}"
    )

    if len(v6) != 47:
        warning(
            f"Expected 47 V6 features, found {len(v6)}"
        )
    else:
        ok("47 V6 behavioral features found")

    if len(v61) != 36:
        warning(
            f"Expected 36 V6.1 features, found {len(v61)}"
        )
    else:
        ok("36 V6.1 behavioral features found")

    behavior_columns = v6 + v61

    if behavior_columns:

        nan_count = int(
            behavior[
                behavior_columns
            ].isna().sum().sum()
        )

        print(
            f"Behavior NaNs    : {nan_count:,}"
        )

        if nan_count:
            error(
                f"Behavior table contains "
                f"{nan_count:,} NaN values"
            )
        else:
            ok("No missing behavioral feature values")


# ============================================================
# FINAL ALERT VALIDATION
# ============================================================

section("7. FINAL ALERT VALIDATION")

alert_path = (
    ROOT /
    "outputs/final_entity_alerts_v61.csv"
)

if alert_path.exists():

    alerts = pd.read_csv(
        alert_path
    )

    required_columns = [
        "entity_id",
        "behavior_probability",
        "novelty_percentile",
        "risk_priority",
        "reason_codes",
    ]

    for column in required_columns:

        if column not in alerts.columns:
            error(
                f"Final alerts missing column: {column}"
            )
        else:
            ok(
                f"Alert column present: {column}"
            )

    if "entity_id" in alerts.columns:

        duplicates = int(
            alerts["entity_id"].duplicated().sum()
        )

        if duplicates:
            error(
                f"Final alerts contain "
                f"{duplicates:,} duplicate entities"
            )
        else:
            ok("Final alert entity IDs are unique")

    if "behavior_probability" in alerts.columns:

        probability = pd.to_numeric(
            alerts["behavior_probability"],
            errors="coerce",
        )

        invalid = (
            probability.isna()
            | (probability < 0)
            | (probability > 1)
        )

        if invalid.any():
            error(
                f"{int(invalid.sum()):,} invalid "
                "behavior probabilities"
            )
        else:
            ok(
                "Behavior probabilities are within [0, 1]"
            )

    if "novelty_percentile" in alerts.columns:

        novelty = pd.to_numeric(
            alerts["novelty_percentile"],
            errors="coerce",
        )

        invalid = (
            novelty.isna()
            | (novelty < 0)
            | (novelty > 1)
        )

        if invalid.any():
            error(
                f"{int(invalid.sum()):,} invalid "
                "novelty percentiles"
            )
        else:
            ok(
                "Novelty percentiles are within [0, 1]"
            )

    if "risk_priority" in alerts.columns:

        allowed = {
            "LOW",
            "MEDIUM",
            "HIGH",
            "CRITICAL",
        }

        observed = set(
            alerts[
                "risk_priority"
            ]
            .dropna()
            .astype(str)
        )

        unknown = observed - allowed

        if unknown:
            error(
                "Unknown risk priorities: "
                + ", ".join(sorted(unknown))
            )

        else:
            ok("Risk priority labels are valid")

        print()
        print("Risk distribution:")

        counts = (
            alerts[
                "risk_priority"
            ]
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
                f"{int(counts.get(priority, 0)):>8,}"
            )


# ============================================================
# GROUND TRUTH
# ============================================================

section("8. GROUND-TRUTH CHECK")

gt_path = (
    ROOT /
    "outputs/ground_truth_resolved_v3.csv"
)

if gt_path.exists():

    gt = pd.read_csv(
        gt_path
    )

    print(
        f"Resolved rows : {len(gt):,}"
    )

    if "is_anomaly" in gt.columns:

        labels = pd.to_numeric(
            gt["is_anomaly"],
            errors="coerce",
        )

        benign = int(
            (labels == 0).sum()
        )

        anomaly = int(
            (labels == 1).sum()
        )

        print(
            f"Benign        : {benign:,}"
        )

        print(
            f"Anomaly       : {anomaly:,}"
        )

        ok("Ground-truth labels readable")


# ============================================================
# FINAL RESULT
# ============================================================

section("VALIDATION RESULT")

print(
    f"Errors   : {len(ERRORS)}"
)

print(
    f"Warnings : {len(WARNINGS)}"
)

if ERRORS:

    print()
    print("PROJECT VALIDATION FAILED")

    print()
    print("Errors:")

    for item in ERRORS:
        print(f"  - {item}")

    sys.exit(1)


if WARNINGS:

    print()
    print(
        "PROJECT VALIDATION PASSED WITH WARNINGS"
    )

    for item in WARNINGS:
        print(f"  - {item}")

    sys.exit(0)


print()
print("PROJECT VALIDATION PASSED")
print("V6.1 artifacts are internally consistent.")

sys.exit(0)

