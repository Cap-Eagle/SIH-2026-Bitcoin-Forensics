#!/usr/bin/env python3

"""
FINAL GROUP-B BITCOIN FORENSICS DETECTOR — V6.1

Primary model:
    Random Forest trained using V6 + V6.1 behavioral features.

Secondary model:
    Existing Isolation Forest novelty score.

Outputs:
    models/final_detector_v61.joblib
    models/final_preprocessor_v61.joblib
    models/final_detector_config_v61.json

    outputs/final_entity_alerts_v61.csv
    outputs/final_detector_evaluation_v61.csv
    outputs/final_detector_summary_v61.txt

IMPORTANT:
    Behavioral features are constructed without ground truth.

    Ground truth is used here only for:
        - supervised training
        - validation threshold selection
        - held-out benchmark evaluation

    The final model is retrained on all labelled entities AFTER
    the held-out benchmark is completed.
"""

from pathlib import Path
import json

import joblib
import numpy as np
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = (
    ROOT / "outputs/entity_behavior_features_v61.csv"
)

GT_FILE = (
    ROOT / "outputs/ground_truth_resolved_v3.csv"
)

IF_FILE = (
    ROOT / "outputs/anomaly_model_v5.csv"
)

MODEL_DIR = ROOT / "models"

MODEL_FILE = (
    MODEL_DIR / "final_detector_v61.joblib"
)

CONFIG_FILE = (
    MODEL_DIR / "final_detector_config_v61.json"
)

ALERT_FILE = (
    ROOT / "outputs/final_entity_alerts_v61.csv"
)

EVAL_FILE = (
    ROOT / "outputs/final_detector_evaluation_v61.csv"
)

SUMMARY_FILE = (
    ROOT / "outputs/final_detector_summary_v61.txt"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42
N_ESTIMATORS = 300

VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

MIN_RECALL = 0.75


# ============================================================
# HELPERS
# ============================================================

def choose_threshold(
    y_true,
    probabilities,
    minimum_recall=0.75,
):
    thresholds = np.unique(
        probabilities
    )

    candidates = []

    for threshold in thresholds:

        prediction = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_true,
            prediction,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            prediction,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            prediction,
            zero_division=0,
        )

        if recall >= minimum_recall:

            candidates.append({
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            })

    if not candidates:
        raise RuntimeError(
            "No validation threshold satisfies minimum recall."
        )

    return sorted(
        candidates,
        key=lambda x: (
            x["precision"],
            x["f1"],
            x["recall"],
        ),
        reverse=True,
    )[0]


def evaluate(
    y_true,
    probability,
    threshold,
):

    prediction = (
        probability >= threshold
    ).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        prediction,
        labels=[0, 1],
    ).ravel()

    return {
        "accuracy":
            accuracy_score(
                y_true,
                prediction,
            ),

        "precision":
            precision_score(
                y_true,
                prediction,
                zero_division=0,
            ),

        "recall":
            recall_score(
                y_true,
                prediction,
                zero_division=0,
            ),

        "f1":
            f1_score(
                y_true,
                prediction,
                zero_division=0,
            ),

        "roc_auc":
            roc_auc_score(
                y_true,
                probability,
            ),

        "average_precision":
            average_precision_score(
                y_true,
                probability,
            ),

        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def reason_codes(row):

    reasons = []

    if (
        row.get(
            "v6_fanout_tx_ratio",
            0,
        ) > 0
    ):
        reasons.append(
            "FAN_OUT_BURST"
        )

    if (
        row.get(
            "v6_burst_interval_ratio",
            0,
        ) >= 0.25
    ):
        reasons.append(
            "TRANSACTION_BURST"
        )

    if (
        row.get(
            "v61_structuring_band_count",
            0,
        ) >= 3
    ):
        reasons.append(
            "STRUCTURING_PATTERN"
        )

    if (
        row.get(
            "v61_small_peel_output_count",
            0,
        ) >= 2
    ):
        reasons.append(
            "PEEL_CHAIN_PATTERN"
        )

    if (
        row.get(
            "v61_rapid_successor_ratio",
            0,
        ) > 0
    ):
        reasons.append(
            "RAPID_FORWARDING"
        )

    if (
        row.get(
            "v61_rapid_chain_depth",
            0,
        ) >= 3
    ):
        reasons.append(
            "RAPID_MULTI_HOP_CHAIN"
        )

    if (
        row.get(
            "v61_layering_successor_count",
            0,
        ) > 0
        and
        row.get(
            "v61_mean_amount_retention",
            0,
        ) >= 0.90
    ):
        reasons.append(
            "LAYERING_PATTERN"
        )

    if (
        row.get(
            "novelty_percentile",
            0,
        ) >= 0.995
    ):
        reasons.append(
            "UNSUPERVISED_NOVELTY"
        )

    if not reasons:
        reasons.append(
            "MULTIVARIATE_BEHAVIORAL_ANOMALY"
        )

    return ";".join(reasons)


# ============================================================
# LOAD
# ============================================================

print("=" * 78)
print("FINAL BITCOIN FORENSICS DETECTOR — V6.1")
print("=" * 78)

features = pd.read_csv(
    FEATURE_FILE
)

gt = pd.read_csv(
    GT_FILE
)

isolation = pd.read_csv(
    IF_FILE
)

print(
    f"\nEntities           : {len(features):,}"
)

print(
    f"Ground truth       : {len(gt):,}"
)


# ============================================================
# FEATURE SET
# ============================================================

BEHAVIOR_FEATURES = [
    c
    for c in features.columns
    if (
        c.startswith("v6_")
        or c.startswith("v61_")
    )
]

print(
    f"Behavior features  : {len(BEHAVIOR_FEATURES):,}"
)


# ============================================================
# PREPARE LABELS
# ============================================================

gt["is_anomaly"] = pd.to_numeric(
    gt["is_anomaly"],
    errors="coerce",
)

gt = gt.dropna(
    subset=["is_anomaly"]
)

gt["is_anomaly"] = (
    gt["is_anomaly"]
    .astype(int)
)


gt_entity = (
    gt.groupby(
        "resolved_entity_id",
        as_index=False,
    )
    .agg(
        is_anomaly=(
            "is_anomaly",
            "max",
        )
    )
)


labelled = features.merge(
    gt_entity,
    left_on="entity_id",
    right_on="resolved_entity_id",
    how="inner",
)


print(
    f"Labelled entities  : {len(labelled):,}"
)

print(
    f"Benign             : "
    f"{(labelled['is_anomaly'] == 0).sum():,}"
)

print(
    f"Anomaly            : "
    f"{(labelled['is_anomaly'] == 1).sum():,}"
)


# ============================================================
# MATRICES
# ============================================================

X = labelled[
    BEHAVIOR_FEATURES
].copy()

X_all = features[
    BEHAVIOR_FEATURES
].copy()

for column in BEHAVIOR_FEATURES:

    X[column] = pd.to_numeric(
        X[column],
        errors="coerce",
    )

    X_all[column] = pd.to_numeric(
        X_all[column],
        errors="coerce",
    )


y = labelled[
    "is_anomaly"
].astype(int)


# ============================================================
# 70 / 15 / 15 SPLIT
# ============================================================

indices = np.arange(
    len(labelled)
)


train_idx, temp_idx = train_test_split(
    indices,
    test_size=(
        VALIDATION_SIZE
        + TEST_SIZE
    ),
    stratify=y,
    random_state=RANDOM_STATE,
)


relative_test_size = (
    TEST_SIZE
    /
    (
        VALIDATION_SIZE
        + TEST_SIZE
    )
)


val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=relative_test_size,
    stratify=y.iloc[temp_idx],
    random_state=RANDOM_STATE,
)


print("\nSplit:")

print(
    f"Train      : {len(train_idx):,}"
)

print(
    f"Validation : {len(val_idx):,}"
)

print(
    f"Test       : {len(test_idx):,}"
)


# ============================================================
# BENCHMARK MODEL
# ============================================================

benchmark_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median"
        ),
    ),
    (
        "scaler",
        RobustScaler(),
    ),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
            min_samples_leaf=2,
        ),
    ),
])


print(
    "\nTraining held-out benchmark model..."
)


benchmark_pipeline.fit(
    X.iloc[train_idx],
    y.iloc[train_idx],
)


val_probability = (
    benchmark_pipeline
    .predict_proba(
        X.iloc[val_idx]
    )[:, 1]
)


threshold_info = choose_threshold(
    y.iloc[val_idx],
    val_probability,
    MIN_RECALL,
)


threshold = (
    threshold_info[
        "threshold"
    ]
)


test_probability = (
    benchmark_pipeline
    .predict_proba(
        X.iloc[test_idx]
    )[:, 1]
)


metrics = evaluate(
    y.iloc[test_idx],
    test_probability,
    threshold,
)


# ============================================================
# PRINT FINAL HELD-OUT RESULT
# ============================================================

print("\n" + "=" * 78)
print("FINAL HELD-OUT BENCHMARK")
print("=" * 78)

print(
    f"Threshold          : {threshold:.6f}"
)

print(
    f"Accuracy           : {metrics['accuracy']:.4f}"
)

print(
    f"Precision          : {metrics['precision']:.4f}"
)

print(
    f"Recall             : {metrics['recall']:.4f}"
)

print(
    f"F1                 : {metrics['f1']:.4f}"
)

print(
    f"ROC-AUC            : {metrics['roc_auc']:.4f}"
)

print(
    f"Average Precision  : {metrics['average_precision']:.4f}"
)

print(
    f"TN / FP / FN / TP  : "
    f"{metrics['tn']} / "
    f"{metrics['fp']} / "
    f"{metrics['fn']} / "
    f"{metrics['tp']}"
)


# ============================================================
# SAVE BENCHMARK
# ============================================================

evaluation = pd.DataFrame([
    {
        "threshold":
            threshold,

        **metrics,
    }
])


evaluation.to_csv(
    EVAL_FILE,
    index=False,
)


# ============================================================
# TRAIN FINAL DEPLOYMENT MODEL
# ============================================================

print(
    "\nTraining final deployment model "
    "using all labelled entities..."
)


final_pipeline = Pipeline([
    (
        "imputer",
        SimpleImputer(
            strategy="median"
        ),
    ),
    (
        "scaler",
        RobustScaler(),
    ),
    (
        "classifier",
        RandomForestClassifier(
            n_estimators=N_ESTIMATORS,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            class_weight="balanced",
            min_samples_leaf=2,
        ),
    ),
])


final_pipeline.fit(
    X,
    y,
)


MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


joblib.dump(
    final_pipeline,
    MODEL_FILE,
)


# ============================================================
# SCORE ALL ENTITIES
# ============================================================

print(
    "Scoring all entities..."
)


probability = (
    final_pipeline
    .predict_proba(
        X_all
    )[:, 1]
)


alerts = features.copy()


alerts[
    "behavior_probability"
] = probability


alerts[
    "behavior_alert"
] = (
    alerts[
        "behavior_probability"
    ]
    >= threshold
).astype(int)


# ============================================================
# ADD ISOLATION FOREST NOVELTY
# ============================================================

isolation = isolation[
    [
        "entity_id",
        "anomaly_score",
    ]
].copy()


isolation = isolation.rename(
    columns={
        "anomaly_score":
            "isolation_raw_score",
    }
)


alerts = alerts.merge(
    isolation,
    on="entity_id",
    how="left",
)


alerts[
    "isolation_raw_score"
] = (
    pd.to_numeric(
        alerts[
            "isolation_raw_score"
        ],
        errors="coerce",
    )
    .fillna(
        alerts[
            "isolation_raw_score"
        ].median()
    )
)


alerts[
    "novelty_percentile"
] = (
    alerts[
        "isolation_raw_score"
    ]
    .rank(
        pct=True,
        method="average",
    )
)


# ============================================================
# ALERT PRIORITY
# ============================================================

def priority(row):

    p = row[
        "behavior_probability"
    ]

    novelty = row[
        "novelty_percentile"
    ]

    if (
        p >= 0.90
        or (
            p >= threshold
            and novelty >= 0.99
        )
    ):
        return "CRITICAL"

    if p >= threshold:
        return "HIGH"

    if (
        p >= 0.50
        or novelty >= 0.995
    ):
        return "MEDIUM"

    return "LOW"


alerts[
    "risk_priority"
] = alerts.apply(
    priority,
    axis=1,
)


# ============================================================
# REASON CODES
# ============================================================

alerts[
    "reason_codes"
] = alerts.apply(
    reason_codes,
    axis=1,
)


# ============================================================
# RANK
# ============================================================

alerts[
    "risk_score"
] = alerts[
    "behavior_probability"
]


alerts = alerts.sort_values(
    [
        "behavior_probability",
        "novelty_percentile",
    ],
    ascending=False,
).reset_index(
    drop=True
)


alerts.insert(
    0,
    "rank",
    np.arange(
        1,
        len(alerts) + 1,
    ),
)


# ============================================================
# SAVE CONFIG
# ============================================================

config = {
    "version":
        "V6.1",

    "primary_model":
        "RandomForestClassifier",

    "primary_features":
        "V6 + V6.1 behavioral features",

    "feature_count":
        len(BEHAVIOR_FEATURES),

    "n_estimators":
        N_ESTIMATORS,

    "random_state":
        RANDOM_STATE,

    "threshold":
        threshold,

    "minimum_validation_recall":
        MIN_RECALL,

    "secondary_model":
        "Isolation Forest novelty score",

    "held_out_metrics":
        metrics,
}


CONFIG_FILE.write_text(
    json.dumps(
        config,
        indent=2,
    ),
    encoding="utf-8",
)


# ============================================================
# SAVE ALERTS
# ============================================================

alerts.to_csv(
    ALERT_FILE,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

priority_counts = (
    alerts[
        "risk_priority"
    ]
    .value_counts()
    .reindex(
        [
            "CRITICAL",
            "HIGH",
            "MEDIUM",
            "LOW",
        ],
        fill_value=0,
    )
)


summary = f"""
FINAL BITCOIN FORENSICS DETECTOR — V6.1
==============================================================================

ARCHITECTURE
------------------------------------------------------------------------------
Primary model:
    Random Forest behavioral classifier

Primary features:
    V6 + V6.1 behavioral features ({len(BEHAVIOR_FEATURES)} features)

Secondary signal:
    Isolation Forest novelty percentile

Feature construction:
    Ground-truth independent

HELD-OUT BENCHMARK
------------------------------------------------------------------------------
Threshold:          {threshold:.6f}

Accuracy:           {metrics['accuracy']:.6f}
Precision:          {metrics['precision']:.6f}
Recall:             {metrics['recall']:.6f}
F1:                 {metrics['f1']:.6f}
ROC-AUC:            {metrics['roc_auc']:.6f}
Average Precision:  {metrics['average_precision']:.6f}

TN: {metrics['tn']}
FP: {metrics['fp']}
FN: {metrics['fn']}
TP: {metrics['tp']}

FINAL DEPLOYMENT
------------------------------------------------------------------------------
Entities scored: {len(alerts):,}

CRITICAL: {priority_counts['CRITICAL']:,}
HIGH:     {priority_counts['HIGH']:,}
MEDIUM:   {priority_counts['MEDIUM']:,}
LOW:      {priority_counts['LOW']:,}

OUTPUT
------------------------------------------------------------------------------
Alerts:
{ALERT_FILE}

Model:
{MODEL_FILE}

Configuration:
{CONFIG_FILE}

IMPORTANT
------------------------------------------------------------------------------
The held-out metrics belong to the 70/15/15 benchmark model.

After benchmark evaluation, the deployment model was retrained using all
1,140 labelled entities so that all available labelled information is used
for the final demonstration/deployment system.

Isolation Forest remains available as a novelty signal for entities whose
behavior differs from patterns represented in the labelled training data.
""".strip()


SUMMARY_FILE.write_text(
    summary,
    encoding="utf-8",
)


print("\n" + "=" * 78)
print("FINAL DETECTOR COMPLETE")
print("=" * 78)

print(summary)

print("\nTop 20 investigative leads:")

print(
    alerts[
        [
            "rank",
            "entity_id",
            "behavior_probability",
            "novelty_percentile",
            "risk_priority",
            "reason_codes",
        ]
    ]
    .head(20)
    .to_string(
        index=False
    )
)

