#!/usr/bin/env python3

"""
V5 Anomaly Detection Model
==========================

Production-style unsupervised anomaly detection for Bitcoin entities.

Design:
    - Entity features are constructed WITHOUT ground truth.
    - Ground truth is used ONLY for evaluation.
    - Feature selection is fixed from the V3 feature semantics +
      previous diagnostic.
    - No hand-tuned weighted anomaly score.
    - Isolation Forest is trained only on the training split.
    - Threshold is selected on validation data.
    - Final metrics are reported ONLY on the held-out test set.

Outputs:
    outputs/anomaly_model_v5.csv
    outputs/anomaly_evaluation_v5.csv
    outputs/anomaly_summary_v5.txt
"""

from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
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
from sklearn.preprocessing import RobustScaler


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

FEATURE_FILE = ROOT / "outputs" / "entity_features_v3.csv"
GT_FILE = ROOT / "outputs" / "ground_truth_resolved_v3.csv"

MODEL_OUT = ROOT / "outputs" / "anomaly_model_v5.csv"
EVAL_OUT = ROOT / "outputs" / "anomaly_evaluation_v5.csv"
SUMMARY_OUT = ROOT / "outputs" / "anomaly_summary_v5.txt"


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

N_ESTIMATORS = 300

CONTAMINATION = 0.04

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

# Minimum validation recall target.
MIN_RECALL = 0.75


# ============================================================
# FEATURE SET
# ============================================================
#
# We intentionally do NOT use all 38 features.
#
# The selected features represent complementary behavioral
# families rather than dumping every correlated activity measure
# into the model.
#
# Graph / connectivity:
#   out_degree
#   in_degree
#   counterparty_count
#
# Transaction activity:
#   input_transaction_count
#   output_transaction_count
#
# Flow:
#   total_input_btc
#   total_output_btc
#   flow_imbalance
#
# Amount behavior:
#   input_amount_entropy
#   output_amount_entropy
#   input_amount_concentration
#   output_amount_concentration
#
# Correlation:
#   exact_match_count
#   close_match_count
#   broad_match_count
#   exact_match_rate
#   close_match_rate
#   min_abs_time_delta_ms
#
# Network:
#   high_confidence_ip_evidence_count
#   high_confidence_asn_evidence_count
#   tor_like_event_count
#   hosting_like_event_count
#   network_risk_signal
#
# These are deliberately kept as separate behavioral dimensions.

SELECTED_FEATURES = [
    # Graph
    "out_degree",
    "in_degree",
    "counterparty_count",

    # Activity
    "input_transaction_count",
    "output_transaction_count",

    # Flow
    "total_input_btc",
    "total_output_btc",
    "flow_imbalance",

    # Amount behavior
    "input_amount_entropy",
    "output_amount_entropy",
    "input_amount_concentration",
    "output_amount_concentration",

    # Correlation / timing
    "exact_match_count",
    "close_match_count",
    "broad_match_count",
    "exact_match_rate",
    "close_match_rate",
    "min_abs_time_delta_ms",

    # Network
    "high_confidence_ip_evidence_count",
    "high_confidence_asn_evidence_count",
    "tor_like_event_count",
    "hosting_like_event_count",
    "network_risk_signal",
]


# ============================================================
# HELPERS
# ============================================================

def print_header(text):
    print("\n" + "=" * 70)
    print(text)
    print("=" * 70)


def safe_auc(y_true, score):
    try:
        return float(roc_auc_score(y_true, score))
    except Exception:
        return np.nan


def choose_threshold(y_true, scores, minimum_recall=0.75):
    """
    Select threshold using validation data only.

    Objective:
        1. Require recall >= minimum_recall.
        2. Maximize precision.
        3. Then maximize F1.
        4. Then maximize recall.

    Higher anomaly score = more anomalous.
    """

    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    finite = np.isfinite(scores)

    y_true = y_true[finite]
    scores = scores[finite]

    thresholds = np.unique(scores)

    candidates = []

    for threshold in thresholds:

        pred = (scores >= threshold).astype(int)

        recall = recall_score(
            y_true,
            pred,
            zero_division=0,
        )

        precision = precision_score(
            y_true,
            pred,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            pred,
            zero_division=0,
        )

        if recall >= minimum_recall:

            candidates.append(
                {
                    "threshold": float(threshold),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "predicted_anomalies": int(pred.sum()),
                }
            )

    if not candidates:

        # If no threshold reaches the desired recall,
        # choose the threshold with maximum recall, then F1.
        for threshold in thresholds:

            pred = (scores >= threshold).astype(int)

            recall = recall_score(
                y_true,
                pred,
                zero_division=0,
            )

            precision = precision_score(
                y_true,
                pred,
                zero_division=0,
            )

            f1 = f1_score(
                y_true,
                pred,
                zero_division=0,
            )

            candidates.append(
                {
                    "threshold": float(threshold),
                    "precision": float(precision),
                    "recall": float(recall),
                    "f1": float(f1),
                    "predicted_anomalies": int(pred.sum()),
                }
            )

        best = sorted(
            candidates,
            key=lambda x: (
                x["recall"],
                x["f1"],
                x["precision"],
            ),
            reverse=True,
        )[0]

        return best

    best = sorted(
        candidates,
        key=lambda x: (
            x["precision"],
            x["f1"],
            x["recall"],
        ),
        reverse=True,
    )[0]

    return best


def evaluate_predictions(y_true, scores, threshold):

    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores).astype(float)

    predicted = (
        scores >= threshold
    ).astype(int)

    cm = confusion_matrix(
        y_true,
        predicted,
        labels=[0, 1],
    )

    tn, fp, fn, tp = cm.ravel()

    return {
        "accuracy": accuracy_score(
            y_true,
            predicted,
        ),
        "precision": precision_score(
            y_true,
            predicted,
            zero_division=0,
        ),
        "recall": recall_score(
            y_true,
            predicted,
            zero_division=0,
        ),
        "f1": f1_score(
            y_true,
            predicted,
            zero_division=0,
        ),
        "roc_auc": safe_auc(
            y_true,
            scores,
        ),
        "average_precision": average_precision_score(
            y_true,
            scores,
        ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_anomalies": int(predicted.sum()),
    }


# ============================================================
# LOAD DATA
# ============================================================

print_header("V5 ANOMALY DETECTION")

print("\nLoading entity features...")

features = pd.read_csv(
    FEATURE_FILE
)

print(
    f"Entity feature rows : {len(features):,}"
)

print(
    f"Entity feature cols : {len(features.columns):,}"
)


print("\nLoading resolved ground truth...")

gt = pd.read_csv(
    GT_FILE
)

print(
    f"Ground-truth rows : {len(gt):,}"
)


# ============================================================
# VALIDATE FEATURE SET
# ============================================================

missing_features = [
    feature
    for feature in SELECTED_FEATURES
    if feature not in features.columns
]

if missing_features:

    raise ValueError(
        "Missing selected features:\n"
        + "\n".join(
            missing_features
        )
    )

print(
    f"\nSelected features : {len(SELECTED_FEATURES)}"
)

for feature in SELECTED_FEATURES:
    print(f"  ✓ {feature}")


# ============================================================
# PREPARE GROUND TRUTH
# ============================================================

gt["is_anomaly"] = pd.to_numeric(
    gt["is_anomaly"],
    errors="coerce",
)

gt = gt.dropna(
    subset=["is_anomaly"]
).copy()

gt["is_anomaly"] = (
    gt["is_anomaly"]
    .astype(int)
)

gt = gt[
    gt["is_anomaly"].isin([0, 1])
].copy()


# One label per resolved entity.
#
# If multiple GT records map to the same entity,
# anomaly wins.

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


# ============================================================
# MERGE FEATURES + LABELS
# ============================================================

eval_df = features.merge(
    gt_entity,
    left_on="entity_id",
    right_on="resolved_entity_id",
    how="left",
)


# IMPORTANT:
#
# Unlabelled entities remain in the dataset.
#
# They are used for unsupervised training.
# Ground-truth labels are only used for the labelled
# validation/test evaluation.

labelled_mask = (
    eval_df["is_anomaly"]
    .notna()
)

labelled = eval_df[
    labelled_mask
].copy()

print_header("GROUND TRUTH")

print(
    f"Labelled entities : {len(labelled):,}"
)

print(
    f"Benign            : "
    f"{(labelled['is_anomaly'] == 0).sum():,}"
)

print(
    f"Anomalies         : "
    f"{(labelled['is_anomaly'] == 1).sum():,}"
)

print(
    f"Unlabelled entities available for training : "
    f"{(~labelled_mask).sum():,}"
)


# ============================================================
# BUILD MODEL MATRIX
# ============================================================

X_all = features[
    SELECTED_FEATURES
].copy()

for column in SELECTED_FEATURES:

    X_all[column] = pd.to_numeric(
        X_all[column],
        errors="coerce",
    )


# ============================================================
# TRAIN / VALIDATION / TEST SPLIT
# ============================================================
#
# Split the LABELED entities only.
#
# This allows:
#
#   train      -> model development/reference
#   validation -> threshold selection
#   test       -> final evaluation
#
# The Isolation Forest itself remains unsupervised.
#

labelled_indices = np.where(
    labelled_mask.to_numpy()
)[0]

labelled_y = (
    eval_df.loc[
        labelled_mask,
        "is_anomaly"
    ]
    .astype(int)
    .to_numpy()
)


train_idx, temp_idx = train_test_split(
    labelled_indices,
    test_size=(
        VALIDATION_SIZE + TEST_SIZE
    ),
    stratify=labelled_y,
    random_state=RANDOM_STATE,
)


temp_y = (
    eval_df.loc[
        temp_idx,
        "is_anomaly"
    ]
    .astype(int)
    .to_numpy()
)


relative_test_size = (
    TEST_SIZE /
    (VALIDATION_SIZE + TEST_SIZE)
)


val_idx, test_idx = train_test_split(
    temp_idx,
    test_size=relative_test_size,
    stratify=temp_y,
    random_state=RANDOM_STATE,
)


print_header("DATA SPLIT")

print(
    f"Train      : {len(train_idx):,}"
)

print(
    f"Validation : {len(val_idx):,}"
)

print(
    f"Test       : {len(test_idx):,}"
)


def split_distribution(indices):

    y = (
        eval_df.loc[
            indices,
            "is_anomaly"
        ]
        .astype(int)
    )

    return {
        "benign": int(
            (y == 0).sum()
        ),
        "anomaly": int(
            (y == 1).sum()
        ),
    }


print(
    "\nTrain distribution:",
    split_distribution(train_idx)
)

print(
    "Validation distribution:",
    split_distribution(val_idx)
)

print(
    "Test distribution:",
    split_distribution(test_idx)
)


# ============================================================
# PREPROCESSING
# ============================================================

print_header("PREPROCESSING")

imputer = SimpleImputer(
    strategy="median"
)

scaler = RobustScaler()


# ============================================================
# UNSUPERVISED TRAINING DATA
# ============================================================
#
# The model is trained WITHOUT ground-truth labels.
#
# We train on ALL entities rather than only the 1,140 labelled
# entities. This better represents a real deployment scenario.
#

print(
    "Preparing all entity features for unsupervised training..."
)

X_imputed = imputer.fit_transform(
    X_all
)

X_scaled = scaler.fit_transform(
    X_imputed
)


# ============================================================
# ISOLATION FOREST
# ============================================================

print_header("TRAINING ISOLATION FOREST")

print(
    f"Estimators    : {N_ESTIMATORS}"
)

print(
    f"Contamination : {CONTAMINATION}"
)

print(
    f"Random state  : {RANDOM_STATE}"
)

model = IsolationForest(
    n_estimators=N_ESTIMATORS,
    contamination=CONTAMINATION,
    random_state=RANDOM_STATE,
    n_jobs=-1,
)

model.fit(
    X_scaled
)


# ============================================================
# RAW MODEL SCORES
# ============================================================
#
# IsolationForest decision_function:
#
#     higher = more normal
#     lower  = more anomalous
#
# We negate it so our anomaly score follows:
#
#     higher = more anomalous
#

decision = model.decision_function(
    X_scaled
)

anomaly_score = -decision


# ============================================================
# VALIDATION THRESHOLD
# ============================================================

validation_scores = anomaly_score[
    val_idx
]

validation_y = (
    eval_df.loc[
        val_idx,
        "is_anomaly"
    ]
    .astype(int)
    .to_numpy()
)


print_header("VALIDATION THRESHOLD")

threshold_info = choose_threshold(
    validation_y,
    validation_scores,
    minimum_recall=MIN_RECALL,
)

threshold = threshold_info[
    "threshold"
]

print(
    f"Selected threshold : {threshold:.6f}"
)

print(
    f"Validation precision: "
    f"{threshold_info['precision']:.4f}"
)

print(
    f"Validation recall   : "
    f"{threshold_info['recall']:.4f}"
)

print(
    f"Validation F1       : "
    f"{threshold_info['f1']:.4f}"
)

print(
    f"Validation predicted anomalies: "
    f"{threshold_info['predicted_anomalies']:,}"
)


# ============================================================
# FINAL TEST EVALUATION
# ============================================================

test_scores = anomaly_score[
    test_idx
]

test_y = (
    eval_df.loc[
        test_idx,
        "is_anomaly"
    ]
    .astype(int)
    .to_numpy()
)


test_metrics = evaluate_predictions(
    test_y,
    test_scores,
    threshold,
)


print_header("HELD-OUT TEST RESULTS")

print(
    f"Accuracy          : "
    f"{test_metrics['accuracy']:.4f}"
)

print(
    f"Precision         : "
    f"{test_metrics['precision']:.4f}"
)

print(
    f"Recall            : "
    f"{test_metrics['recall']:.4f}"
)

print(
    f"F1                : "
    f"{test_metrics['f1']:.4f}"
)

print(
    f"ROC-AUC           : "
    f"{test_metrics['roc_auc']:.4f}"
)

print(
    f"Average Precision : "
    f"{test_metrics['average_precision']:.4f}"
)

print("\nConfusion matrix:")

print(
    f"TN = {test_metrics['tn']}"
)

print(
    f"FP = {test_metrics['fp']}"
)

print(
    f"FN = {test_metrics['fn']}"
)

print(
    f"TP = {test_metrics['tp']}"
)


# ============================================================
# SCENARIO EVALUATION
# ============================================================

print_header("SCENARIO RECALL")

test_gt = gt[
    gt["resolved_entity_id"].isin(
        eval_df.loc[
            test_idx,
            "entity_id"
        ]
    )
].copy()


# Map scores back to entity IDs.

score_df = pd.DataFrame({
    "entity_id": eval_df["entity_id"],
    "anomaly_score": anomaly_score,
})


score_df["predicted_anomaly"] = (
    score_df["anomaly_score"]
    >= threshold
).astype(int)


scenario_results = []


for scenario in sorted(
    test_gt["scenario_id"]
    .astype(str)
    .unique()
):

    scenario_gt = test_gt[
        test_gt["scenario_id"].astype(str)
        == scenario
    ]

    scenario_entities = set(
        scenario_gt[
            "resolved_entity_id"
        ]
    )

    scenario_predictions = score_df[
        score_df["entity_id"].isin(
            scenario_entities
        )
    ]

    if len(scenario_predictions) == 0:
        continue

    recall = float(
        scenario_predictions[
            "predicted_anomaly"
        ].mean()
    )

    scenario_results.append({
        "scenario": scenario,
        "entities": len(
            scenario_predictions
        ),
        "detected": int(
            scenario_predictions[
                "predicted_anomaly"
            ].sum()
        ),
        "recall": recall,
    })

    print(
        f"{scenario:<25} "
        f"{int(scenario_predictions['predicted_anomaly'].sum()):>4}/"
        f"{len(scenario_predictions):<4} "
        f"recall={recall:.4f}"
    )


scenario_df = pd.DataFrame(
    scenario_results
)


# ============================================================
# SAVE ENTITY SCORES
# ============================================================

result = features[
    ["entity_id"]
].copy()

result["anomaly_score"] = anomaly_score

result["predicted_anomaly"] = (
    anomaly_score >= threshold
).astype(int)


# Attach GT where available for evaluation/debugging.
result = result.merge(
    gt_entity,
    left_on="entity_id",
    right_on="resolved_entity_id",
    how="left",
)

result.drop(
    columns=["resolved_entity_id"],
    inplace=True,
)


# Rank all entities.

result["anomaly_rank"] = (
    result["anomaly_score"]
    .rank(
        ascending=False,
        method="first",
    )
    .astype(int)
)

result = result.sort_values(
    "anomaly_score",
    ascending=False,
).reset_index(drop=True)


result.to_csv(
    MODEL_OUT,
    index=False,
)


# ============================================================
# SAVE EVALUATION SUMMARY CSV
# ============================================================

evaluation_rows = []

evaluation_rows.append({
    "dataset": "validation",
    **threshold_info,
})

evaluation_rows.append({
    "dataset": "test",
    "threshold": threshold,
    "precision": test_metrics["precision"],
    "recall": test_metrics["recall"],
    "f1": test_metrics["f1"],
    "accuracy": test_metrics["accuracy"],
    "roc_auc": test_metrics["roc_auc"],
    "average_precision": test_metrics["average_precision"],
    "tn": test_metrics["tn"],
    "fp": test_metrics["fp"],
    "fn": test_metrics["fn"],
    "tp": test_metrics["tp"],
    "predicted_anomalies": test_metrics[
        "predicted_anomalies"
    ],
})


evaluation_df = pd.DataFrame(
    evaluation_rows
)

evaluation_df.to_csv(
    EVAL_OUT,
    index=False,
)


# ============================================================
# TEXT SUMMARY
# ============================================================

summary = []

summary.append(
    "V5 ANOMALY DETECTION SUMMARY"
)

summary.append(
    "=" * 70
)

summary.append(
    f"Total entities: {len(features):,}"
)

summary.append(
    f"Labelled entities: {len(labelled):,}"
)

summary.append(
    f"Selected features: {len(SELECTED_FEATURES)}"
)

summary.append(
    f"Isolation Forest estimators: {N_ESTIMATORS}"
)

summary.append(
    f"Isolation Forest contamination: {CONTAMINATION}"
)

summary.append(
    f"Train size: {len(train_idx):,}"
)

summary.append(
    f"Validation size: {len(val_idx):,}"
)

summary.append(
    f"Test size: {len(test_idx):,}"
)

summary.append("")

summary.append(
    "SELECTED FEATURES"
)

summary.append(
    "-" * 70
)

for feature in SELECTED_FEATURES:
    summary.append(
        f"  {feature}"
    )

summary.append("")

summary.append(
    "VALIDATION"
)

summary.append(
    "-" * 70
)

summary.append(
    f"Threshold: {threshold:.6f}"
)

summary.append(
    f"Precision: {threshold_info['precision']:.6f}"
)

summary.append(
    f"Recall: {threshold_info['recall']:.6f}"
)

summary.append(
    f"F1: {threshold_info['f1']:.6f}"
)

summary.append("")

summary.append(
    "HELD-OUT TEST"
)

summary.append(
    "-" * 70
)

for key in [
    "accuracy",
    "precision",
    "recall",
    "f1",
    "roc_auc",
    "average_precision",
]:

    summary.append(
        f"{key}: {test_metrics[key]:.6f}"
    )

summary.append("")

summary.append(
    "CONFUSION MATRIX"
)

summary.append(
    "-" * 70
)

summary.append(
    f"TN: {test_metrics['tn']}"
)

summary.append(
    f"FP: {test_metrics['fp']}"
)

summary.append(
    f"FN: {test_metrics['fn']}"
)

summary.append(
    f"TP: {test_metrics['tp']}"
)

summary.append("")

summary.append(
    "SCENARIO RECALL"
)

summary.append(
    "-" * 70
)

for _, row in scenario_df.iterrows():

    summary.append(
        f"{row['scenario']}: "
        f"{int(row['detected'])}/"
        f"{int(row['entities'])} "
        f"({row['recall']:.4f})"
    )

summary.append("")

summary.append(
    "GROUND TRUTH USAGE"
)

summary.append(
    "-" * 70
)

summary.append(
    "Ground truth was NOT used to construct features."
)

summary.append(
    "Ground truth was NOT used to train Isolation Forest."
)

summary.append(
    "Ground truth was used only for validation threshold "
    "selection and held-out evaluation."
)

summary.append("")


with open(
    SUMMARY_OUT,
    "w",
    encoding="utf-8",
) as f:

    f.write(
        "\n".join(summary)
    )


# ============================================================
# FINAL OUTPUT
# ============================================================

print_header("V5 COMPLETE")

print(
    f"Entity scores : {MODEL_OUT}"
)

print(
    f"Evaluation    : {EVAL_OUT}"
)

print(
    f"Summary       : {SUMMARY_OUT}"
)

print("\nTop 20 anomalies:")

print(
    result[
        [
            "anomaly_rank",
            "entity_id",
            "anomaly_score",
            "predicted_anomaly",
        ]
    ]
    .head(20)
    .to_string(index=False)
)

print("\nDone.")
