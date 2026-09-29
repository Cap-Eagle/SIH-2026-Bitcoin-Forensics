#!/usr/bin/env python3

"""
V6.1 Behavioral Feature Benchmark
=================================

Purpose:
    Compare three supervised diagnostic feature configurations:

        A. V5_BASELINE
           Original 23 V5 features.

        B. V61_BEHAVIOR_ONLY
           V6 + V6.1 behavioral features.

        C. V5_PLUS_V61
           Original V5 features + V6/V6.1 behavioral features.

IMPORTANT:
    This is a DIAGNOSTIC benchmark only.

    Ground truth is used for:
        - train / validation / test splitting
        - supervised Random Forest training
        - validation threshold selection
        - held-out evaluation

    Ground truth is NOT used to construct V6/V6.1 features.

Methodology:
    - Same labelled entities for all experiments
    - Same 70/15/15 stratified split
    - Same Random Forest configuration
    - Same validation threshold-selection policy
    - Same held-out test entities

This makes the three experiments directly comparable.

Inputs:
    outputs/entity_features_v3.csv
    outputs/entity_behavior_features_v61.csv
    outputs/ground_truth_resolved_v3.csv

Outputs:
    outputs/behavior_benchmark_v61.csv
    outputs/behavior_benchmark_scenarios_v61.csv
    outputs/behavior_feature_importance_v61.csv
    outputs/behavior_benchmark_summary_v61.txt
"""

from pathlib import Path

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
from sklearn.preprocessing import RobustScaler


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

V5_FEATURE_FILE = (
    ROOT / "outputs" / "entity_features_v3.csv"
)

V61_FEATURE_FILE = (
    ROOT / "outputs" / "entity_behavior_features_v61.csv"
)

GT_FILE = (
    ROOT / "outputs" / "ground_truth_resolved_v3.csv"
)

RESULT_FILE = (
    ROOT / "outputs" / "behavior_benchmark_v61.csv"
)

SCENARIO_FILE = (
    ROOT / "outputs" / "behavior_benchmark_scenarios_v61.csv"
)

IMPORTANCE_FILE = (
    ROOT / "outputs" / "behavior_feature_importance_v61.csv"
)

SUMMARY_FILE = (
    ROOT / "outputs" / "behavior_benchmark_summary_v61.txt"
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_STATE = 42

N_ESTIMATORS = 300

TRAIN_SIZE = 0.70
VALIDATION_SIZE = 0.15
TEST_SIZE = 0.15

MIN_RECALL = 0.75


# ============================================================
# ORIGINAL V5 FEATURES
# ============================================================

V5_FEATURES = [

    # Graph / connectivity
    "out_degree",
    "in_degree",
    "counterparty_count",

    # Transaction activity
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

def safe_auc(y_true, scores):

    try:

        return float(
            roc_auc_score(
                y_true,
                scores,
            )
        )

    except Exception:

        return np.nan


def choose_threshold(
    y_true,
    probabilities,
    minimum_recall=0.75,
):
    """
    Same threshold-selection policy as V5.

    Priority:
        1. Recall >= minimum target
        2. Highest precision
        3. Highest F1
        4. Highest recall

    Validation data ONLY.
    """

    y_true = np.asarray(
        y_true
    ).astype(int)

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    thresholds = np.unique(
        probabilities
    )

    candidates = []

    for threshold in thresholds:

        predicted = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_true,
            predicted,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            predicted,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            predicted,
            zero_division=0,
        )

        if recall >= minimum_recall:

            candidates.append({
                "threshold": float(
                    threshold
                ),
                "precision": float(
                    precision
                ),
                "recall": float(
                    recall
                ),
                "f1": float(
                    f1
                ),
                "predicted_anomalies": int(
                    predicted.sum()
                ),
            })

    if candidates:

        return sorted(
            candidates,
            key=lambda x: (
                x["precision"],
                x["f1"],
                x["recall"],
            ),
            reverse=True,
        )[0]

    # Fallback.

    candidates = []

    for threshold in thresholds:

        predicted = (
            probabilities >= threshold
        ).astype(int)

        precision = precision_score(
            y_true,
            predicted,
            zero_division=0,
        )

        recall = recall_score(
            y_true,
            predicted,
            zero_division=0,
        )

        f1 = f1_score(
            y_true,
            predicted,
            zero_division=0,
        )

        candidates.append({
            "threshold": float(
                threshold
            ),
            "precision": float(
                precision
            ),
            "recall": float(
                recall
            ),
            "f1": float(
                f1
            ),
            "predicted_anomalies": int(
                predicted.sum()
            ),
        })

    return sorted(
        candidates,
        key=lambda x: (
            x["recall"],
            x["f1"],
            x["precision"],
        ),
        reverse=True,
    )[0]


def evaluate(
    y_true,
    probabilities,
    threshold,
):

    y_true = np.asarray(
        y_true
    ).astype(int)

    probabilities = np.asarray(
        probabilities,
        dtype=float,
    )

    predicted = (
        probabilities >= threshold
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
            probabilities,
        ),
        "average_precision":
            average_precision_score(
                y_true,
                probabilities,
            ),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "predicted_anomalies": int(
            predicted.sum()
        ),
    }


# ============================================================
# START
# ============================================================

print("=" * 78)
print("V6.1 BEHAVIORAL FEATURE BENCHMARK")
print("=" * 78)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading original V5 features...")

v5 = pd.read_csv(
    V5_FEATURE_FILE
)

print(
    f"V5 entities       : {len(v5):,}"
)

print(
    f"V5 columns        : {len(v5.columns):,}"
)


print("\nLoading V6.1 behavioral features...")

v61 = pd.read_csv(
    V61_FEATURE_FILE
)

print(
    f"V6.1 entities     : {len(v61):,}"
)

print(
    f"V6.1 columns      : {len(v61.columns):,}"
)


print("\nLoading resolved ground truth...")

gt = pd.read_csv(
    GT_FILE
)

print(
    f"Ground-truth rows : {len(gt):,}"
)


# ============================================================
# VALIDATE ENTITY IDS
# ============================================================

for name, df in [
    ("V5", v5),
    ("V6.1", v61),
]:

    if "entity_id" not in df.columns:

        raise ValueError(
            f"{name} has no entity_id column."
        )

    duplicates = int(
        df["entity_id"].duplicated().sum()
    )

    if duplicates:

        raise ValueError(
            f"{name} contains "
            f"{duplicates:,} duplicate entity IDs."
        )


# ============================================================
# VALIDATE V5 FEATURES
# ============================================================

missing_v5 = [
    feature
    for feature in V5_FEATURES
    if feature not in v5.columns
]

if missing_v5:

    raise ValueError(
        "Missing V5 features:\n"
        + "\n".join(missing_v5)
    )


# ============================================================
# DETECT V6 / V6.1 FEATURES
# ============================================================

BEHAVIOR_FEATURES = [
    column
    for column in v61.columns
    if (
        column.startswith("v6_")
        or column.startswith("v61_")
    )
]


if not BEHAVIOR_FEATURES:

    raise ValueError(
        "No v6_ or v61_ behavioral features "
        "were found."
    )


V6_FEATURES = [
    column
    for column in BEHAVIOR_FEATURES
    if column.startswith("v6_")
]


V61_NEW_FEATURES = [
    column
    for column in BEHAVIOR_FEATURES
    if column.startswith("v61_")
]


print("\nBehavior feature discovery:")

print(
    f"V6 features       : "
    f"{len(V6_FEATURES):,}"
)

print(
    f"V6.1 new features : "
    f"{len(V61_NEW_FEATURES):,}"
)

print(
    f"Behavior total    : "
    f"{len(BEHAVIOR_FEATURES):,}"
)


# ============================================================
# MERGE FEATURE TABLES
# ============================================================

print("\nMerging feature tables...")


behavior_subset = v61[
    ["entity_id"] + BEHAVIOR_FEATURES
].copy()


features = v5.merge(
    behavior_subset,
    on="entity_id",
    how="inner",
    validate="one_to_one",
)


print(
    f"Merged entities   : {len(features):,}"
)


if len(features) != len(v5):

    print(
        "WARNING: merged entity count differs "
        "from original V5 entity count."
    )


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
# MERGE LABELS
# ============================================================

labelled = features.merge(
    gt_entity,
    left_on="entity_id",
    right_on="resolved_entity_id",
    how="inner",
)


print("\n" + "=" * 78)
print("LABELLED DATASET")
print("=" * 78)

print(
    f"Labelled entities : {len(labelled):,}"
)

print(
    f"Benign            : "
    f"{(labelled['is_anomaly'] == 0).sum():,}"
)

print(
    f"Anomaly           : "
    f"{(labelled['is_anomaly'] == 1).sum():,}"
)


# ============================================================
# CREATE ONE SHARED SPLIT
# ============================================================

print("\nCreating ONE shared stratified 70/15/15 split...")


indices = np.arange(
    len(labelled)
)

labels = labelled[
    "is_anomaly"
].astype(int).to_numpy()


train_idx, temp_idx = train_test_split(
    indices,
    test_size=(
        VALIDATION_SIZE
        + TEST_SIZE
    ),
    stratify=labels,
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
    stratify=labels[temp_idx],
    random_state=RANDOM_STATE,
)


print(
    f"Train      : {len(train_idx):,}"
)

print(
    f"Validation : {len(val_idx):,}"
)

print(
    f"Test       : {len(test_idx):,}"
)


print("\nSplit class distribution:")

for name, idx in [
    ("Train", train_idx),
    ("Validation", val_idx),
    ("Test", test_idx),
]:

    subset_y = labels[idx]

    print(
        f"{name:<10} "
        f"benign={(subset_y == 0).sum():>4} "
        f"anomaly={(subset_y == 1).sum():>4}"
    )


# ============================================================
# EXPERIMENT CONFIGURATION
# ============================================================

EXPERIMENTS = {

    "V5_BASELINE":
        V5_FEATURES,

    "V61_BEHAVIOR_ONLY":
        BEHAVIOR_FEATURES,

    "V5_PLUS_V61":
        V5_FEATURES
        + BEHAVIOR_FEATURES,
}


# ============================================================
# RESULT COLLECTION
# ============================================================

result_rows = []
scenario_rows = []
importance_rows = []

experiment_test_probabilities = {}


# ============================================================
# RUN EXPERIMENTS
# ============================================================

for experiment_name, selected_features in (
    EXPERIMENTS.items()
):

    print("\n")
    print("=" * 78)
    print(
        f"EXPERIMENT: {experiment_name}"
    )
    print("=" * 78)

    print(
        f"Feature count : "
        f"{len(selected_features):,}"
    )


    # --------------------------------------------------------
    # BUILD MATRIX
    # --------------------------------------------------------

    X = labelled[
        selected_features
    ].copy()


    for column in selected_features:

        X[column] = pd.to_numeric(
            X[column],
            errors="coerce",
        )


    y = labelled[
        "is_anomaly"
    ].astype(int).to_numpy()


    X_train = X.iloc[
        train_idx
    ].copy()

    X_val = X.iloc[
        val_idx
    ].copy()

    X_test = X.iloc[
        test_idx
    ].copy()


    y_train = y[
        train_idx
    ]

    y_val = y[
        val_idx
    ]

    y_test = y[
        test_idx
    ]


    # --------------------------------------------------------
    # PREPROCESSING
    # --------------------------------------------------------

    print("Preprocessing...")


    imputer = SimpleImputer(
        strategy="median"
    )

    scaler = RobustScaler()


    X_train_processed = (
        imputer.fit_transform(
            X_train
        )
    )

    X_val_processed = (
        imputer.transform(
            X_val
        )
    )

    X_test_processed = (
        imputer.transform(
            X_test
        )
    )


    X_train_processed = (
        scaler.fit_transform(
            X_train_processed
        )
    )

    X_val_processed = (
        scaler.transform(
            X_val_processed
        )
    )

    X_test_processed = (
        scaler.transform(
            X_test_processed
        )
    )


    # --------------------------------------------------------
    # RANDOM FOREST
    # --------------------------------------------------------

    print("Training Random Forest...")


    model = RandomForestClassifier(
        n_estimators=N_ESTIMATORS,
        random_state=RANDOM_STATE,
        n_jobs=-1,
        class_weight="balanced",
        min_samples_leaf=2,
    )


    model.fit(
        X_train_processed,
        y_train,
    )


    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    val_probabilities = (
        model.predict_proba(
            X_val_processed
        )[:, 1]
    )


    threshold_info = choose_threshold(
        y_val,
        val_probabilities,
        minimum_recall=MIN_RECALL,
    )


    threshold = (
        threshold_info[
            "threshold"
        ]
    )


    print("\nValidation:")

    print(
        f"Threshold : {threshold:.6f}"
    )

    print(
        f"Precision : "
        f"{threshold_info['precision']:.4f}"
    )

    print(
        f"Recall    : "
        f"{threshold_info['recall']:.4f}"
    )

    print(
        f"F1        : "
        f"{threshold_info['f1']:.4f}"
    )


    # --------------------------------------------------------
    # TEST
    # --------------------------------------------------------

    test_probabilities = (
        model.predict_proba(
            X_test_processed
        )[:, 1]
    )


    test_metrics = evaluate(
        y_test,
        test_probabilities,
        threshold,
    )


    experiment_test_probabilities[
        experiment_name
    ] = test_probabilities


    print("\nHELD-OUT TEST:")

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


    # --------------------------------------------------------
    # RESULT ROW
    # --------------------------------------------------------

    result_rows.append({

        "experiment":
            experiment_name,

        "feature_count":
            len(selected_features),

        "threshold":
            threshold,

        "validation_precision":
            threshold_info[
                "precision"
            ],

        "validation_recall":
            threshold_info[
                "recall"
            ],

        "validation_f1":
            threshold_info[
                "f1"
            ],

        "test_accuracy":
            test_metrics[
                "accuracy"
            ],

        "test_precision":
            test_metrics[
                "precision"
            ],

        "test_recall":
            test_metrics[
                "recall"
            ],

        "test_f1":
            test_metrics[
                "f1"
            ],

        "test_roc_auc":
            test_metrics[
                "roc_auc"
            ],

        "test_average_precision":
            test_metrics[
                "average_precision"
            ],

        "tn":
            test_metrics[
                "tn"
            ],

        "fp":
            test_metrics[
                "fp"
            ],

        "fn":
            test_metrics[
                "fn"
            ],

        "tp":
            test_metrics[
                "tp"
            ],
    })


    # --------------------------------------------------------
    # FEATURE IMPORTANCE
    # --------------------------------------------------------

    importance_df = pd.DataFrame({

        "feature":
            selected_features,

        "importance":
            model.feature_importances_,
    })


    importance_df = (
        importance_df
        .sort_values(
            "importance",
            ascending=False,
        )
        .reset_index(drop=True)
    )


    print("\nTop 15 features:")

    for rank, row in (
        importance_df
        .head(15)
        .iterrows()
    ):

        print(
            f"{rank + 1:2d}. "
            f"{row['feature']:<45} "
            f"{row['importance']:.6f}"
        )


    for rank, row in (
        importance_df.iterrows()
    ):

        importance_rows.append({

            "experiment":
                experiment_name,

            "rank":
                rank + 1,

            "feature":
                row["feature"],

            "importance":
                row["importance"],
        })


    # --------------------------------------------------------
    # SCENARIO EVALUATION
    # --------------------------------------------------------

    test_prediction = (
        test_probabilities
        >= threshold
    ).astype(int)


    test_entities = (
        labelled.iloc[
            test_idx
        ]["entity_id"]
        .to_numpy()
    )


    test_prediction_df = pd.DataFrame({

        "entity_id":
            test_entities,

        "prediction":
            test_prediction,

        "probability":
            test_probabilities,

        "is_anomaly":
            y_test,
    })


    print("\nScenario performance:")


    for scenario in sorted(
        gt["scenario_id"]
        .dropna()
        .astype(str)
        .unique()
    ):

        scenario_gt = gt[
            gt["scenario_id"]
            .astype(str)
            == scenario
        ]


        scenario_entities = set(
            scenario_gt[
                "resolved_entity_id"
            ]
        )


        scenario_test = (
            test_prediction_df[
                test_prediction_df[
                    "entity_id"
                ].isin(
                    scenario_entities
                )
            ]
        )


        if len(scenario_test) == 0:

            continue


        detected = int(
            scenario_test[
                "prediction"
            ].sum()
        )

        total = len(
            scenario_test
        )

        recall = (
            detected / total
        )


        scenario_rows.append({

            "experiment":
                experiment_name,

            "scenario":
                scenario,

            "test_entities":
                total,

            "detected":
                detected,

            "recall":
                recall,
        })


        print(
            f"{scenario:<25} "
            f"{detected:>4}/{total:<4} "
            f"recall={recall:.4f}"
        )


# ============================================================
# BUILD RESULTS
# ============================================================

result_df = pd.DataFrame(
    result_rows
)

scenario_df = pd.DataFrame(
    scenario_rows
)

importance_output = pd.DataFrame(
    importance_rows
)


# ============================================================
# SAVE
# ============================================================

result_df.to_csv(
    RESULT_FILE,
    index=False,
)

scenario_df.to_csv(
    SCENARIO_FILE,
    index=False,
)

importance_output.to_csv(
    IMPORTANCE_FILE,
    index=False,
)


# ============================================================
# COMPARISON
# ============================================================

print("\n")
print("=" * 78)
print("FINAL MODEL COMPARISON")
print("=" * 78)


comparison_columns = [
    "experiment",
    "feature_count",
    "test_accuracy",
    "test_precision",
    "test_recall",
    "test_f1",
    "test_roc_auc",
    "test_average_precision",
    "fp",
    "fn",
]


print(
    result_df[
        comparison_columns
    ].to_string(
        index=False
    )
)


# ============================================================
# DELTA VS V5
# ============================================================

baseline_row = result_df[
    result_df["experiment"]
    == "V5_BASELINE"
].iloc[0]


print("\n")
print("=" * 78)
print("CHANGE RELATIVE TO V5 BASELINE")
print("=" * 78)


for _, row in result_df.iterrows():

    if row["experiment"] == "V5_BASELINE":

        continue

    print(
        f"\n{row['experiment']}"
    )

    print(
        f"  ROC-AUC delta : "
        f"{row['test_roc_auc'] - baseline_row['test_roc_auc']:+.4f}"
    )

    print(
        f"  AP delta      : "
        f"{row['test_average_precision'] - baseline_row['test_average_precision']:+.4f}"
    )

    print(
        f"  F1 delta      : "
        f"{row['test_f1'] - baseline_row['test_f1']:+.4f}"
    )

    print(
        f"  Precision     : "
        f"{row['test_precision'] - baseline_row['test_precision']:+.4f}"
    )

    print(
        f"  Recall        : "
        f"{row['test_recall'] - baseline_row['test_recall']:+.4f}"
    )

    print(
        f"  FP delta      : "
        f"{int(row['fp'] - baseline_row['fp']):+d}"
    )

    print(
        f"  FN delta      : "
        f"{int(row['fn'] - baseline_row['fn']):+d}"
    )


# ============================================================
# SCENARIO PIVOT
# ============================================================

print("\n")
print("=" * 78)
print("SCENARIO COMPARISON")
print("=" * 78)


if not scenario_df.empty:

    scenario_pivot = (
        scenario_df.pivot(
            index="scenario",
            columns="experiment",
            values="recall",
        )
    )

    print(
        scenario_pivot.to_string()
    )


# ============================================================
# SUMMARY FILE
# ============================================================

summary = []

summary.append(
    "V6.1 BEHAVIORAL FEATURE BENCHMARK"
)

summary.append(
    "=" * 78
)

summary.append("")

summary.append(
    "Purpose:"
)

summary.append(
    "Compare the original V5 feature set against "
    "the new V6/V6.1 behavioral features using "
    "the same labelled entities, split, classifier, "
    "and threshold-selection methodology."
)

summary.append("")

summary.append(
    f"Labelled entities: {len(labelled):,}"
)

summary.append(
    f"Train: {len(train_idx):,}"
)

summary.append(
    f"Validation: {len(val_idx):,}"
)

summary.append(
    f"Test: {len(test_idx):,}"
)

summary.append(
    f"Random Forest estimators: {N_ESTIMATORS}"
)

summary.append(
    f"Minimum validation recall: {MIN_RECALL:.2f}"
)

summary.append("")

summary.append(
    "FEATURE SETS"
)

summary.append(
    "-" * 78
)

summary.append(
    f"V5_BASELINE: {len(V5_FEATURES)}"
)

summary.append(
    f"V61_BEHAVIOR_ONLY: {len(BEHAVIOR_FEATURES)}"
)

summary.append(
    f"V5_PLUS_V61: "
    f"{len(V5_FEATURES) + len(BEHAVIOR_FEATURES)}"
)

summary.append("")

summary.append(
    "HELD-OUT TEST RESULTS"
)

summary.append(
    "-" * 78
)


for _, row in result_df.iterrows():

    summary.append("")

    summary.append(
        row["experiment"]
    )

    summary.append(
        f"  Features: "
        f"{int(row['feature_count'])}"
    )

    summary.append(
        f"  Threshold: "
        f"{row['threshold']:.6f}"
    )

    summary.append(
        f"  Accuracy: "
        f"{row['test_accuracy']:.6f}"
    )

    summary.append(
        f"  Precision: "
        f"{row['test_precision']:.6f}"
    )

    summary.append(
        f"  Recall: "
        f"{row['test_recall']:.6f}"
    )

    summary.append(
        f"  F1: "
        f"{row['test_f1']:.6f}"
    )

    summary.append(
        f"  ROC-AUC: "
        f"{row['test_roc_auc']:.6f}"
    )

    summary.append(
        f"  Average Precision: "
        f"{row['test_average_precision']:.6f}"
    )

    summary.append(
        f"  TN: {int(row['tn'])}"
    )

    summary.append(
        f"  FP: {int(row['fp'])}"
    )

    summary.append(
        f"  FN: {int(row['fn'])}"
    )

    summary.append(
        f"  TP: {int(row['tp'])}"
    )


summary.append("")

summary.append(
    "CHANGE RELATIVE TO V5"
)

summary.append(
    "-" * 78
)


for _, row in result_df.iterrows():

    if row["experiment"] == "V5_BASELINE":

        continue

    summary.append("")

    summary.append(
        row["experiment"]
    )

    summary.append(
        f"  ROC-AUC delta: "
        f"{row['test_roc_auc'] - baseline_row['test_roc_auc']:+.6f}"
    )

    summary.append(
        f"  Average Precision delta: "
        f"{row['test_average_precision'] - baseline_row['test_average_precision']:+.6f}"
    )

    summary.append(
        f"  F1 delta: "
        f"{row['test_f1'] - baseline_row['test_f1']:+.6f}"
    )

    summary.append(
        f"  Precision delta: "
        f"{row['test_precision'] - baseline_row['test_precision']:+.6f}"
    )

    summary.append(
        f"  Recall delta: "
        f"{row['test_recall'] - baseline_row['test_recall']:+.6f}"
    )

    summary.append(
        f"  FP delta: "
        f"{int(row['fp'] - baseline_row['fp']):+d}"
    )

    summary.append(
        f"  FN delta: "
        f"{int(row['fn'] - baseline_row['fn']):+d}"
    )


summary.append("")

summary.append(
    "SCENARIO PERFORMANCE"
)

summary.append(
    "-" * 78
)


if not scenario_df.empty:

    for scenario in sorted(
        scenario_df[
            "scenario"
        ].unique()
    ):

        summary.append("")

        summary.append(
            scenario
        )

        scenario_subset = (
            scenario_df[
                scenario_df[
                    "scenario"
                ] == scenario
            ]
        )

        for _, row in (
            scenario_subset.iterrows()
        ):

            summary.append(
                f"  {row['experiment']}: "
                f"{int(row['detected'])}/"
                f"{int(row['test_entities'])} "
                f"({row['recall']:.6f})"
            )


summary.append("")

summary.append(
    "IMPORTANT"
)

summary.append(
    "-" * 78
)

summary.append(
    "This benchmark is diagnostic and supervised."
)

summary.append(
    "It does NOT replace the production anomaly detector."
)

summary.append(
    "Ground truth was not used to construct "
    "V6 or V6.1 behavioral features."
)

summary.append(
    "All experiments used the same labelled entities "
    "and identical train/validation/test indices."
)


SUMMARY_FILE.write_text(
    "\n".join(summary),
    encoding="utf-8",
)


# ============================================================
# DONE
# ============================================================

print("\n")
print("=" * 78)
print("V6.1 BENCHMARK COMPLETE")
print("=" * 78)

print(
    f"\nResults    : {RESULT_FILE}"
)

print(
    f"Scenarios  : {SCENARIO_FILE}"
)

print(
    f"Importance : {IMPORTANCE_FILE}"
)

print(
    f"Summary    : {SUMMARY_FILE}"
)

print("\nDone.")

