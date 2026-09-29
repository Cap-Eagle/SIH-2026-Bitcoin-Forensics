#!/usr/bin/env python3

"""
GROUP B — V3 FEATURE ENGINEERING

Builds an entity-level feature table from:

1. Canonical transaction data
2. P2P correlation evidence
3. Address -> entity mapping
4. Entity clusters
5. Entity flow graph

IMPORTANT:
- Ground truth is NEVER loaded here.
- No labels are used.
- Features are constructed independently of evaluation.
- This script is intended to run before train/validation/test splitting.

Input:
    data/raw/blockchain_transactions.csv
    data/correlated/group_b_features.csv
    outputs/address_to_entity.csv
    outputs/entity_clusters.csv
    outputs/entity_edges.csv

Output:
    outputs/entity_features_v3.csv
"""

import ast
import json
import math
import pickle
from collections import defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CORRELATED_DIR = DATA_DIR / "correlated"
OUT_DIR = ROOT / "outputs"

TX_FILE = RAW_DIR / "blockchain_transactions.csv"
CORRELATION_FEATURE_FILE = (
    CORRELATED_DIR / "group_b_features.csv"
)
ADDRESS_ENTITY_FILE = OUT_DIR / "address_to_entity.csv"
ENTITY_CLUSTER_FILE = OUT_DIR / "entity_clusters.csv"
ENTITY_EDGE_FILE = OUT_DIR / "entity_edges.csv"

OUTPUT_FILE = OUT_DIR / "entity_features_v3.csv"


# ============================================================
# HELPERS
# ============================================================

def parse_list(value):
    """Safely parse JSON/Python-style list values."""

    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, float) and math.isnan(value):
        return []

    text = str(value).strip()

    if text in {"", "[]", "nan", "None", "null"}:
        return []

    try:
        parsed = json.loads(text)

        if isinstance(parsed, list):
            return parsed

    except (json.JSONDecodeError, TypeError):
        pass

    try:
        parsed = ast.literal_eval(text)

        if isinstance(parsed, list):
            return parsed

    except (ValueError, SyntaxError):
        pass

    return []


def safe_float(value, default=0.0):
    try:
        value = float(value)

        if math.isnan(value) or math.isinf(value):
            return default

        return value

    except (TypeError, ValueError):
        return default


def safe_int(value, default=0):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def entropy(values):
    """
    Shannon entropy of a list of positive weights.

    Used for measuring concentration/diversity of flows.
    """

    values = [
        safe_float(v)
        for v in values
        if safe_float(v) > 0
    ]

    if not values:
        return 0.0

    total = sum(values)

    if total <= 0:
        return 0.0

    result = 0.0

    for value in values:
        p = value / total

        if p > 0:
            result -= p * math.log2(p)

    return result


def concentration(values):
    """
    Herfindahl-style concentration measure.

    1.0 = completely concentrated
    lower values = more distributed
    """

    values = [
        safe_float(v)
        for v in values
        if safe_float(v) > 0
    ]

    if not values:
        return 0.0

    total = sum(values)

    if total <= 0:
        return 0.0

    return sum((value / total) ** 2 for value in values)


# ============================================================
# VALIDATE INPUTS
# ============================================================

required_files = [
    TX_FILE,
    CORRELATION_FEATURE_FILE,
    ADDRESS_ENTITY_FILE,
    ENTITY_CLUSTER_FILE,
    ENTITY_EDGE_FILE,
]

for path in required_files:
    if not path.exists():
        raise FileNotFoundError(
            f"Required V3 input does not exist:\n{path}"
        )


print("=" * 70)
print("GROUP B FEATURE ENGINEERING — V3")
print("=" * 70)
print()

print("Loading:")
print(f"  Transactions       : {TX_FILE}")
print(f"  Correlation        : {CORRELATION_FEATURE_FILE}")
print(f"  Address -> Entity  : {ADDRESS_ENTITY_FILE}")
print(f"  Entity Clusters    : {ENTITY_CLUSTER_FILE}")
print(f"  Entity Edges       : {ENTITY_EDGE_FILE}")
print()


# ============================================================
# LOAD TRANSACTIONS
# ============================================================

print("Loading transactions...")

tx = pd.read_csv(TX_FILE)

print(f"Transactions loaded  : {len(tx):,}")

required_tx_columns = [
    "txid",
    "tx_timestamp",
    "input_addresses",
    "output_addresses",
    "input_amounts_btc",
    "output_amounts_btc",
    "fee_btc",
    "script_type",
]

missing = [
    c for c in required_tx_columns
    if c not in tx.columns
]

if missing:
    raise ValueError(
        "Missing transaction columns: "
        + ", ".join(missing)
    )


# ============================================================
# LOAD CORRELATION FEATURES
# ============================================================

print("Loading correlation features...")

corr = pd.read_csv(
    CORRELATION_FEATURE_FILE
)

print(
    f"Correlation feature rows : {len(corr):,}"
)

required_corr_columns = [
    "txid",
    "candidate_count",
    "exact_match_count",
    "close_candidate_count",
    "broad_candidate_count",
    "weighted_evidence_score",
    "max_match_score",
    "min_abs_time_delta_ms",
    "unique_src_ip_count",
    "unique_dst_ip_count",
    "unique_src_country_count",
    "unique_src_asn_count",
    "high_confidence_src_ip_count",
    "high_confidence_src_asn_count",
    "unique_message_type_count",
    "tor_like_event_count",
    "hosting_like_event_count",
]

missing = [
    c for c in required_corr_columns
    if c not in corr.columns
]

if missing:
    raise ValueError(
        "Missing correlation feature columns: "
        + ", ".join(missing)
    )


# Keep only feature columns that should be joined.
corr_features = corr[
    required_corr_columns
].copy()

corr_features = corr_features.drop_duplicates(
    subset=["txid"]
)

tx = tx.merge(
    corr_features,
    on="txid",
    how="left",
    suffixes=("", "_corr"),
)


# ============================================================
# FILL CORRELATION DEFAULTS
# ============================================================

correlation_numeric = [
    "candidate_count",
    "exact_match_count",
    "close_candidate_count",
    "broad_candidate_count",
    "weighted_evidence_score",
    "max_match_score",
    "min_abs_time_delta_ms",
    "unique_src_ip_count",
    "unique_dst_ip_count",
    "unique_src_country_count",
    "unique_src_asn_count",
    "high_confidence_src_ip_count",
    "high_confidence_src_asn_count",
    "unique_message_type_count",
    "tor_like_event_count",
    "hosting_like_event_count",
]

for column in correlation_numeric:

    if column not in tx.columns:
        tx[column] = 0.0

    tx[column] = pd.to_numeric(
        tx[column],
        errors="coerce",
    ).fillna(0.0)


# ============================================================
# LOAD ADDRESS -> ENTITY MAPPING
# ============================================================

print("Loading address/entity mapping...")

address_entity = pd.read_csv(
    ADDRESS_ENTITY_FILE
)

print(
    f"Address mappings       : {len(address_entity):,}"
)

address_to_entity = dict(
    zip(
        address_entity["address"].astype(str),
        address_entity["derived_entity_id"].astype(str),
    )
)


# ============================================================
# LOAD ENTITY CLUSTERS
# ============================================================

print("Loading entity clusters...")

clusters = pd.read_csv(
    ENTITY_CLUSTER_FILE
)

print(
    f"Entity clusters        : {len(clusters):,}"
)

cluster_size_map = dict(
    zip(
        clusters["derived_entity_id"].astype(str),
        pd.to_numeric(
            clusters["cluster_size"],
            errors="coerce",
        ).fillna(1),
    )
)


# ============================================================
# LOAD ENTITY EDGES
# ============================================================

print("Loading entity edges...")

edges = pd.read_csv(
    ENTITY_EDGE_FILE
)

print(
    f"Entity edge records    : {len(edges):,}"
)


# ============================================================
# BUILD ENTITY-LEVEL AGGREGATION STRUCTURES
# ============================================================

print()
print("Building entity transaction statistics...")


entity_tx_count = defaultdict(int)
entity_input_volume = defaultdict(float)
entity_output_volume = defaultdict(float)

entity_input_tx_count = defaultdict(int)
entity_output_tx_count = defaultdict(int)

entity_fee_total = defaultdict(float)

entity_counterparties = defaultdict(set)

entity_input_amounts = defaultdict(list)
entity_output_amounts = defaultdict(list)

entity_correlated_candidates = defaultdict(float)
entity_exact_matches = defaultdict(float)
entity_close_matches = defaultdict(float)
entity_broad_matches = defaultdict(float)
entity_weighted_evidence = defaultdict(float)

entity_max_match_score = defaultdict(float)
entity_min_time_delta = defaultdict(lambda: None)

entity_src_ips = defaultdict(set)
entity_dst_ips = defaultdict(set)
entity_src_countries = defaultdict(set)
entity_src_asns = defaultdict(set)

entity_high_conf_ips = defaultdict(set)
entity_high_conf_asns = defaultdict(set)

entity_message_types = defaultdict(set)

entity_tor_events = defaultdict(float)
entity_hosting_events = defaultdict(float)

entity_script_types = defaultdict(set)


# ============================================================
# PROCESS TRANSACTIONS
# ============================================================

for _, row in tx.iterrows():

    txid = str(row["txid"])

    input_addresses = [
        str(a)
        for a in parse_list(row["input_addresses"])
        if str(a).strip()
    ]

    output_addresses = [
        str(a)
        for a in parse_list(row["output_addresses"])
        if str(a).strip()
    ]

    input_amounts = [
        safe_float(v)
        for v in parse_list(row["input_amounts_btc"])
    ]

    output_amounts = [
        safe_float(v)
        for v in parse_list(row["output_amounts_btc"])
    ]

    input_entities = {
        address_to_entity[a]
        for a in input_addresses
        if a in address_to_entity
    }

    output_entities = {
        address_to_entity[a]
        for a in output_addresses
        if a in address_to_entity
    }

    involved_entities = (
        input_entities | output_entities
    )

    if not involved_entities:
        continue

    total_input = sum(input_amounts)
    total_output = sum(output_amounts)
    fee = safe_float(row["fee_btc"])

    script_type = str(
        row.get("script_type", "UNKNOWN")
    )

    # --------------------------------------------------------
    # Transaction statistics
    # --------------------------------------------------------

    for entity in involved_entities:

        entity_tx_count[entity] += 1

        entity_fee_total[entity] += fee

        entity_script_types[entity].add(
            script_type
        )

    # --------------------------------------------------------
    # Input statistics
    # --------------------------------------------------------

    for entity in input_entities:

        entity_input_tx_count[entity] += 1

        entity_input_volume[entity] += total_input

        for amount in input_amounts:
            entity_input_amounts[entity].append(
                amount
            )

    # --------------------------------------------------------
    # Output statistics
    # --------------------------------------------------------

    for entity in output_entities:

        entity_output_tx_count[entity] += 1

        entity_output_volume[entity] += total_output

        for amount in output_amounts:
            entity_output_amounts[entity].append(
                amount
            )

    # --------------------------------------------------------
    # Entity counterparties
    # --------------------------------------------------------

    for source in input_entities:

        for destination in output_entities:

            if source != destination:

                entity_counterparties[source].add(
                    destination
                )

                entity_counterparties[destination].add(
                    source
                )

    # --------------------------------------------------------
    # Correlation evidence
    # --------------------------------------------------------

    candidates = safe_float(
        row["candidate_count"]
    )

    exact = safe_float(
        row["exact_match_count"]
    )

    close = safe_float(
        row["close_candidate_count"]
    )

    broad = safe_float(
        row["broad_candidate_count"]
    )

    weighted = safe_float(
        row["weighted_evidence_score"]
    )

    max_score = safe_float(
        row["max_match_score"]
    )

    min_delta = safe_float(
        row["min_abs_time_delta_ms"]
    )

    for entity in involved_entities:

        entity_correlated_candidates[entity] += candidates

        entity_exact_matches[entity] += exact

        entity_close_matches[entity] += close

        entity_broad_matches[entity] += broad

        entity_weighted_evidence[entity] += weighted

        entity_max_match_score[entity] = max(
            entity_max_match_score[entity],
            max_score,
        )

        if min_delta > 0:

            current = entity_min_time_delta[entity]

            if current is None or min_delta < current:
                entity_min_time_delta[entity] = min_delta

        # ----------------------------------------------------
        # Network diversity
        # ----------------------------------------------------

        entity_src_ips[entity].update(
            range(
                int(
                    safe_float(
                        row["unique_src_ip_count"]
                    )
                )
            )
        )

        entity_dst_ips[entity].update(
            range(
                int(
                    safe_float(
                        row["unique_dst_ip_count"]
                    )
                )
            )
        )

        entity_src_countries[entity].update(
            range(
                int(
                    safe_float(
                        row["unique_src_country_count"]
                    )
                )
            )
        )

        entity_src_asns[entity].update(
            range(
                int(
                    safe_float(
                        row["unique_src_asn_count"]
                    )
                )
            )
        )

        entity_high_conf_ips[entity].update(
            range(
                int(
                    safe_float(
                        row["high_confidence_src_ip_count"]
                    )
                )
            )
        )

        entity_high_conf_asns[entity].update(
            range(
                int(
                    safe_float(
                        row["high_confidence_src_asn_count"]
                    )
                )
            )
        )

        entity_message_types[entity].update(
            range(
                int(
                    safe_float(
                        row["unique_message_type_count"]
                    )
                )
            )
        )

        entity_tor_events[entity] += safe_float(
            row["tor_like_event_count"]
        )

        entity_hosting_events[entity] += safe_float(
            row["hosting_like_event_count"]
        )


# ============================================================
# BUILD ENTITY SET
# ============================================================

all_entities = set(cluster_size_map)

all_entities.update(
    entity_tx_count.keys()
)

all_entities.update(
    address_entity["derived_entity_id"].astype(str)
)

print(
    f"Entities represented      : {len(all_entities):,}"
)


# ============================================================
# BUILD GRAPH-LEVEL FEATURES
# ============================================================

print("Building graph features...")

entity_graph = nx.DiGraph()

for _, row in edges.iterrows():

    source = str(row["source_entity"])
    target = str(row["target_entity"])

    if source == target:
        continue

    entity_graph.add_edge(
        source,
        target,
    )


# Add isolated entities.
for entity in all_entities:

    if entity not in entity_graph:
        entity_graph.add_node(entity)


# ============================================================
# CONSTRUCT FEATURE TABLE
# ============================================================

print("Constructing feature table...")

records = []

for entity in sorted(all_entities):

    cluster_size = safe_int(
        cluster_size_map.get(entity, 1),
        1,
    )

    tx_count = entity_tx_count.get(
        entity,
        0,
    )

    input_volume = entity_input_volume.get(
        entity,
        0.0,
    )

    output_volume = entity_output_volume.get(
        entity,
        0.0,
    )

    fee_total = entity_fee_total.get(
        entity,
        0.0,
    )

    input_tx_count = entity_input_tx_count.get(
        entity,
        0,
    )

    output_tx_count = entity_output_tx_count.get(
        entity,
        0,
    )

    counterparties = len(
        entity_counterparties.get(
            entity,
            set(),
        )
    )

    in_degree = entity_graph.in_degree(entity)

    out_degree = entity_graph.out_degree(entity)

    degree = in_degree + out_degree

    # --------------------------------------------------------
    # Flow balance
    # --------------------------------------------------------

    gross_flow = (
        input_volume + output_volume
    )

    if gross_flow > 0:

        flow_imbalance = abs(
            input_volume - output_volume
        ) / gross_flow

    else:

        flow_imbalance = 0.0

    # --------------------------------------------------------
    # Fee behaviour
    # --------------------------------------------------------

    average_fee = (
        fee_total / tx_count
        if tx_count > 0
        else 0.0
    )

    # --------------------------------------------------------
    # Amount concentration
    # --------------------------------------------------------

    input_amounts = entity_input_amounts.get(
        entity,
        [],
    )

    output_amounts = entity_output_amounts.get(
        entity,
        [],
    )

    input_entropy = entropy(
        input_amounts
    )

    output_entropy = entropy(
        output_amounts
    )

    input_concentration = concentration(
        input_amounts
    )

    output_concentration = concentration(
        output_amounts
    )

    # --------------------------------------------------------
    # Correlation evidence
    # --------------------------------------------------------

    candidate_count = (
        entity_correlated_candidates.get(
            entity,
            0.0,
        )
    )

    exact_matches = (
        entity_exact_matches.get(
            entity,
            0.0,
        )
    )

    close_matches = (
        entity_close_matches.get(
            entity,
            0.0,
        )
    )

    broad_matches = (
        entity_broad_matches.get(
            entity,
            0.0,
        )
    )

    weighted_evidence = (
        entity_weighted_evidence.get(
            entity,
            0.0,
        )
    )

    max_match_score = (
        entity_max_match_score.get(
            entity,
            0.0,
        )
    )

    min_time_delta = (
        entity_min_time_delta.get(
            entity
        )
    )

    if min_time_delta is None:
        min_time_delta = 0.0

    # --------------------------------------------------------
    # Network evidence
    #
    # NOTE:
    # The original correlation table contains aggregate
    # diversity counts rather than individual identities.
    # Therefore these remain aggregate sums, not fake IDs.
    # --------------------------------------------------------

    src_ip_count = sum(
        safe_float(
            x
        )
        for x in entity_src_ips.get(
            entity,
            set(),
        )
    )

    dst_ip_count = sum(
        safe_float(
            x
        )
        for x in entity_dst_ips.get(
            entity,
            set(),
        )
    )

    country_count = sum(
        safe_float(
            x
        )
        for x in entity_src_countries.get(
            entity,
            set(),
        )
    )

    asn_count = sum(
        safe_float(
            x
        )
        for x in entity_src_asns.get(
            entity,
            set(),
        )
    )

    high_conf_ip_count = sum(
        safe_float(
            x
        )
        for x in entity_high_conf_ips.get(
            entity,
            set(),
        )
    )

    high_conf_asn_count = sum(
        safe_float(
            x
        )
        for x in entity_high_conf_asns.get(
            entity,
            set(),
        )
    )

    message_type_count = sum(
        safe_float(
            x
        )
        for x in entity_message_types.get(
            entity,
            set(),
        )
    )

    tor_events = entity_tor_events.get(
        entity,
        0.0,
    )

    hosting_events = entity_hosting_events.get(
        entity,
        0.0,
    )

    # --------------------------------------------------------
    # Derived ratios
    # --------------------------------------------------------

    exact_match_rate = (
        exact_matches / candidate_count
        if candidate_count > 0
        else 0.0
    )

    close_match_rate = (
        close_matches / candidate_count
        if candidate_count > 0
        else 0.0
    )

    network_risk_signal = (
        tor_events + hosting_events
    )

    # --------------------------------------------------------
    # Entity record
    # --------------------------------------------------------

    records.append(
        {
            "entity_id": entity,

            # Entity structure
            "cluster_size": cluster_size,
            "is_multi_address_entity": int(
                cluster_size > 1
            ),

            # Transaction activity
            "transaction_count": tx_count,
            "input_transaction_count": input_tx_count,
            "output_transaction_count": output_tx_count,

            # Value flow
            "total_input_btc": input_volume,
            "total_output_btc": output_volume,
            "gross_flow_btc": gross_flow,
            "flow_imbalance": flow_imbalance,

            # Fees
            "total_fee_btc": fee_total,
            "average_fee_btc": average_fee,

            # Graph topology
            "in_degree": in_degree,
            "out_degree": out_degree,
            "total_degree": degree,
            "counterparty_count": counterparties,

            # Flow distribution
            "input_amount_entropy": input_entropy,
            "output_amount_entropy": output_entropy,
            "input_amount_concentration": input_concentration,
            "output_amount_concentration": output_concentration,

            # Correlation evidence
            "candidate_count": candidate_count,
            "exact_match_count": exact_matches,
            "close_match_count": close_matches,
            "broad_match_count": broad_matches,
            "weighted_evidence_score": weighted_evidence,
            "max_match_score": max_match_score,
            "min_abs_time_delta_ms": min_time_delta,
            "exact_match_rate": exact_match_rate,
            "close_match_rate": close_match_rate,

            # Network diversity
            "src_ip_evidence_count": src_ip_count,
            "dst_ip_evidence_count": dst_ip_count,
            "src_country_evidence_count": country_count,
            "src_asn_evidence_count": asn_count,
            "high_confidence_ip_evidence_count":
                high_conf_ip_count,
            "high_confidence_asn_evidence_count":
                high_conf_asn_count,
            "message_type_evidence_count":
                message_type_count,

            # Network risk indicators
            "tor_like_event_count": tor_events,
            "hosting_like_event_count": hosting_events,
            "network_risk_signal": network_risk_signal,
        }
    )


# ============================================================
# DATAFRAME
# ============================================================

features = pd.DataFrame(records)

features = features.sort_values(
    "entity_id"
).reset_index(drop=True)


# ============================================================
# SANITY CHECKS
# ============================================================

print()
print("Running feature sanity checks...")

if features.empty:
    raise RuntimeError(
        "Feature table is empty."
    )

if features["entity_id"].duplicated().any():
    raise RuntimeError(
        "Duplicate entity IDs detected."
    )

numeric_columns = [
    column
    for column in features.columns
    if column != "entity_id"
]

for column in numeric_columns:

    if not pd.api.types.is_numeric_dtype(
        features[column]
    ):
        raise RuntimeError(
            f"Non-numeric feature detected: {column}"
        )

    if not features[column].notna().all():
        raise RuntimeError(
            f"NaN values detected in feature: {column}"
        )

    if not pd.Series(
        features[column]
    ).map(math.isfinite).all():
        raise RuntimeError(
            f"Non-finite values detected: {column}"
        )


# ============================================================
# SAVE
# ============================================================

OUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

features.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("FEATURE ENGINEERING COMPLETE")
print("=" * 70)
print()

print(
    f"Entities                  : {len(features):,}"
)

print(
    f"Features                  : {len(features.columns):,}"
)

print(
    f"Multi-address entities    : "
    f"{int(features['is_multi_address_entity'].sum()):,}"
)

print(
    f"Entities with correlation : "
    f"{int((features['candidate_count'] > 0).sum()):,}"
)

print(
    f"Entities with exact match : "
    f"{int((features['exact_match_count'] > 0).sum()):,}"
)

print(
    f"Entities with Tor evidence: "
    f"{int((features['tor_like_event_count'] > 0).sum()):,}"
)

print(
    f"Entities with hosting evidence: "
    f"{int((features['hosting_like_event_count'] > 0).sum()):,}"
)

print()
print("Feature columns:")
print()

for index, column in enumerate(
    features.columns,
    start=1,
):
    print(
        f"  {index:02d}. {column}"
    )

print()
print(
    f"Output: {OUTPUT_FILE}"
)

print()
print("=" * 70)
