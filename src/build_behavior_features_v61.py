#!/usr/bin/env python3

"""
GROUP B — BEHAVIORAL FEATURE ENGINEERING V6.1

Purpose
-------
Extend the existing V6 entity behavioral features with additional
scenario-independent forensic features for:

1. Cross-entity IP sharing
2. Near-threshold / structuring-like payments
3. Small peel-like outputs
4. Cross-entity rapid transaction chains
5. Amount-preserving hops
6. Sequential / graph-based transaction behavior

IMPORTANT
---------
Ground truth is NEVER loaded by this script.

No scenario labels, anomaly labels, related_txids, or synthetic
ground-truth information are used to construct features.

Inputs
------
outputs/entity_behavior_features_v6.csv
data/raw/blockchain_transactions.csv
data/correlated/correlations.csv
outputs/address_to_entity.csv
outputs/entity_edges.csv

Output
------
outputs/entity_behavior_features_v61.csv
"""

from __future__ import annotations

import ast
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CORRELATED_DIR = DATA_DIR / "correlated"
OUTPUT_DIR = ROOT / "outputs"

V6_FILE = OUTPUT_DIR / "entity_behavior_features_v6.csv"

TX_FILE = RAW_DIR / "blockchain_transactions.csv"

CORRELATIONS_FILE = (
    CORRELATED_DIR / "correlations.csv"
)

ADDRESS_ENTITY_FILE = (
    OUTPUT_DIR / "address_to_entity.csv"
)

ENTITY_EDGES_FILE = (
    OUTPUT_DIR / "entity_edges.csv"
)

OUTPUT_FILE = (
    OUTPUT_DIR / "entity_behavior_features_v61.csv"
)


# ============================================================
# CONFIGURATION
# ============================================================

# Broad detector ranges are intentionally used instead of
# memorizing the exact synthetic generator parameters.

STRUCTURING_LOWER_BTC = 0.90
STRUCTURING_UPPER_BTC = 1.00

PEEL_MIN_BTC = 0.01
PEEL_MAX_BTC = 0.05

# Rapid-hop / layering timing windows.
VERY_RAPID_CHAIN_SECONDS = 180.0
RAPID_CHAIN_SECONDS = 300.0
LAYERING_CHAIN_SECONDS = 900.0

# Amount-retention thresholds.
HIGH_AMOUNT_RETENTION = 0.90
VERY_HIGH_AMOUNT_RETENTION = 0.97

# An IP is considered shared when at least this many
# distinct entities are associated with it.
SHARED_IP_ENTITY_THRESHOLD = 3


# ============================================================
# HELPERS
# ============================================================

def parse_list(value):
    """
    Safely parse a JSON/Python-style list.
    """

    if value is None:
        return []

    if isinstance(value, list):
        return value

    if isinstance(value, tuple):
        return list(value)

    if isinstance(value, float) and math.isnan(value):
        return []

    text = str(value).strip()

    if text in {
        "",
        "[]",
        "nan",
        "None",
        "null",
        "<NA>",
    }:
        return []

    try:
        result = json.loads(text)

        if isinstance(result, list):
            return result

    except (
        json.JSONDecodeError,
        TypeError,
    ):
        pass

    try:
        result = ast.literal_eval(text)

        if isinstance(result, list):
            return result

    except (
        ValueError,
        SyntaxError,
        TypeError,
    ):
        pass

    return []


def safe_float(value, default=0.0):
    """
    Convert a value safely to float.
    """

    try:
        result = float(value)

        if not np.isfinite(result):
            return default

        return result

    except (
        TypeError,
        ValueError,
    ):
        return default


def safe_ratio(
    numerator,
    denominator,
):
    """
    Safe ratio calculation.
    """

    denominator = safe_float(
        denominator,
        0.0,
    )

    if denominator <= 0:
        return 0.0

    return (
        safe_float(
            numerator,
            0.0,
        )
        / denominator
    )


def safe_mean(values):
    """
    Safe mean.
    """

    values = [
        safe_float(v)
        for v in values
        if np.isfinite(
            safe_float(v)
        )
    ]

    if not values:
        return 0.0

    return float(
        np.mean(values)
    )


def safe_median(values):
    """
    Safe median.
    """

    values = [
        safe_float(v)
        for v in values
        if np.isfinite(
            safe_float(v)
        )
    ]

    if not values:
        return 0.0

    return float(
        np.median(values)
    )


def safe_std(values):
    """
    Safe standard deviation.
    """

    values = [
        safe_float(v)
        for v in values
        if np.isfinite(
            safe_float(v)
        )
    ]

    if len(values) < 2:
        return 0.0

    return float(
        np.std(values)
    )


def safe_min(values):
    """
    Safe minimum.
    """

    if not values:
        return 0.0

    return float(
        min(values)
    )


def safe_max(values):
    """
    Safe maximum.
    """

    if not values:
        return 0.0

    return float(
        max(values)
    )


def parse_timestamp(value):
    """
    Safely convert timestamp to UTC Timestamp.
    """

    try:
        result = pd.to_datetime(
            value,
            utc=True,
            errors="coerce",
        )

        if pd.isna(result):
            return None

        return result

    except Exception:
        return None


# ============================================================
# VALIDATE INPUTS
# ============================================================

required_files = [
    V6_FILE,
    TX_FILE,
    CORRELATIONS_FILE,
    ADDRESS_ENTITY_FILE,
    ENTITY_EDGES_FILE,
]

for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"Required input does not exist:\n"
            f"{path}"
        )


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print(
    "GROUP B BEHAVIORAL FEATURE ENGINEERING — V6.1"
)
print("=" * 70)
print()

print("Loading data...")


# ============================================================
# LOAD V6 FEATURES
# ============================================================

v6 = pd.read_csv(
    V6_FILE
)

if "entity_id" not in v6.columns:

    raise ValueError(
        "V6 feature file does not contain entity_id"
    )

print(
    f"V6 entities         : "
    f"{len(v6):,}"
)

print(
    f"V6 feature columns  : "
    f"{len(v6.columns) - 1:,}"
)


# ============================================================
# LOAD TRANSACTIONS
# ============================================================

tx = pd.read_csv(
    TX_FILE
)

print(
    f"Transactions        : "
    f"{len(tx):,}"
)


required_tx_columns = [
    "txid",
    "tx_timestamp",
    "is_coinbase",
    "input_addresses",
    "output_addresses",
    "input_amounts_btc",
    "output_amounts_btc",
]

missing = [
    c
    for c in required_tx_columns
    if c not in tx.columns
]

if missing:

    raise ValueError(
        "Missing transaction columns: "
        + ", ".join(missing)
    )


# ============================================================
# LOAD CORRELATIONS
# ============================================================

correlations = pd.read_csv(
    CORRELATIONS_FILE
)

print(
    f"Correlation events  : "
    f"{len(correlations):,}"
)


required_corr_columns = [
    "src_ip",
    "candidate_txid",
    "input_addresses",
]

missing = [
    c
    for c in required_corr_columns
    if c not in correlations.columns
]

if missing:

    raise ValueError(
        "Missing correlation columns: "
        + ", ".join(missing)
    )


# ============================================================
# LOAD ADDRESS -> ENTITY MAP
# ============================================================

address_map = pd.read_csv(
    ADDRESS_ENTITY_FILE
)

required_address_columns = [
    "address",
    "derived_entity_id",
]

missing = [
    c
    for c in required_address_columns
    if c not in address_map.columns
]

if missing:

    raise ValueError(
        "Missing address-map columns: "
        + ", ".join(missing)
    )


print(
    f"Address mappings    : "
    f"{len(address_map):,}"
)


address_to_entity = dict(
    zip(
        address_map["address"].astype(str),
        address_map[
            "derived_entity_id"
        ].astype(str),
    )
)


# ============================================================
# LOAD ENTITY EDGES
# ============================================================

edges = pd.read_csv(
    ENTITY_EDGES_FILE
)

required_edge_columns = [
    "source_entity",
    "target_entity",
    "txid",
]

missing = [
    c
    for c in required_edge_columns
    if c not in edges.columns
]

if missing:

    raise ValueError(
        "Missing entity-edge columns: "
        + ", ".join(missing)
    )


print(
    f"Entity edges        : "
    f"{len(edges):,}"
)


# ============================================================
# ENTITY SET
# ============================================================

all_entities = (
    v6["entity_id"]
    .astype(str)
    .tolist()
)

entity_set = set(
    all_entities
)


# ============================================================
# TRANSACTION LOOKUPS
# ============================================================

print()
print(
    "Building transaction lookup tables..."
)


tx_timestamp = {}
tx_total_input = {}
tx_total_output = {}
tx_output_amounts = {}
tx_input_amounts = {}
tx_source_entities = {}
tx_target_entities = {}


for row in tx.itertuples(
    index=False
):

    txid = str(
        row.txid
    )

    timestamp = parse_timestamp(
        row.tx_timestamp
    )

    input_addresses = parse_list(
        row.input_addresses
    )

    output_addresses = parse_list(
        row.output_addresses
    )

    input_amounts = [
        safe_float(x)
        for x in parse_list(
            row.input_amounts_btc
        )
    ]

    output_amounts = [
        safe_float(x)
        for x in parse_list(
            row.output_amounts_btc
        )
    ]

    source_entities = {
        address_to_entity[address]
        for address
        in input_addresses
        if address in address_to_entity
    }

    target_entities = {
        address_to_entity[address]
        for address
        in output_addresses
        if address in address_to_entity
    }

    tx_timestamp[txid] = timestamp

    tx_total_input[txid] = float(
        sum(input_amounts)
    )

    tx_total_output[txid] = float(
        sum(output_amounts)
    )

    tx_input_amounts[txid] = (
        input_amounts
    )

    tx_output_amounts[txid] = (
        output_amounts
    )

    tx_source_entities[txid] = (
        source_entities
    )

    tx_target_entities[txid] = (
        target_entities
    )


# ============================================================
# ENTITY TRANSACTION ACTIVITY
# ============================================================

print(
    "Building entity transaction behavior..."
)


entity_spending_txids = defaultdict(
    list
)

entity_receiving_txids = defaultdict(
    list
)

entity_output_amounts = defaultdict(
    list
)

entity_structuring_amounts = defaultdict(
    list
)

entity_peel_amounts = defaultdict(
    list
)

entity_structuring_times = defaultdict(
    list
)

entity_peel_times = defaultdict(
    list
)


for row in tx.itertuples(
    index=False
):

    txid = str(
        row.txid
    )

    # Coinbase transactions do not have
    # normal spending entities.

    is_coinbase = bool(
        row.is_coinbase
    )

    timestamp = tx_timestamp.get(
        txid
    )

    outputs = tx_output_amounts.get(
        txid,
        [],
    )

    source_entities = (
        tx_source_entities.get(
            txid,
            set(),
        )
    )

    target_entities = (
        tx_target_entities.get(
            txid,
            set(),
        )
    )

    # ----------------------------------------
    # RECEIVING ACTIVITY
    # ----------------------------------------

    for entity in target_entities:

        if entity in entity_set:

            entity_receiving_txids[
                entity
            ].append(
                txid
            )

    if is_coinbase:
        continue

    # ----------------------------------------
    # SPENDING ACTIVITY
    # ----------------------------------------

    for entity in source_entities:

        if entity not in entity_set:
            continue

        entity_spending_txids[
            entity
        ].append(
            txid
        )

        for amount in outputs:

            entity_output_amounts[
                entity
            ].append(
                amount
            )

            # --------------------------------
            # STRUCTURING BAND
            # --------------------------------

            if (
                STRUCTURING_LOWER_BTC
                <= amount
                <= STRUCTURING_UPPER_BTC
            ):

                entity_structuring_amounts[
                    entity
                ].append(
                    amount
                )

                if timestamp is not None:

                    entity_structuring_times[
                        entity
                    ].append(
                        timestamp
                    )

            # --------------------------------
            # SMALL PEEL-LIKE OUTPUT
            # --------------------------------

            if (
                PEEL_MIN_BTC
                <= amount
                <= PEEL_MAX_BTC
            ):

                entity_peel_amounts[
                    entity
                ].append(
                    amount
                )

                if timestamp is not None:

                    entity_peel_times[
                        entity
                    ].append(
                        timestamp
                    )


# ============================================================
# STRUCTURING TEMPORAL FEATURES
# ============================================================

print(
    "Calculating structuring temporal behavior..."
)


entity_structuring_intervals = defaultdict(
    list
)


for entity, timestamps in (
    entity_structuring_times.items()
):

    timestamps = sorted(
        timestamps
    )

    for previous, current in zip(
        timestamps,
        timestamps[1:],
    ):

        delta = (
            current - previous
        ).total_seconds()

        if delta >= 0:

            entity_structuring_intervals[
                entity
            ].append(
                float(delta)
            )


# ============================================================
# PEEL TEMPORAL FEATURES
# ============================================================

print(
    "Calculating peel temporal behavior..."
)


entity_peel_intervals = defaultdict(
    list
)


for entity, timestamps in (
    entity_peel_times.items()
):

    timestamps = sorted(
        timestamps
    )

    for previous, current in zip(
        timestamps,
        timestamps[1:],
    ):

        delta = (
            current - previous
        ).total_seconds()

        if delta >= 0:

            entity_peel_intervals[
                entity
            ].append(
                float(delta)
            )


# ============================================================
# CROSS-ENTITY SOURCE-IP SHARING
# ============================================================

print(
    "Building cross-entity source-IP sharing..."
)


ip_entities = defaultdict(
    set
)

entity_ips = defaultdict(
    set
)


for row in correlations.itertuples(
    index=False
):

    src_ip = getattr(
        row,
        "src_ip",
        None,
    )

    if src_ip is None:
        continue

    if pd.isna(src_ip):
        continue

    src_ip = str(
        src_ip
    ).strip()

    if not src_ip:
        continue

    addresses = parse_list(
        getattr(
            row,
            "input_addresses",
            None,
        )
    )

    entities_seen = {
        address_to_entity[address]
        for address
        in addresses
        if address in address_to_entity
    }

    for entity in entities_seen:

        if entity not in entity_set:
            continue

        ip_entities[
            src_ip
        ].add(
            entity
        )

        entity_ips[
            entity
        ].add(
            src_ip
        )


# ============================================================
# IP SHARING ENTITY FEATURES
# ============================================================

entity_shared_ip_count = {}
entity_max_entities_per_ip = {}
entity_mean_entities_per_ip = {}
entity_shared_ip_ratio = {}
entity_unique_ip_count = {}


for entity in all_entities:

    ips = entity_ips.get(
        entity,
        set(),
    )

    entity_unique_ip_count[
        entity
    ] = len(
        ips
    )

    if not ips:

        entity_shared_ip_count[
            entity
        ] = 0

        entity_max_entities_per_ip[
            entity
        ] = 0

        entity_mean_entities_per_ip[
            entity
        ] = 0.0

        entity_shared_ip_ratio[
            entity
        ] = 0.0

        continue

    sharing_counts = [
        len(
            ip_entities[ip]
        )
        for ip in ips
    ]

    shared_counts = [
        count
        for count in sharing_counts
        if count
        >= SHARED_IP_ENTITY_THRESHOLD
    ]

    entity_shared_ip_count[
        entity
    ] = len(
        shared_counts
    )

    entity_max_entities_per_ip[
        entity
    ] = int(
        max(
            sharing_counts
        )
    )

    entity_mean_entities_per_ip[
        entity
    ] = safe_mean(
        sharing_counts
    )

    entity_shared_ip_ratio[
        entity
    ] = safe_ratio(
        len(shared_counts),
        len(ips),
    )


# ============================================================
# TRANSACTION GRAPH
# ============================================================

print(
    "Building temporal entity graph..."
)


# Store edge events:
#
# source entity
# target entity
# txid
# timestamp
# transferred/output amount approximation

entity_outgoing_edges = defaultdict(
    list
)

entity_incoming_edges = defaultdict(
    list
)


for row in edges.itertuples(
    index=False
):

    source = str(
        row.source_entity
    )

    target = str(
        row.target_entity
    )

    txid = str(
        row.txid
    )

    timestamp = tx_timestamp.get(
        txid
    )

    if timestamp is None:
        continue

    total_output = tx_total_output.get(
        txid,
        0.0,
    )

    event = {
        "source": source,
        "target": target,
        "txid": txid,
        "timestamp": timestamp,
        "amount": total_output,
    }

    entity_outgoing_edges[
        source
    ].append(
        event
    )

    entity_incoming_edges[
        target
    ].append(
        event
    )


# Sort events by time.

for entity in list(
    entity_outgoing_edges
):

    entity_outgoing_edges[
        entity
    ].sort(
        key=lambda x: x[
            "timestamp"
        ]
    )


for entity in list(
    entity_incoming_edges
):

    entity_incoming_edges[
        entity
    ].sort(
        key=lambda x: x[
            "timestamp"
        ]
    )


# ============================================================
# CROSS-ENTITY SUCCESSOR FEATURES
# ============================================================

print(
    "Calculating cross-entity successor timing..."
)


entity_successor_delays = defaultdict(
    list
)

entity_rapid_successor_count = defaultdict(
    int
)

entity_very_rapid_successor_count = defaultdict(
    int
)

entity_layering_successor_count = defaultdict(
    int
)

entity_amount_retention = defaultdict(
    list
)


# For each outgoing transfer A -> B,
# look for B's first subsequent spend.

for source in all_entities:

    outgoing = entity_outgoing_edges.get(
        source,
        [],
    )

    for edge in outgoing:

        target = edge[
            "target"
        ]

        incoming_time = edge[
            "timestamp"
        ]

        target_spends = (
            entity_outgoing_edges.get(
                target,
                [],
            )
        )

        next_spend = None

        for candidate in target_spends:

            if (
                candidate["timestamp"]
                > incoming_time
            ):

                next_spend = candidate
                break

        if next_spend is None:
            continue

        delay = (
            next_spend["timestamp"]
            - incoming_time
        ).total_seconds()

        if delay < 0:
            continue

        entity_successor_delays[
            source
        ].append(
            float(delay)
        )

        if (
            delay
            <= VERY_RAPID_CHAIN_SECONDS
        ):

            entity_very_rapid_successor_count[
                source
            ] += 1

        if (
            delay
            <= RAPID_CHAIN_SECONDS
        ):

            entity_rapid_successor_count[
                source
            ] += 1

        if (
            delay
            <= LAYERING_CHAIN_SECONDS
        ):

            entity_layering_successor_count[
                source
            ] += 1

        incoming_amount = safe_float(
            edge["amount"]
        )

        outgoing_amount = safe_float(
            next_spend["amount"]
        )

        if (
            incoming_amount > 0
            and outgoing_amount > 0
        ):

            retention = min(
                outgoing_amount
                / incoming_amount,
                1.5,
            )

            entity_amount_retention[
                source
            ].append(
                retention
            )


# ============================================================
# RAPID PATH DEPTH
# ============================================================

print(
    "Calculating rapid-chain path depth..."
)


def rapid_path_depth(
    start_entity,
    max_depth=10,
):
    """
    Follow temporally ordered entity edges.

    This is not a ground-truth path lookup. It uses only
    observed entity graph structure and transaction timing.
    """

    best_depth = 0

    initial_edges = (
        entity_outgoing_edges.get(
            start_entity,
            [],
        )
    )

    stack = []

    for edge in initial_edges:

        stack.append(
            (
                edge["target"],
                edge["timestamp"],
                1,
                {start_entity},
            )
        )

    while stack:

        (
            current_entity,
            previous_time,
            depth,
            visited,
        ) = stack.pop()

        best_depth = max(
            best_depth,
            depth,
        )

        if depth >= max_depth:
            continue

        if current_entity in visited:
            continue

        next_visited = set(
            visited
        )

        next_visited.add(
            current_entity
        )

        for edge in (
            entity_outgoing_edges.get(
                current_entity,
                [],
            )
        ):

            if (
                edge["timestamp"]
                <= previous_time
            ):
                continue

            delay = (
                edge["timestamp"]
                - previous_time
            ).total_seconds()

            if (
                delay
                > RAPID_CHAIN_SECONDS
            ):
                break

            stack.append(
                (
                    edge["target"],
                    edge["timestamp"],
                    depth + 1,
                    next_visited,
                )
            )

    return best_depth


entity_rapid_chain_depth = {}


entities_with_outgoing = set(
    entity_outgoing_edges.keys()
)


for index, entity in enumerate(
    all_entities,
    start=1,
):

    if entity not in entities_with_outgoing:

        entity_rapid_chain_depth[
            entity
        ] = 0

        continue

    entity_rapid_chain_depth[
        entity
    ] = rapid_path_depth(
        entity
    )

    if (
        index % 10000
        == 0
    ):

        print(
            f"  rapid-path entities processed: "
            f"{index:,}/{len(all_entities):,}"
        )


# ============================================================
# BUILD V6.1 FEATURE RECORDS
# ============================================================

print()
print(
    "Building V6.1 entity features..."
)


records = []


for entity in all_entities:

    output_amounts = (
        entity_output_amounts.get(
            entity,
            [],
        )
    )

    structuring_amounts = (
        entity_structuring_amounts.get(
            entity,
            [],
        )
    )

    peel_amounts = (
        entity_peel_amounts.get(
            entity,
            [],
        )
    )

    structuring_intervals = (
        entity_structuring_intervals.get(
            entity,
            [],
        )
    )

    peel_intervals = (
        entity_peel_intervals.get(
            entity,
            [],
        )
    )

    successor_delays = (
        entity_successor_delays.get(
            entity,
            [],
        )
    )

    retention_values = (
        entity_amount_retention.get(
            entity,
            [],
        )
    )

    outgoing_edges = (
        entity_outgoing_edges.get(
            entity,
            [],
        )
    )

    # ----------------------------------------
    # Structuring amount consistency
    # ----------------------------------------

    structuring_mean = safe_mean(
        structuring_amounts
    )

    structuring_std = safe_std(
        structuring_amounts
    )

    if structuring_mean > 0:

        structuring_cv = (
            structuring_std
            / structuring_mean
        )

    else:

        structuring_cv = 0.0

    # ----------------------------------------
    # Peel amount consistency
    # ----------------------------------------

    peel_mean = safe_mean(
        peel_amounts
    )

    peel_std = safe_std(
        peel_amounts
    )

    if peel_mean > 0:

        peel_cv = (
            peel_std
            / peel_mean
        )

    else:

        peel_cv = 0.0

    # ----------------------------------------
    # Amount retention
    # ----------------------------------------

    high_retention_count = sum(
        1
        for value
        in retention_values
        if value
        >= HIGH_AMOUNT_RETENTION
    )

    very_high_retention_count = sum(
        1
        for value
        in retention_values
        if value
        >= VERY_HIGH_AMOUNT_RETENTION
    )

    # ----------------------------------------
    # Record
    # ----------------------------------------

    records.append(
        {
            "entity_id":
                entity,

            # ==================================
            # STRUCTURING
            # ==================================

            "v61_structuring_band_count":
                len(
                    structuring_amounts
                ),

            "v61_structuring_band_ratio":
                safe_ratio(
                    len(
                        structuring_amounts
                    ),
                    len(
                        output_amounts
                    ),
                ),

            "v61_structuring_amount_mean":
                structuring_mean,

            "v61_structuring_amount_std":
                structuring_std,

            "v61_structuring_amount_cv":
                structuring_cv,

            "v61_structuring_median_interval_sec":
                safe_median(
                    structuring_intervals
                ),

            "v61_structuring_min_interval_sec":
                safe_min(
                    structuring_intervals
                ),

            # ==================================
            # PEEL-LIKE BEHAVIOR
            # ==================================

            "v61_small_peel_output_count":
                len(
                    peel_amounts
                ),

            "v61_small_peel_output_ratio":
                safe_ratio(
                    len(
                        peel_amounts
                    ),
                    len(
                        output_amounts
                    ),
                ),

            "v61_small_peel_amount_mean":
                peel_mean,

            "v61_small_peel_amount_std":
                peel_std,

            "v61_small_peel_amount_cv":
                peel_cv,

            "v61_peel_median_interval_sec":
                safe_median(
                    peel_intervals
                ),

            "v61_peel_min_interval_sec":
                safe_min(
                    peel_intervals
                ),

            # ==================================
            # CROSS-ENTITY IP SHARING
            # ==================================

            "v61_unique_source_ip_count":
                entity_unique_ip_count.get(
                    entity,
                    0,
                ),

            "v61_shared_ip_count":
                entity_shared_ip_count.get(
                    entity,
                    0,
                ),

            "v61_max_entities_per_ip":
                entity_max_entities_per_ip.get(
                    entity,
                    0,
                ),

            "v61_mean_entities_per_ip":
                entity_mean_entities_per_ip.get(
                    entity,
                    0.0,
                ),

            "v61_shared_ip_ratio":
                entity_shared_ip_ratio.get(
                    entity,
                    0.0,
                ),

            # ==================================
            # CROSS-ENTITY TEMPORAL BEHAVIOR
            # ==================================

            "v61_successor_observation_count":
                len(
                    successor_delays
                ),

            "v61_min_successor_delay_sec":
                safe_min(
                    successor_delays
                ),

            "v61_median_successor_delay_sec":
                safe_median(
                    successor_delays
                ),

            "v61_very_rapid_successor_count":
                entity_very_rapid_successor_count.get(
                    entity,
                    0,
                ),

            "v61_rapid_successor_count":
                entity_rapid_successor_count.get(
                    entity,
                    0,
                ),

            "v61_layering_successor_count":
                entity_layering_successor_count.get(
                    entity,
                    0,
                ),

            "v61_rapid_successor_ratio":
                safe_ratio(
                    entity_rapid_successor_count.get(
                        entity,
                        0,
                    ),
                    len(
                        successor_delays
                    ),
                ),

            "v61_layering_successor_ratio":
                safe_ratio(
                    entity_layering_successor_count.get(
                        entity,
                        0,
                    ),
                    len(
                        successor_delays
                    ),
                ),

            # ==================================
            # RAPID GRAPH PATH
            # ==================================

            "v61_rapid_chain_depth":
                entity_rapid_chain_depth.get(
                    entity,
                    0,
                ),

            # ==================================
            # AMOUNT RETENTION
            # ==================================

            "v61_amount_retention_observations":
                len(
                    retention_values
                ),

            "v61_mean_amount_retention":
                safe_mean(
                    retention_values
                ),

            "v61_median_amount_retention":
                safe_median(
                    retention_values
                ),

            "v61_high_amount_retention_count":
                high_retention_count,

            "v61_high_amount_retention_ratio":
                safe_ratio(
                    high_retention_count,
                    len(
                        retention_values
                    ),
                ),

            "v61_very_high_amount_retention_count":
                very_high_retention_count,

            "v61_very_high_amount_retention_ratio":
                safe_ratio(
                    very_high_retention_count,
                    len(
                        retention_values
                    ),
                ),

            # ==================================
            # GRAPH ACTIVITY
            # ==================================

            "v61_temporal_outgoing_edge_count":
                len(
                    outgoing_edges
                ),
        }
    )


v61_features = pd.DataFrame(
    records
)


# ============================================================
# VALIDATE NEW FEATURE TABLE
# ============================================================

if (
    v61_features[
        "entity_id"
    ].duplicated().any()
):

    raise ValueError(
        "Duplicate entity IDs generated "
        "in V6.1 feature table."
    )


# ============================================================
# MERGE V6 + V6.1
# ============================================================

result = v6.merge(
    v61_features,
    on="entity_id",
    how="left",
    validate="one_to_one",
)


v61_columns = [
    column
    for column in result.columns
    if column.startswith(
        "v61_"
    )
]


# Replace invalid numeric values.

for column in v61_columns:

    result[column] = (
        pd.to_numeric(
            result[column],
            errors="coerce",
        )
        .replace(
            [
                np.inf,
                -np.inf,
            ],
            np.nan,
        )
        .fillna(
            0.0
        )
    )


# ============================================================
# FINAL VALIDATION
# ============================================================

duplicate_count = (
    result[
        "entity_id"
    ]
    .duplicated()
    .sum()
)

nan_count = (
    result[
        v61_columns
    ]
    .isna()
    .sum()
    .sum()
)


if len(result) != len(v6):

    raise ValueError(
        "Entity count changed during V6.1 merge."
    )


if duplicate_count != 0:

    raise ValueError(
        "Duplicate entity IDs detected "
        "after V6.1 merge."
    )


if nan_count != 0:

    raise ValueError(
        "NaN values remain in V6.1 features."
    )


# ============================================================
# SAVE
# ============================================================

result.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# REPORT
# ============================================================

print()
print("=" * 70)
print(
    "V6.1 FEATURE ENGINEERING COMPLETE"
)
print("=" * 70)

print(
    f"Entities                 : "
    f"{len(result):,}"
)

print(
    f"Original V6 features     : "
    f"{len(v6.columns) - 1:,}"
)

print(
    f"New V6.1 features        : "
    f"{len(v61_columns):,}"
)

print(
    f"Total feature columns    : "
    f"{len(result.columns) - 1:,}"
)

print(
    f"Duplicate entity IDs     : "
    f"{duplicate_count:,}"
)

print(
    f"NaN values               : "
    f"{nan_count:,}"
)

print()
print(
    "V6.1 non-zero coverage:"
)

for column in v61_columns:

    nonzero = int(
        (
            result[column]
            != 0
        ).sum()
    )

    percentage = (
        100.0
        * nonzero
        / len(result)
    )

    print(
        f"{column:45} "
        f"{nonzero:>8,} "
        f"({percentage:6.2f}%)"
    )


print()
print(
    "Selected V6.1 statistics:"
)

important_columns = [
    "v61_structuring_band_count",
    "v61_small_peel_output_count",
    "v61_shared_ip_count",
    "v61_max_entities_per_ip",
    "v61_rapid_successor_count",
    "v61_rapid_successor_ratio",
    "v61_rapid_chain_depth",
    "v61_mean_amount_retention",
]

existing_important = [
    column
    for column
    in important_columns
    if column in result.columns
]

print(
    result[
        existing_important
    ]
    .describe()
    .T
    .to_string()
)


print()
print(
    f"Saved: {OUTPUT_FILE}"
)

print("=" * 70)
