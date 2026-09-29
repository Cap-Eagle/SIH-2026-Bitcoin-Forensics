#!/usr/bin/env python3

"""
GROUP B — V6 BEHAVIORAL FEATURE ENGINEERING

Build entity-level temporal and behavioral features directly from
transaction history, transaction ancestry, entity mappings, and
transaction/network correlation evidence.

IMPORTANT:
    Ground truth is NEVER loaded here.
    Labels are NEVER used here.

Inputs:
    data/raw/blockchain_transactions.csv
    data/correlated/group_b_features.csv
    outputs/address_to_entity.csv
    outputs/entity_features_v3.csv

Output:
    outputs/entity_behavior_features_v6.csv
"""

from pathlib import Path
from collections import defaultdict
import ast
import json
import math

import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

TX_FILE = ROOT / "data/raw/blockchain_transactions.csv"
CORR_FILE = ROOT / "data/correlated/group_b_features.csv"
MAP_FILE = ROOT / "outputs/address_to_entity.csv"
BASE_FEATURE_FILE = ROOT / "outputs/entity_features_v3.csv"

OUTPUT_FILE = ROOT / "outputs/entity_behavior_features_v6.csv"


# ============================================================
# CONFIG
# ============================================================

RAPID_HOP_SECONDS = 10 * 60
VERY_RAPID_SECONDS = 2 * 60

BURST_SECONDS = 5 * 60

FAN_THRESHOLD = 4

AMOUNT_ROUND_DECIMALS = 4

EPS = 1e-12


# ============================================================
# HELPERS
# ============================================================

def parse_list(value):
    if value is None:
        return []

    if isinstance(value, list):
        return value

    try:
        if pd.isna(value):
            return []
    except Exception:
        pass

    text = str(value).strip()

    if text in {"", "[]", "nan", "None", "null"}:
        return []

    try:
        result = json.loads(text)
        if isinstance(result, list):
            return result
    except Exception:
        pass

    try:
        result = ast.literal_eval(text)
        if isinstance(result, list):
            return result
    except Exception:
        pass

    return []


def safe_float(value, default=0.0):
    try:
        x = float(value)

        if not math.isfinite(x):
            return default

        return x

    except Exception:
        return default


def safe_mean(values):
    if not values:
        return 0.0

    return float(np.mean(values))


def safe_median(values):
    if not values:
        return 0.0

    return float(np.median(values))


def safe_std(values):
    if len(values) < 2:
        return 0.0

    return float(np.std(values))


def safe_max(values):
    if not values:
        return 0.0

    return float(np.max(values))


def ratio(a, b):
    if b <= 0:
        return 0.0

    return float(a) / float(b)


def entropy(values):
    values = np.asarray(
        [safe_float(v) for v in values if safe_float(v) > 0],
        dtype=float,
    )

    if len(values) == 0:
        return 0.0

    total = values.sum()

    if total <= 0:
        return 0.0

    p = values / total
    p = p[p > 0]

    return float(-(p * np.log2(p)).sum())


def concentration(values):
    values = np.asarray(
        [safe_float(v) for v in values if safe_float(v) > 0],
        dtype=float,
    )

    if len(values) == 0:
        return 0.0

    total = values.sum()

    if total <= 0:
        return 0.0

    p = values / total

    return float(np.square(p).sum())


# ============================================================
# VALIDATE
# ============================================================

for path in [
    TX_FILE,
    CORR_FILE,
    MAP_FILE,
    BASE_FEATURE_FILE,
]:
    if not path.exists():
        raise FileNotFoundError(path)


print("=" * 70)
print("GROUP B BEHAVIORAL FEATURE ENGINEERING — V6")
print("=" * 70)

print("\nLoading data...")


# ============================================================
# LOAD
# ============================================================

tx = pd.read_csv(TX_FILE)

corr = pd.read_csv(CORR_FILE)

mapping = pd.read_csv(MAP_FILE)

base = pd.read_csv(BASE_FEATURE_FILE)


print(f"Transactions       : {len(tx):,}")
print(f"Correlation rows   : {len(corr):,}")
print(f"Address mappings   : {len(mapping):,}")
print(f"Entities           : {len(base):,}")


# ============================================================
# NORMALIZE TIMESTAMPS
# ============================================================

tx["timestamp"] = pd.to_datetime(
    tx["tx_timestamp"],
    utc=True,
    errors="coerce",
)

tx_timestamp = dict(
    zip(
        tx["txid"].astype(str),
        tx["timestamp"],
    )
)


# ============================================================
# ADDRESS -> ENTITY
# ============================================================

address_to_entity = dict(
    zip(
        mapping["address"].astype(str),
        mapping["derived_entity_id"].astype(str),
    )
)


# ============================================================
# TRANSACTION -> ENTITY PARTICIPATION
# ============================================================

entity_outgoing_tx = defaultdict(set)
entity_incoming_tx = defaultdict(set)

entity_timestamps = defaultdict(list)

entity_output_counts = defaultdict(list)
entity_input_counts = defaultdict(list)

entity_output_amounts = defaultdict(list)
entity_input_amounts = defaultdict(list)

entity_two_output_count = defaultdict(int)

entity_total_noncoinbase_tx = defaultdict(int)


print("\nExpanding transaction behavior...")


for row in tx.itertuples(index=False):

    txid = str(row.txid)

    timestamp = tx_timestamp.get(txid)

    inputs = parse_list(row.input_addresses)
    outputs = parse_list(row.output_addresses)

    input_amounts = [
        safe_float(x)
        for x in parse_list(row.input_amounts_btc)
    ]

    output_amounts = [
        safe_float(x)
        for x in parse_list(row.output_amounts_btc)
    ]

    input_entities = {
        address_to_entity[a]
        for a in inputs
        if a in address_to_entity
    }

    output_entities = {
        address_to_entity[a]
        for a in outputs
        if a in address_to_entity
    }

    # -----------------------------------------------
    # OUTGOING / SPENDING ACTIVITY
    # -----------------------------------------------

    for entity in input_entities:

        entity_outgoing_tx[entity].add(txid)

        entity_input_counts[entity].append(len(inputs))
        entity_output_counts[entity].append(len(outputs))

        entity_input_amounts[entity].extend(input_amounts)
        entity_output_amounts[entity].extend(output_amounts)

        entity_total_noncoinbase_tx[entity] += 1

        if len(outputs) == 2:
            entity_two_output_count[entity] += 1

        if pd.notna(timestamp):
            entity_timestamps[entity].append(timestamp)

    # -----------------------------------------------
    # RECEIVING ACTIVITY
    # -----------------------------------------------

    for entity in output_entities:

        entity_incoming_tx[entity].add(txid)

        if pd.notna(timestamp):
            entity_timestamps[entity].append(timestamp)


# ============================================================
# RAPID-SPEND / TRANSACTION ANCESTRY
# ============================================================

entity_spend_delays = defaultdict(list)


print("Calculating transaction ancestry timing...")


for row in tx.itertuples(index=False):

    current_txid = str(row.txid)

    current_time = tx_timestamp.get(current_txid)

    if pd.isna(current_time):
        continue

    input_addresses = parse_list(row.input_addresses)

    input_entities = {
        address_to_entity[a]
        for a in input_addresses
        if a in address_to_entity
    }

    if not input_entities:
        continue

    parent_txids = parse_list(row.input_txids)

    for parent_txid in parent_txids:

        parent_time = tx_timestamp.get(str(parent_txid))

        if parent_time is None or pd.isna(parent_time):
            continue

        delay = (
            current_time - parent_time
        ).total_seconds()

        if delay < 0:
            continue

        for entity in input_entities:
            entity_spend_delays[entity].append(delay)


# ============================================================
# NETWORK / CORRELATION AGGREGATION
# ============================================================

print("Aggregating network evidence...")


corr_tx = corr.set_index("txid", drop=False)


entity_network = defaultdict(
    lambda: {
        "rows": 0,
        "candidate": 0.0,
        "exact": 0.0,
        "close": 0.0,
        "broad": 0.0,
        "weighted": 0.0,
        "src_ip": 0.0,
        "dst_ip": 0.0,
        "src_country": 0.0,
        "src_asn": 0.0,
        "high_ip": 0.0,
        "high_asn": 0.0,
        "tor": 0.0,
        "hosting": 0.0,
        "time_delta": [],
    }
)


for entity in base["entity_id"].astype(str):

    txids = (
        entity_outgoing_tx.get(entity, set())
        | entity_incoming_tx.get(entity, set())
    )

    for txid in txids:

        if txid not in corr_tx.index:
            continue

        row = corr_tx.loc[txid]

        # Defensive handling in case txid duplicates exist.
        if isinstance(row, pd.DataFrame):
            rows = [
                r for _, r in row.iterrows()
            ]
        else:
            rows = [row]

        for r in rows:

            d = entity_network[entity]

            d["rows"] += 1

            d["candidate"] += safe_float(
                r.get("candidate_count", 0)
            )

            d["exact"] += safe_float(
                r.get("exact_match_count", 0)
            )

            d["close"] += safe_float(
                r.get("close_candidate_count", 0)
            )

            d["broad"] += safe_float(
                r.get("broad_candidate_count", 0)
            )

            d["weighted"] += safe_float(
                r.get("weighted_evidence_score", 0)
            )

            d["src_ip"] += safe_float(
                r.get("unique_src_ip_count", 0)
            )

            d["dst_ip"] += safe_float(
                r.get("unique_dst_ip_count", 0)
            )

            d["src_country"] += safe_float(
                r.get("unique_src_country_count", 0)
            )

            d["src_asn"] += safe_float(
                r.get("unique_src_asn_count", 0)
            )

            d["high_ip"] += safe_float(
                r.get(
                    "high_confidence_src_ip_count",
                    0,
                )
            )

            d["high_asn"] += safe_float(
                r.get(
                    "high_confidence_src_asn_count",
                    0,
                )
            )

            d["tor"] += safe_float(
                r.get("tor_like_event_count", 0)
            )

            d["hosting"] += safe_float(
                r.get("hosting_like_event_count", 0)
            )

            td = safe_float(
                r.get("min_abs_time_delta_ms", np.nan),
                default=np.nan,
            )

            if np.isfinite(td):
                d["time_delta"].append(td)


# ============================================================
# BUILD ENTITY FEATURES
# ============================================================

print("Building V6 entity behavioral features...")


records = []


for entity in base["entity_id"].astype(str):

    outgoing = entity_outgoing_tx.get(entity, set())
    incoming = entity_incoming_tx.get(entity, set())

    timestamps = sorted(
        set(entity_timestamps.get(entity, []))
    )

    output_counts = entity_output_counts.get(
        entity,
        [],
    )

    input_counts = entity_input_counts.get(
        entity,
        [],
    )

    output_amounts = entity_output_amounts.get(
        entity,
        [],
    )

    input_amounts = entity_input_amounts.get(
        entity,
        [],
    )

    spend_delays = entity_spend_delays.get(
        entity,
        [],
    )

    network = entity_network[entity]

    # --------------------------------------------------------
    # TEMPORAL ACTIVITY
    # --------------------------------------------------------

    interarrival = []

    if len(timestamps) >= 2:

        for a, b in zip(
            timestamps[:-1],
            timestamps[1:],
        ):
            interarrival.append(
                (b - a).total_seconds()
            )

    burst_count = sum(
        x <= BURST_SECONDS
        for x in interarrival
    )

    # --------------------------------------------------------
    # RAPID HOP
    # --------------------------------------------------------

    rapid_count = sum(
        d <= RAPID_HOP_SECONDS
        for d in spend_delays
    )

    very_rapid_count = sum(
        d <= VERY_RAPID_SECONDS
        for d in spend_delays
    )

    # --------------------------------------------------------
    # FAN IN / FAN OUT
    # --------------------------------------------------------

    fanout_transactions = sum(
        c >= FAN_THRESHOLD
        for c in output_counts
    )

    fanin_transactions = sum(
        c >= FAN_THRESHOLD
        for c in input_counts
    )

    # --------------------------------------------------------
    # PEEL-LIKE OUTPUT STRUCTURE
    # --------------------------------------------------------

    two_output_count = entity_two_output_count.get(
        entity,
        0,
    )

    spending_tx_count = max(
        len(outgoing),
        1,
    )

    # --------------------------------------------------------
    # STRUCTURING
    # --------------------------------------------------------

    rounded_outputs = [
        round(v, AMOUNT_ROUND_DECIMALS)
        for v in output_amounts
        if v > 0
    ]

    if rounded_outputs:

        counts = pd.Series(
            rounded_outputs
        ).value_counts()

        repeated_amount_count = int(
            counts[counts > 1].sum()
        )

        unique_output_amounts = int(
            len(counts)
        )

        max_repeated_amount_frequency = int(
            counts.max()
        )

    else:

        repeated_amount_count = 0
        unique_output_amounts = 0
        max_repeated_amount_frequency = 0

    # --------------------------------------------------------
    # NETWORK RATIOS
    # --------------------------------------------------------

    candidate = network["candidate"]

    exact_rate = ratio(
        network["exact"],
        candidate,
    )

    high_ip_rate = ratio(
        network["high_ip"],
        max(network["src_ip"], 1.0),
    )

    high_asn_rate = ratio(
        network["high_asn"],
        max(network["src_asn"], 1.0),
    )

    # --------------------------------------------------------
    # FLOW THROUGH
    # --------------------------------------------------------

    total_input = sum(input_amounts)
    total_output = sum(output_amounts)

    if total_input > 0 and total_output > 0:
        flow_through_ratio = (
            min(total_input, total_output)
            / max(total_input, total_output)
        )
    else:
        flow_through_ratio = 0.0

    # --------------------------------------------------------
    # RECORD
    # --------------------------------------------------------

    records.append(
        {
            "entity_id": entity,

            # activity
            "v6_incoming_tx_count": len(incoming),
            "v6_outgoing_tx_count": len(outgoing),
            "v6_active_tx_count": len(
                incoming | outgoing
            ),

            # temporal
            "v6_median_interarrival_sec":
                safe_median(interarrival),

            "v6_mean_interarrival_sec":
                safe_mean(interarrival),

            "v6_min_interarrival_sec":
                min(interarrival)
                if interarrival else 0.0,

            "v6_burst_interval_count":
                burst_count,

            "v6_burst_interval_ratio":
                ratio(
                    burst_count,
                    len(interarrival),
                ),

            # rapid hopping
            "v6_spend_delay_observations":
                len(spend_delays),

            "v6_median_spend_delay_sec":
                safe_median(spend_delays),

            "v6_mean_spend_delay_sec":
                safe_mean(spend_delays),

            "v6_min_spend_delay_sec":
                min(spend_delays)
                if spend_delays else 0.0,

            "v6_rapid_spend_count":
                rapid_count,

            "v6_rapid_spend_ratio":
                ratio(
                    rapid_count,
                    len(spend_delays),
                ),

            "v6_very_rapid_spend_ratio":
                ratio(
                    very_rapid_count,
                    len(spend_delays),
                ),

            # fan behavior
            "v6_max_outputs_per_spend":
                safe_max(output_counts),

            "v6_mean_outputs_per_spend":
                safe_mean(output_counts),

            "v6_fanout_tx_count":
                fanout_transactions,

            "v6_fanout_tx_ratio":
                ratio(
                    fanout_transactions,
                    len(output_counts),
                ),

            "v6_max_inputs_per_spend":
                safe_max(input_counts),

            "v6_mean_inputs_per_spend":
                safe_mean(input_counts),

            "v6_fanin_tx_count":
                fanin_transactions,

            "v6_fanin_tx_ratio":
                ratio(
                    fanin_transactions,
                    len(input_counts),
                ),

            # peel structure
            "v6_two_output_tx_count":
                two_output_count,

            "v6_two_output_tx_ratio":
                ratio(
                    two_output_count,
                    spending_tx_count,
                ),

            # amounts
            "v6_output_amount_entropy":
                entropy(output_amounts),

            "v6_output_amount_concentration":
                concentration(output_amounts),

            "v6_input_amount_entropy":
                entropy(input_amounts),

            "v6_input_amount_concentration":
                concentration(input_amounts),

            "v6_unique_output_amounts":
                unique_output_amounts,

            "v6_repeated_output_amount_count":
                repeated_amount_count,

            "v6_repeated_output_amount_ratio":
                ratio(
                    repeated_amount_count,
                    len(rounded_outputs),
                ),

            "v6_max_repeated_amount_frequency":
                max_repeated_amount_frequency,

            # flow
            "v6_flow_through_ratio":
                flow_through_ratio,

            # network
            "v6_network_candidate_count":
                candidate,

            "v6_network_exact_match_count":
                network["exact"],

            "v6_network_exact_match_rate":
                exact_rate,

            "v6_network_weighted_evidence":
                network["weighted"],

            "v6_network_src_ip_count":
                network["src_ip"],

            "v6_network_dst_ip_count":
                network["dst_ip"],

            "v6_network_src_country_count":
                network["src_country"],

            "v6_network_src_asn_count":
                network["src_asn"],

            "v6_network_high_conf_ip_rate":
                high_ip_rate,

            "v6_network_high_conf_asn_rate":
                high_asn_rate,

            "v6_network_tor_event_count":
                network["tor"],

            "v6_network_hosting_event_count":
                network["hosting"],

            "v6_network_median_time_delta_ms":
                safe_median(
                    network["time_delta"]
                ),
        }
    )


# ============================================================
# SAVE
# ============================================================

features = pd.DataFrame(records)


numeric_columns = [
    c
    for c in features.columns
    if c != "entity_id"
]


features[numeric_columns] = (
    features[numeric_columns]
    .replace([np.inf, -np.inf], np.nan)
    .fillna(0.0)
)


OUTPUT_FILE.parent.mkdir(
    parents=True,
    exist_ok=True,
)

features.to_csv(
    OUTPUT_FILE,
    index=False,
)


# ============================================================
# VALIDATION
# ============================================================

print("\n" + "=" * 70)
print("V6 FEATURE ENGINEERING COMPLETE")
print("=" * 70)

print(f"Entities                 : {len(features):,}")
print(f"Behavior features        : {len(features.columns) - 1}")
print(
    "Duplicate entity IDs    :",
    features["entity_id"].duplicated().sum(),
)

print(
    "NaN values              :",
    int(features.isna().sum().sum()),
)


print("\nNon-zero coverage:")

for column in numeric_columns:

    count = int(
        (features[column] != 0).sum()
    )

    pct = (
        count / len(features) * 100
        if len(features)
        else 0
    )

    print(
        f"{column:42s}"
        f"{count:8,d} "
        f"({pct:6.2f}%)"
    )


print("\nKey behavioral statistics:")

key_columns = [
    "v6_rapid_spend_ratio",
    "v6_very_rapid_spend_ratio",
    "v6_fanout_tx_ratio",
    "v6_fanin_tx_ratio",
    "v6_two_output_tx_ratio",
    "v6_repeated_output_amount_ratio",
    "v6_flow_through_ratio",
    "v6_burst_interval_ratio",
]

print(
    features[key_columns]
    .describe()
    .T
    .to_string()
)


print(f"\nSaved: {OUTPUT_FILE}")

print("=" * 70)
