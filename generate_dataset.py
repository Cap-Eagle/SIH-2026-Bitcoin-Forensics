"""PS 26146 Group A — upgraded synthetic Bitcoin/P2P forensic dataset generator.

Creates TARGET_TOTAL_TX synthetic Bitcoin-like transactions (configurable —
default scaled to ~30,000) with:
- UTXO-consistent transactions: each non-coinbase input spends an earlier output.
- Multi-input activity for common-input ownership clustering.
- Separate simulated P2P network events for temporal correlation.
- Local deterministic GeoIP/ASN enrichment.
- SEVEN planted pattern types, each with INSTANCES_PER_SCENARIO instances
  (default 8, not 1 — a single planted example per pattern can't support a
  meaningful recall statistic): fan_out_burst, ip_sharing, structuring,
  rapid_hop, fan_in_collection, multi_hop_layering (new), and peel_chain (new).
- Private ground truth and private entity-to-wallet mapping for evaluation.
- CSV, JSON, and XML ingestion demonstration files.

Dependencies:
    pip install numpy

Run:
    python generate_dataset_v2.py

All wallet IDs, IPs, ASNs, country labels, and associations are synthetic.
Documentation-only IP ranges are used. P2P correlation is simulated relay
observation evidence and is not proof of wallet ownership.
"""
from __future__ import annotations

import csv
import hashlib
import ipaddress
import json
import random
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

import numpy as np

# ==================== CONFIGURATION ====================
SEED = 42

# --- SCALE: bump these two to grow the dataset. 10x gives ~30,000 tx. ---
SCALE_FACTOR = 30
TARGET_TOTAL_TX = 3_000 * SCALE_FACTOR
NUM_ENTITIES = 700 * SCALE_FACTOR

BASE_TIME = datetime(2026, 1, 1, 0, 0, 0, tzinfo=timezone.utc)
# Each transaction advances the clock by a random ~15-100s gap (next_time()).
# The time window MUST scale with transaction count or generation fails
# ("Generated timeline exceeded configured span") well before 30k tx —
# confirmed by actually running this at scale, not assumed.
TIME_SPAN_SECONDS = 72 * 3600 * SCALE_FACTOR
SATOSHIS = 100_000_000
SCRIPT_TYPES = ["P2PKH", "P2SH", "P2WPKH"]
OUT_DIR = Path("output")
PRIVATE_DIR = OUT_DIR / "private"
EVENTS_PER_TX = (3, 6)
DECOY_EVENT_FRACTION = 0.15
CORRELATION_WINDOW_SECONDS = 30

# --- ANOMALY PLANTING: how many instances of EACH scenario type. ---
# Previously exactly 1 per scenario (statistically meaningless for recall).
# 8 gives a real per-scenario sample; raise further if you want tighter
# confidence intervals on your recall numbers.
INSTANCES_PER_SCENARIO = 30

# Per-scenario internal sizes (kept as named config, not magic numbers).
FAN_OUT_TARGET_COUNT = 20
IP_SHARING_GROUP_SIZE = 7
STRUCTURING_TX_COUNT = 10
RAPID_HOP_CHAIN_LENGTH = 7
FAN_IN_DONOR_COUNT = 12
MULTI_HOP_LAYERING_CHAIN_LENGTH = 4       # new scenario
PEEL_CHAIN_TX_COUNT = 15                  # new scenario

BENIGN_CONTROL_COUNT = 25 * SCALE_FACTOR

random.seed(SEED)
np.random.seed(SEED)

# RFC 5737 documentation-only IPv4 ranges and private-use ASN values.
IP_PROFILES = [
    {
        "country_code": "IN",
        "asn": 64512,
        "asn_org": "Synthetic ISP India",
        "prefix": "192.0.2.0/24",
        "node_profile": "wallet_node",
        "is_tor_like": False,
        "is_hosting_like": False,
    },
    {
        "country_code": "DE",
        "asn": 64513,
        "asn_org": "Synthetic Hosting Germany",
        "prefix": "198.51.100.0/24",
        "node_profile": "relay",
        "is_tor_like": False,
        "is_hosting_like": True,
    },
    {
        "country_code": "NL",
        "asn": 64514,
        "asn_org": "Synthetic Privacy Relay Netherlands",
        "prefix": "203.0.113.0/24",
        "node_profile": "tor_like",
        "is_tor_like": True,
        "is_hosting_like": False,
    },
]

_wallet_counter = 0
_tx_counter = 0
_event_counter = 0


# ==================== BASIC HELPERS ====================
def to_btc(satoshis: int) -> float:
    return round(satoshis / SATOSHIS, 8)


def iso_z(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def parse_z(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def new_wallet(kind: str = "wallet") -> str:
    global _wallet_counter
    _wallet_counter += 1
    return f"{kind}_syn_{_wallet_counter:07d}"


def new_txid() -> str:
    global _tx_counter
    _tx_counter += 1
    return hashlib.sha256(f"{SEED}|tx|{_tx_counter}".encode()).hexdigest()


def new_event_id() -> str:
    global _event_counter
    _event_counter += 1
    return f"evt_syn_{_event_counter:08d}"


def next_time(previous: datetime, low: int = 15, high: int = 100) -> datetime:
    return previous + timedelta(seconds=random.randint(low, high))


def build_ip_pool() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for profile in IP_PROFILES:
        network = ipaddress.ip_network(profile["prefix"])
        for host in list(network.hosts())[9:79]:
            row = {key: value for key, value in profile.items() if key != "prefix"}
            row["ip"] = str(host)
            rows.append(row)
    return rows


def choose_ip(ip_pool: list[dict[str, Any]], node_profile: str | None = None) -> str:
    candidates = [
        row for row in ip_pool
        if node_profile is None or row["node_profile"] == node_profile
    ]
    if not candidates:
        raise ValueError(f"No IP candidates for profile {node_profile}")
    return random.choice(candidates)["ip"]


# ==================== UTXO / TRANSACTION FUNCTIONS ====================
def validate_transaction(tx: dict[str, Any]) -> None:
    """Validate record shape and exact satoshi accounting."""
    if tx["is_coinbase"]:
        if any([
            tx["input_txids"], tx["input_vout_indexes"],
            tx["input_addresses"], tx["input_amounts_btc"],
        ]):
            raise ValueError(f"Coinbase transaction has inputs: {tx['txid']}")
        if not tx["output_addresses"] or tx["fee_btc"] != 0.0:
            raise ValueError(f"Invalid coinbase transaction: {tx['txid']}")
        return

    if not tx["input_addresses"] or not tx["output_addresses"]:
        raise ValueError(f"Missing inputs or outputs: {tx['txid']}")
    if not (
        len(tx["input_txids"])
        == len(tx["input_vout_indexes"])
        == len(tx["input_addresses"])
        == len(tx["input_amounts_btc"])
    ):
        raise ValueError(f"Input arrays do not align: {tx['txid']}")
    if len(tx["output_addresses"]) != len(tx["output_amounts_btc"]):
        raise ValueError(f"Output arrays do not align: {tx['txid']}")

    input_sats = sum(round(value * SATOSHIS) for value in tx["input_amounts_btc"])
    output_sats = sum(round(value * SATOSHIS) for value in tx["output_amounts_btc"])
    fee_sats = round(tx["fee_btc"] * SATOSHIS)

    if input_sats != output_sats + fee_sats:
        raise ValueError(f"Value conservation failed: {tx['txid']}")
    if min(tx["input_amounts_btc"] + tx["output_amounts_btc"]) <= 0 or fee_sats <= 0:
        raise ValueError(f"Non-positive amount or fee: {tx['txid']}")
    if tx["script_type"] not in SCRIPT_TYPES:
        raise ValueError(f"Unsupported script type: {tx['txid']}")


def create_transaction(
    timestamp: datetime,
    inputs: list[dict[str, Any]],
    outputs: list[tuple[str, int]],
    fee_sats: int,
    origin_entity: str,
    scenario: str | None,
    is_coinbase: bool = False,
) -> dict[str, Any]:
    txid = new_txid()

    if is_coinbase:
        tx = {
            "txid": txid,
            "tx_timestamp": iso_z(timestamp),
            "block_height": 900_000,
            "is_coinbase": True,
            "input_txids": [],
            "input_vout_indexes": [],
            "input_addresses": [],
            "output_addresses": [address for address, _ in outputs],
            "input_amounts_btc": [],
            "output_amounts_btc": [to_btc(amount) for _, amount in outputs],
            "fee_btc": 0.0,
            "script_type": "P2PKH",
            "_origin_entity": origin_entity,
            "_scenario": scenario,
        }
    else:
        total_input_sats = sum(utxo["amount_sats"] for utxo in inputs)
        total_output_sats = sum(amount for _, amount in outputs)
        if total_input_sats != total_output_sats + fee_sats:
            raise ValueError("Caller supplied unbalanced inputs, outputs, and fee")

        tx = {
            "txid": txid,
            "tx_timestamp": iso_z(timestamp),
            "block_height": 900_000,
            "is_coinbase": False,
            "input_txids": [utxo["txid"] for utxo in inputs],
            "input_vout_indexes": [utxo["vout_index"] for utxo in inputs],
            "input_addresses": [utxo["address"] for utxo in inputs],
            "output_addresses": [address for address, _ in outputs],
            "input_amounts_btc": [to_btc(utxo["amount_sats"]) for utxo in inputs],
            "output_amounts_btc": [to_btc(amount) for _, amount in outputs],
            "fee_btc": to_btc(fee_sats),
            "script_type": random.choice(SCRIPT_TYPES),
            "_origin_entity": origin_entity,
            "_scenario": scenario,
        }

    validate_transaction(tx)
    return tx


def register_transaction(
    tx: dict[str, Any],
    spent_inputs: list[dict[str, Any]],
    utxos: dict[tuple[str, int], dict[str, Any]],
) -> None:
    for utxo in spent_inputs:
        if utxo["spent"]:
            raise ValueError(f"Double-spend attempted: {(utxo['txid'], utxo['vout_index'])}")
        utxo["spent"] = True

    for index, (address, amount_btc) in enumerate(
        zip(tx["output_addresses"], tx["output_amounts_btc"])
    ):
        utxos[(tx["txid"], index)] = {
            "txid": tx["txid"],
            "vout_index": index,
            "address": address,
            "amount_sats": round(amount_btc * SATOSHIS),
            "created_at": tx["tx_timestamp"],
            "spent": False,
        }


def spendable_for_entity(
    entity: str,
    utxos: dict[tuple[str, int], dict[str, Any]],
    wallet_entity: dict[str, str],
) -> list[dict[str, Any]]:
    return [
        utxo for utxo in utxos.values()
        if not utxo["spent"] and wallet_entity[utxo["address"]] == entity
    ]


def add_address(
    entity: str,
    kind: str,
    entity_wallets: dict[str, list[str]],
    wallet_entity: dict[str, str],
    wallet_ip: dict[str, str],
    entity_map: list[dict[str, str]],
    entity_type: str = "normal",
) -> str:
    address = new_wallet(kind)
    entity_wallets[entity].append(address)
    wallet_entity[address] = entity
    wallet_ip[address] = wallet_ip[entity_wallets[entity][0]]
    entity_map.append({
        "entity_id": entity,
        "wallet_id": address,
        "entity_type": entity_type,
        "is_synthetic": "true",
    })
    return address


def spend(
    entity: str,
    timestamp: datetime,
    recipient_address: str,
    entity_wallets: dict[str, list[str]],
    wallet_entity: dict[str, str],
    wallet_ip: dict[str, str],
    entity_map: list[dict[str, str]],
    utxos: dict[tuple[str, int], dict[str, Any]],
    scenario: str | None = None,
    recipient_amount_sats: int | None = None,
    force_input_count: int | None = None,
) -> dict[str, Any]:
    candidates = spendable_for_entity(entity, utxos, wallet_entity)
    if not candidates:
        raise RuntimeError(f"No unspent UTXO for {entity}")

    candidates.sort(key=lambda item: item["amount_sats"], reverse=True)
    input_count = force_input_count or random.choices(
        [1, 2, 3, 4], weights=[0.72, 0.18, 0.07, 0.03], k=1
    )[0]
    inputs = candidates[:min(input_count, len(candidates))]
    total_input_sats = sum(item["amount_sats"] for item in inputs)
    fee_sats = random.randint(1_000, min(30_000, max(1_000, total_input_sats // 100)))

    if recipient_amount_sats is None:
        recipient_amount_sats = random.randint(
            max(10_000, total_input_sats // 10),
            max(10_001, int((total_input_sats - fee_sats) * 0.70)),
        )
    if recipient_amount_sats + fee_sats >= total_input_sats:
        recipient_amount_sats = total_input_sats - fee_sats - 1

    change_sats = total_input_sats - recipient_amount_sats - fee_sats
    outputs = [(recipient_address, recipient_amount_sats)]
    if change_sats > 0:
        change_address = add_address(
            entity, "change", entity_wallets, wallet_entity,
            wallet_ip, entity_map,
        )
        outputs.append((change_address, change_sats))

    tx = create_transaction(timestamp, inputs, outputs, fee_sats, entity, scenario)
    register_transaction(tx, inputs, utxos)
    return tx


# ==================== ENTITY / SCENARIO GENERATION ====================
def make_entities(ip_pool: list[dict[str, Any]]) -> tuple[
    dict[str, list[str]], dict[str, str], dict[str, str], list[dict[str, str]]
]:
    entity_wallets: dict[str, list[str]] = {}
    wallet_entity: dict[str, str] = {}
    wallet_ip: dict[str, str] = {}
    entity_map: list[dict[str, str]] = []

    for index in range(NUM_ENTITIES):
        entity = f"entity_syn_{index + 1:04d}"
        wallet = new_wallet("wallet")
        entity_wallets[entity] = [wallet]
        wallet_entity[wallet] = entity
        wallet_ip[wallet] = choose_ip(ip_pool, "wallet_node")
        entity_map.append({
            "entity_id": entity,
            "wallet_id": wallet,
            "entity_type": "normal",
            "is_synthetic": "true",
        })

    return entity_wallets, wallet_entity, wallet_ip, entity_map


def generate_dataset() -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, str]],
]:
    ip_pool = build_ip_pool()
    entity_wallets, wallet_entity, wallet_ip, entity_map = make_entities(ip_pool)
    entities = list(entity_wallets)
    utxos: dict[tuple[str, int], dict[str, Any]] = {}
    transactions: list[dict[str, Any]] = []
    ground_truth: list[dict[str, Any]] = []
    current_time = BASE_TIME

    # Each entity receives one exported seed output. Future non-coinbase inputs cite these outputs.
    for entity in entities:
        tx = create_transaction(
            current_time,
            [],
            [(entity_wallets[entity][0], 2_000_000_000)],  # 20 BTC synthetic funding
            0,
            entity,
            "seed_funding",
            is_coinbase=True,
        )
        register_transaction(tx, [], utxos)
        transactions.append(tx)
        current_time = next_time(current_time)

    # ---- Entity allocation cursor -------------------------------------
    # Every "protagonist" entity (hub, structurer, chain member, donor...)
    # must be unique across ALL scenario instances, or ground-truth labels
    # collide. Recipients/counterparties can safely reuse the shared pool.
    # Reserve protagonists from the FRONT of the entity list; recipients
    # draw from a pool that starts well past the last protagonist; benign
    # controls are drawn from the very END, guaranteeing no overlap
    # regardless of how INSTANCES_PER_SCENARIO or NUM_ENTITIES are tuned.
    _cursor = 0

    def reserve_entities(n: int) -> list[str]:
        nonlocal _cursor
        block = entities[_cursor:_cursor + n]
        if len(block) < n:
            raise RuntimeError(
                f"NUM_ENTITIES={NUM_ENTITIES} is too small for the configured "
                f"INSTANCES_PER_SCENARIO={INSTANCES_PER_SCENARIO}. Increase NUM_ENTITIES."
            )
        _cursor += n
        return block

    RECIPIENT_POOL_START = max(
        300,
        INSTANCES_PER_SCENARIO * (
            1 + IP_SHARING_GROUP_SIZE + 1 + RAPID_HOP_CHAIN_LENGTH
            + (1 + FAN_IN_DONOR_COUNT) + MULTI_HOP_LAYERING_CHAIN_LENGTH + 1
        ) + 50,
    )
    recipient_pool = entities[RECIPIENT_POOL_START:-BENIGN_CONTROL_COUNT]
    if len(recipient_pool) < 20:
        raise RuntimeError("NUM_ENTITIES too small to leave a usable recipient pool — increase it.")

    # 1. FAN-OUT BURST — hub distributes funds to many fresh addresses.
    for _ in range(INSTANCES_PER_SCENARIO):
        hub = reserve_entities(1)[0]
        hub_input = spendable_for_entity(hub, utxos, wallet_entity)[:1]
        hub_total = hub_input[0]["amount_sats"]
        fanout_fee = 15_000
        amount_each = (hub_total - fanout_fee) // FAN_OUT_TARGET_COUNT
        fanout_outputs = []
        for i in range(FAN_OUT_TARGET_COUNT):
            target_entity = recipient_pool[(i + _cursor) % len(recipient_pool)]
            target_address = add_address(
                target_entity, "fresh", entity_wallets, wallet_entity,
                wallet_ip, entity_map, "fan_out_target",
            )
            fanout_outputs.append((target_address, amount_each))

        fanout_change = hub_total - fanout_fee - (amount_each * FAN_OUT_TARGET_COUNT)
        if fanout_change > 0:
            change_address = add_address(hub, "change", entity_wallets, wallet_entity, wallet_ip, entity_map)
            fanout_outputs.append((change_address, fanout_change))

        fanout_tx = create_transaction(
            current_time, hub_input, fanout_outputs, fanout_fee, hub, "fan_out_burst"
        )
        register_transaction(fanout_tx, hub_input, utxos)
        transactions.append(fanout_tx)
        current_time = next_time(current_time)
        ground_truth.append({
            "target_type": "entity", "target_id": hub,
            "scenario_id": "fan_out_burst", "is_anomaly": "true", "severity": "high",
            "related_txids": fanout_tx["txid"],
            "expected_signals": "high_fan_out;fresh_output_ratio",
            "detail": f"One hub transaction distributes to {FAN_OUT_TARGET_COUNT} fresh output addresses.",
        })

    # 2. IP SHARING — a group of entities uses only two synthetic source IPs.
    for _ in range(INSTANCES_PER_SCENARIO):
        shared_entities = reserve_entities(IP_SHARING_GROUP_SIZE)
        shared_ips = random.sample(
            [row["ip"] for row in ip_pool if row["node_profile"] == "wallet_node"], 2
        )
        for index, entity in enumerate(shared_entities):
            wallet_ip[entity_wallets[entity][0]] = shared_ips[index % 2]
            related = []
            for _ in range(3):
                recipient_entity = random.choice(recipient_pool)
                tx = spend(
                    entity, current_time, entity_wallets[recipient_entity][0],
                    entity_wallets, wallet_entity, wallet_ip, entity_map, utxos,
                    scenario="ip_sharing",
                )
                transactions.append(tx)
                related.append(tx["txid"])
                current_time = next_time(current_time)
            ground_truth.append({
                "target_type": "entity", "target_id": entity,
                "scenario_id": "ip_sharing", "is_anomaly": "true", "severity": "medium",
                "related_txids": "|".join(related),
                "expected_signals": "shared_source_ip;shared_asn",
                "detail": f"Entity uses one of two P2P source IPs shared by {IP_SHARING_GROUP_SIZE} entities.",
            })

    # 3. STRUCTURING — repeated transfers just under a synthetic 1 BTC threshold.
    for _ in range(INSTANCES_PER_SCENARIO):
        structurer = reserve_entities(1)[0]
        structuring_ids = []
        for _ in range(STRUCTURING_TX_COUNT):
            recipient_entity = random.choice(recipient_pool)
            amount = random.randint(92_000_000, 99_500_000)  # 0.92–0.995 BTC
            tx = spend(
                structurer, current_time, entity_wallets[recipient_entity][0],
                entity_wallets, wallet_entity, wallet_ip, entity_map, utxos,
                scenario="structuring", recipient_amount_sats=amount,
            )
            transactions.append(tx)
            structuring_ids.append(tx["txid"])
            current_time = next_time(current_time, 90, 900)
        ground_truth.append({
            "target_type": "entity", "target_id": structurer,
            "scenario_id": "structuring", "is_anomaly": "true", "severity": "high",
            "related_txids": "|".join(structuring_ids),
            "expected_signals": "near_threshold_amounts;high_frequency",
            "detail": f"{STRUCTURING_TX_COUNT} transfers between 0.92 and 0.995 BTC within a short period.",
        })

    # 4. RAPID HOP — funds move through a chain of entities inside minutes.
    for _ in range(INSTANCES_PER_SCENARIO):
        hop_entities = reserve_entities(RAPID_HOP_CHAIN_LENGTH)
        hop_ids = []
        for index, entity in enumerate(hop_entities[:-1]):
            recipient_entity = hop_entities[index + 1]
            tx = spend(
                entity, current_time, entity_wallets[recipient_entity][0],
                entity_wallets, wallet_entity, wallet_ip, entity_map, utxos,
                scenario="rapid_hop", force_input_count=1,
            )
            transactions.append(tx)
            hop_ids.append(tx["txid"])
            current_time = next_time(current_time, 30, 180)
        ground_truth.append({
            "target_type": "entity", "target_id": hop_entities[0],
            "scenario_id": "rapid_hop", "is_anomaly": "true", "severity": "high",
            "related_txids": "|".join(hop_ids),
            "expected_signals": "short_interhop_time;directed_chain",
            "detail": f"Funds traverse {RAPID_HOP_CHAIN_LENGTH} entities with 30–180 second gaps.",
        })

    # 5. FAN-IN COLLECTION — many donors send to one collector quickly.
    for _ in range(INSTANCES_PER_SCENARIO):
        collector = reserve_entities(1)[0]
        donors = reserve_entities(FAN_IN_DONOR_COUNT)
        fanin_ids = []
        for donor in donors:
            tx = spend(
                donor, current_time, entity_wallets[collector][0],
                entity_wallets, wallet_entity, wallet_ip, entity_map, utxos,
                scenario="fan_in_collection",
            )
            transactions.append(tx)
            fanin_ids.append(tx["txid"])
            current_time = next_time(current_time, 30, 180)
        ground_truth.append({
            "target_type": "entity", "target_id": collector,
            "scenario_id": "fan_in_collection", "is_anomaly": "true", "severity": "high",
            "related_txids": "|".join(fanin_ids),
            "expected_signals": "high_fan_in;many_distinct_senders",
            "detail": f"{FAN_IN_DONOR_COUNT} independent donors fund a single collector quickly.",
        })

    # 6. MULTI-HOP LAYERING (new) — funds pass through several intermediary
    # wallets with amount decay at each hop, simulating classic layering
    # rather than rapid_hop's speed-focused single pass-through chain.
    # Each hop skims a small fee-like amount and forwards the rest.
    for _ in range(INSTANCES_PER_SCENARIO):
        layer_entities = reserve_entities(MULTI_HOP_LAYERING_CHAIN_LENGTH)
        layer_ids = []
        for index, entity in enumerate(layer_entities[:-1]):
            recipient_entity = layer_entities[index + 1]
            candidates = spendable_for_entity(entity, utxos, wallet_entity)
            if not candidates:
                continue
            total_available = sum(c["amount_sats"] for c in candidates)
            # Forward ~90-96% onward, keep the rest as an implicit fee/skim.
            forward_fraction = random.uniform(0.90, 0.96)
            tx = spend(
                entity, current_time, entity_wallets[recipient_entity][0],
                entity_wallets, wallet_entity, wallet_ip, entity_map, utxos,
                scenario="multi_hop_layering", force_input_count=1,
                recipient_amount_sats=int(total_available * forward_fraction),
            )
            transactions.append(tx)
            layer_ids.append(tx["txid"])
            current_time = next_time(current_time, 60, 400)
        ground_truth.append({
            "target_type": "entity", "target_id": layer_entities[0],
            "scenario_id": "multi_hop_layering", "is_anomaly": "true", "severity": "high",
            "related_txids": "|".join(layer_ids),
            "expected_signals": "multi_hop_chain;amount_decay;layering_depth",
            "detail": f"Funds pass through {MULTI_HOP_LAYERING_CHAIN_LENGTH} intermediary wallets with decaying amounts.",
        })

    # 7. PEEL CHAIN (new) — one entity repeatedly skims a small, fairly
    # consistent amount off a large balance across many transactions,
    # keeping the change each time (classic peeling behavior).
    for _ in range(INSTANCES_PER_SCENARIO):
        peeler = reserve_entities(1)[0]
        peel_ids = []
        for _ in range(PEEL_CHAIN_TX_COUNT):
            candidates = spendable_for_entity(peeler, utxos, wallet_entity)
            if not candidates:
                break
            recipient_entity = random.choice(recipient_pool)
            peel_amount = random.randint(1_500_000, 3_500_000)  # ~0.015-0.035 BTC skim
            tx = spend(
                peeler, current_time, entity_wallets[recipient_entity][0],
                entity_wallets, wallet_entity, wallet_ip, entity_map, utxos,
                scenario="peel_chain", force_input_count=1,
                recipient_amount_sats=peel_amount,
            )
            transactions.append(tx)
            peel_ids.append(tx["txid"])
            current_time = next_time(current_time, 120, 600)
        ground_truth.append({
            "target_type": "entity", "target_id": peeler,
            "scenario_id": "peel_chain", "is_anomaly": "true", "severity": "medium",
            "related_txids": "|".join(peel_ids),
            "expected_signals": "repeated_small_peels;decreasing_balance;single_source_chain",
            "detail": f"Entity peels ~0.015-0.035 BTC off its balance across {len(peel_ids)} transactions.",
        })

    # Fill the remainder with normal activity. The weighted choice deliberately creates multi-input txs.
    while len(transactions) < TARGET_TOTAL_TX:
        sender = random.choice(entities)
        if not spendable_for_entity(sender, utxos, wallet_entity):
            continue
        recipient = random.choice([entity for entity in entities if entity != sender])
        tx = spend(
            sender, current_time, entity_wallets[recipient][0],
            entity_wallets, wallet_entity, wallet_ip, entity_map, utxos,
            scenario=None,
        )
        transactions.append(tx)
        current_time = next_time(current_time)

    if len(transactions) != TARGET_TOTAL_TX:
        raise AssertionError(f"Expected {TARGET_TOTAL_TX}, generated {len(transactions)}")
    if current_time > BASE_TIME + timedelta(seconds=TIME_SPAN_SECONDS):
        raise RuntimeError("Generated timeline exceeded configured 72-hour span")
    if len({tx["txid"] for tx in transactions}) != len(transactions):
        raise RuntimeError("Duplicate TXID generated")

    for tx in transactions:
        validate_transaction(tx)

    # Private negative labels allow a real false-positive evaluation.
    # Drawn from the tail of the entity list — guaranteed not to overlap
    # with any reserved protagonist entity above, regardless of scale.
    for entity in entities[-BENIGN_CONTROL_COUNT:]:
        related = [tx["txid"] for tx in transactions if tx["_origin_entity"] == entity][:10]
        ground_truth.append({
            "target_type": "entity", "target_id": entity,
            "scenario_id": "benign_control", "is_anomaly": "false", "severity": "none",
            "related_txids": "|".join(related),
            "expected_signals": "none",
            "detail": "Baseline entity reserved for false-positive evaluation.",
        })

    events = generate_network_events(transactions, ip_pool, entity_wallets, wallet_ip)
    return transactions, events, ip_pool, ground_truth, entity_map


# ==================== P2P NETWORK-EVENT GENERATION ====================
def generate_network_events(
    transactions: list[dict[str, Any]],
    ip_pool: list[dict[str, Any]],
    entity_wallets: dict[str, list[str]],
    wallet_ip: dict[str, str],
) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    all_txids = [tx["txid"] for tx in transactions]

    for tx in transactions:
        if tx["is_coinbase"]:
            continue
        transaction_time = parse_z(tx["tx_timestamp"])
        source_ip = wallet_ip[entity_wallets[tx["_origin_entity"]][0]]
        relay_ip = choose_ip(ip_pool, "relay")

        for index in range(random.randint(*EVENTS_PER_TX)):
            message_type = ["inv", "getdata", "tx"][index % 3]
            event_time = transaction_time + timedelta(
                milliseconds=random.randint(50, CORRELATION_WINDOW_SECONDS * 1000 - 100)
            )
            if message_type == "getdata":
                src_ip, dst_ip = relay_ip, source_ip
                src_port, dst_port = 8333, random.randint(49152, 65535)
                relay_role = "relay_to_wallet"
            else:
                src_ip, dst_ip = source_ip, relay_ip
                src_port, dst_port = random.randint(49152, 65535), 8333
                relay_role = "wallet_to_relay"

            events.append({
                "event_id": new_event_id(),
                "event_timestamp": iso_z(event_time),
                "src_ip": src_ip,
                "dst_ip": dst_ip,
                "src_port": src_port,
                "dst_port": dst_port,
                "message_type": message_type,
                "observed_txid": tx["txid"],
                "relay_role": relay_role,
                "_is_decoy": False,
            })

    # Add timing-decoy observations. Some are inside ±30s but refer to no/another TXID;
    # others are just outside the window to verify window-boundary enforcement.
    for _ in range(int(len(events) * DECOY_EVENT_FRACTION)):
        anchor = random.choice(transactions)
        anchor_time = parse_z(anchor["tx_timestamp"])
        if random.random() < 0.5:
            offset = random.uniform(-29, 29)
        else:
            offset = random.choice([-1, 1]) * random.uniform(31, 150)
        observed = "" if random.random() < 0.5 else random.choice(
            [txid for txid in all_txids if txid != anchor["txid"]]
        )
        events.append({
            "event_id": new_event_id(),
            "event_timestamp": iso_z(anchor_time + timedelta(seconds=offset)),
            "src_ip": choose_ip(ip_pool),
            "dst_ip": choose_ip(ip_pool),
            "src_port": random.randint(49152, 65535),
            "dst_port": 8333,
            "message_type": random.choice(["inv", "getdata", "tx", "addr"]),
            "observed_txid": observed,
            "relay_role": "background_decoy",
            "_is_decoy": True,
        })

    random.shuffle(events)
    return events


# ==================== EXPORT FUNCTIONS ====================
def public_record(record: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in record.items() if not key.startswith("_")}


def export_all(
    transactions: list[dict[str, Any]],
    events: list[dict[str, Any]],
    ip_pool: list[dict[str, Any]],
    ground_truth: list[dict[str, Any]],
    entity_map: list[dict[str, str]],
) -> None:
    if OUT_DIR.is_symlink():
        # Preserve the symlink and clear the directory it targets.
        for child in OUT_DIR.iterdir():
            if child.is_dir() and not child.is_symlink():
                shutil.rmtree(child)
            else:
                child.unlink()
    elif OUT_DIR.exists():
        shutil.rmtree(OUT_DIR)

    PRIVATE_DIR.mkdir(parents=True, exist_ok=True)

    tx_fields = [
        "txid", "tx_timestamp", "block_height", "is_coinbase",
        "input_txids", "input_vout_indexes", "input_addresses", "output_addresses",
        "input_amounts_btc", "output_amounts_btc", "fee_btc", "script_type",
    ]
    with (OUT_DIR / "blockchain_transactions.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=tx_fields)
        writer.writeheader()
        for tx in transactions:
            row = public_record(tx)
            for field in [
                "input_txids", "input_vout_indexes", "input_addresses",
                "output_addresses", "input_amounts_btc", "output_amounts_btc",
            ]:
                row[field] = json.dumps(row[field])
            writer.writerow(row)

    event_fields = [
        "event_id", "event_timestamp", "src_ip", "dst_ip", "src_port", "dst_port",
        "message_type", "observed_txid", "relay_role",
    ]
    with (OUT_DIR / "network_events.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=event_fields)
        writer.writeheader()
        for event in events:
            writer.writerow(public_record(event))

    ip_fields = [
        "ip", "country_code", "asn", "asn_org", "node_profile",
        "is_tor_like", "is_hosting_like",
    ]
    with (OUT_DIR / "ip_enrichment.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=ip_fields)
        writer.writeheader()
        writer.writerows(ip_pool)

    # Flat source demonstrates CSV ingestion of every mandatory challenge field.
    events_by_txid: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for event in events:
        if event["observed_txid"]:
            events_by_txid[event["observed_txid"]].append(event)
    flat_fields = [
        "timestamp", "src_ip", "dst_ip", "src_port", "dst_port", "txid",
        "input_addresses", "output_addresses", "input_amounts", "output_amounts",
        "fee", "script_type",
    ]
    with (OUT_DIR / "dataset.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=flat_fields)
        writer.writeheader()
        for tx in transactions:
            if tx["is_coinbase"]:
                continue
            event = events_by_txid[tx["txid"]][0]
            writer.writerow({
                "timestamp": tx["tx_timestamp"], "src_ip": event["src_ip"],
                "dst_ip": event["dst_ip"], "src_port": event["src_port"],
                "dst_port": event["dst_port"], "txid": tx["txid"],
                "input_addresses": "|".join(tx["input_addresses"]),
                "output_addresses": "|".join(tx["output_addresses"]),
                "input_amounts": "|".join(map(str, tx["input_amounts_btc"])),
                "output_amounts": "|".join(map(str, tx["output_amounts_btc"])),
                "fee": tx["fee_btc"], "script_type": tx["script_type"],
            })

    payload = {
        "metadata": {
            "generator": "PS 26146 Group A v2",
            "synthetic": True,
            "transaction_count": len(transactions),
            "network_correlation": "simulated relay observations, not ownership attribution",
        },
        "blockchain_transactions": [public_record(tx) for tx in transactions],
        "network_events": [public_record(event) for event in events],
    }
    (OUT_DIR / "dataset.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")

    root = ET.Element("bitcoin_forensics_dataset", {"synthetic": "true", "version": "2"})
    tx_root = ET.SubElement(root, "blockchain_transactions")
    for tx in transactions:
        tx_node = ET.SubElement(tx_root, "transaction")
        for key, value in public_record(tx).items():
            ET.SubElement(tx_node, key).text = json.dumps(value) if isinstance(value, list) else str(value)
    event_root = ET.SubElement(root, "network_events")
    for event in events:
        event_node = ET.SubElement(event_root, "network_event")
        for key, value in public_record(event).items():
            ET.SubElement(event_node, key).text = str(value)
    tree = ET.ElementTree(root)
    if hasattr(ET, "indent"):
        ET.indent(tree, space="  ")
    tree.write(OUT_DIR / "dataset.xml", encoding="utf-8", xml_declaration=True)

    gt_fields = [
        "target_type", "target_id", "scenario_id", "is_anomaly", "severity",
        "related_txids", "expected_signals", "detail",
    ]
    with (PRIVATE_DIR / "ground_truth.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=gt_fields)
        writer.writeheader()
        writer.writerows(ground_truth)

    with (PRIVATE_DIR / "entity_wallet_map.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["entity_id", "wallet_id", "entity_type", "is_synthetic"],
        )
        writer.writeheader()
        writer.writerows(entity_map)

    report = {
        "transactions": len(transactions),
        "coinbase_transactions": sum(tx["is_coinbase"] for tx in transactions),
        "non_coinbase_transactions": sum(not tx["is_coinbase"] for tx in transactions),
        "network_events": len(events),
        "decoy_network_events": sum(event["_is_decoy"] for event in events),
        "unique_txids": len({tx["txid"] for tx in transactions}),
        "value_conservation_failures": 0,
        "ground_truth_targets": len(ground_truth),
        "private_entity_wallet_mappings": len(entity_map),
    }
    (OUT_DIR / "dataset_quality_report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    (OUT_DIR / "README.md").write_text(
        f"# PS 26146 Group A Dataset v2\n\n"
        f"- Transactions: {len(transactions):,}\n"
        f"- Network events: {len(events):,}\n"
        f"- UTXO-consistent non-coinbase inputs: yes\n"
        f"- Private labels and entity mapping: `private/` (do not provide to Group B)\n",
        encoding="utf-8",
    )


def main() -> None:
    transactions, events, ip_pool, ground_truth, entity_map = generate_dataset()
    export_all(transactions, events, ip_pool, ground_truth, entity_map)
    print(f"Transactions written: {len(transactions):,}")
    print(f"Network events written: {len(events):,}")
    print(f"Private ground-truth targets: {len(ground_truth):,}")
    print(f"Private entity-wallet mappings: {len(entity_map):,}")
    print(f"Output directory: ./{OUT_DIR}")
    print("IMPORTANT: output_v2/private is evaluation-only; do not hand it to Group B.")


if __name__ == "__main__":
    main()
