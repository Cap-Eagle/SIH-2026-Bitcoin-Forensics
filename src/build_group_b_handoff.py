import csv
import json
from pathlib import Path


# ============================================================
# V3 PROJECT PATHS
# ============================================================

# File location:
#   btc-forensics_v3/src/build_group_b_handoff.py
#
# parents[0] -> src/
# parents[1] -> btc-forensics_v3/
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CORRELATED_DIR = DATA_DIR / "correlated"
OUT_DIR = ROOT / "outputs"


# Input files
CORRELATIONS = CORRELATED_DIR / "correlations.csv"
TRANSACTIONS = RAW_DIR / "blockchain_transactions.csv"

# Output file
OUTPUT = CORRELATED_DIR / "group_b_features.csv"


# ============================================================
# DIRECTORY SETUP
# ============================================================

CORRELATED_DIR.mkdir(parents=True, exist_ok=True)
OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# INPUT VALIDATION
# ============================================================

if not CORRELATIONS.exists():
    raise FileNotFoundError(
        f"Missing correlation file:\n{CORRELATIONS}\n\n"
        "Run the correlation stage before running this script."
    )

if not TRANSACTIONS.exists():
    raise FileNotFoundError(
        f"Missing blockchain transaction file:\n{TRANSACTIONS}\n\n"
        "Expected blockchain_transactions.csv inside data/raw/."
    )


# ============================================================
# LOAD CANONICAL TRANSACTIONS
# ============================================================
#
# Start from the canonical blockchain transaction table so that
# every transaction is represented, including transactions that
# have no corresponding P2P relay evidence.
#
# This is important because absence of correlation evidence should
# not cause a transaction to disappear from the Group B dataset.
# ============================================================

features = {}

with TRANSACTIONS.open(newline="", encoding="utf-8") as handle:

    reader = csv.DictReader(handle)

    for row in reader:

        input_addresses = json.loads(row["input_addresses"])
        output_addresses = json.loads(row["output_addresses"])

        input_amounts = json.loads(row["input_amounts_btc"])
        output_amounts = json.loads(row["output_amounts_btc"])

        total_input = sum(float(value) for value in input_amounts)
        total_output = sum(float(value) for value in output_amounts)

        fee = float(row["fee_btc"])

        features[row["txid"]] = {

            # ------------------------------------------------
            # Identity / blockchain information
            # ------------------------------------------------

            "txid": row["txid"],
            "tx_timestamp": row["tx_timestamp"],
            "block_height": row["block_height"],
            "is_coinbase": row["is_coinbase"],

            # ------------------------------------------------
            # Transaction structure
            # ------------------------------------------------

            "input_txids": row["input_txids"],
            "input_vout_indexes": row["input_vout_indexes"],

            "input_addresses": row["input_addresses"],
            "output_addresses": row["output_addresses"],

            "input_amounts_btc": row["input_amounts_btc"],
            "output_amounts_btc": row["output_amounts_btc"],

            "fee_btc": row["fee_btc"],
            "script_type": row["script_type"],

            "input_count": len(input_addresses),
            "output_count": len(output_addresses),

            "total_input_btc": round(total_input, 8),
            "total_output_btc": round(total_output, 8),

            "fee_rate": (
                round(fee / total_input, 10)
                if total_input > 0
                else 0.0
            ),

            "max_output_btc": round(max(output_amounts), 8),
            "min_output_btc": round(min(output_amounts), 8),

            # ------------------------------------------------
            # Correlation evidence
            # ------------------------------------------------

            "candidate_count": 0,
            "exact_match_count": 0,
            "close_candidate_count": 0,
            "broad_candidate_count": 0,

            "max_match_score": 0.0,

            "min_abs_time_delta_ms": "",

            # ------------------------------------------------
            # Network diversity
            # ------------------------------------------------

            "src_ips": set(),
            "dst_ips": set(),

            "src_countries": set(),
            "src_asns": set(),

            "high_confidence_src_ips": set(),
            "high_confidence_src_asns": set(),

            "message_types": set(),

            # ------------------------------------------------
            # Network risk indicators
            # ------------------------------------------------

            "tor_like_event_count": 0,
            "hosting_like_event_count": 0,
        }


# ============================================================
# LOAD CORRELATION EVIDENCE
# ============================================================

with CORRELATIONS.open(newline="", encoding="utf-8") as handle:

    reader = csv.DictReader(handle)

    for row in reader:

        txid = row["candidate_txid"]

        # Ignore correlations referring to unknown transactions.
        if txid not in features:
            continue

        item = features[txid]

        # ----------------------------------------------------
        # Candidate statistics
        # ----------------------------------------------------

        item["candidate_count"] += 1

        score = float(row["match_score"])

        item["max_match_score"] = max(
            item["max_match_score"],
            score
        )

        # ----------------------------------------------------
        # Time delta
        # ----------------------------------------------------

        delta = abs(int(row["time_delta_ms"]))

        current_min = item["min_abs_time_delta_ms"]

        if current_min == "" or delta < current_min:
            item["min_abs_time_delta_ms"] = delta

        # ----------------------------------------------------
        # Match reason
        # ----------------------------------------------------

        if row["match_reason"] == "exact_txid_and_time":

            item["exact_match_count"] += 1

        elif row["match_reason"] == "close_time_candidate":

            item["close_candidate_count"] += 1

        elif row["match_reason"] == "time_window_candidate":

            item["broad_candidate_count"] += 1

        # ----------------------------------------------------
        # Network identifiers
        # ----------------------------------------------------

        if row["src_ip"]:
            item["src_ips"].add(row["src_ip"])

        if row["dst_ip"]:
            item["dst_ips"].add(row["dst_ip"])

        if row["src_country"]:
            item["src_countries"].add(row["src_country"])

        if row["src_asn"]:
            item["src_asns"].add(row["src_asn"])

        if row["message_type"]:
            item["message_types"].add(row["message_type"])

        # ----------------------------------------------------
        # High-confidence diversity
        #
        # Only exact / close evidence should contribute here.
        # ----------------------------------------------------

        if score >= 0.65:

            if row["src_ip"]:
                item["high_confidence_src_ips"].add(row["src_ip"])

            if row["src_asn"]:
                item["high_confidence_src_asns"].add(row["src_asn"])

        # ----------------------------------------------------
        # TOR / hosting indicators
        # ----------------------------------------------------

        if row["src_is_tor_like"].strip().lower() == "true":
            item["tor_like_event_count"] += 1

        if row["src_is_hosting_like"].strip().lower() == "true":
            item["hosting_like_event_count"] += 1


# ============================================================
# OUTPUT SCHEMA
# ============================================================

fields = [

    # Transaction identity
    "txid",
    "tx_timestamp",
    "block_height",
    "is_coinbase",

    # Transaction structure
    "input_txids",
    "input_vout_indexes",

    "input_addresses",
    "output_addresses",

    "input_amounts_btc",
    "output_amounts_btc",

    "fee_btc",
    "script_type",

    "input_count",
    "output_count",

    "total_input_btc",
    "total_output_btc",

    "fee_rate",

    "max_output_btc",
    "min_output_btc",

    # Correlation evidence
    "candidate_count",
    "exact_match_count",
    "close_candidate_count",
    "broad_candidate_count",

    "weighted_evidence_score",
    "max_match_score",

    "min_abs_time_delta_ms",

    # Network diversity
    "unique_src_ip_count",
    "unique_dst_ip_count",

    "unique_src_country_count",
    "unique_src_asn_count",

    "high_confidence_src_ip_count",
    "high_confidence_src_asn_count",

    "unique_message_type_count",

    # Risk indicators
    "tor_like_event_count",
    "hosting_like_event_count",
]


# ============================================================
# WRITE GROUP B FEATURE TABLE
# ============================================================

with OUTPUT.open(
    "w",
    newline="",
    encoding="utf-8"
) as handle:

    writer = csv.DictWriter(
        handle,
        fieldnames=fields
    )

    writer.writeheader()

    for txid in sorted(features):

        item = features[txid]

        # ----------------------------------------------------
        # Weighted correlation evidence
        #
        # Exact match       = 1.00
        # Close candidate   = 0.65
        # Broad candidate   = 0.35
        # ----------------------------------------------------

        weighted_evidence = (
            1.00 * item["exact_match_count"]
            + 0.65 * item["close_candidate_count"]
            + 0.35 * item["broad_candidate_count"]
        )

        writer.writerow({

            # Transaction identity
            "txid": item["txid"],
            "tx_timestamp": item["tx_timestamp"],
            "block_height": item["block_height"],
            "is_coinbase": item["is_coinbase"],

            # Transaction structure
            "input_txids": item["input_txids"],
            "input_vout_indexes": item["input_vout_indexes"],

            "input_addresses": item["input_addresses"],
            "output_addresses": item["output_addresses"],

            "input_amounts_btc": item["input_amounts_btc"],
            "output_amounts_btc": item["output_amounts_btc"],

            "fee_btc": item["fee_btc"],
            "script_type": item["script_type"],

            "input_count": item["input_count"],
            "output_count": item["output_count"],

            "total_input_btc": item["total_input_btc"],
            "total_output_btc": item["total_output_btc"],

            "fee_rate": item["fee_rate"],

            "max_output_btc": item["max_output_btc"],
            "min_output_btc": item["min_output_btc"],

            # Correlation evidence
            "candidate_count": item["candidate_count"],
            "exact_match_count": item["exact_match_count"],
            "close_candidate_count": item["close_candidate_count"],
            "broad_candidate_count": item["broad_candidate_count"],

            "weighted_evidence_score": round(
                weighted_evidence,
                4
            ),

            "max_match_score": item["max_match_score"],

            "min_abs_time_delta_ms": item[
                "min_abs_time_delta_ms"
            ],

            # Network diversity
            "unique_src_ip_count": len(
                item["src_ips"]
            ),

            "unique_dst_ip_count": len(
                item["dst_ips"]
            ),

            "unique_src_country_count": len(
                item["src_countries"]
            ),

            "unique_src_asn_count": len(
                item["src_asns"]
            ),

            "high_confidence_src_ip_count": len(
                item["high_confidence_src_ips"]
            ),

            "high_confidence_src_asn_count": len(
                item["high_confidence_src_asns"]
            ),

            "unique_message_type_count": len(
                item["message_types"]
            ),

            # Risk indicators
            "tor_like_event_count": item[
                "tor_like_event_count"
            ],

            "hosting_like_event_count": item[
                "hosting_like_event_count"
            ],
        })


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("GROUP B HANDOFF — V3")
print("=" * 60)
print()
print(f"Canonical transactions loaded : {len(features):,}")
print(f"Feature rows written           : {len(features):,}")
print()
print(f"Input transactions             : {TRANSACTIONS}")
print(f"Input correlations             : {CORRELATIONS}")
print(f"Output                         : {OUTPUT}")
print()
print("Handoff generation complete.")
print("=" * 60)
