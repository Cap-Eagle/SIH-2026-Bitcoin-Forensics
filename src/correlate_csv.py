import argparse
import csv
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from pathlib import Path


# ============================================================
# V3 PROJECT PATHS
# ============================================================

# File:
#   btc-forensics_v3/src/correlate_csv.py
#
# parents[0] -> src/
# parents[1] -> btc-forensics_v3/
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CORRELATED_DIR = DATA_DIR / "correlated"


# ============================================================
# TIME HELPERS
# ============================================================

def parse_time(value: str) -> datetime:
    return (
        datetime.fromisoformat(
            value.replace("Z", "+00:00")
        )
        .astimezone(timezone.utc)
    )


def iso_z(value: datetime) -> str:
    return (
        value
        .astimezone(timezone.utc)
        .isoformat()
        .replace("+00:00", "Z")
    )


# ============================================================
# COMMAND-LINE ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description=(
        "Offline correlation of synthetic P2P events "
        "and Bitcoin-like transactions."
    )
)

parser.add_argument(
    "--input-dir",
    default=str(RAW_DIR),
    help="Directory containing blockchain_transactions.csv, "
         "network_events.csv and ip_enrichment.csv."
)

parser.add_argument(
    "--output-dir",
    default=str(CORRELATED_DIR),
    help="Directory where correlations.csv will be written."
)

parser.add_argument(
    "--window-seconds",
    type=int,
    default=30,
    help="Maximum transaction/event time difference."
)

args = parser.parse_args()


# ============================================================
# RESOLVE DIRECTORIES
# ============================================================

input_dir = Path(args.input_dir)
output_dir = Path(args.output_dir)

output_dir.mkdir(
    parents=True,
    exist_ok=True
)

window = timedelta(
    seconds=args.window_seconds
)


# ============================================================
# INPUT FILES
# ============================================================

tx_path = input_dir / "blockchain_transactions.csv"
event_path = input_dir / "network_events.csv"
ip_path = input_dir / "ip_enrichment.csv"


required_files = [
    tx_path,
    event_path,
    ip_path,
]


for path in required_files:

    if not path.exists():

        raise FileNotFoundError(
            f"\nMissing required input:\n"
            f"  {path}\n\n"
            f"Expected v3 input directory:\n"
            f"  {input_dir}\n"
        )


# ============================================================
# LOAD LOCAL SYNTHETIC IP / ASN ENRICHMENT
# ============================================================

ip_lookup = {}

with ip_path.open(
    newline="",
    encoding="utf-8"
) as handle:

    for row in csv.DictReader(handle):

        ip_lookup[row["ip"]] = row


# ============================================================
# LOAD TRANSACTIONS
# ============================================================

transactions = []

transactions_by_second = defaultdict(list)


with tx_path.open(
    newline="",
    encoding="utf-8"
) as handle:

    for row in csv.DictReader(handle):

        row["tx_dt"] = parse_time(
            row["tx_timestamp"]
        )

        transactions.append(row)

        transactions_by_second[
            int(row["tx_dt"].timestamp())
        ].append(row)


# ============================================================
# OUTPUT SCHEMA
# ============================================================

fieldnames = [

    # Correlation identity
    "correlation_id",
    "event_id",

    # Timing
    "event_timestamp",

    # Network
    "src_ip",
    "dst_ip",
    "src_port",
    "dst_port",

    "message_type",
    "relay_role",

    # Transaction identity
    "observed_txid",
    "candidate_txid",

    # Transaction timing
    "tx_timestamp",

    # Transaction structure
    "input_addresses",
    "output_addresses",

    "input_amounts_btc",
    "output_amounts_btc",

    "fee_btc",
    "script_type",

    # Source enrichment
    "src_country",
    "src_asn",
    "src_asn_org",
    "src_node_profile",

    "src_is_tor_like",
    "src_is_hosting_like",

    # Destination enrichment
    "dst_country",
    "dst_asn",
    "dst_asn_org",
    "dst_node_profile",

    # Correlation metrics
    "time_delta_ms",
    "match_reason",
    "match_score",
]


# ============================================================
# OUTPUT
# ============================================================

output_path = (
    output_dir / "correlations.csv"
)


# ============================================================
# CORRELATION COUNTERS
# ============================================================

candidate_pairs = 0

exact_matches = 0

close_candidates = 0

broad_candidates = 0


# ============================================================
# CORRELATE NETWORK EVENTS WITH TRANSACTIONS
# ============================================================

with event_path.open(
    newline="",
    encoding="utf-8"
) as event_handle, \
     output_path.open(
         "w",
         newline="",
         encoding="utf-8"
     ) as output_handle:

    writer = csv.DictWriter(
        output_handle,
        fieldnames=fieldnames
    )

    writer.writeheader()


    # --------------------------------------------------------
    # Process each network event
    # --------------------------------------------------------

    for event in csv.DictReader(event_handle):

        event_dt = parse_time(
            event["event_timestamp"]
        )

        event_second = int(
            event_dt.timestamp()
        )


        # ----------------------------------------------------
        # IP enrichment
        # ----------------------------------------------------

        src = ip_lookup.get(
            event["src_ip"],
            {}
        )

        dst = ip_lookup.get(
            event["dst_ip"],
            {}
        )


        # ----------------------------------------------------
        # Search transactions inside time window
        # ----------------------------------------------------

        for second in range(
            event_second - args.window_seconds,
            event_second + args.window_seconds + 1,
        ):

            for tx in transactions_by_second.get(
                second,
                []
            ):

                # ------------------------------------------------
                # Exact time difference
                # ------------------------------------------------

                delta_ms = round(
                    (
                        event_dt - tx["tx_dt"]
                    ).total_seconds()
                    * 1000
                )


                # ------------------------------------------------
                # Defensive boundary check
                # ------------------------------------------------

                if (
                    abs(delta_ms)
                    > args.window_seconds * 1000
                ):

                    continue


                # ------------------------------------------------
                # Determine match quality
                # ------------------------------------------------

                observed_txid = (
                    event["observed_txid"]
                )


                if (
                    observed_txid
                    and observed_txid == tx["txid"]
                ):

                    match_reason = (
                        "exact_txid_and_time"
                    )

                    match_score = 1.00

                    exact_matches += 1


                elif abs(delta_ms) <= 5_000:

                    match_reason = (
                        "close_time_candidate"
                    )

                    match_score = 0.65

                    close_candidates += 1


                else:

                    match_reason = (
                        "time_window_candidate"
                    )

                    match_score = 0.35

                    broad_candidates += 1


                candidate_pairs += 1


                # ------------------------------------------------
                # Write correlation
                # ------------------------------------------------

                writer.writerow({

                    "correlation_id":
                        f"corr_{candidate_pairs:09d}",

                    "event_id":
                        event["event_id"],

                    "event_timestamp":
                        event["event_timestamp"],

                    "src_ip":
                        event["src_ip"],

                    "dst_ip":
                        event["dst_ip"],

                    "src_port":
                        event["src_port"],

                    "dst_port":
                        event["dst_port"],

                    "message_type":
                        event["message_type"],

                    "relay_role":
                        event["relay_role"],

                    "observed_txid":
                        observed_txid,

                    "candidate_txid":
                        tx["txid"],

                    "tx_timestamp":
                        tx["tx_timestamp"],

                    "input_addresses":
                        tx["input_addresses"],

                    "output_addresses":
                        tx["output_addresses"],

                    "input_amounts_btc":
                        tx["input_amounts_btc"],

                    "output_amounts_btc":
                        tx["output_amounts_btc"],

                    "fee_btc":
                        tx["fee_btc"],

                    "script_type":
                        tx["script_type"],

                    "src_country":
                        src.get(
                            "country_code",
                            "UNKNOWN"
                        ),

                    "src_asn":
                        src.get(
                            "asn",
                            ""
                        ),

                    "src_asn_org":
                        src.get(
                            "asn_org",
                            "UNKNOWN"
                        ),

                    "src_node_profile":
                        src.get(
                            "node_profile",
                            "UNKNOWN"
                        ),

                    "src_is_tor_like":
                        src.get(
                            "is_tor_like",
                            "false"
                        ),

                    "src_is_hosting_like":
                        src.get(
                            "is_hosting_like",
                            "false"
                        ),

                    "dst_country":
                        dst.get(
                            "country_code",
                            "UNKNOWN"
                        ),

                    "dst_asn":
                        dst.get(
                            "asn",
                            ""
                        ),

                    "dst_asn_org":
                        dst.get(
                            "asn_org",
                            "UNKNOWN"
                        ),

                    "dst_node_profile":
                        dst.get(
                            "node_profile",
                            "UNKNOWN"
                        ),

                    "time_delta_ms":
                        delta_ms,

                    "match_reason":
                        match_reason,

                    "match_score":
                        match_score,
                })


# ============================================================
# SUMMARY
# ============================================================

print()

print("=" * 60)
print("GROUP B CORRELATION — V3")
print("=" * 60)

print()

print(
    f"Window                     : "
    f"±{args.window_seconds} seconds"
)

print(
    f"Transactions loaded        : "
    f"{len(transactions):,}"
)

print(
    f"Candidate pairs written    : "
    f"{candidate_pairs:,}"
)

print(
    f"Exact TXID/time matches    : "
    f"{exact_matches:,}"
)

print(
    f"Close-time candidates      : "
    f"{close_candidates:,}"
)

print(
    f"Broad time-window matches  : "
    f"{broad_candidates:,}"
)

print()

print(
    f"Input directory            : "
    f"{input_dir}"
)

print(
    f"Output                     : "
    f"{output_path}"
)

print()

print("Correlation generation complete.")

print("=" * 60)
