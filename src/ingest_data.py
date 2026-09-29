#!/usr/bin/env python3

"""
SIH V6.1 Universal Data Ingestion
=================================

Purpose
-------
Accept investigation datasets in CSV, JSON, or XML and normalize them
into the canonical CSV structure expected by the SIH V6.1 pipeline.

Supported operational datasets:
    transactions
    network
    ip-enrichment

Ground-truth files are intentionally NOT required for production
inference. They belong to training / benchmark workflows.

Examples
--------
python src/ingest_data.py \
    --transactions transactions.csv \
    --network network_events.json \
    --ip-enrichment ip_enrichment.xml

python src/ingest_data.py \
    --transactions transactions.csv \
    --network network_events.csv

python src/ingest_data.py --validate-existing
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

import pandas as pd


# ============================================================
# PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = ROOT / "data" / "raw"

TRANSACTION_OUTPUT = RAW_DIR / "blockchain_transactions.csv"
NETWORK_OUTPUT = RAW_DIR / "network_events.csv"
IP_OUTPUT = RAW_DIR / "ip_enrichment.csv"


# ============================================================
# CANONICAL SCHEMAS
# ============================================================

TRANSACTION_REQUIRED = [
    "txid",
    "tx_timestamp",
    "block_height",
    "is_coinbase",
    "input_txids",
    "input_vout_indexes",
    "input_addresses",
    "output_addresses",
    "input_amounts_btc",
    "output_amounts_btc",
    "fee_btc",
    "script_type",
]

NETWORK_REQUIRED = [
    "event_timestamp",
    "src_ip",
    "dst_ip",
    "observed_txid",
]

IP_REQUIRED = [
    "ip",
]


# Columns that should contain JSON-style lists in the canonical
# transaction representation.

LIST_COLUMNS = [
    "input_txids",
    "input_vout_indexes",
    "input_addresses",
    "output_addresses",
    "input_amounts_btc",
    "output_amounts_btc",
]


# ============================================================
# UTILITIES
# ============================================================

def banner(title: str) -> None:
    print()
    print("=" * 72)
    print(title)
    print("=" * 72)


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path)


def load_json(path: Path) -> pd.DataFrame:
    """
    Supports:
        JSON array of objects
        JSON object
        common nested keys such as:
            data
            records
            transactions
            events
    """

    with path.open(
        "r",
        encoding="utf-8",
    ) as handle:
        obj = json.load(handle)

    if isinstance(obj, list):
        return pd.json_normalize(obj)

    if isinstance(obj, dict):

        for key in [
            "data",
            "records",
            "transactions",
            "events",
            "items",
        ]:
            value = obj.get(key)

            if isinstance(value, list):
                return pd.json_normalize(value)

        return pd.json_normalize([obj])

    raise ValueError(
        "Unsupported JSON structure. "
        "Expected an object or array of objects."
    )


def load_xml(path: Path) -> pd.DataFrame:
    """
    pandas.read_xml handles common record-oriented XML layouts.
    """

    try:
        return pd.read_xml(path)

    except Exception as exc:
        raise ValueError(
            f"Unable to parse XML file: {exc}"
        ) from exc


def load_dataset(path_string: str) -> pd.DataFrame:

    path = Path(path_string).expanduser().resolve()

    if not path.exists():
        raise FileNotFoundError(
            f"Input file does not exist: {path}"
        )

    suffix = path.suffix.lower()

    print(f"Reading: {path}")
    print(f"Format : {suffix or 'unknown'}")

    if suffix == ".csv":
        df = load_csv(path)

    elif suffix == ".json":
        df = load_json(path)

    elif suffix == ".xml":
        df = load_xml(path)

    else:
        raise ValueError(
            f"Unsupported format '{suffix}'. "
            "Supported formats: CSV, JSON, XML."
        )

    if df.empty:
        raise ValueError(
            f"Dataset contains no rows: {path}"
        )

    # Normalize column names.

    df.columns = [
        str(column).strip()
        for column in df.columns
    ]

    return df


# ============================================================
# SCHEMA VALIDATION
# ============================================================

def validate_columns(
    df: pd.DataFrame,
    required: list[str],
    dataset_name: str,
) -> None:

    missing = [
        column
        for column in required
        if column not in df.columns
    ]

    if missing:

        print()
        print(
            f"ERROR: {dataset_name} schema is incompatible."
        )

        print("\nMissing required columns:")

        for column in missing:
            print(f"  - {column}")

        print("\nColumns actually found:")

        for column in df.columns:
            print(f"  - {column}")

        raise ValueError(
            f"{dataset_name} failed schema validation."
        )


# ============================================================
# TRANSACTION NORMALIZATION
# ============================================================

def normalize_list_value(value):

    if isinstance(value, list):
        return json.dumps(value)

    if pd.isna(value):
        return "[]"

    value = str(value).strip()

    if not value:
        return "[]"

    # Already JSON?

    try:
        parsed = json.loads(value)

        if isinstance(parsed, list):
            return json.dumps(parsed)

    except Exception:
        pass

    # Conservative fallback:
    # preserve one scalar as a one-element list.

    return json.dumps([value])


def normalize_transactions(
    df: pd.DataFrame,
) -> pd.DataFrame:

    validate_columns(
        df,
        TRANSACTION_REQUIRED,
        "Blockchain transaction dataset",
    )

    result = df.copy()

    for column in LIST_COLUMNS:
        result[column] = (
            result[column]
            .apply(normalize_list_value)
        )

    result["txid"] = (
        result["txid"]
        .astype(str)
        .str.strip()
    )

    result["tx_timestamp"] = pd.to_datetime(
        result["tx_timestamp"],
        errors="coerce",
        utc=True,
    )

    bad_timestamp = (
        result["tx_timestamp"].isna().sum()
    )

    if bad_timestamp:
        raise ValueError(
            f"{bad_timestamp:,} transactions contain "
            "invalid timestamps."
        )

    result["tx_timestamp"] = (
        result["tx_timestamp"]
        .dt.strftime("%Y-%m-%dT%H:%M:%SZ")
    )

    result["block_height"] = pd.to_numeric(
        result["block_height"],
        errors="coerce",
    )

    result["fee_btc"] = pd.to_numeric(
        result["fee_btc"],
        errors="coerce",
    )

    # Normalize coinbase values.

    result["is_coinbase"] = (
        result["is_coinbase"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({
            "true": True,
            "false": False,
            "1": True,
            "0": False,
            "yes": True,
            "no": False,
        })
    )

    if result["is_coinbase"].isna().any():
        raise ValueError(
            "Invalid values detected in is_coinbase."
        )

    if result["txid"].duplicated().any():

        count = int(
            result["txid"].duplicated().sum()
        )

        raise ValueError(
            f"Duplicate transaction IDs detected: {count:,}"
        )

    return result


# ============================================================
# NETWORK NORMALIZATION
# ============================================================

def normalize_network(
    df: pd.DataFrame,
) -> pd.DataFrame:

    validate_columns(
        df,
        NETWORK_REQUIRED,
        "Network event dataset",
    )

    result = df.copy()

    result["event_timestamp"] = pd.to_datetime(
        result["event_timestamp"],
        errors="coerce",
        utc=True,
    )

    bad_timestamp = (
        result["event_timestamp"].isna().sum()
    )

    if bad_timestamp:
        raise ValueError(
            f"{bad_timestamp:,} network events contain "
            "invalid timestamps."
        )

    result["event_timestamp"] = (
        result["event_timestamp"]
        .dt.strftime(
            "%Y-%m-%dT%H:%M:%S.%fZ"
        )
    )

    for column in [
        "src_ip",
        "dst_ip",
        "observed_txid",
    ]:
        result[column] = (
            result[column]
            .astype(str)
            .str.strip()
        )

    return result


# ============================================================
# IP ENRICHMENT NORMALIZATION
# ============================================================

def normalize_ip(
    df: pd.DataFrame,
) -> pd.DataFrame:

    validate_columns(
        df,
        IP_REQUIRED,
        "IP enrichment dataset",
    )

    result = df.copy()

    result["ip"] = (
        result["ip"]
        .astype(str)
        .str.strip()
    )

    return result


# ============================================================
# OUTPUT
# ============================================================

def save_dataset(
    df: pd.DataFrame,
    output: Path,
    dataset_name: str,
) -> None:

    output.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        output,
        index=False,
    )

    print(
        f"✓ {dataset_name:<24} "
        f"{len(df):>10,} rows"
    )

    print(
        f"  Saved: {output}"
    )


# ============================================================
# EXISTING DATA VALIDATION
# ============================================================

def validate_existing() -> None:

    banner(
        "VALIDATING EXISTING OPERATIONAL DATA"
    )

    checks = [
        (
            TRANSACTION_OUTPUT,
            TRANSACTION_REQUIRED,
            "Blockchain transactions",
        ),
        (
            NETWORK_OUTPUT,
            NETWORK_REQUIRED,
            "Network events",
        ),
        (
            IP_OUTPUT,
            IP_REQUIRED,
            "IP enrichment",
        ),
    ]

    errors = 0

    for path, required, name in checks:

        if not path.exists():

            print(
                f"✗ {name}: missing"
            )

            errors += 1
            continue

        try:

            df = pd.read_csv(
                path,
                nrows=10,
            )

            validate_columns(
                df,
                required,
                name,
            )

            print(
                f"✓ {name}: schema valid"
            )

        except Exception as exc:

            print(
                f"✗ {name}: {exc}"
            )

            errors += 1

    print()

    if errors:
        raise SystemExit(
            f"Validation failed with "
            f"{errors} error(s)."
        )

    print(
        "Operational input schemas are valid."
    )


# ============================================================
# COMMAND LINE
# ============================================================

def parse_arguments():

    parser = argparse.ArgumentParser(
        description=(
            "Normalize CSV, JSON, or XML investigation "
            "datasets for the SIH V6.1 Bitcoin "
            "forensics pipeline."
        )
    )

    parser.add_argument(
        "--transactions",
        help=(
            "Blockchain transaction dataset "
            "(CSV, JSON, or XML)."
        ),
    )

    parser.add_argument(
        "--network",
        help=(
            "Network-event dataset "
            "(CSV, JSON, or XML)."
        ),
    )

    parser.add_argument(
        "--ip-enrichment",
        dest="ip_enrichment",
        help=(
            "IP-enrichment dataset "
            "(CSV, JSON, or XML)."
        ),
    )

    parser.add_argument(
        "--validate-existing",
        action="store_true",
        help=(
            "Validate the canonical datasets already "
            "stored under data/raw/."
        ),
    )

    return parser.parse_args()


# ============================================================
# MAIN
# ============================================================

def main():

    args = parse_arguments()

    print("=" * 72)
    print("SIH V6.1 UNIVERSAL DATA INGESTION")
    print("=" * 72)

    if args.validate_existing:

        validate_existing()

        return

    if not any([
        args.transactions,
        args.network,
        args.ip_enrichment,
    ]):

        print(
            "\nNo input datasets supplied."
        )

        print(
            "\nUse --help for usage information."
        )

        raise SystemExit(1)

    RAW_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # TRANSACTIONS
    # --------------------------------------------------------

    if args.transactions:

        banner(
            "INGESTING BLOCKCHAIN TRANSACTIONS"
        )

        df = load_dataset(
            args.transactions
        )

        print(
            f"Input rows : {len(df):,}"
        )

        df = normalize_transactions(
            df
        )

        save_dataset(
            df,
            TRANSACTION_OUTPUT,
            "Transactions",
        )

    # --------------------------------------------------------
    # NETWORK
    # --------------------------------------------------------

    if args.network:

        banner(
            "INGESTING NETWORK EVENTS"
        )

        df = load_dataset(
            args.network
        )

        print(
            f"Input rows : {len(df):,}"
        )

        df = normalize_network(
            df
        )

        save_dataset(
            df,
            NETWORK_OUTPUT,
            "Network events",
        )

    # --------------------------------------------------------
    # IP ENRICHMENT
    # --------------------------------------------------------

    if args.ip_enrichment:

        banner(
            "INGESTING IP ENRICHMENT"
        )

        df = load_dataset(
            args.ip_enrichment
        )

        print(
            f"Input rows : {len(df):,}"
        )

        df = normalize_ip(
            df
        )

        save_dataset(
            df,
            IP_OUTPUT,
            "IP enrichment",
        )

    banner(
        "INGESTION COMPLETE"
    )

    print(
        "Canonical operational datasets are ready."
    )

    print(
        "\nNext step:"
    )

    print(
        "    ./run_pipeline.sh"
    )


if __name__ == "__main__":

    try:
        main()

    except KeyboardInterrupt:

        print(
            "\nInterrupted."
        )

        sys.exit(130)

    except Exception as exc:

        print()
        print("=" * 72)
        print("INGESTION FAILED")
        print("=" * 72)
        print(exc)

        sys.exit(1)

