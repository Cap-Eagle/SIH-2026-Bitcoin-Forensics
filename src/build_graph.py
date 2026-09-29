import ast
import json
from itertools import combinations
import pickle
from pathlib import Path

import networkx as nx
import pandas as pd


# ============================================================
# V3 PROJECT PATHS
# ============================================================

# File:
#   btc-forensics_v3/src/build_graph.py
#
# parents[0] -> src/
# parents[1] -> btc-forensics_v3/
ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"

OUT = ROOT / "outputs"

OUT.mkdir(parents=True, exist_ok=True)

TX_FILE = RAW_DIR / "blockchain_transactions.csv"


# ============================================================
# PARSING HELPERS
# ============================================================

def parse_list(value):
    """
    Parse JSON or Python-style list values stored in CSV columns.
    """

    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    text = str(value).strip()

    if text in {"", "[]", "nan", "None"}:
        return []

    # First try JSON.
    try:
        parsed = json.loads(text)

        if isinstance(parsed, list):
            return parsed

        return []

    except (json.JSONDecodeError, TypeError):
        pass

    # Fall back to Python literal syntax.
    try:
        parsed = ast.literal_eval(text)

        if isinstance(parsed, list):
            return parsed

        return []

    except (ValueError, SyntaxError):
        return []


def parse_bool(value) -> bool:
    """
    Safely parse boolean-like values from CSV.
    """

    if isinstance(value, bool):
        return value

    return str(value).strip().lower() in {
        "true",
        "1",
        "yes",
        "y",
    }


def safe_float(value, default: float = 0.0) -> float:
    """
    Convert a value to float without crashing graph construction.
    """

    try:
        return float(value)

    except (TypeError, ValueError):

        return default


# ============================================================
# INPUT VALIDATION
# ============================================================

if not TX_FILE.exists():

    raise FileNotFoundError(
        f"\nCould not find canonical transaction file:\n"
        f"  {TX_FILE}\n\n"
        f"Expected V3 input location:\n"
        f"  {RAW_DIR}\n"
    )


# ============================================================
# LOAD TRANSACTIONS
# ============================================================

print()
print("=" * 60)
print("GROUP B GRAPH BUILD — V3")
print("=" * 60)
print()

print(
    f"Reading transactions from:\n"
    f"  {TX_FILE}"
)

print(
    f"Writing graph outputs to:\n"
    f"  {OUT}"
)

print()


df = pd.read_csv(TX_FILE)


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "txid",
    "input_addresses",
    "input_amounts_btc",
    "output_addresses",
    "output_amounts_btc",
]


missing_columns = [
    column
    for column in required_columns
    if column not in df.columns
]


if missing_columns:

    raise ValueError(
        "blockchain_transactions.csv is missing "
        "required columns: "
        + ", ".join(missing_columns)
    )


# ============================================================
# OPTIONAL COLUMNS
# ============================================================

timestamp_column = (
    "tx_timestamp"
    if "tx_timestamp" in df.columns
    else (
        "timestamp"
        if "timestamp" in df.columns
        else None
    )
)


block_height_column = (
    "block_height"
    if "block_height" in df.columns
    else None
)


# ============================================================
# GRAPH 1 — TRANSACTION GRAPH
# ============================================================
#
# Structure:
#
#       address
#          │
#          ▼
#     transaction
#          │
#          ▼
#       address
#
# Input:
#       address -> transaction
#
# Output:
#       transaction -> address
#
# This preserves transaction-level provenance.
# ============================================================

print("Building transaction graph...")


G = nx.MultiDiGraph()


for _, row in df.iterrows():

    txid = str(row["txid"])

    tx_node = f"tx:{txid}"


    # --------------------------------------------------------
    # Transaction metadata
    # --------------------------------------------------------

    timestamp = (
        row[timestamp_column]
        if timestamp_column
        else ""
    )


    if (
        block_height_column
        and pd.notna(row[block_height_column])
    ):

        try:

            block_height = int(
                row[block_height_column]
            )

        except (TypeError, ValueError):

            block_height = -1

    else:

        block_height = -1


    is_coinbase = parse_bool(
        row.get(
            "is_coinbase",
            False
        )
    )


    # --------------------------------------------------------
    # Add transaction node
    # --------------------------------------------------------

    G.add_node(

        tx_node,

        node_type="transaction",

        timestamp=timestamp,

        block_height=block_height,

        is_coinbase=is_coinbase,

    )


    # --------------------------------------------------------
    # Parse transaction addresses/amounts
    # --------------------------------------------------------

    input_addresses = parse_list(
        row["input_addresses"]
    )

    input_amounts = parse_list(
        row["input_amounts_btc"]
    )

    output_addresses = parse_list(
        row["output_addresses"]
    )

    output_amounts = parse_list(
        row["output_amounts_btc"]
    )


    # --------------------------------------------------------
    # Input edges
    # --------------------------------------------------------

    for address, amount in zip(
        input_addresses,
        input_amounts
    ):

        address = str(address)

        G.add_node(
            address,
            node_type="address"
        )

        G.add_edge(

            address,

            tx_node,

            amount_btc=safe_float(amount),

            direction="input",

        )


    # --------------------------------------------------------
    # Output edges
    # --------------------------------------------------------

    for address, amount in zip(
        output_addresses,
        output_amounts
    ):

        address = str(address)

        G.add_node(
            address,
            node_type="address"
        )

        G.add_edge(

            tx_node,

            address,

            amount_btc=safe_float(amount),

            direction="output",

        )


# ============================================================
# SAVE TRANSACTION GRAPH
# ============================================================

transaction_graph_path = (
    OUT / "transaction_graph.gpickle"
)


with transaction_graph_path.open(
    "wb"
) as file:

    pickle.dump(
        G,
        file,
        protocol=pickle.HIGHEST_PROTOCOL
    )


# ============================================================
# GRAPH 2 — ADDRESS GRAPH
# ============================================================
#
# Structure:
#
#       input address
#             │
#             ▼
#       output address
#
# An edge means the two addresses participated
# in the same transaction.
#
# It does NOT claim an exact value transfer
# between that individual pair.
# ============================================================

print("Building address graph...")


A = nx.DiGraph()


for _, row in df.iterrows():

    txid = str(row["txid"])


    input_addresses = parse_list(
        row["input_addresses"]
    )

    output_addresses = parse_list(
        row["output_addresses"]
    )


    for source in input_addresses:

        source = str(source)


        for destination in output_addresses:

            destination = str(destination)


            # Avoid self-loops.
            if source == destination:
                continue


            A.add_node(
                source,
                node_type="address"
            )

            A.add_node(
                destination,
                node_type="address"
            )


            # ------------------------------------------------
            # Existing edge
            # ------------------------------------------------

            if A.has_edge(
                source,
                destination
            ):

                A[source][destination][
                    "tx_count"
                ] += 1

                A[source][destination][
                    "txids"
                ].append(txid)


            # ------------------------------------------------
            # New edge
            # ------------------------------------------------

            else:

                A.add_edge(

                    source,

                    destination,

                    tx_count=1,

                    txids=[txid],

                )


# ============================================================
# SAVE ADDRESS GRAPH
# ============================================================

address_graph_path = (
    OUT / "address_graph.gpickle"
)


with address_graph_path.open(
    "wb"
) as file:

    pickle.dump(
        A,
        file,
        protocol=pickle.HIGHEST_PROTOCOL
    )


# ============================================================
# GRAPH 3 — OWNERSHIP GRAPH
# ============================================================
#
# Common-input ownership heuristic:
#
# If two addresses appear as inputs
# in the same non-coinbase transaction,
# they receive an undirected edge.
#
# IMPORTANT:
# This is heuristic evidence.
# It is NOT proof of common ownership.
# ============================================================

print("Building ownership graph...")


O = nx.Graph()


for _, row in df.iterrows():

    # --------------------------------------------------------
    # Coinbase transactions do not provide useful
    # common-input ownership evidence.
    # --------------------------------------------------------

    if parse_bool(
        row.get(
            "is_coinbase",
            False
        )
    ):

        continue


    # --------------------------------------------------------
    # Normalize and deduplicate input addresses.
    # --------------------------------------------------------

    input_addresses = sorted(

        {
            str(address)

            for address in parse_list(
                row["input_addresses"]
            )

            if str(address).strip()

        }

    )


    # --------------------------------------------------------
    # Preserve isolated ownership nodes.
    # --------------------------------------------------------

    for address in input_addresses:

        O.add_node(

            address,

            node_type="address"

        )


    # --------------------------------------------------------
    # Pairwise co-spend edges.
    # --------------------------------------------------------

    for left, right in combinations(
        input_addresses,
        2
    ):

        txid = str(
            row["txid"]
        )


        # ----------------------------------------------------
        # Existing edge
        # ----------------------------------------------------

        if O.has_edge(
            left,
            right
        ):

            O[left][right][
                "tx_count"
            ] += 1

            O[left][right][
                "txids"
            ].append(txid)


        # ----------------------------------------------------
        # New edge
        # ----------------------------------------------------

        else:

            O.add_edge(

                left,

                right,

                tx_count=1,

                txids=[txid],

            )


# ============================================================
# SAVE OWNERSHIP GRAPH
# ============================================================

ownership_graph_path = (
    OUT / "ownership_graph.gpickle"
)


with ownership_graph_path.open(
    "wb"
) as file:

    pickle.dump(
        O,
        file,
        protocol=pickle.HIGHEST_PROTOCOL
    )


# ============================================================
# SUMMARY
# ============================================================

print()

print("=" * 60)
print("GRAPH BUILD COMPLETE")
print("=" * 60)

print()

print(
    f"Transactions processed     : "
    f"{len(df):,}"
)

print()

print(
    f"Transaction graph nodes    : "
    f"{G.number_of_nodes():,}"
)

print(
    f"Transaction graph edges    : "
    f"{G.number_of_edges():,}"
)

print()

print(
    f"Address graph nodes        : "
    f"{A.number_of_nodes():,}"
)

print(
    f"Address graph edges        : "
    f"{A.number_of_edges():,}"
)

print()

print(
    f"Ownership graph nodes      : "
    f"{O.number_of_nodes():,}"
)

print(
    f"Ownership graph edges      : "
    f"{O.number_of_edges():,}"
)

print()

print(
    f"Saved:\n"
    f"  {transaction_graph_path}\n"
    f"  {address_graph_path}\n"
    f"  {ownership_graph_path}"
)

print()

print("=" * 60)
