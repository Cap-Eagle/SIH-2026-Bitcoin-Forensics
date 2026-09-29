"""
Group B V3 — Stage 2: Entity Resolution

Derives logical entities from common-input ownership evidence.

Important:
- Entity IDs are DERIVED from transaction structure.
- Ground-truth labels are NOT used.
- This stage must not read ground_truth.
- V3 paths are completely isolated from btc-forensics_v2.

Inputs:
    data/raw/blockchain_transactions.csv
    outputs/ownership_graph.gpickle

Outputs:
    outputs/address_to_entity.csv
    outputs/entity_clusters.csv
    outputs/entity_graph.gpickle
    outputs/entity_edges.csv
"""

import json
import pickle
from collections import defaultdict
from pathlib import Path

import networkx as nx
import pandas as pd


# ============================================================
# V3 PATHS
# ============================================================

ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = ROOT / "outputs"

TRANSACTIONS_FILE = RAW_DIR / "blockchain_transactions.csv"
OWNERSHIP_GRAPH_FILE = OUT_DIR / "ownership_graph.gpickle"

OUT_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# HELPERS
# ============================================================

class UnionFind:
    """Simple Union-Find / Disjoint Set Union implementation."""

    def __init__(self):
        self.parent = {}
        self.rank = {}

    def find(self, value):
        if value not in self.parent:
            self.parent[value] = value
            self.rank[value] = 0
            return value

        if self.parent[value] != value:
            self.parent[value] = self.find(self.parent[value])

        return self.parent[value]

    def union(self, left, right):
        left_root = self.find(left)
        right_root = self.find(right)

        if left_root == right_root:
            return

        if self.rank[left_root] < self.rank[right_root]:
            self.parent[left_root] = right_root

        elif self.rank[left_root] > self.rank[right_root]:
            self.parent[right_root] = left_root

        else:
            self.parent[right_root] = left_root
            self.rank[left_root] += 1


def parse_list(value):
    """Parse JSON list values stored in CSV columns."""

    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    text = str(value).strip()

    if text in {"", "[]", "nan", "None"}:
        return []

    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else []

    except (json.JSONDecodeError, TypeError):
        return []


def safe_float(value, default=0.0):
    """Convert value to float without crashing."""

    try:
        return float(value)

    except (TypeError, ValueError):
        return default


def build_long_addresses(transactions):
    """
    Convert list-valued transaction columns into address-level rows.

    Each row contains:

        txid
        tx_timestamp
        address
        role
        amount_btc
    """

    rows = []

    for _, tx in transactions.iterrows():

        txid = str(tx["txid"])
        timestamp = tx.get("tx_timestamp", "")

        input_addresses = parse_list(
            tx.get("input_addresses", "[]")
        )

        input_amounts = parse_list(
            tx.get("input_amounts_btc", "[]")
        )

        output_addresses = parse_list(
            tx.get("output_addresses", "[]")
        )

        output_amounts = parse_list(
            tx.get("output_amounts_btc", "[]")
        )

        # -------------------------
        # INPUTS
        # -------------------------

        for address, amount in zip(
            input_addresses,
            input_amounts
        ):

            address = str(address).strip()

            if address:

                rows.append(
                    {
                        "txid": txid,
                        "tx_timestamp": timestamp,
                        "address": address,
                        "role": "input",
                        "amount_btc": safe_float(amount),
                    }
                )

        # -------------------------
        # OUTPUTS
        # -------------------------

        for address, amount in zip(
            output_addresses,
            output_amounts
        ):

            address = str(address).strip()

            if address:

                rows.append(
                    {
                        "txid": txid,
                        "tx_timestamp": timestamp,
                        "address": address,
                        "role": "output",
                        "amount_btc": safe_float(amount),
                    }
                )

    return pd.DataFrame(
        rows,
        columns=[
            "txid",
            "tx_timestamp",
            "address",
            "role",
            "amount_btc",
        ],
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 70)
    print("GROUP B ENTITY RESOLUTION — V3")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # Validate inputs
    # --------------------------------------------------------

    if not TRANSACTIONS_FILE.exists():

        raise FileNotFoundError(
            f"Missing transaction file:\n"
            f"  {TRANSACTIONS_FILE}"
        )

    if not OWNERSHIP_GRAPH_FILE.exists():

        raise FileNotFoundError(
            f"Missing ownership graph:\n"
            f"  {OWNERSHIP_GRAPH_FILE}\n\n"
            f"Run first:\n"
            f"  python3 src/build_graph.py"
        )

    print("Input transaction file:")
    print(f"  {TRANSACTIONS_FILE}")

    print()
    print("Input ownership graph:")
    print(f"  {OWNERSHIP_GRAPH_FILE}")

    print()
    print("Output directory:")
    print(f"  {OUT_DIR}")

    # --------------------------------------------------------
    # Load transactions
    # --------------------------------------------------------

    print()
    print("Loading transactions...")

    transactions = pd.read_csv(
        TRANSACTIONS_FILE
    )

    print(
        f"Transactions loaded: "
        f"{len(transactions):,}"
    )

    # --------------------------------------------------------
    # Expand addresses
    # --------------------------------------------------------

    print()
    print("Expanding transaction address lists...")

    long_addresses = build_long_addresses(
        transactions
    )

    if long_addresses.empty:

        raise ValueError(
            "No address rows were created."
        )

    print(
        f"Address-level rows: "
        f"{len(long_addresses):,}"
    )

    # --------------------------------------------------------
    # Load ownership graph
    # --------------------------------------------------------

    print()
    print("Loading ownership graph...")

    with open(
        OWNERSHIP_GRAPH_FILE,
        "rb"
    ) as file:

        ownership_graph = pickle.load(file)

    print(
        f"Ownership graph nodes: "
        f"{ownership_graph.number_of_nodes():,}"
    )

    print(
        f"Ownership graph edges: "
        f"{ownership_graph.number_of_edges():,}"
    )

    # --------------------------------------------------------
    # 1. Union-Find
    # --------------------------------------------------------

    print()
    print("Building entity clusters using Union-Find...")

    uf = UnionFind()

    for source, destination in ownership_graph.edges():

        uf.union(
            source,
            destination
        )

    # --------------------------------------------------------
    # 2. Collect all addresses
    # --------------------------------------------------------

    all_addresses = (
        long_addresses["address"]
        .dropna()
        .astype(str)
        .unique()
        .tolist()
    )

    print(
        f"Unique addresses: "
        f"{len(all_addresses):,}"
    )

    # --------------------------------------------------------
    # 3. Assign deterministic entity IDs
    # --------------------------------------------------------

    root_to_entity = {}
    address_to_entity = {}

    entity_counter = 0

    for address in sorted(all_addresses):

        root = uf.find(address)

        if root not in root_to_entity:

            entity_counter += 1

            root_to_entity[root] = (
                f"derived_entity_{entity_counter:06d}"
            )

        address_to_entity[address] = (
            root_to_entity[root]
        )

    # --------------------------------------------------------
    # 4. Build entity clusters
    # --------------------------------------------------------

    print()
    print("Building entity clusters...")

    cluster_rows = defaultdict(list)

    for address, entity_id in address_to_entity.items():

        cluster_rows[entity_id].append(
            address
        )

    entity_clusters = []

    for entity_id in sorted(cluster_rows):

        addresses = sorted(
            cluster_rows[entity_id]
        )

        entity_clusters.append(
            {
                "derived_entity_id": entity_id,
                "cluster_size": len(addresses),
                "addresses": json.dumps(
                    addresses
                ),
            }
        )

    entity_clusters_df = pd.DataFrame(
        entity_clusters
    )

    # --------------------------------------------------------
    # 5. Address → Entity mapping
    # --------------------------------------------------------

    address_to_entity_df = pd.DataFrame(
        [
            {
                "address": address,
                "derived_entity_id": entity_id,
            }

            for address, entity_id
            in sorted(
                address_to_entity.items()
            )
        ]
    )

    # --------------------------------------------------------
    # 6. Entity flow graph
    # --------------------------------------------------------

    print()
    print("Building entity flow graph...")

    entity_graph = nx.DiGraph()

    entity_edges = []

    grouped = long_addresses.groupby(
        "txid",
        sort=False
    )

    for txid, group in grouped:

        inputs = (
            group.loc[
                group["role"]
                .astype(str)
                .str.lower()
                == "input",
                "address",
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        outputs = (
            group.loc[
                group["role"]
                .astype(str)
                .str.lower()
                == "output",
                "address",
            ]
            .dropna()
            .astype(str)
            .unique()
            .tolist()
        )

        input_entities = {
            address_to_entity[address]

            for address in inputs

            if address in address_to_entity
        }

        output_entities = {
            address_to_entity[address]

            for address in outputs

            if address in address_to_entity
        }

        for source_entity in input_entities:

            for destination_entity in output_entities:

                if (
                    source_entity
                    == destination_entity
                ):
                    continue

                if entity_graph.has_edge(
                    source_entity,
                    destination_entity
                ):

                    entity_graph[
                        source_entity
                    ][
                        destination_entity
                    ]["tx_count"] += 1

                    entity_graph[
                        source_entity
                    ][
                        destination_entity
                    ]["txids"].append(
                        str(txid)
                    )

                else:

                    entity_graph.add_edge(
                        source_entity,
                        destination_entity,
                        tx_count=1,
                        txids=[str(txid)],
                    )

                entity_edges.append(
                    {
                        "source_entity":
                            source_entity,

                        "target_entity":
                            destination_entity,

                        "txid":
                            str(txid),
                    }
                )

    # --------------------------------------------------------
    # 7. Save outputs
    # --------------------------------------------------------

    print()
    print("Saving outputs...")

    address_to_entity_df.to_csv(
        OUT_DIR / "address_to_entity.csv",
        index=False,
    )

    entity_clusters_df.to_csv(
        OUT_DIR / "entity_clusters.csv",
        index=False,
    )

    with open(
        OUT_DIR / "entity_graph.gpickle",
        "wb"
    ) as file:

        pickle.dump(
            entity_graph,
            file
        )

    pd.DataFrame(
        entity_edges,
        columns=[
            "source_entity",
            "target_entity",
            "txid",
        ],
    ).to_csv(
        OUT_DIR / "entity_edges.csv",
        index=False,
    )

    # --------------------------------------------------------
    # 8. Diagnostics
    # --------------------------------------------------------

    cluster_sizes = (
        entity_clusters_df[
            "cluster_size"
        ]
        if not entity_clusters_df.empty
        else pd.Series(dtype=int)
    )

    print()
    print("=" * 70)
    print("ENTITY RESOLUTION COMPLETE")
    print("=" * 70)

    print()
    print(
        f"Transactions processed     : "
        f"{len(transactions):,}"
    )

    print(
        f"Address-level rows         : "
        f"{len(long_addresses):,}"
    )

    print(
        f"Unique addresses           : "
        f"{len(address_to_entity_df):,}"
    )

    print(
        f"Derived entities           : "
        f"{len(entity_clusters_df):,}"
    )

    print(
        f"Entity graph nodes         : "
        f"{entity_graph.number_of_nodes():,}"
    )

    print(
        f"Entity graph edges         : "
        f"{entity_graph.number_of_edges():,}"
    )

    if not cluster_sizes.empty:

        print()
        print("Entity cluster statistics:")

        print(
            f"  Mean cluster size        : "
            f"{cluster_sizes.mean():.2f}"
        )

        print(
            f"  Median cluster size      : "
            f"{cluster_sizes.median():.2f}"
        )

        print(
            f"  Maximum cluster size     : "
            f"{cluster_sizes.max():,}"
        )

        print(
            f"  Single-address entities  : "
            f"{(cluster_sizes == 1).sum():,}"
        )

        print(
            f"  Multi-address entities   : "
            f"{(cluster_sizes > 1).sum():,}"
        )

    print()
    print("Saved:")

    print(
        f"  {OUT_DIR / 'address_to_entity.csv'}"
    )

    print(
        f"  {OUT_DIR / 'entity_clusters.csv'}"
    )

    print(
        f"  {OUT_DIR / 'entity_graph.gpickle'}"
    )

    print(
        f"  {OUT_DIR / 'entity_edges.csv'}"
    )

    print()
    print("=" * 70)


if __name__ == "__main__":
    main()
