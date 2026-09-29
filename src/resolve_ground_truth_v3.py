"""
Group B Ground Truth Resolution — V3

Resolves synthetic ground-truth entities to derived entities produced
by the V3 entity-resolution pipeline.

Path:

ground_truth.csv
    ↓
entity_wallet_map.csv
    ↓
wallet/address occurrences in blockchain_transactions.csv
    ↓
address_to_entity.csv
    ↓
derived_entity_id
"""

from pathlib import Path
import ast
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]

GT_FILE = ROOT / "data" / "ground_truth" / "ground_truth.csv"
MAP_FILE = ROOT / "data" / "ground_truth" / "entity_wallet_map.csv"
TX_FILE = ROOT / "data" / "raw" / "blockchain_transactions.csv"
ADDR_ENTITY_FILE = ROOT / "outputs" / "address_to_entity.csv"

OUTPUT_FILE = ROOT / "outputs" / "ground_truth_resolved_v3.csv"


def parse_list(value):
    """
    Parse list-like CSV fields safely.
    """
    if pd.isna(value):
        return []

    if isinstance(value, list):
        return value

    try:
        parsed = ast.literal_eval(value)
        if isinstance(parsed, list):
            return parsed
        return []
    except Exception:
        return []


def main():

    print("=" * 70)
    print("GROUP B GROUND TRUTH RESOLUTION — V3")
    print("=" * 70)

    # --------------------------------------------------------
    # LOAD FILES
    # --------------------------------------------------------

    print()
    print("Loading ground truth...")
    gt = pd.read_csv(GT_FILE)

    print(f"Ground truth rows        : {len(gt):,}")

    print()
    print("Loading entity/wallet map...")
    mapping = pd.read_csv(MAP_FILE)

    print(f"Mapping rows             : {len(mapping):,}")

    print()
    print("Loading transactions...")
    tx = pd.read_csv(TX_FILE)

    print(f"Transactions             : {len(tx):,}")

    print()
    print("Loading address/entity mapping...")
    addr_entity = pd.read_csv(ADDR_ENTITY_FILE)

    print(f"Address mappings         : {len(addr_entity):,}")

    # --------------------------------------------------------
    # BUILD WALLET -> DERIVED ENTITY LOOKUP
    # --------------------------------------------------------

    print()
    print("Building wallet/address → derived entity lookup...")

    address_to_entity = dict(
        zip(
            addr_entity["address"].astype(str),
            addr_entity["derived_entity_id"].astype(str),
        )
    )

    # --------------------------------------------------------
    # FIND WALLET OCCURRENCES IN TRANSACTIONS
    # --------------------------------------------------------

    print()
    print("Expanding transaction addresses...")

    occurrences = []

    for _, row in tx.iterrows():

        inputs = parse_list(row["input_addresses"])
        outputs = parse_list(row["output_addresses"])

        for address in inputs:
            occurrences.append(
                (str(address), row["txid"])
            )

        for address in outputs:
            occurrences.append(
                (str(address), row["txid"])
            )

    occurrences_df = pd.DataFrame(
        occurrences,
        columns=["address", "txid"],
    )

    print(
        f"Address occurrences      : "
        f"{len(occurrences_df):,}"
    )

    # --------------------------------------------------------
    # MAP OCCURRENCES TO DERIVED ENTITIES
    # --------------------------------------------------------

    occurrences_df["derived_entity_id"] = (
        occurrences_df["address"]
        .map(address_to_entity)
    )

    occurrences_df = occurrences_df.dropna(
        subset=["derived_entity_id"]
    )

    # --------------------------------------------------------
    # MAP WALLET -> SYNTHETIC ENTITY
    # --------------------------------------------------------

    wallet_to_gt = dict(
        zip(
            mapping["wallet_id"].astype(str),
            mapping["entity_id"].astype(str),
        )
    )

    occurrences_df["ground_truth_entity"] = (
        occurrences_df["address"]
        .map(wallet_to_gt)
    )

    occurrences_df = occurrences_df.dropna(
        subset=["ground_truth_entity"]
    )

    print(
        f"Ground-truth occurrences : "
        f"{len(occurrences_df):,}"
    )

    # --------------------------------------------------------
    # BUILD GT ENTITY -> DERIVED ENTITY RELATIONSHIP
    # --------------------------------------------------------

    print()
    print("Resolving synthetic entities...")

    relationships = (
        occurrences_df
        .groupby(
            [
                "ground_truth_entity",
                "derived_entity_id",
            ]
        )
        .agg(
            occurrence_count=("address", "size"),
            unique_addresses=("address", "nunique"),
            unique_txids=("txid", "nunique"),
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # KEEP BEST DERIVED ENTITY PER GROUND-TRUTH ENTITY
    # --------------------------------------------------------

    relationships = relationships.sort_values(
        [
            "ground_truth_entity",
            "occurrence_count",
            "unique_addresses",
            "unique_txids",
        ],
        ascending=[True, False, False, False],
    )

    best = (
        relationships
        .drop_duplicates(
            subset=["ground_truth_entity"],
            keep="first",
        )
        .copy()
    )

    best = best.rename(
        columns={
            "derived_entity_id": "resolved_entity_id"
        }
    )

    # --------------------------------------------------------
    # MERGE WITH GROUND TRUTH
    # --------------------------------------------------------

    result = gt.merge(
        best[
            [
                "ground_truth_entity",
                "resolved_entity_id",
                "occurrence_count",
                "unique_addresses",
                "unique_txids",
            ]
        ],
        left_on="target_id",
        right_on="ground_truth_entity",
        how="left",
    )

    result.drop(
        columns=["ground_truth_entity"],
        inplace=True,
        errors="ignore",
    )

    # --------------------------------------------------------
    # RESOLUTION STATUS
    # --------------------------------------------------------

    result["resolution_status"] = result[
        "resolved_entity_id"
    ].notna()

    # --------------------------------------------------------
    # VALIDATION
    # --------------------------------------------------------

    resolved = int(
        result["resolution_status"].sum()
    )

    unresolved = len(result) - resolved

    print()
    print("=" * 70)
    print("GROUND TRUTH RESOLUTION COMPLETE")
    print("=" * 70)

    print(
        f"Ground truth cases       : {len(result):,}"
    )

    print(
        f"Resolved                 : {resolved:,}"
    )

    print(
        f"Unresolved               : {unresolved:,}"
    )

    print(
        f"Resolution rate          : "
        f"{resolved / len(result) * 100:.2f}%"
    )

    print()
    print("Resolution by scenario:")

    scenario_stats = (
        result
        .groupby("scenario_id")
        .agg(
            cases=("target_id", "size"),
            resolved=("resolution_status", "sum"),
        )
    )

    scenario_stats["resolution_rate"] = (
        scenario_stats["resolved"]
        / scenario_stats["cases"]
        * 100
    )

    print(
        scenario_stats.to_string()
    )

    print()
    print("Sample resolved cases:")

    print(
        result[
            [
                "target_id",
                "scenario_id",
                "is_anomaly",
                "resolved_entity_id",
                "occurrence_count",
                "unique_addresses",
                "unique_txids",
                "resolution_status",
            ]
        ]
        .head(20)
        .to_string(index=False)
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    result.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print()
    print(f"Saved: {OUTPUT_FILE}")
    print("=" * 70)


if __name__ == "__main__":
    main()
