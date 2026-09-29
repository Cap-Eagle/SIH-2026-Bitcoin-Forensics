#!/usr/bin/env python3
"""
Chain Custody — SIH26146 Bitcoin Behavioral Forensics Console (V6.1)

Place at:
    SIH_V6/app/dashboard.py

Run from repository root:
    streamlit run app/dashboard.py

Primary inputs:
    outputs/final_entity_alerts_v61.csv
    outputs/entity_behavior_features_v61.csv
    outputs/entity_edges.csv
    outputs/address_to_entity.csv

Supporting evidence:
    data/raw/blockchain_transactions.csv
    data/raw/network_events.csv
    data/raw/ip_enrichment.csv
    data/correlated/correlations.csv
    data/correlated/group_b_features.csv

Optional analysis artifacts:
    outputs/behavior_feature_importance_v61.csv
    outputs/final_detector_summary_v61.txt

Design goals:
- entity-first investigation rather than wallet-first scoring
- no ground-truth labels in operational investigation views
- explain behavioral risk using graph, temporal, amount and network evidence
- fully offline; no external fonts, APIs or web calls
- lazy-load large evidence files so the overview stays responsive
"""

from __future__ import annotations

import ast
import html
import json
import math
from collections import Counter
from pathlib import Path
from typing import Iterable

import networkx as nx
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# =============================================================================
# Paths and constants
# =============================================================================

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent if APP_DIR.name == "app" else APP_DIR

DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
CORRELATED_DIR = DATA_DIR / "correlated"
OUTPUTS_DIR = ROOT / "outputs"
MODELS_DIR = ROOT / "models"

ALERT_FILE = OUTPUTS_DIR / "final_entity_alerts_v61.csv"
BEHAVIOR_FEATURE_FILE = OUTPUTS_DIR / "entity_behavior_features_v61.csv"
ENTITY_EDGE_FILE = OUTPUTS_DIR / "entity_edges.csv"
ADDRESS_ENTITY_FILE = OUTPUTS_DIR / "address_to_entity.csv"
ENTITY_CLUSTER_FILE = OUTPUTS_DIR / "entity_clusters.csv"
FEATURE_IMPORTANCE_FILE = OUTPUTS_DIR / "behavior_feature_importance_v61.csv"
FINAL_SUMMARY_FILE = OUTPUTS_DIR / "final_detector_summary_v61.txt"

BLOCKCHAIN_TX_FILE = RAW_DIR / "blockchain_transactions.csv"
NETWORK_EVENTS_FILE = RAW_DIR / "network_events.csv"
IP_ENRICHMENT_FILE = RAW_DIR / "ip_enrichment.csv"
CORRELATIONS_FILE = CORRELATED_DIR / "correlations.csv"
GROUP_B_FEATURES_FILE = CORRELATED_DIR / "group_b_features.csv"

PREVIEW_ROWS = 20
MAX_GRAPH_NODES = 110

RISK_ORDER = ["CRITICAL", "HIGH", "MEDIUM", "LOW", "UNSCORED"]
RISK_COLORS = {
    "CRITICAL": "#E05252",
    "HIGH": "#EA7B39",
    "MEDIUM": "#D8A829",
    "LOW": "#43B581",
    "UNSCORED": "#94A3B8",
    "UNKNOWN": "#64748B",
}

PATTERN_LABELS = {
    "RAPID_FORWARDING": "Rapid forwarding",
    "RAPID_MULTI_HOP_CHAIN": "Rapid multi-hop chain",
    "LAYERING_PATTERN": "Layering",
    "TRANSACTION_BURST": "Transaction burst",
    "STRUCTURING_PATTERN": "Structuring",
    "PEEL_CHAIN_PATTERN": "Peel-like behavior",
    "FAN_OUT_BURST": "Fan-out burst",
    "FAN_IN_COLLECTION": "Fan-in collection",
    "UNSUPERVISED_NOVELTY": "Novel / outlier behavior",
    "MULTIVARIATE_BEHAVIORAL_ANOMALY": "Multivariate anomaly",
}

# Curated evidence fields. Missing fields are simply skipped, so this dashboard
# remains compatible with small feature-name changes between V6.1 iterations.
EVIDENCE_GROUPS: dict[str, list[tuple[str, tuple[str, ...]]]] = {
    "Temporal / chain": [
        ("Rapid chain depth", ("v61_rapid_chain_depth",)),
        ("Minimum successor delay (s)", ("v61_min_successor_delay_sec",)),
        ("Rapid successor count", ("v61_rapid_successor_count",)),
        ("Rapid successor ratio", ("v61_rapid_successor_ratio",)),
        ("Very-rapid successor count", ("v61_very_rapid_successor_count",)),
        ("Layering successor count", ("v61_layering_successor_count",)),
        ("Layering successor ratio", ("v61_layering_successor_ratio",)),
    ],
    "Amount behavior": [
        ("Structuring-band count", ("v61_structuring_band_count",)),
        ("Structuring-band ratio", ("v61_structuring_band_ratio",)),
        ("Small peel-output count", ("v61_small_peel_output_count",)),
        ("Small peel-output ratio", ("v61_small_peel_output_ratio",)),
        ("Mean amount retention", ("v61_mean_amount_retention", "v61_mean_amount_retention_ratio")),
        ("High-retention ratio", ("v61_high_amount_retention_ratio",)),
        ("Very-high-retention ratio", ("v61_very_high_amount_retention_ratio",)),
    ],
    "Graph / flow": [
        ("Fan-out transaction count", ("v6_fanout_tx_count",)),
        ("Fan-out transaction ratio", ("v6_fanout_tx_ratio",)),
        ("Fan-in transaction count", ("v6_fanin_tx_count",)),
        ("Fan-in transaction ratio", ("v6_fanin_tx_ratio",)),
        ("Temporal outgoing edges", ("v61_temporal_outgoing_edge_count",)),
        ("Output amount entropy", ("v6_output_amount_entropy",)),
    ],
    "Network / IP": [
        ("Unique source IPs", ("v61_unique_source_ip_count", "v61_unique_src_ip_count")),
        ("Shared IP count", ("v61_shared_ip_count",)),
        ("Shared-IP ratio", ("v61_shared_ip_ratio",)),
        ("Maximum entities per IP", ("v61_max_entities_per_ip",)),
        ("Mean entities per IP", ("v61_mean_entities_per_ip",)),
    ],
}


# =============================================================================
# Basic helpers
# =============================================================================


def safe_float(value, default: float = 0.0) -> float:
    try:
        if value is None or pd.isna(value):
            return default
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return default
        return value
    except (TypeError, ValueError):
        return default


def safe_int(value, default: int = 0) -> int:
    try:
        if value is None or pd.isna(value):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def parse_list_field(value) -> list:
    if isinstance(value, list):
        return value
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    text = str(value).strip()
    if not text or text.lower() in {"nan", "none", "null", "[]"}:
        return []
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, list) else []
    except (json.JSONDecodeError, TypeError):
        pass
    try:
        parsed = ast.literal_eval(text)
        return parsed if isinstance(parsed, list) else []
    except (ValueError, SyntaxError):
        return []


def short_id(value: object, limit: int = 24) -> str:
    text = str(value)
    if len(text) <= limit:
        return text
    return f"{text[:limit]}…"


def normalize_entity_column(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df
    df = df.copy()
    if "entity_id" in df.columns:
        df["entity_id"] = df["entity_id"].astype(str)
        return df
    for candidate in ["derived_entity_id", "resolved_entity_id"]:
        if candidate in df.columns:
            df = df.rename(columns={candidate: "entity_id"})
            df["entity_id"] = df["entity_id"].astype(str)
            return df
    return df


def reason_list(value: object) -> list[str]:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return []
    return [part.strip() for part in str(value).split(";") if part.strip()]


def pattern_label(code: str) -> str:
    return PATTERN_LABELS.get(code, code.replace("_", " ").title())


def feature_value(row: pd.Series | None, *names: str) -> float:
    if row is None:
        return 0.0
    for name in names:
        if name in row.index:
            return safe_float(row.get(name))
    return 0.0


def risk_badge(level: str) -> str:
    level = str(level or "UNKNOWN").upper()
    color = RISK_COLORS.get(level, RISK_COLORS["UNKNOWN"])
    return (
        f'<span class="risk-chip" style="background:{color};">'
        f'{html.escape(level)}</span>'
    )


def pattern_badge(code: str) -> str:
    label = pattern_label(code)
    return f'<span class="pattern-chip">{html.escape(label)}</span>'


def human_number(value: float) -> str:
    value = safe_float(value)
    if abs(value) >= 1_000_000:
        return f"{value / 1_000_000:.1f}M"
    if abs(value) >= 1_000:
        return f"{value / 1_000:.1f}K"
    return f"{value:.0f}"


def entity_display_id(entity_id: str) -> str:
    return str(entity_id)


# =============================================================================
# Offline visual theme — V3 visual language, V6.1 semantics
# =============================================================================


def inject_theme() -> None:
    st.markdown(
        """
        <style>
        :root {
            --bg:#EEF1F8;
            --panel:#FFFFFF;
            --panel2:#F8FAFD;
            --ink:#101827;
            --soft:#3E485C;
            --muted:#758096;
            --line:#DDE4EF;
            --brand:#2B5BE7;
            --brand2:#4F7DFF;
            --brandTint:#EAF0FF;
            --sidebar:#08101F;
            --sidebar2:#172448;
            --table:#11141C;
            --table2:#181C26;
        }

        html, body, [class*="css"] {
            font-family: Inter, ui-sans-serif, system-ui, -apple-system,
                         BlinkMacSystemFont, "Segoe UI", sans-serif;
        }
        .stApp { background:var(--bg); color:var(--ink); }
        [data-testid="stHeader"] { background:transparent; }
        .block-container { max-width:1560px; padding:1.35rem 2rem 3.2rem; }

        [data-testid="stMain"] h1,
        [data-testid="stMain"] h2,
        [data-testid="stMain"] h3,
        [data-testid="stMain"] h4,
        [data-testid="stMain"] p,
        [data-testid="stMain"] span,
        [data-testid="stMain"] label { color:var(--ink); }

        .hero-card {
            background:var(--panel);
            border:1px solid var(--line);
            border-left:4px solid var(--brand);
            border-radius:16px;
            padding:1.25rem 1.55rem;
            margin-bottom:1rem;
            box-shadow:0 8px 28px rgba(15,23,42,.07);
        }
        .hero-kicker {
            color:var(--brand)!important;
            font-size:.68rem;
            font-weight:850;
            letter-spacing:.12em;
            text-transform:uppercase;
        }
        .hero-title {
            color:var(--ink)!important;
            font-size:1.95rem;
            font-weight:850;
            letter-spacing:-.035em;
            margin-top:.35rem;
        }
        .hero-sub {
            color:var(--muted)!important;
            font-size:.93rem;
            margin-top:.3rem;
        }

        .section-title {
            color:var(--ink)!important;
            font-size:1.12rem;
            font-weight:820;
            border-left:3px solid var(--brand);
            padding-left:.68rem;
            margin:1.5rem 0 .8rem;
        }
        .section-subtitle {
            color:var(--muted)!important;
            font-size:.82rem;
            margin:-.5rem 0 .8rem .78rem;
        }
        .soft-note {
            color:var(--muted)!important;
            font-size:.82rem;
            line-height:1.55;
        }

        .risk-chip {
            color:#fff!important;
            display:inline-block;
            padding:.25rem .62rem;
            border-radius:999px;
            font-size:.7rem;
            font-weight:850;
            letter-spacing:.04em;
            margin:.08rem .22rem .08rem 0;
        }
        .pattern-chip {
            display:inline-block;
            color:#3150A3!important;
            background:#EAF0FF;
            border:1px solid #CAD8FF;
            padding:.24rem .55rem;
            border-radius:999px;
            font-size:.7rem;
            font-weight:720;
            margin:.08rem .22rem .08rem 0;
        }

        .narrative-card,
        .evidence-card {
            background:var(--panel);
            border:1px solid var(--line);
            border-radius:13px;
            padding:.85rem 1rem;
            box-shadow:0 4px 15px rgba(15,23,42,.04);
        }
        .narrative-card { border-left:4px solid var(--brand); }
        .evidence-card b { color:var(--ink)!important; }
        .evidence-card small { color:var(--muted)!important; }

        div[data-testid="stMetric"] {
            background:var(--panel);
            border:1px solid var(--line);
            border-radius:14px;
            padding:.78rem 1rem;
            min-height:104px;
            box-shadow:0 5px 18px rgba(15,23,42,.05);
        }
        div[data-testid="stMetricLabel"] {
            color:var(--soft)!important;
            font-size:.8rem!important;
            font-weight:620!important;
        }
        div[data-testid="stMetricValue"] {
            color:var(--ink)!important;
            font-size:1.7rem!important;
            font-weight:850!important;
            letter-spacing:-.03em;
        }

        .stButton > button,
        .stDownloadButton > button {
            border-radius:9px;
            border:1px solid #D5DDEC;
            background:#fff;
            color:#274DBF;
            font-weight:720;
            min-height:40px;
        }
        .stButton > button:hover,
        .stDownloadButton > button:hover {
            border-color:var(--brand2);
            background:var(--brandTint);
            color:var(--brand);
        }

        input, textarea, [data-baseweb="select"] > div {
            border-radius:9px!important;
        }
        [data-testid="stTextInput"] input,
        [data-testid="stSelectbox"] [data-baseweb="select"] > div,
        [data-testid="stMultiSelect"] [data-baseweb="select"] > div {
            min-height:43px!important;
        }

        .dark-table-wrap {
            width:100%;
            overflow:auto;
            border:1px solid #272D3A;
            border-radius:11px;
            box-shadow:0 4px 15px rgba(15,23,42,.08);
            background:var(--table);
        }
        table.dark-table {
            border-collapse:collapse;
            width:100%;
            min-width:900px;
            color:#E6E9F0;
            font-size:.78rem;
        }
        .dark-table th {
            position:sticky;
            top:0;
            z-index:1;
            background:#1A1F2A;
            color:#AAB3C4!important;
            text-align:left;
            padding:.62rem .6rem;
            border-bottom:1px solid #303746;
            white-space:nowrap;
            font-weight:650;
        }
        .dark-table td {
            color:#E6E9F0!important;
            background:#11151D;
            padding:.55rem .6rem;
            border-bottom:1px solid #252B36;
            border-right:1px solid #222833;
            vertical-align:top;
            white-space:nowrap;
        }
        .dark-table tr:hover td { background:#171D27; }
        .dark-table .reason-cell {
            white-space:normal;
            min-width:310px;
            line-height:1.4;
            color:#D5DAE4!important;
        }

        [data-testid="stSidebar"] {
            background:linear-gradient(180deg,#08101F,#111A33 58%,#1A2850)!important;
            border-right:0;
            min-width:255px!important;
            width:255px!important;
            box-shadow:4px 0 18px rgba(10,15,30,.16)!important;
        }
        [data-testid="stSidebar"] * { color:#D6DDEE; }
        [data-testid="stSidebar"] hr { border-color:rgba(255,255,255,.09); }

        .cc-brand {
            display:flex;
            gap:.7rem;
            align-items:center;
            padding:.55rem .35rem 1.25rem;
            margin-bottom:.35rem;
            border-bottom:1px solid rgba(255,255,255,.09);
        }
        .cc-mark {
            width:40px;
            height:40px;
            border-radius:10px;
            background:linear-gradient(135deg,#4F7DFF,#2B5BE7);
            display:flex;
            align-items:center;
            justify-content:center;
            color:#fff!important;
            font-size:1.25rem;
            font-weight:850;
            box-shadow:0 5px 16px rgba(43,91,231,.45);
        }
        .cc-brand b {
            display:block;
            color:#fff!important;
            font-size:1.2rem;
            line-height:1.2;
        }
        .cc-brand small {
            display:block;
            color:#9DA8C3!important;
            font-size:.72rem;
            margin-top:.2rem;
        }
        .cc-nav-label {
            color:#9DA8C3!important;
            font-size:.67rem;
            font-weight:850;
            letter-spacing:.12em;
            text-transform:uppercase;
            margin:1.18rem .3rem .38rem;
        }
        [data-testid="stSidebar"] .stButton { margin:0!important; }
        [data-testid="stSidebar"] .stButton > button {
            width:100%;
            background:transparent;
            border:0;
            box-shadow:none;
            color:#D6DDEE;
            text-align:left;
            min-height:43px;
            padding:.45rem .62rem;
            border-radius:9px;
        }
        [data-testid="stSidebar"] .stButton > button:hover {
            background:rgba(79,125,255,.18);
            color:#fff;
        }
        .offline-pill {
            color:#9DA8C3!important;
            font-size:.72rem;
            margin:1rem .3rem .4rem;
        }
        .status-ok { color:#43B581!important; }
        .status-missing { color:#E05252!important; }
        </style>
        """,
        unsafe_allow_html=True,
    )


# =============================================================================
# Cached data loaders
# =============================================================================


def _read_csv(path: Path, **kwargs) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    try:
        return pd.read_csv(path, **kwargs)
    except Exception:
        return pd.DataFrame()


@st.cache_data(show_spinner=False)
def load_text(path_string: str) -> str:
    path = Path(path_string)
    if not path.exists():
        return ""
    try:
        return path.read_text(encoding="utf-8", errors="replace")
    except Exception:
        return ""


@st.cache_data(show_spinner=False)
def load_alerts() -> pd.DataFrame:
    df = normalize_entity_column(_read_csv(ALERT_FILE))
    if df.empty or "entity_id" not in df.columns:
        return pd.DataFrame()
    df = df.copy()
    for col in ["behavior_probability", "novelty_percentile", "risk_score"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)
    if "behavior_probability" not in df.columns:
        df["behavior_probability"] = pd.to_numeric(df.get("risk_score", 0.0), errors="coerce").fillna(0.0)
    if "risk_score" not in df.columns:
        df["risk_score"] = df["behavior_probability"]
    if "novelty_percentile" not in df.columns:
        df["novelty_percentile"] = 0.0
    if "risk_priority" not in df.columns:
        df["risk_priority"] = "UNSCORED"
    df["risk_priority"] = df["risk_priority"].astype(str).str.upper().str.strip()
    if "reason_codes" not in df.columns:
        df["reason_codes"] = "MULTIVARIATE_BEHAVIORAL_ANOMALY"
    if "rank" not in df.columns:
        df = df.sort_values("behavior_probability", ascending=False).reset_index(drop=True)
        df.insert(0, "rank", range(1, len(df) + 1))
    return df.sort_values("rank").reset_index(drop=True)


@st.cache_data(show_spinner=False)
def load_behavior_features() -> pd.DataFrame:
    df = normalize_entity_column(_read_csv(BEHAVIOR_FEATURE_FILE))
    if not df.empty and "entity_id" in df.columns:
        df = df.drop_duplicates("entity_id", keep="first")
    return df


@st.cache_data(show_spinner=False)
def load_address_entity() -> pd.DataFrame:
    df = _read_csv(ADDRESS_ENTITY_FILE)
    if df.empty:
        return df
    if "derived_entity_id" in df.columns and "entity_id" not in df.columns:
        df = df.rename(columns={"derived_entity_id": "entity_id"})
    if "address" in df.columns:
        df["address"] = df["address"].astype(str)
    if "entity_id" in df.columns:
        df["entity_id"] = df["entity_id"].astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_entity_edges() -> pd.DataFrame:
    df = _read_csv(ENTITY_EDGE_FILE)
    if df.empty:
        return df
    rename = {}
    if "source_entity" not in df.columns:
        for c in ["source", "src_entity", "source_entity_id"]:
            if c in df.columns:
                rename[c] = "source_entity"
                break
    if "target_entity" not in df.columns:
        for c in ["target", "dst_entity", "target_entity_id"]:
            if c in df.columns:
                rename[c] = "target_entity"
                break
    df = df.rename(columns=rename)
    for col in ["source_entity", "target_entity", "txid"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_transactions() -> pd.DataFrame:
    wanted = {
        "txid", "tx_timestamp", "block_height", "is_coinbase",
        "input_addresses", "output_addresses", "input_amounts_btc",
        "output_amounts_btc", "fee_btc", "script_type",
    }
    if not BLOCKCHAIN_TX_FILE.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(BLOCKCHAIN_TX_FILE, usecols=lambda c: c in wanted)
    except Exception:
        df = _read_csv(BLOCKCHAIN_TX_FILE)
    if df.empty or "txid" not in df.columns:
        return pd.DataFrame()
    df["txid"] = df["txid"].astype(str)
    if "tx_timestamp" in df.columns:
        df["parsed_timestamp"] = pd.to_datetime(df["tx_timestamp"], errors="coerce", utc=True)
    else:
        df["parsed_timestamp"] = pd.NaT
    return df


@st.cache_data(show_spinner=False)
def load_correlations_minimal() -> pd.DataFrame:
    wanted = {
        "event_timestamp", "src_ip", "dst_ip", "observed_txid", "candidate_txid",
        "match_score", "abs_time_delta_ms", "message_type", "relay_role",
    }
    if not CORRELATIONS_FILE.exists():
        return pd.DataFrame()
    try:
        df = pd.read_csv(CORRELATIONS_FILE, usecols=lambda c: c in wanted)
    except Exception:
        df = _read_csv(CORRELATIONS_FILE)
    for col in ["observed_txid", "candidate_txid"]:
        if col in df.columns:
            df[col] = df[col].astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_ip_enrichment() -> pd.DataFrame:
    df = _read_csv(IP_ENRICHMENT_FILE)
    if "ip" in df.columns:
        df["ip"] = df["ip"].astype(str)
    return df


@st.cache_data(show_spinner=False)
def load_feature_importance() -> pd.DataFrame:
    return _read_csv(FEATURE_IMPORTANCE_FILE)


@st.cache_resource(show_spinner=False)
def load_entity_graph() -> nx.DiGraph:
    edges = load_entity_edges()
    graph = nx.DiGraph()
    if edges.empty or not {"source_entity", "target_entity"}.issubset(edges.columns):
        return graph
    for row in edges[["source_entity", "target_entity"]].itertuples(index=False):
        source, target = str(row[0]), str(row[1])
        if source == "nan" or target == "nan":
            continue
        if graph.has_edge(source, target):
            graph[source][target]["tx_count"] += 1
        else:
            graph.add_edge(source, target, tx_count=1)
    return graph


# =============================================================================
# Entity lookup helpers
# =============================================================================


def get_entity_alert(entity_id: str, alerts: pd.DataFrame) -> pd.Series | None:
    matches = alerts[alerts["entity_id"].astype(str) == str(entity_id)]
    return None if matches.empty else matches.iloc[0]


def get_entity_features(entity_id: str, features: pd.DataFrame) -> pd.Series | None:
    if features.empty or "entity_id" not in features.columns:
        return None
    matches = features[features["entity_id"].astype(str) == str(entity_id)]
    return None if matches.empty else matches.iloc[0]


def addresses_for_entity(entity_id: str, mapping: pd.DataFrame) -> list[str]:
    if mapping.empty or not {"entity_id", "address"}.issubset(mapping.columns):
        return []
    return mapping.loc[mapping["entity_id"] == str(entity_id), "address"].astype(str).drop_duplicates().tolist()


def txids_for_entity(entity_id: str) -> list[str]:
    edges = load_entity_edges()
    if edges.empty or "txid" not in edges.columns:
        return []
    mask = pd.Series(False, index=edges.index)
    if "source_entity" in edges.columns:
        mask |= edges["source_entity"].astype(str).eq(str(entity_id))
    if "target_entity" in edges.columns:
        mask |= edges["target_entity"].astype(str).eq(str(entity_id))
    return edges.loc[mask, "txid"].astype(str).drop_duplicates().tolist()


def transaction_roles_for_entity(entity_id: str) -> pd.DataFrame:
    edges = load_entity_edges()
    if edges.empty or "txid" not in edges.columns:
        return pd.DataFrame(columns=["txid", "direction"])
    frames = []
    if "source_entity" in edges.columns:
        out = edges[edges["source_entity"] == entity_id][["txid"]].copy()
        out["direction"] = "Outgoing"
        frames.append(out)
    if "target_entity" in edges.columns:
        inc = edges[edges["target_entity"] == entity_id][["txid"]].copy()
        inc["direction"] = "Incoming"
        frames.append(inc)
    if not frames:
        return pd.DataFrame(columns=["txid", "direction"])
    roles = pd.concat(frames, ignore_index=True)
    roles = roles.groupby("txid", as_index=False)["direction"].agg(lambda s: "/".join(sorted(set(s))))
    return roles


def entity_neighborhood(entity_id: str, radius: int = 1, max_nodes: int = MAX_GRAPH_NODES) -> nx.DiGraph:
    graph = load_entity_graph()
    if entity_id not in graph:
        return nx.DiGraph()
    undirected = graph.to_undirected()
    distances = nx.single_source_shortest_path_length(undirected, entity_id, cutoff=radius)
    nodes = list(distances.keys())
    if len(nodes) > max_nodes:
        ranked = sorted(
            nodes,
            key=lambda n: (distances.get(n, 99), -undirected.degree[n]),
        )
        keep = set(ranked[: max_nodes - 1]) | {entity_id}
    else:
        keep = set(nodes)
    return graph.subgraph(keep).copy()


# =============================================================================
# Session state / navigation
# =============================================================================


def initialize_state() -> None:
    defaults = {
        "section": "overview",
        "selected_entity": None,
        "selected_txid": None,
        "cases": [],
        "case_name": "SIH26146 Investigation",
        "case_notes": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def navigate(section: str, entity_id: str | None = None, txid: str | None = None) -> None:
    st.session_state["section"] = section
    if entity_id is not None:
        st.session_state["selected_entity"] = str(entity_id)
    if txid is not None:
        st.session_state["selected_txid"] = str(txid)
    st.rerun()


def reset_case() -> None:
    st.session_state["cases"] = []
    st.session_state["case_name"] = "SIH26146 Investigation"
    st.session_state["case_notes"] = ""
    st.session_state["selected_entity"] = None
    st.session_state["selected_txid"] = None


# =============================================================================
# Reusable UI helpers
# =============================================================================


def render_dark_table(df: pd.DataFrame, max_rows: int = PREVIEW_ROWS, reason_columns: Iterable[str] = ()) -> None:
    if df.empty:
        st.info("No records match the current selection.")
        return
    view = df.head(max_rows).copy()
    reason_columns = set(reason_columns)
    head = "".join(f"<th>{html.escape(str(c))}</th>" for c in view.columns)
    body_rows = []
    for _, row in view.iterrows():
        cells = []
        for column in view.columns:
            value = row[column]
            if isinstance(value, float):
                if math.isnan(value):
                    text = ""
                elif abs(value) < 10:
                    text = f"{value:.4f}"
                else:
                    text = f"{value:,.2f}"
            else:
                text = str(value)
            cls = "reason-cell" if column in reason_columns else ""
            cells.append(f'<td class="{cls}">{html.escape(text)}</td>')
        body_rows.append("<tr>" + "".join(cells) + "</tr>")
    st.markdown(
        '<div class="dark-table-wrap"><table class="dark-table">'
        f'<thead><tr>{head}</tr></thead><tbody>{"".join(body_rows)}</tbody>'
        '</table></div>',
        unsafe_allow_html=True,
    )
    if len(df) > max_rows:
        st.caption(f"Previewing {max_rows:,} of {len(df):,} matching records.")


def section_title(title: str, subtitle: str | None = None) -> None:
    st.markdown(f'<div class="section-title">{html.escape(title)}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-subtitle">{html.escape(subtitle)}</div>', unsafe_allow_html=True)


def artifact_status() -> dict[str, bool]:
    return {
        "Final alerts": ALERT_FILE.exists(),
        "Behavior features": BEHAVIOR_FEATURE_FILE.exists(),
        "Entity edges": ENTITY_EDGE_FILE.exists(),
        "Address mapping": ADDRESS_ENTITY_FILE.exists(),
        "Transactions": BLOCKCHAIN_TX_FILE.exists(),
        "Correlations": CORRELATIONS_FILE.exists(),
    }


def render_sidebar(alerts: pd.DataFrame) -> None:
    priority_queue = int(alerts["risk_priority"].isin(["CRITICAL", "HIGH"]).sum()) if not alerts.empty else 0
    with st.sidebar:
        st.markdown(
            '<div class="cc-brand">'
            '<div class="cc-mark">✓</div>'
            '<div><b>Chain Custody</b><small>Behavioral Forensics · BTC · V6.1</small></div>'
            '</div>',
            unsafe_allow_html=True,
        )

        groups = [
            ("Investigate", [
                ("overview", "▦  Overview"),
                ("alerts", f"◇  Entity Alerts  {priority_queue:,}"),
                ("entity_search", "⌕  Entity Search"),
                ("transactions", "◎  Transaction Explorer"),
                ("graph", "⌘  Entity Graph"),
            ]),
            ("Analysis", [
                ("temporal", "◷  Temporal Analysis"),
                ("network", "⌁  Network / IP Evidence"),
                ("patterns", "⛓  Behavioral Patterns"),
            ]),
            ("Workspace", [
                ("cases", f"□  Case Files  {len(st.session_state['cases'])}"),
            ]),
        ]

        for group_name, items in groups:
            st.markdown(f'<div class="cc-nav-label">{group_name}</div>', unsafe_allow_html=True)
            for key, label in items:
                prefix = "●  " if st.session_state["section"] == key else ""
                if st.button(prefix + label, key=f"nav_{key}", use_container_width=True):
                    navigate(key)

        with st.expander("System status", expanded=False):
            for label, ok in artifact_status().items():
                icon = "●" if ok else "●"
                cls = "status-ok" if ok else "status-missing"
                state = "ready" if ok else "missing"
                st.markdown(f'<span class="{cls}">{icon}</span> {label}: {state}', unsafe_allow_html=True)

        st.markdown(
            '<div class="offline-pill"><span class="status-ok">●</span> Offline · synthetic evaluation dataset</div>',
            unsafe_allow_html=True,
        )


def render_header() -> None:
    st.markdown(
        """
        <div class="hero-card">
            <div class="hero-kicker">Chain Custody / SIH26146 / Offline Bitcoin Forensics</div>
            <div class="hero-title">Case Overview — Behavioral Investigation Console</div>
            <div class="hero-sub">Entity resolution · blockchain/P2P correlation · temporal graph analysis · V6/V6.1 behavioral features · Random Forest detection · Isolation Forest novelty analysis</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_top_controls(alerts: pd.DataFrame, mapping: pd.DataFrame) -> None:
    c1, c2, c3, c4 = st.columns([1.0, 1.15, .72, 1.55])
    with c1:
        st.selectbox(
            "Dataset",
            ["SIH26146 synthetic investigation dataset"],
            disabled=True,
            label_visibility="collapsed",
        )
    with c2:
        st.selectbox(
            "Network",
            ["Bitcoin — mainnet semantics (synthetic)"],
            disabled=True,
            label_visibility="collapsed",
        )
    with c3:
        if st.button("＋  New case", use_container_width=True):
            reset_case()
            st.success("New case workspace created.")
    with c4:
        query = st.text_input(
            "Global search",
            placeholder="Search entity ID, wallet address, reason code, or transaction ID…",
            key="global_search",
            label_visibility="collapsed",
        ).strip()

    if not query:
        return

    q = query.lower()
    entity_columns = [c for c in ["entity_id", "risk_priority", "reason_codes"] if c in alerts.columns]
    entity_hits = alerts[
        alerts[entity_columns]
        .astype(str)
        .apply(lambda col: col.str.lower().str.contains(q, na=False))
        .any(axis=1)
    ].head(8) if entity_columns else pd.DataFrame()

    wallet_hits = pd.DataFrame()
    if not mapping.empty and "address" in mapping.columns:
        wallet_hits = mapping[mapping["address"].str.lower().str.contains(q, na=False)].head(8)

    tx_hits = pd.DataFrame()
    # Only touch the transaction file after the user has actually typed a query.
    tx = load_transactions()
    if not tx.empty and "txid" in tx.columns:
        tx_hits = tx[tx["txid"].str.lower().str.contains(q, na=False)].head(8)

    if entity_hits.empty and wallet_hits.empty and tx_hits.empty:
        st.info(f"No result found for '{query}'.")
        return

    with st.expander("Search results", expanded=True):
        a, b, c = st.columns(3)
        with a:
            st.caption(f"Entities ({len(entity_hits)})")
            for i, row in entity_hits.reset_index(drop=True).iterrows():
                if st.button(
                    f"{short_id(row['entity_id'])} · {row.get('risk_priority', 'UNSCORED')}",
                    key=f"search_entity_{i}",
                    use_container_width=True,
                ):
                    navigate("entity_search", entity_id=row["entity_id"])
        with b:
            st.caption(f"Wallets ({len(wallet_hits)})")
            for i, row in wallet_hits.reset_index(drop=True).iterrows():
                entity = str(row.get("entity_id", ""))
                if st.button(
                    f"{short_id(row.get('address', ''))} → {short_id(entity)}",
                    key=f"search_wallet_{i}",
                    use_container_width=True,
                ):
                    navigate("entity_search", entity_id=entity)
        with c:
            st.caption(f"Transactions ({len(tx_hits)})")
            for i, row in tx_hits.reset_index(drop=True).iterrows():
                if st.button(short_id(row["txid"]), key=f"search_tx_{i}", use_container_width=True):
                    navigate("transactions", txid=row["txid"])


# =============================================================================
# Graph rendering
# =============================================================================


def draw_entity_graph(
    graph: nx.Graph,
    alerts: pd.DataFrame,
    title: str,
    focus: set[str] | None = None,
    height: int = 520,
) -> None:
    if graph is None or graph.number_of_nodes() == 0:
        st.info("No graph records are available for this selection.")
        return

    focus = {str(v) for v in (focus or set())}
    risk_lookup = dict(zip(alerts["entity_id"].astype(str), alerts["risk_priority"].astype(str)))
    prob_lookup = dict(zip(alerts["entity_id"].astype(str), alerts["behavior_probability"]))
    novelty_lookup = dict(zip(alerts["entity_id"].astype(str), alerts["novelty_percentile"]))
    reason_lookup = dict(zip(alerts["entity_id"].astype(str), alerts["reason_codes"].astype(str)))

    undirected = graph.to_undirected()
    k = max(0.35, 1.4 / math.sqrt(max(undirected.number_of_nodes(), 2)))
    positions = nx.spring_layout(undirected, seed=17, k=k, iterations=70)

    edge_x, edge_y = [], []
    for source, target in graph.edges():
        if source not in positions or target not in positions:
            continue
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edge_trace = go.Scatter(
        x=edge_x,
        y=edge_y,
        mode="lines",
        hoverinfo="none",
        line=dict(width=1.0, color="rgba(135,151,177,.50)"),
    )

    node_x, node_y, colors, sizes, hover = [], [], [], [], []
    for node in graph.nodes():
        node_s = str(node)
        risk = risk_lookup.get(node_s, "UNKNOWN")
        probability = safe_float(prob_lookup.get(node_s, 0.0))
        novelty = safe_float(novelty_lookup.get(node_s, 0.0))
        reasons = reason_list(reason_lookup.get(node_s, ""))[:3]
        degree = graph.degree[node]

        node_x.append(positions[node][0])
        node_y.append(positions[node][1])
        colors.append(RISK_COLORS.get(risk, RISK_COLORS["UNKNOWN"]))
        sizes.append(24 if node_s in focus else 7 + min(8, degree * 0.6))
        hover.append(
            f"<b>{node_s}</b><br>Priority: {risk}"
            f"<br>Behavior probability: {probability:.4f}"
            f"<br>Novelty percentile: {novelty * 100:.2f}%"
            f"<br>Degree: {degree}"
            + ("<br>" + "<br>".join(pattern_label(r) for r in reasons) if reasons else "")
        )

    node_trace = go.Scatter(
        x=node_x,
        y=node_y,
        mode="markers",
        hoverinfo="text",
        text=hover,
        marker=dict(
            color=colors,
            size=sizes,
            line=dict(width=1, color="#445066"),
            opacity=.96,
        ),
    )

    fig = go.Figure([edge_trace, node_trace])
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color="#5F6B80")),
        showlegend=False,
        height=height,
        margin=dict(l=10, r=10, t=42, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        hoverlabel=dict(bgcolor="#11151D", font_color="#FFFFFF"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


# =============================================================================
# Alert filtering / reusable queue
# =============================================================================


def available_patterns(alerts: pd.DataFrame) -> list[str]:
    counter = Counter()
    if "reason_codes" in alerts.columns:
        for value in alerts["reason_codes"]:
            counter.update(reason_list(value))
    return [code for code, _ in counter.most_common()]


def alert_filters(alerts: pd.DataFrame, key_prefix: str) -> pd.DataFrame:
    patterns = available_patterns(alerts)
    c1, c2, c3, c4 = st.columns([1, 1, 1, .8])
    with c1:
        priorities = st.multiselect(
            "Risk priority",
            RISK_ORDER[:-1],
            default=["CRITICAL", "HIGH", "MEDIUM"],
            key=f"{key_prefix}_priority",
        )
    with c2:
        pattern = st.selectbox(
            "Behavior",
            ["ALL"] + patterns,
            format_func=lambda x: "ALL" if x == "ALL" else pattern_label(x),
            key=f"{key_prefix}_pattern",
        )
    with c3:
        min_prob = st.slider(
            "Minimum behavior probability",
            0.0, 1.0, 0.50, 0.01,
            key=f"{key_prefix}_prob",
        )
    with c4:
        min_novelty = st.slider(
            "Minimum novelty",
            0.0, 1.0, 0.0, 0.01,
            key=f"{key_prefix}_novelty",
        )

    search = st.text_input(
        "Search entity / reason code",
        placeholder="derived_entity_066961 or RAPID_FORWARDING",
        key=f"{key_prefix}_search",
    ).strip()

    filtered = alerts.copy()
    if priorities:
        filtered = filtered[filtered["risk_priority"].isin(priorities)]
    filtered = filtered[filtered["behavior_probability"] >= min_prob]
    filtered = filtered[filtered["novelty_percentile"] >= min_novelty]
    if pattern != "ALL":
        filtered = filtered[filtered["reason_codes"].astype(str).str.contains(pattern, na=False, regex=False)]
    if search:
        q = search.lower()
        filtered = filtered[
            filtered[["entity_id", "reason_codes"]]
            .astype(str)
            .apply(lambda col: col.str.lower().str.contains(q, na=False))
            .any(axis=1)
        ]
    return filtered.sort_values("rank")


def alert_preview_frame(filtered: pd.DataFrame) -> pd.DataFrame:
    columns = [
        c for c in [
            "rank", "entity_id", "behavior_probability", "novelty_percentile",
            "risk_priority", "reason_codes"
        ]
        if c in filtered.columns
    ]
    view = filtered[columns].copy()
    if "behavior_probability" in view.columns:
        view["behavior_probability"] = view["behavior_probability"].round(4)
    if "novelty_percentile" in view.columns:
        view["novelty_percentile"] = (view["novelty_percentile"] * 100).round(2).astype(str) + "%"
    return view


# =============================================================================
# Overview
# =============================================================================


def render_overview(alerts: pd.DataFrame) -> None:
    counts = alerts["risk_priority"].value_counts()
    critical = safe_int(counts.get("CRITICAL", 0))
    high = safe_int(counts.get("HIGH", 0))
    medium = safe_int(counts.get("MEDIUM", 0))
    non_low = critical + high + medium

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Entities analyzed", f"{len(alerts):,}")
    c2.metric("Critical entities", f"{critical:,}")
    c3.metric("Critical + High", f"{critical + high:,}")
    c4.metric("Non-low entities", f"{non_low:,}")

    section_title(
        "High-risk entity network overview",
        "Large nodes are the top investigative leads; surrounding nodes are direct entity-flow neighbors.",
    )
    graph = load_entity_graph()
    if graph.number_of_nodes() == 0:
        st.info("Entity graph unavailable. Run graph construction and entity resolution first.")
    else:
        focus = [e for e in alerts.head(8)["entity_id"].tolist() if e in graph]
        nodes: set[str] = set(focus)
        for entity in focus:
            # Take the most connected direct neighbors so the overview remains readable.
            neighbors = set(graph.predecessors(entity)) | set(graph.successors(entity))
            ranked = sorted(neighbors, key=lambda n: graph.degree[n], reverse=True)[:15]
            nodes.update(ranked)
        subgraph = graph.subgraph(nodes).copy()
        draw_entity_graph(subgraph, alerts, "Top risk entities and connected flow neighbors", set(focus), height=555)
        st.caption("Red = critical · orange = high · yellow = medium · green = low. Entity clusters are investigative heuristics, not confirmed real-world identities.")

    section_title("Behavioral signal snapshot", "How often major reason codes occur in the active investigation queue.")
    active = alerts[alerts["risk_priority"].isin(["CRITICAL", "HIGH", "MEDIUM"])]
    reason_counter = Counter()
    for value in active["reason_codes"]:
        reason_counter.update(reason_list(value))
    signal_cards = [
        ("Rapid forwarding", reason_counter.get("RAPID_FORWARDING", 0)),
        ("Layering", reason_counter.get("LAYERING_PATTERN", 0)),
        ("Structuring", reason_counter.get("STRUCTURING_PATTERN", 0)),
        ("Novel behavior", reason_counter.get("UNSUPERVISED_NOVELTY", 0)),
    ]
    cols = st.columns(4)
    for col, (label, value) in zip(cols, signal_cards):
        col.metric(label, f"{value:,}")

    section_title("Entity alert filters")
    filtered = alert_filters(alerts, "overview")
    st.caption(f"{len(filtered):,} entities match the current filters.")

    if not filtered.empty:
        selected = st.selectbox(
            "Open an entity case file",
            ["(none)"] + filtered["entity_id"].head(750).tolist(),
            key="overview_open_entity",
        )
        if selected != "(none)" and st.button("Open entity investigation", key="overview_open_button"):
            navigate("entity_search", entity_id=selected)

    section_title("Entity alert preview")
    render_dark_table(
        alert_preview_frame(filtered),
        max_rows=PREVIEW_ROWS,
        reason_columns={"reason_codes"},
    )


# =============================================================================
# Alerts page
# =============================================================================


def render_alerts(alerts: pd.DataFrame) -> None:
    section_title("Entity alerts & triage", "Prioritize entities by behavioral probability, novelty and forensic reason codes.")
    filtered = alert_filters(alerts, "alerts")

    c1, c2, c3 = st.columns(3)
    c1.metric("Matching entities", f"{len(filtered):,}")
    c2.metric("Critical", f"{int((filtered['risk_priority'] == 'CRITICAL').sum()):,}")
    c3.metric("Critical + High", f"{int(filtered['risk_priority'].isin(['CRITICAL', 'HIGH']).sum()):,}")

    section_title("Ranked investigation queue")
    render_dark_table(alert_preview_frame(filtered), max_rows=100, reason_columns={"reason_codes"})

    if not filtered.empty:
        entity = st.selectbox("Open entity", filtered["entity_id"].head(1000).tolist(), key="alerts_open_entity")
        if st.button("Investigate selected entity"):
            navigate("entity_search", entity_id=entity)


# =============================================================================
# Entity investigation
# =============================================================================


def build_entity_narrative(entity_id: str, alert: pd.Series, feat: pd.Series | None) -> str:
    priority = str(alert.get("risk_priority", "UNSCORED"))
    probability = safe_float(alert.get("behavior_probability"))
    novelty = safe_float(alert.get("novelty_percentile"))
    reasons = reason_list(alert.get("reason_codes"))

    phrases = []
    if "RAPID_FORWARDING" in reasons:
        delay = feature_value(feat, "v61_min_successor_delay_sec")
        if delay > 0:
            phrases.append(f"rapid forwarding with a minimum observed successor delay of about {delay:.0f} seconds")
        else:
            phrases.append("rapid forwarding between entities")
    if "RAPID_MULTI_HOP_CHAIN" in reasons:
        depth = feature_value(feat, "v61_rapid_chain_depth")
        phrases.append(f"a rapid multi-hop path with depth {depth:.0f}" if depth > 0 else "a rapid multi-hop path")
    if "LAYERING_PATTERN" in reasons:
        phrases.append("layering-like successor behavior")
    if "STRUCTURING_PATTERN" in reasons:
        phrases.append("repeated near-threshold output amounts consistent with structuring behavior")
    if "PEEL_CHAIN_PATTERN" in reasons:
        phrases.append("small repeated peel-like outputs")
    if "TRANSACTION_BURST" in reasons:
        phrases.append("bursty transaction timing")
    if "UNSUPERVISED_NOVELTY" in reasons:
        phrases.append("strong unsupervised novelty relative to the broader entity population")

    evidence = ", ".join(phrases[:4]) if phrases else "a multivariate combination of behavioral signals"
    return (
        f"{entity_id} is currently prioritized as {priority}. The behavioral classifier assigns "
        f"a probability of {probability:.3f}, while its novelty percentile is {novelty * 100:.2f}%. "
        f"The main investigative evidence is {evidence}. This is a triage lead, not a determination "
        f"of criminal ownership or identity."
    )


def evidence_rows(feat: pd.Series | None, group_name: str) -> pd.DataFrame:
    rows = []
    if feat is None:
        return pd.DataFrame()
    for label, aliases in EVIDENCE_GROUPS[group_name]:
        present = next((name for name in aliases if name in feat.index), None)
        if present is None:
            continue
        rows.append({"Signal": label, "Value": safe_float(feat.get(present)), "Feature": present})
    return pd.DataFrame(rows)


def render_entity_search(alerts: pd.DataFrame, features: pd.DataFrame, mapping: pd.DataFrame) -> None:
    section_title("Entity search", "Search by derived entity ID or any wallet address assigned to that entity.")
    query = st.text_input(
        "Entity ID or wallet address",
        value=st.session_state.get("selected_entity") or "",
        key="entity_search_query",
    ).strip()

    entity_id: str | None = None
    if query:
        if not mapping.empty and "address" in mapping.columns:
            exact_wallet = mapping[mapping["address"].astype(str) == query]
            if not exact_wallet.empty:
                entity_id = str(exact_wallet.iloc[0]["entity_id"])
        exact_entity = alerts[alerts["entity_id"].astype(str) == query]
        if entity_id is None and not exact_entity.empty:
            entity_id = query
        if entity_id is None:
            fuzzy = alerts[alerts["entity_id"].str.contains(query, case=False, na=False)].head(20)
            if not fuzzy.empty:
                entity_id = st.selectbox("Matching entities", fuzzy["entity_id"].tolist())

    if entity_id is None:
        st.info("Enter an entity ID such as derived_entity_066961 or one of its wallet addresses.")
        return

    st.session_state["selected_entity"] = entity_id
    render_entity_detail(entity_id, alerts, features, mapping)


def render_entity_detail(entity_id: str, alerts: pd.DataFrame, features: pd.DataFrame, mapping: pd.DataFrame) -> None:
    alert = get_entity_alert(entity_id, alerts)
    feat = get_entity_features(entity_id, features)
    if alert is None:
        st.warning("This entity is not present in final_entity_alerts_v61.csv.")
        return

    section_title(f"Entity case file — {entity_id}")
    st.markdown(risk_badge(alert.get("risk_priority", "UNSCORED")), unsafe_allow_html=True)

    addresses = addresses_for_entity(entity_id, mapping)
    txids = txids_for_entity(entity_id)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Behavior probability", f"{safe_float(alert.get('behavior_probability')):.4f}")
    c2.metric("Novelty percentile", f"{safe_float(alert.get('novelty_percentile')) * 100:.2f}%")
    c3.metric("Risk rank", f"#{safe_int(alert.get('rank')):,}")
    c4.metric("Clustered wallets", f"{len(addresses):,}")
    c5.metric("Transactions touched", f"{len(txids):,}")

    narrative = build_entity_narrative(entity_id, alert, feat)
    st.markdown(
        f'<div class="narrative-card"><b>Investigation summary</b><br><span class="soft-note">{html.escape(narrative)}</span></div>',
        unsafe_allow_html=True,
    )

    reasons = reason_list(alert.get("reason_codes"))
    section_title("Detected behaviors")
    st.markdown(" ".join(pattern_badge(r) for r in reasons) if reasons else "No reason codes recorded.", unsafe_allow_html=True)

    evidence_tab, graph_tab, wallets_tab, tx_tab = st.tabs([
        "Behavioral evidence", "Entity flow", "Clustered wallets", "Transactions"
    ])

    with evidence_tab:
        for group_name in EVIDENCE_GROUPS:
            frame = evidence_rows(feat, group_name)
            if frame.empty:
                continue
            st.markdown(f"**{group_name}**")
            # Keep zeroes visible because absence can itself be useful forensic context.
            st.dataframe(frame, use_container_width=True, hide_index=True)

    with graph_tab:
        radius = st.slider("Graph radius", 1, 2, 1, key=f"entity_radius_{entity_id}")
        subgraph = entity_neighborhood(entity_id, radius=radius)
        draw_entity_graph(subgraph, alerts, "Selected entity and transaction-flow neighborhood", {entity_id}, height=520)

    with wallets_tab:
        if addresses:
            st.dataframe(pd.DataFrame({"wallet_address": addresses}), use_container_width=True, hide_index=True, height=320)
        else:
            st.info("No address-to-entity mapping was found for this entity.")

    with tx_tab:
        tx = load_transactions()
        if txids and not tx.empty:
            entity_tx = tx[tx["txid"].isin(txids)].sort_values("parsed_timestamp", ascending=False)
            columns = [c for c in ["txid", "tx_timestamp", "block_height", "fee_btc", "script_type"] if c in entity_tx.columns]
            st.dataframe(entity_tx[columns].head(100), use_container_width=True, hide_index=True)
            selected_tx = st.selectbox("Open transaction", entity_tx["txid"].head(500).tolist(), key=f"entity_tx_select_{entity_id}")
            if st.button("Open transaction explorer", key=f"entity_tx_open_{entity_id}"):
                navigate("transactions", txid=selected_tx)
        else:
            st.info("No transaction mapping is available for this entity.")

    if entity_id not in st.session_state["cases"]:
        if st.button("＋ Add entity to case file", key=f"case_add_{entity_id}"):
            st.session_state["cases"].append(entity_id)
            st.success("Entity added to the case workspace.")
    else:
        st.success("This entity is already in the current case workspace.")

    st.markdown(
        '<div class="soft-note">Entity scope: entity IDs are heuristic wallet clusters derived from blockchain transaction relationships. '
        'They are investigative abstractions and do not constitute confirmed real-world identity attribution.</div>',
        unsafe_allow_html=True,
    )


# =============================================================================
# Transaction explorer — evidence, not a second transaction-risk model
# =============================================================================


def build_address_flow_table(
    addresses: list[str],
    amounts: list,
    mapping: pd.DataFrame,
    alerts: pd.DataFrame,
    role: str,
) -> pd.DataFrame:
    entity_lookup = dict(zip(mapping["address"], mapping["entity_id"])) if not mapping.empty else {}
    risk_lookup = dict(zip(alerts["entity_id"], alerts["risk_priority"])) if not alerts.empty else {}
    prob_lookup = dict(zip(alerts["entity_id"], alerts["behavior_probability"])) if not alerts.empty else {}
    rows = []
    for index, address in enumerate(addresses):
        entity = entity_lookup.get(address, "UNRESOLVED")
        rows.append({
            "role": role,
            "address": address,
            "amount_btc": safe_float(amounts[index]) if index < len(amounts) else 0.0,
            "entity_id": entity,
            "entity_priority": risk_lookup.get(entity, "UNSCORED"),
            "entity_probability": safe_float(prob_lookup.get(entity, 0.0)),
        })
    return pd.DataFrame(rows)


def render_transaction_lineage(
    txid: str,
    input_entities: list[str],
    output_entities: list[str],
    alerts: pd.DataFrame,
) -> None:
    graph = nx.DiGraph()
    tx_node = f"tx:{txid}"
    for entity in input_entities:
        graph.add_edge(entity, tx_node)
    for entity in output_entities:
        graph.add_edge(tx_node, entity)
    if graph.number_of_nodes() <= 1:
        return

    risk_lookup = dict(zip(alerts["entity_id"], alerts["risk_priority"]))
    prob_lookup = dict(zip(alerts["entity_id"], alerts["behavior_probability"]))
    positions = nx.spring_layout(graph, seed=9, k=1.05)

    edge_x, edge_y = [], []
    for source, target in graph.edges():
        x0, y0 = positions[source]
        x1, y1 = positions[target]
        edge_x.extend([x0, x1, None])
        edge_y.extend([y0, y1, None])

    edges = go.Scatter(
        x=edge_x, y=edge_y, mode="lines", hoverinfo="none",
        line=dict(width=1.3, color="rgba(120,137,165,.55)"),
    )

    node_x, node_y, colors, sizes, hover = [], [], [], [], []
    for node in graph.nodes():
        node_x.append(positions[node][0])
        node_y.append(positions[node][1])
        if node == tx_node:
            colors.append("#2B5BE7")
            sizes.append(28)
            hover.append(f"<b>Transaction</b><br>{txid}")
        else:
            risk = risk_lookup.get(node, "UNKNOWN")
            colors.append(RISK_COLORS.get(risk, RISK_COLORS["UNKNOWN"]))
            sizes.append(16)
            hover.append(
                f"<b>{node}</b><br>Priority: {risk}<br>Behavior probability: {safe_float(prob_lookup.get(node, 0.0)):.4f}"
            )

    nodes = go.Scatter(
        x=node_x, y=node_y, mode="markers", hoverinfo="text", text=hover,
        marker=dict(color=colors, size=sizes, line=dict(width=1, color="#445066")),
    )
    fig = go.Figure([edges, nodes])
    fig.update_layout(
        height=390,
        showlegend=False,
        margin=dict(l=10, r=10, t=30, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
        hoverlabel=dict(bgcolor="#11151D", font_color="#FFFFFF"),
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


def render_transaction_explorer(alerts: pd.DataFrame, mapping: pd.DataFrame) -> None:
    section_title(
        "Transaction explorer",
        "Inspect blockchain flow, resolved entities and P2P evidence. Transaction risk is not separately inferred here.",
    )
    tx = load_transactions()
    if tx.empty:
        st.warning("data/raw/blockchain_transactions.csv is unavailable.")
        return

    query = st.text_input("Transaction ID", value=st.session_state.get("selected_txid") or "", key="tx_query").strip()
    if not query:
        st.info("Enter a transaction ID to inspect its flow and supporting network evidence.")
        return

    matches = tx[tx["txid"] == query]
    if matches.empty:
        fuzzy = tx[tx["txid"].str.contains(query, case=False, na=False)].head(20)
        if fuzzy.empty:
            st.warning("No transaction matched that ID.")
            return
        query = st.selectbox("Matching transactions", fuzzy["txid"].tolist())
        matches = tx[tx["txid"] == query]

    st.session_state["selected_txid"] = query
    record = matches.iloc[0]
    inputs = [str(v) for v in parse_list_field(record.get("input_addresses"))]
    outputs = [str(v) for v in parse_list_field(record.get("output_addresses"))]
    input_amounts = parse_list_field(record.get("input_amounts_btc"))
    output_amounts = parse_list_field(record.get("output_amounts_btc"))

    input_frame = build_address_flow_table(inputs, input_amounts, mapping, alerts, "INPUT")
    output_frame = build_address_flow_table(outputs, output_amounts, mapping, alerts, "OUTPUT")
    flow = pd.concat([input_frame, output_frame], ignore_index=True)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Inputs", len(inputs))
    c2.metric("Outputs", len(outputs))
    c3.metric("Input BTC", f"{sum(safe_float(v) for v in input_amounts):.6f}")
    c4.metric("Output BTC", f"{sum(safe_float(v) for v in output_amounts):.6f}")
    c5.metric("Fee BTC", f"{safe_float(record.get('fee_btc')):.8f}")

    left, right = st.columns([1.1, 1])
    with left:
        section_title("Resolved address flow")
        st.dataframe(flow, use_container_width=True, hide_index=True, height=360)
    with right:
        section_title("Transaction lineage")
        input_entities = sorted({e for e in input_frame.get("entity_id", []) if e != "UNRESOLVED"})
        output_entities = sorted({e for e in output_frame.get("entity_id", []) if e != "UNRESOLVED"})
        render_transaction_lineage(query, input_entities, output_entities, alerts)

    section_title("Transaction facts")
    facts = pd.DataFrame([
        ("Timestamp", record.get("tx_timestamp", "N/A")),
        ("Block height", record.get("block_height", "N/A")),
        ("Coinbase / seed", record.get("is_coinbase", "N/A")),
        ("Script type", record.get("script_type", "N/A")),
    ], columns=["Field", "Value"])
    st.dataframe(facts, use_container_width=True, hide_index=True)

    section_title("P2P correlation evidence")
    corr = load_correlations_minimal()
    if corr.empty:
        st.info("No correlations.csv evidence is available.")
    else:
        mask = pd.Series(False, index=corr.index)
        if "candidate_txid" in corr.columns:
            mask |= corr["candidate_txid"].eq(query)
        if "observed_txid" in corr.columns:
            mask |= corr["observed_txid"].eq(query)
        evidence = corr[mask].copy()
        if evidence.empty:
            st.caption("No P2P correlation rows reference this transaction.")
        else:
            m1, m2, m3 = st.columns(3)
            m1.metric("Correlation rows", f"{len(evidence):,}")
            m2.metric("Unique source IPs", f"{evidence['src_ip'].nunique():,}" if "src_ip" in evidence.columns else "N/A")
            m3.metric("Unique destination IPs", f"{evidence['dst_ip'].nunique():,}" if "dst_ip" in evidence.columns else "N/A")
            st.dataframe(evidence.head(250), use_container_width=True, hide_index=True)


# =============================================================================
# Entity graph page
# =============================================================================


def render_graph_page(alerts: pd.DataFrame) -> None:
    section_title("Entity transaction-flow graph", "Explore resolved entities and their direct transaction relationships.")
    graph = load_entity_graph()
    if graph.number_of_nodes() == 0:
        st.warning("Entity graph is unavailable.")
        return

    candidates = alerts[alerts["risk_priority"].isin(["CRITICAL", "HIGH", "MEDIUM"])]["entity_id"].head(4000).tolist()
    default = st.session_state.get("selected_entity")
    index = candidates.index(default) if default in candidates else 0
    entity = st.selectbox("Focus entity", candidates, index=index, key="graph_entity")
    radius = st.slider("Neighborhood radius", 1, 2, 1)
    max_nodes = st.slider("Maximum nodes", 30, 140, 90, 10)

    st.session_state["selected_entity"] = entity
    sub = entity_neighborhood(entity, radius=radius, max_nodes=max_nodes)
    draw_entity_graph(sub, alerts, f"Entity neighborhood around {entity}", {entity}, height=650)

    if sub.number_of_nodes() > 1:
        neighbor_rows = []
        for node in sub.nodes():
            if str(node) == entity:
                continue
            alert = get_entity_alert(str(node), alerts)
            neighbor_rows.append({
                "entity_id": str(node),
                "degree_in_view": sub.degree[node],
                "risk_priority": "UNSCORED" if alert is None else alert.get("risk_priority"),
                "behavior_probability": 0.0 if alert is None else safe_float(alert.get("behavior_probability")),
            })
        neighbors = pd.DataFrame(neighbor_rows)
        if not neighbors.empty:
            neighbors = neighbors.sort_values(["behavior_probability", "degree_in_view"], ascending=False)
            section_title("Connected entities")
            st.dataframe(neighbors.head(100), use_container_width=True, hide_index=True)


# =============================================================================
# Temporal analysis
# =============================================================================


def render_temporal_page(alerts: pd.DataFrame, features: pd.DataFrame) -> None:
    section_title("Temporal analysis", "Inspect rapid forwarding, successor timing and transaction chronology for a selected entity.")
    candidates = alerts[alerts["risk_priority"].isin(["CRITICAL", "HIGH", "MEDIUM"])]["entity_id"].head(4000).tolist()
    if not candidates:
        st.info("No investigation entities are available.")
        return
    default = st.session_state.get("selected_entity")
    index = candidates.index(default) if default in candidates else 0
    entity = st.selectbox("Entity", candidates, index=index, key="temporal_entity")
    st.session_state["selected_entity"] = entity

    feat = get_entity_features(entity, features)
    alert = get_entity_alert(entity, alerts)

    metrics = [
        ("Rapid chain depth", feature_value(feat, "v61_rapid_chain_depth")),
        ("Min successor delay (s)", feature_value(feat, "v61_min_successor_delay_sec")),
        ("Rapid successors", feature_value(feat, "v61_rapid_successor_count")),
        ("Layering successors", feature_value(feat, "v61_layering_successor_count")),
        ("Very-rapid successors", feature_value(feat, "v61_very_rapid_successor_count")),
    ]
    cols = st.columns(5)
    for col, (label, value) in zip(cols, metrics):
        col.metric(label, f"{value:.4f}" if value % 1 else f"{int(value):,}")

    roles = transaction_roles_for_entity(entity)
    tx = load_transactions()
    if not roles.empty and not tx.empty:
        timeline = roles.merge(tx, on="txid", how="left")
        timeline = timeline.dropna(subset=["parsed_timestamp"]).sort_values("parsed_timestamp")
        if not timeline.empty:
            timeline["total_output_btc"] = timeline["output_amounts_btc"].apply(
                lambda value: sum(safe_float(v) for v in parse_list_field(value))
            ) if "output_amounts_btc" in timeline.columns else 0.0

            colors = timeline["direction"].map({
                "Outgoing": "#E05252",
                "Incoming": "#43B581",
                "Incoming/Outgoing": "#2B5BE7",
            }).fillna("#94A3B8")

            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=timeline["parsed_timestamp"],
                y=timeline["total_output_btc"],
                mode="markers",
                marker=dict(color=colors, size=9, opacity=.82),
                text=[
                    f"<b>{row.txid}</b><br>{row.direction}<br>Output: {safe_float(row.total_output_btc):.6f} BTC"
                    for row in timeline.itertuples()
                ],
                hoverinfo="text",
            ))
            fig.update_layout(
                height=430,
                xaxis_title="Transaction time",
                yaxis_title="Total output BTC",
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,.35)",
                margin=dict(l=20, r=20, t=20, b=30),
                hoverlabel=dict(bgcolor="#11151D", font_color="#FFFFFF"),
            )
            section_title("Entity transaction timeline")
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
            st.caption("Green = incoming · red = outgoing · blue = both. Timing-based V6.1 successor features are summarized above.")
        else:
            st.info("No timestamped transaction rows were found for this entity.")
    else:
        st.info("No transaction-role timeline is available for this entity.")

    if alert is not None:
        section_title("Temporal reason codes")
        temporal_reasons = [
            r for r in reason_list(alert.get("reason_codes"))
            if any(token in r for token in ["RAPID", "LAYER", "BURST"])
        ]
        st.markdown(" ".join(pattern_badge(r) for r in temporal_reasons) or "No temporal reason codes active.", unsafe_allow_html=True)

    sub = entity_neighborhood(entity, radius=2, max_nodes=75)
    section_title("Flow context")
    draw_entity_graph(sub, alerts, "Two-hop flow context", {entity}, height=520)


# =============================================================================
# Network / IP evidence
# =============================================================================


def render_network_page(alerts: pd.DataFrame, features: pd.DataFrame) -> None:
    section_title("Network / IP evidence", "Review P2P relay observations associated with transactions touching an entity.")
    candidates = alerts[alerts["risk_priority"].isin(["CRITICAL", "HIGH", "MEDIUM"])]["entity_id"].head(4000).tolist()
    if not candidates:
        st.info("No investigation entities are available.")
        return
    default = st.session_state.get("selected_entity")
    index = candidates.index(default) if default in candidates else 0
    entity = st.selectbox("Entity", candidates, index=index, key="network_entity")
    st.session_state["selected_entity"] = entity

    feat = get_entity_features(entity, features)
    network_features = evidence_rows(feat, "Network / IP")
    if not network_features.empty:
        section_title("Entity-level network behavior")
        st.dataframe(network_features, use_container_width=True, hide_index=True)

    txids = set(txids_for_entity(entity))
    corr = load_correlations_minimal()
    if not txids or corr.empty:
        st.info("No raw correlation evidence is available for this entity.")
        return

    mask = pd.Series(False, index=corr.index)
    if "candidate_txid" in corr.columns:
        mask |= corr["candidate_txid"].isin(txids)
    if "observed_txid" in corr.columns:
        mask |= corr["observed_txid"].isin(txids)
    evidence = corr[mask].copy()

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Correlation rows", f"{len(evidence):,}")
    c2.metric("Unique source IPs", f"{evidence['src_ip'].nunique():,}" if "src_ip" in evidence.columns else "N/A")
    c3.metric("Unique destination IPs", f"{evidence['dst_ip'].nunique():,}" if "dst_ip" in evidence.columns else "N/A")
    c4.metric("Entity transactions", f"{len(txids):,}")

    if evidence.empty:
        st.caption("No correlation rows reference this entity's transactions.")
        return

    left, right = st.columns([1, 1])
    with left:
        if "src_ip" in evidence.columns:
            counts = evidence["src_ip"].value_counts().head(15).sort_values()
            fig = go.Figure(go.Bar(x=counts.values, y=counts.index, orientation="h"))
            fig.update_layout(
                title="Most observed source IPs",
                height=390,
                margin=dict(l=20, r=20, t=45, b=20),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,.35)",
                xaxis_title="Correlation events",
                yaxis_title="",
            )
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})
    with right:
        enrichment = load_ip_enrichment()
        if not enrichment.empty and "src_ip" in evidence.columns and "ip" in enrichment.columns:
            ips = pd.DataFrame({"ip": evidence["src_ip"].dropna().astype(str).unique()})
            enriched = ips.merge(enrichment, on="ip", how="left")
            section_title("Source IP enrichment")
            st.dataframe(enriched.head(100), use_container_width=True, hide_index=True, height=330)
        else:
            st.info("IP enrichment data is unavailable.")

    section_title("Raw P2P evidence")
    st.dataframe(evidence.head(300), use_container_width=True, hide_index=True)


# =============================================================================
# Behavioral patterns / triage analytics
# =============================================================================


def render_patterns_page(alerts: pd.DataFrame) -> None:
    section_title("Behavioral patterns", "Population-level view of the reason codes and signals driving the investigation queue.")

    counter = Counter()
    active = alerts[alerts["risk_priority"].isin(["CRITICAL", "HIGH", "MEDIUM"])]
    for value in active["reason_codes"]:
        counter.update(reason_list(value))
    pattern_df = pd.DataFrame(counter.most_common(), columns=["Pattern", "Entities"])
    if not pattern_df.empty:
        pattern_df["Label"] = pattern_df["Pattern"].map(pattern_label)
        top = pattern_df.head(15).sort_values("Entities")
        fig = go.Figure(go.Bar(x=top["Entities"], y=top["Label"], orientation="h"))
        fig.update_layout(
            height=430,
            margin=dict(l=20, r=20, t=20, b=30),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(255,255,255,.35)",
            xaxis_title="Entities",
            yaxis_title="",
        )
        st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    section_title("Behavior probability vs novelty", "Upper-right entities combine known suspicious behavior with strong novelty.")
    non_low = alerts[alerts["risk_priority"].isin(["CRITICAL", "HIGH", "MEDIUM"])]
    low = alerts[alerts["risk_priority"] == "LOW"]
    if len(low) > 3000:
        low = low.sample(3000, random_state=42)
    plot_df = pd.concat([non_low, low], ignore_index=True)

    fig = go.Figure()
    for risk in ["LOW", "MEDIUM", "HIGH", "CRITICAL"]:
        subset = plot_df[plot_df["risk_priority"] == risk]
        if subset.empty:
            continue
        fig.add_trace(go.Scattergl(
            x=subset["behavior_probability"],
            y=subset["novelty_percentile"] * 100,
            mode="markers",
            name=risk,
            marker=dict(color=RISK_COLORS[risk], size=6 if risk == "LOW" else 8, opacity=.58 if risk == "LOW" else .82),
            text=[f"{eid}<br>{reason}" for eid, reason in zip(subset["entity_id"], subset["reason_codes"])],
            hovertemplate="%{text}<br>Behavior=%{x:.3f}<br>Novelty=%{y:.2f}%<extra></extra>",
        ))
    fig.update_layout(
        height=500,
        xaxis_title="Behavior probability",
        yaxis_title="Novelty percentile",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(255,255,255,.35)",
        margin=dict(l=20, r=20, t=20, b=35),
        legend_title="Risk priority",
    )
    st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})

    importance = load_feature_importance()
    if not importance.empty:
        # Support common column names produced by the benchmark script.
        feature_col = next((c for c in importance.columns if "feature" in c.lower()), None)
        importance_col = next((c for c in importance.columns if "importance" in c.lower()), None)
        if feature_col and importance_col:
            top_imp = importance[[feature_col, importance_col]].head(15).sort_values(importance_col)
            section_title("Top behavioral features")
            fig = go.Figure(go.Bar(
                x=top_imp[importance_col],
                y=top_imp[feature_col].astype(str).str.replace("v61_", "", regex=False).str.replace("v6_", "", regex=False),
                orientation="h",
            ))
            fig.update_layout(
                height=460,
                margin=dict(l=20, r=20, t=20, b=30),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(255,255,255,.35)",
                xaxis_title="Random Forest importance",
                yaxis_title="",
            )
            st.plotly_chart(fig, use_container_width=True, config={"displaylogo": False})


# =============================================================================
# Case workspace
# =============================================================================


def render_cases_page(alerts: pd.DataFrame, features: pd.DataFrame, mapping: pd.DataFrame) -> None:
    section_title("Investigation case files", "Collect entities, add investigator notes and export a portable evidence package.")

    st.session_state["case_name"] = st.text_input("Case name", value=st.session_state["case_name"])
    st.session_state["case_notes"] = st.text_area(
        "Investigator notes",
        value=st.session_state["case_notes"],
        height=110,
        placeholder="Record hypotheses, follow-up steps, or observations…",
    )

    cases = list(st.session_state["cases"])
    if not cases:
        st.info("No entities have been added yet. Open an entity case file and choose 'Add entity to case file'.")
        return

    case_rows = alerts[alerts["entity_id"].isin(cases)].copy()
    columns = [
        c for c in ["entity_id", "risk_priority", "behavior_probability", "novelty_percentile", "reason_codes"]
        if c in case_rows.columns
    ]
    render_dark_table(case_rows[columns], max_rows=100, reason_columns={"reason_codes"})

    selected = st.selectbox("Selected case entity", cases)
    c1, c2 = st.columns(2)
    with c1:
        if st.button("Open entity investigation", use_container_width=True):
            navigate("entity_search", entity_id=selected)
    with c2:
        if st.button("Remove selected entity", use_container_width=True):
            st.session_state["cases"] = [e for e in cases if e != selected]
            st.rerun()

    export_records = []
    for entity in cases:
        alert = get_entity_alert(entity, alerts)
        feat = get_entity_features(entity, features)
        selected_features = {}
        if feat is not None:
            for group_name in EVIDENCE_GROUPS:
                for _, aliases in EVIDENCE_GROUPS[group_name]:
                    for feature_name in aliases:
                        if feature_name in feat.index:
                            selected_features[feature_name] = safe_float(feat.get(feature_name))
                            break
        export_records.append({
            "entity_id": entity,
            "risk_priority": None if alert is None else alert.get("risk_priority"),
            "behavior_probability": None if alert is None else safe_float(alert.get("behavior_probability")),
            "novelty_percentile": None if alert is None else safe_float(alert.get("novelty_percentile")),
            "reason_codes": [] if alert is None else reason_list(alert.get("reason_codes")),
            "wallet_addresses": addresses_for_entity(entity, mapping),
            "transaction_ids": txids_for_entity(entity),
            "selected_behavior_features": selected_features,
        })

    package = {
        "case_name": st.session_state["case_name"],
        "investigator_notes": st.session_state["case_notes"],
        "entity_count": len(export_records),
        "entities": export_records,
        "scope_note": (
            "Entity IDs are heuristic wallet clusters. Risk outputs are investigative triage signals and do not constitute identity attribution or proof of criminality."
        ),
    }

    c1, c2 = st.columns(2)
    with c1:
        st.download_button(
            "Download case JSON",
            data=json.dumps(package, indent=2),
            file_name="chain_custody_case.json",
            mime="application/json",
            use_container_width=True,
        )
    with c2:
        csv_export = case_rows[columns].to_csv(index=False) if not case_rows.empty else ""
        st.download_button(
            "Download entity list CSV",
            data=csv_export,
            file_name="chain_custody_case_entities.csv",
            mime="text/csv",
            use_container_width=True,
        )


# =============================================================================
# Main
# =============================================================================


def main() -> None:
    st.set_page_config(
        page_title="Chain Custody | Bitcoin Forensics V6.1",
        page_icon="⛨",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    inject_theme()
    initialize_state()

    alerts = load_alerts()
    if alerts.empty:
        st.error("Final V6.1 entity alerts are missing or invalid.")
        st.code("python src/final_detector_v61.py", language="bash")
        st.caption(f"Expected file: {ALERT_FILE}")
        st.stop()

    mapping = load_address_entity()

    render_sidebar(alerts)
    render_header()
    render_top_controls(alerts, mapping)

    section = st.session_state["section"]

    # Heavy feature matrix is loaded only on pages that actually need it.
    if section == "overview":
        render_overview(alerts)
    elif section == "alerts":
        render_alerts(alerts)
    elif section == "entity_search":
        render_entity_search(alerts, load_behavior_features(), mapping)
    elif section == "transactions":
        render_transaction_explorer(alerts, mapping)
    elif section == "graph":
        render_graph_page(alerts)
    elif section == "temporal":
        render_temporal_page(alerts, load_behavior_features())
    elif section == "network":
        render_network_page(alerts, load_behavior_features())
    elif section == "patterns":
        render_patterns_page(alerts)
    elif section == "cases":
        render_cases_page(alerts, load_behavior_features(), mapping)
    else:
        st.session_state["section"] = "overview"
        st.rerun()

    st.divider()
    st.caption(
        "Chain Custody SIH_V6 demonstration console · Offline synthetic evaluation environment. "
        "Entity clusters are heuristic investigative groupings. Ground-truth labels are used for supervised "
        "training/evaluation only and are not part of V6/V6.1 feature construction or operational evidence views."
    )


if __name__ == "__main__":
    main()
