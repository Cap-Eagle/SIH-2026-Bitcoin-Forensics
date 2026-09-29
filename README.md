# SIH 2026 — SIH26146 Bitcoin Forensics V6.1

An offline Bitcoin forensic intelligence system for correlating blockchain and P2P network evidence, resolving transaction behavior, detecting suspicious activity, and generating explainable investigative leads.

---

## Goal

Build an end-to-end forensic pipeline capable of transforming raw Bitcoin transaction and network metadata into prioritized investigative alerts.

```text
Blockchain transactions + P2P network events
                    │
                    ▼
         Blockchain ↔ P2P Correlation
                    │
                    ▼
          Transaction Graph Construction
                    │
                    ▼
              Entity Resolution
                    │
                    ▼
             Feature Engineering
        V3 base + V6 + V6.1 behavior
                    │
                    ▼
        ┌─────────────────────────┐
        │ Random Forest Detector  │
        │ Isolation Forest        │
        │ Novelty Signal          │
        └────────────┬────────────┘
                     │
                     ▼
          Explainable Risk Scoring
                     │
                     ▼
       CRITICAL / HIGH / MEDIUM / LOW
                     │
                     ▼
        Forensic Investigation Dashboard
```

---

## Key Capabilities

- Bitcoin transaction graph construction
- Blockchain/P2P network correlation
- Address and entity resolution
- Behavioral feature engineering
- Temporal transaction analysis
- Rapid forwarding detection
- Multi-hop chain analysis
- Layering-pattern detection
- Structuring-pattern detection
- Peel-chain behavior analysis
- Fan-in and fan-out detection
- Shared source-IP analysis
- Network evidence aggregation
- Random Forest behavioral classification
- Isolation Forest novelty detection
- Explainable reason codes
- Risk-based investigative prioritization
- Interactive Streamlit investigation dashboard
- End-to-end project validation
- Synthetic dataset generation for reproducibility

---

# Quick Start

## Linux / macOS

Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
python -m pip install -r requirements.txt
```

Run the complete project:

```bash
chmod +x run_full_project.sh run_pipeline.sh
./run_full_project.sh
```

Launch the dashboard:

```bash
streamlit run app/dashboard.py
```

---

## Windows

Create and activate a virtual environment:

```bat
python -m venv .venv
.venv\Scripts\activate
```

Install dependencies:

```bat
python -m pip install -r requirements.txt
```

Run the complete project:

```bat
run_full_project.bat
```

Launch the dashboard:

```bat
streamlit run app\dashboard.py
```

---

# Dataset

Large generated datasets and intermediate artifacts do not need to be stored directly in Git.

The repository contains:

```text
generate_dataset.py
```

which generates the synthetic dataset required for the complete demonstration pipeline.

The generated operational inputs include:

```text
data/raw/blockchain_transactions.csv
data/raw/network_events.csv
data/raw/ip_enrichment.csv

data/ground_truth/ground_truth.csv
data/ground_truth/entity_wallet_map.csv
```

Existing operational data can be checked with:

```bash
python src/ingest_data.py --validate-existing
```

---

# Important Modelling Choice

The authoritative Bitcoin transaction representation preserves:

```text
Address → Transaction → Address
```

This preserves multi-input and multi-output Bitcoin transactions without inventing an arbitrary mapping between individual transaction inputs and outputs.

Coinbase/seed transactions can therefore remain represented without creating normal incoming address relationships where no input address exists.

Entity resolution is performed separately to construct the entity-level representation used for behavioral analysis.

---

# Behavioral Intelligence Features

The final detector uses **83 V6/V6.1 behavioral features**.

They capture multiple classes of forensic behavior, including:

### Transaction Activity

- incoming/outgoing transaction activity
- transaction inter-arrival timing
- burst behavior
- rapid spending

### Flow Behavior

- flow-through behavior
- amount concentration
- amount entropy
- repeated output amounts
- input/output structure

### Graph Behavior

- fan-in patterns
- fan-out patterns
- transaction successors
- multi-hop behavior
- temporal outgoing relationships

### Temporal Behavior

- successor delays
- rapid successor activity
- rapid-chain depth
- transaction ancestry timing

### Layering and Structuring

- structuring-band activity
- repeated structured amounts
- layering successors
- amount retention across transaction chains

### Peel Behavior

- small peel outputs
- peel timing
- peel amount characteristics

### Network Evidence

- source-IP behavior
- destination-IP evidence
- country/ASN evidence
- Tor-related events
- hosting-related events
- blockchain/network timing correlation
- cross-entity IP sharing

---

# Detection Architecture

## Behavioral Detector

The primary detector is a Random Forest classifier operating on the V6/V6.1 behavioral feature representation.

It identifies patterns represented by the labelled evaluation data.

## Novelty Detector

An Isolation Forest provides a secondary unsupervised novelty signal.

This allows the system to highlight entities whose behavior differs significantly from the observed behavioral population.

The novelty detector supplements the primary behavioral classifier rather than replacing it.

---

# Explainable Alerts

The final detector generates:

```text
outputs/final_entity_alerts_v61.csv
```

Each entity receives information including:

```text
entity_id
behavior_probability
novelty_percentile
risk_priority
reason_codes
```

Risk priorities are represented as:

```text
CRITICAL
HIGH
MEDIUM
LOW
```

Example explanation codes include:

```text
RAPID_FORWARDING
RAPID_MULTI_HOP_CHAIN
LAYERING_PATTERN
TRANSACTION_BURST
UNSUPERVISED_NOVELTY
```

This allows investigators to understand why an entity was prioritized instead of receiving only an opaque anomaly score.

---

# Evaluation

The supplied synthetic benchmark contains:

```text
Labelled entities : 1,140
Benign            : 750
Anomalous         : 390
```

The final held-out benchmark produced:

| Metric | Result |
|---|---:|
| Accuracy | 99.42% |
| Precision | 100.00% |
| Recall | 98.31% |
| F1 | 99.15% |
| ROC-AUC | 99.98% |
| Average Precision | 99.97% |

Confusion matrix:

```text
TN = 112
FP = 0
FN = 1
TP = 58
```

These results apply to the supplied synthetic evaluation dataset and should **not** be interpreted as equivalent performance on arbitrary real-world Bitcoin traffic.

After held-out evaluation, the deployment model is retrained using all available labelled entities.

---

# Project Validation

The repository includes an integrity validator:

```bash
python src/validate_project.py
```

A successful project build should end with:

```text
PROJECT VALIDATION PASSED
V6.1 artifacts are internally consistent.
```

The validator checks areas including:

- project structure
- required input files
- pipeline outputs
- model artifacts
- entity-table consistency
- behavioral feature counts
- missing feature values
- alert schema
- probability ranges
- risk labels
- ground-truth readability

---

# Repository Structure

```text
SIH_V6/
│
├── app/
│   └── dashboard.py
│
├── src/
│   ├── ingest_data.py
│   ├── correlate_csv.py
│   ├── build_group_b_handoff.py
│   ├── build_graph.py
│   ├── resolve_entities.py
│   ├── build_features_v3.py
│   ├── resolve_ground_truth_v3.py
│   ├── build_behavior_features_v6.py
│   ├── build_behavior_features_v61.py
│   ├── train_anomaly_model_v5.py
│   ├── final_detector_v61.py
│   ├── benchmark_behavior_v61.py
│   └── validate_project.py
│
├── data/
│   ├── raw/
│   ├── correlated/
│   └── ground_truth/
│
├── outputs/
│
├── models/
│
├── generate_dataset.py
├── requirements.txt
├── run_pipeline.sh
├── run_pipeline.bat
├── run_full_project.sh
├── run_full_project.bat
└── README.md
```

---

# Pipeline Stages

The complete V6.1 pipeline executes:

```text
01  Blockchain ↔ P2P Correlation
02  Group B Correlation Handoff
03  Blockchain Graph Construction
04  Entity Resolution
05  V3 Base Entity Feature Engineering
06  Ground-Truth Entity Resolution
07  V6 Behavioral Feature Engineering
08  V6.1 Temporal / Cross-Entity Feature Engineering
09  Isolation Forest Novelty Model
10  Final V6.1 Forensic Detector
```

Linux/macOS:

```bash
./run_pipeline.sh
```

Windows:

```bat
run_pipeline.bat
```

---

# Dashboard

Launch the investigation console using:

```bash
streamlit run app/dashboard.py
```

The dashboard provides an investigator-oriented interface for exploring risk-prioritized entities, behavioral explanations, graph relationships, network evidence, and forensic leads.

Model evaluation is deliberately kept separate from the primary investigation workflow so that the operational interface focuses on investigation rather than ML experimentation.

---

# Team Development

The project is structured so individual pipeline stages can also be executed independently.

For example:

```bash
python src/correlate_csv.py
python src/build_behavior_features_v6.py
python src/build_behavior_features_v61.py
python src/final_detector_v61.py
```

This allows team members to work on correlation, graph analysis, feature engineering, detection, or visualization independently while retaining a common end-to-end pipeline.

---

# Reproducibility

The repository is designed so that a fresh clone can reconstruct the demonstration environment from source rather than requiring hundreds of megabytes of generated CSV artifacts to be committed to Git.

The intended workflow is:

```text
Clone repository
      ↓
Install dependencies
      ↓
Generate dataset
      ↓
Validate inputs
      ↓
Run forensic pipeline
      ↓
Validate artifacts
      ↓
Launch dashboard
```

---

# Scope and Limitations

The included dataset is synthetic and intended for system development, evaluation, and demonstration.

The system produces investigative leads and behavioral risk indicators. A high-risk classification should therefore be treated as evidence for further investigation rather than proof that a wallet or entity is controlled by a criminal actor.

Real-world deployment would require additional data-quality controls, blockchain data ingestion, network observation infrastructure, threat-intelligence integration, model monitoring, and operational validation.
