#!/usr/bin/env python3

import importlib
import sys

PACKAGES = {
    "pandas": "pandas",
    "numpy": "numpy",
    "sklearn": "scikit-learn",
    "networkx": "networkx",
    "joblib": "joblib",
    "streamlit": "streamlit",
    "plotly": "plotly",
    "lxml": "lxml",
}

print("=" * 70)
print("SIH V6.1 ENVIRONMENT CHECK")
print("=" * 70)

print(f"\nPython: {sys.version.split()[0]}\n")

failed = []

for module, package in PACKAGES.items():
    try:
        imported = importlib.import_module(module)
        version = getattr(imported, "__version__", "installed")
        print(f"✓ {package:<20} {version}")
    except Exception:
        print(f"✗ {package:<20} MISSING")
        failed.append(package)

print()

if failed:
    print("Missing packages:")
    for package in failed:
        print(f"  - {package}")

    print("\nRun:")
    print("  python -m pip install -r requirements.txt")
    raise SystemExit(1)

print("ENVIRONMENT READY")
