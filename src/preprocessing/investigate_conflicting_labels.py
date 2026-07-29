"""
=========================================================
Project:
Privacy-Preserving Federated Transfer Learning
for Mammography Classification

Script:
investigate_conflicting_labels.py

Purpose:
Extract mammograms that have conflicting lesion labels
for manual inspection and final label policy definition.

Author:
Harshit Garg
=========================================================
"""

from pathlib import Path
import pandas as pd

# --------------------------------------------------------
# Paths
# --------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MASTER_FILE = PROCESSED_DIR / "master_labels.csv"

OUTPUT_FILE = PROCESSED_DIR / "conflicting_mammograms.csv"

# --------------------------------------------------------
# Load
# --------------------------------------------------------

print("=" * 70)
print("INVESTIGATING CONFLICTING LABELS")
print("=" * 70)

df = pd.read_csv(MASTER_FILE)

# --------------------------------------------------------
# Find duplicate images
# --------------------------------------------------------

duplicates = df[df.duplicated(subset="image_path", keep=False)].copy()

# --------------------------------------------------------
# Keep only conflicting pathologies
# --------------------------------------------------------

conflicting = duplicates.groupby("image_path").filter(
    lambda x: x["pathology"].nunique() > 1
)

# --------------------------------------------------------
# Sort nicely
# --------------------------------------------------------

conflicting = conflicting.sort_values(
    by=[
        "patient_id",
        "image_path",
        "assessment",
        "abnormality_type"
    ]
)

# --------------------------------------------------------
# Save
# --------------------------------------------------------

conflicting.to_csv(OUTPUT_FILE, index=False)

# --------------------------------------------------------
# Summary
# --------------------------------------------------------

print()

print(f"Conflicting Mammograms : {conflicting['image_path'].nunique()}")

print(f"Total Metadata Rows    : {len(conflicting)}")

print()

print("Pathology Distribution")

print(conflicting["pathology"].value_counts())

print()

print("Saved")

print(OUTPUT_FILE)

print("=" * 70)