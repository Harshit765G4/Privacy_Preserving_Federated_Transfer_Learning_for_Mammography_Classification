"""
======================================================================
Project:
Privacy-Preserving Federated Transfer Learning
for Mammography Classification

Script:
create_patient_folds.py

Purpose:
Create patient-wise stratified 5-fold cross validation.

Author:
Harshit Garg
======================================================================
"""

from pathlib import Path
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_FILE = PROCESSED_DIR / "mammogram_dataset.csv"

OUTPUT_FILE = PROCESSED_DIR / "mammogram_dataset_with_folds.csv"

REPORT_FILE = PROCESSED_DIR / "fold_summary.txt"

# ============================================================
# LOAD
# ============================================================

print("=" * 70)
print("CREATING PATIENT-WISE FOLDS")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"Loaded Mammograms : {len(df)}")

# ============================================================
# CREATE FOLDS
# ============================================================

sgkf = StratifiedGroupKFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

df["fold"] = -1

X = df["processed_image_path"]
y = df["label"]
groups = df["patient_id"]

for fold, (_, val_idx) in enumerate(sgkf.split(X, y, groups)):
    df.loc[val_idx, "fold"] = fold

# ============================================================
# SAVE DATASET
# ============================================================

df.to_csv(OUTPUT_FILE, index=False)

# ============================================================
# CREATE REPORT
# ============================================================

lines = []

lines.append("=" * 60)
lines.append("PATIENT-WISE FOLD SUMMARY")
lines.append("=" * 60)
lines.append("")

for fold in sorted(df["fold"].unique()):

    fold_df = df[df["fold"] == fold]

    benign = (fold_df["label"] == 0).sum()
    malignant = (fold_df["label"] == 1).sum()

    patients = fold_df["patient_id"].nunique()

    lines.append(f"Fold {fold}")
    lines.append(f"Images      : {len(fold_df)}")
    lines.append(f"Patients    : {patients}")
    lines.append(f"Benign      : {benign}")
    lines.append(f"Malignant   : {malignant}")
    lines.append("")

with open(REPORT_FILE, "w") as f:
    f.write("\n".join(lines))

# ============================================================
# SANITY CHECK
# ============================================================

duplicate_patients = 0

for patient in df["patient_id"].unique():

    if df[df["patient_id"] == patient]["fold"].nunique() > 1:
        duplicate_patients += 1

print()

print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"Total Mammograms : {len(df)}")
print(f"Unique Patients  : {df['patient_id'].nunique()}")
print(f"Duplicate Patients Across Folds : {duplicate_patients}")

print()

print("Fold Distribution")

print(df["fold"].value_counts().sort_index())

print()

print("Saved Dataset")

print(OUTPUT_FILE)

print()

print("Saved Report")

print(REPORT_FILE)

print("=" * 70)