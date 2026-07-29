
"""
======================================================================
Project:
Privacy-Preserving Federated Transfer Learning for Mammography Classification

Script:
create_training_dataset.py

Purpose:
Creates the final training dataset by matching metadata with
preprocessed CLAHE images.

Author:
Harshit Garg
======================================================================
"""

from pathlib import Path
import pandas as pd

# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_CSV = PROCESSED_DIR / "mammogram_dataset_with_folds.csv"

CLAHE_DIR = PROCESSED_DIR / "images_clahe"

OUTPUT_CSV = PROCESSED_DIR / "training_dataset.csv"

MISSING_REPORT = PROCESSED_DIR / "missing_processed_images.csv"

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("CREATING FINAL TRAINING DATASET")
print("=" * 70)

df = pd.read_csv(INPUT_CSV)

print(f"Metadata Records : {len(df)}")

# ============================================================
# BUILD EXPECTED CLAHE FILENAME
# ============================================================

df["clahe_filename"] = (
    df["patient_id"].astype(str)
    + "_"
    + df["breast_side"].astype(str)
    + "_"
    + df["image_view"].astype(str)
    + ".png"
)

df["clahe_path"] = df["clahe_filename"].apply(
    lambda x: str(CLAHE_DIR / x)
)

# ============================================================
# CHECK IMAGE EXISTS
# ============================================================

df["processed_exists"] = df["clahe_path"].apply(
    lambda x: Path(x).exists()
)

missing = df[~df["processed_exists"]].copy()

training_df = df[df["processed_exists"]].copy()

# ============================================================
# SAVE
# ============================================================

training_df.to_csv(OUTPUT_CSV, index=False)
missing.to_csv(MISSING_REPORT, index=False)

# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("SUMMARY")
print("=" * 70)

print(f"Original Metadata Rows : {len(df)}")
print(f"Training Rows          : {len(training_df)}")
print(f"Missing CLAHE Images   : {len(missing)}")

if len(missing):

    print("\nMissing Image Examples:\n")

    print(
        missing[
            [
                "patient_id",
                "breast_side",
                "image_view",
                "pathology"
            ]
        ].head(10)
    )

print()

print("Saved Training Dataset")

print(OUTPUT_CSV)

print()

print("Saved Missing Report")

print(MISSING_REPORT)

print("=" * 70)