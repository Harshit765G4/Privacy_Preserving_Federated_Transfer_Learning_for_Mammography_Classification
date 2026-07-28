"""
=========================================================
Project:
Privacy-Preserving Federated Transfer Learning
for Mammography Classification

Script:
create_master_labels.py

Purpose:
Create the final clean master_labels.csv for all
downstream ML and Federated Learning experiments.

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

INPUT_FILE = PROCESSED_DIR / "mapped_jpeg_images.csv"

OUTPUT_FILE = PROCESSED_DIR / "master_labels.csv"

# --------------------------------------------------------
# Load
# --------------------------------------------------------

print("=" * 70)
print("CREATING FINAL MASTER LABELS DATASET")
print("=" * 70)

df = pd.read_csv(INPUT_FILE)

print(f"Loaded Records : {len(df)}")

# --------------------------------------------------------
# Merge breast density columns
# --------------------------------------------------------

if "breast density" in df.columns:
    df["breast_density"] = df["breast_density"].fillna(df["breast density"])
    df.drop(columns=["breast density"], inplace=True)

# --------------------------------------------------------
# Remove unwanted columns
# --------------------------------------------------------

drop_columns = [
    "SeriesInstanceUID_x",
    "SeriesInstanceUID_y",
    "cropped image file path",
    "ROI mask file path",
    "image file path",
    "image_exists",
    "SeriesDescription",
]

for col in drop_columns:
    if col in df.columns:
        df.drop(columns=col, inplace=True)

# --------------------------------------------------------
# Remove duplicate JPEG path column
# --------------------------------------------------------

if "image_path" in df.columns:
    df.drop(columns=["image_path"], inplace=True)

# --------------------------------------------------------
# Rename columns
# --------------------------------------------------------

rename_columns = {
    "left or right breast": "breast_side",
    "image view": "image_view",
    "abnormality type": "abnormality_type",
    "local_jpg_path": "image_path",
}

df.rename(columns=rename_columns, inplace=True)

# --------------------------------------------------------
# Select final columns
# --------------------------------------------------------

master_df = df[
    [
        "patient_id",
        "pathology",
        "abnormality_type",
        "dataset_type",
        "dataset_split",
        "breast_density",
        "breast_side",
        "image_view",
        "assessment",
        "subtlety",
        "full_image_series_uid",
        "image_path",
    ]
].copy()

# --------------------------------------------------------
# Dataset Quality Checks
# --------------------------------------------------------

print("\nChecking Missing Values\n")

missing = master_df.isnull().sum()

print(missing)

print("\nChecking Image Path Existence...")

master_df["image_exists"] = master_df["image_path"].apply(
    lambda x: Path(x).exists()
)

missing_images = (~master_df["image_exists"]).sum()

print(f"Missing Images : {missing_images}")

duplicate_paths = master_df.duplicated(
    subset="image_path"
).sum()

duplicate_patients = master_df.duplicated(
    subset=["patient_id"]
).sum()

print(f"Duplicate Image Paths : {duplicate_paths}")

print(f"Duplicate Patients    : {duplicate_patients}")

# --------------------------------------------------------
# Investigate duplicate image paths
# --------------------------------------------------------

duplicate_df = master_df[
    master_df.duplicated(
        subset="image_path",
        keep=False
    )
]

duplicate_report = (
    PROCESSED_DIR /
    "duplicate_image_report.csv"
)

duplicate_df.to_csv(
    duplicate_report,
    index=False
)

# --------------------------------------------------------
# Remove temporary column
# --------------------------------------------------------

master_df.drop(
    columns=["image_exists"],
    inplace=True
)

# --------------------------------------------------------
# Save
# --------------------------------------------------------

master_df.to_csv(
    OUTPUT_FILE,
    index=False
)

# --------------------------------------------------------
# Summary
# --------------------------------------------------------

print("\n" + "=" * 70)

print("MASTER DATASET SUMMARY")

print("=" * 70)

print(f"Total Images            : {len(master_df)}")

print(f"Unique Patients         : {master_df['patient_id'].nunique()}")

print("\nPathology Distribution")

print(master_df["pathology"].value_counts())

print("\nAbnormality Distribution")

print(master_df["abnormality_type"].value_counts())

print("\nImage View Distribution")

print(master_df["image_view"].value_counts())

print("\nBreast Side Distribution")

print(master_df["breast_side"].value_counts())

print("\nOutput")

print(OUTPUT_FILE)

print("\nDuplicate Report")

print(duplicate_report)

print("=" * 70)