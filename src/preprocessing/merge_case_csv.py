"""
=========================================================
Project:
Privacy-Preserving Federated Transfer Learning
for Mammography Classification

Script:
merge_case_csv.py

Purpose:
Merge all CBIS-DDSM case description CSV files into
one master metadata file.

Author:
Harshit Garg

=========================================================
"""

from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# Project Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_CSV_DIR = PROJECT_ROOT / "data" / "raw" / "csv"

OUTPUT_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_FILE = OUTPUT_DIR / "merged_case_descriptions.csv"


# ---------------------------------------------------------
# CSV Files
# ---------------------------------------------------------

CSV_FILES = [
    ("mass_case_description_train_set.csv", "train", "mass"),
    ("mass_case_description_test_set.csv", "test", "mass"),
    ("calc_case_description_train_set.csv", "train", "calc"),
    ("calc_case_description_test_set.csv", "test", "calc"),
]


# ---------------------------------------------------------
# Read and Label CSVs
# ---------------------------------------------------------

merged_data = []

print("=" * 60)
print("CBIS-DDSM CASE DESCRIPTION MERGE")
print("=" * 60)

for filename, split, abnormality in CSV_FILES:

    file_path = RAW_CSV_DIR / filename

    if not file_path.exists():
        raise FileNotFoundError(f"\nMissing file:\n{file_path}")

    df = pd.read_csv(file_path)

    df["dataset_split"] = split
    df["dataset_type"] = abnormality

    merged_data.append(df)

    print(f"{filename:<40} {len(df):>6} rows")


# ---------------------------------------------------------
# Merge
# ---------------------------------------------------------

master_df = pd.concat(merged_data, ignore_index=True)


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

master_df.to_csv(OUTPUT_FILE, index=False)


# ---------------------------------------------------------
# Summary
# ---------------------------------------------------------

print("\n" + "-" * 60)
print(f"Total Records : {len(master_df)}")
print(f"Total Columns : {master_df.shape[1]}")
print(f"Output File   : {OUTPUT_FILE}")

print("-" * 60)

print("\nDataset Split Distribution")

print(master_df["dataset_split"].value_counts())

print("\nDataset Type Distribution")

print(master_df["dataset_type"].value_counts())

print("\nPathology Distribution")

print(master_df["pathology"].value_counts())

print("\nMerge completed successfully.")
