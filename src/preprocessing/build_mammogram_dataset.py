"""
======================================================================
Project:
Privacy-Preserving Federated Transfer Learning
for Mammography Classification

Script:
build_mammogram_dataset.py

Purpose:
Convert lesion-level annotations into mammogram-level dataset.

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

MASTER_FILE = PROCESSED_DIR / "master_labels_clahe.csv"

OUTPUT_DATASET = PROCESSED_DIR / "mammogram_dataset.csv"

OUTPUT_REPORT = PROCESSED_DIR / "mammogram_dataset_report.txt"

# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("BUILDING MAMMOGRAM DATASET")
print("=" * 70)

df = pd.read_csv(MASTER_FILE)

print(f"Loaded Records : {len(df)}")

# ============================================================
# Merge BENIGN_WITHOUT_CALLBACK
# ============================================================

df["pathology"] = df["pathology"].replace(
    "BENIGN_WITHOUT_CALLBACK",
    "BENIGN"
)

# ============================================================
# Build Mammogram Dataset
# ============================================================

records = []

for image_path, group in df.groupby("image_path"):

    if "MALIGNANT" in group["pathology"].values:
        final_pathology = "MALIGNANT"
        label = 1
    else:
        final_pathology = "BENIGN"
        label = 0

    first = group.iloc[0]

    records.append({

        "patient_id": first["patient_id"],

        "image_path": image_path,

        "processed_image_path":
            first["processed_image_path"],

        "pathology": final_pathology,

        "label": label,

        "breast_side": first["breast_side"],

        "image_view": first["image_view"],

        "breast_density": first["breast_density"],

        "dataset_type": first["dataset_type"],

        "dataset_split": first["dataset_split"],

        "assessment": group["assessment"].max(),

        "subtlety": group["subtlety"].max(),

        "abnormality_count": len(group)

    })

mammogram_df = pd.DataFrame(records)

# ============================================================
# SAVE CSV
# ============================================================

mammogram_df.to_csv(
    OUTPUT_DATASET,
    index=False
)

# ============================================================
# CREATE REPORT
# ============================================================

report = []

report.append("=" * 60)

report.append("MAMMOGRAM DATASET REPORT")

report.append("=" * 60)

report.append(f"Original Metadata Rows : {len(df)}")

report.append(f"Final Mammograms       : {len(mammogram_df)}")

report.append(
    f"Removed Duplicate Rows : {len(df)-len(mammogram_df)}"
)

report.append("")

report.append("Label Distribution")

report.append(
    str(
        mammogram_df["pathology"].value_counts()
    )
)

report.append("")

report.append("Binary Labels")

report.append(
    str(
        mammogram_df["label"].value_counts()
    )
)

report.append("")

report.append("Average Lesions / Mammogram")

report.append(
    str(
        round(
            mammogram_df["abnormality_count"].mean(),
            2
        )
    )
)

with open(OUTPUT_REPORT, "w") as f:

    f.write("\n".join(report))

# ============================================================
# SUMMARY
# ============================================================

print()

print("=" * 70)

print("SUMMARY")

print("=" * 70)

print(f"Original Rows : {len(df)}")

print(f"Final Images  : {len(mammogram_df)}")

print()

print("Class Distribution")

print(
    mammogram_df["pathology"].value_counts()
)

print()

print("Binary Labels")

print(
    mammogram_df["label"].value_counts()
)

print()

print("Saved Dataset")

print(OUTPUT_DATASET)

print()

print("Saved Report")

print(OUTPUT_REPORT)

print("=" * 70)
