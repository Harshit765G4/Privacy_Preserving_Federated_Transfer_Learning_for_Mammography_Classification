"""
=========================================================
Project:
Privacy-Preserving Federated Transfer Learning
for Mammography Classification

Script:
analyze_duplicate_annotations.py

Purpose:
Analyze duplicate mammogram annotations and determine
whether duplicate metadata rows can be safely merged.

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

REPORT_FILE = PROCESSED_DIR / "duplicate_annotation_analysis.csv"

# --------------------------------------------------------
# Load
# --------------------------------------------------------

print("=" * 70)
print("ANALYZING DUPLICATE ANNOTATIONS")
print("=" * 70)

df = pd.read_csv(MASTER_FILE)

# --------------------------------------------------------
# Duplicate Mammograms
# --------------------------------------------------------

duplicates = df[df.duplicated(subset="image_path", keep=False)].copy()

print(f"Duplicate Metadata Rows : {len(duplicates)}")

print(f"Unique Duplicate Images : {duplicates['image_path'].nunique()}")

# --------------------------------------------------------
# Group Analysis
# --------------------------------------------------------

analysis = []

for image_path, group in duplicates.groupby("image_path"):

    analysis.append({

        "image_path": image_path,

        "num_annotations": len(group),

        "num_patients": group["patient_id"].nunique(),

        "num_pathologies": group["pathology"].nunique(),

        "pathology_labels":
            ", ".join(sorted(group["pathology"].unique())),

        "num_abnormality_types":
            group["abnormality_type"].nunique(),

        "abnormality_types":
            ", ".join(sorted(group["abnormality_type"].unique()))

    })

report = pd.DataFrame(analysis)

report.to_csv(REPORT_FILE, index=False)

# --------------------------------------------------------
# Summary
# --------------------------------------------------------

print()

print("=" * 70)

print("SUMMARY")

print("=" * 70)

print(f"Duplicate Images               : {len(report)}")

print(f"Images with Same Label         : {(report['num_pathologies']==1).sum()}")

print(f"Images with Conflicting Labels : {(report['num_pathologies']>1).sum()}")

print()

print("Distribution of Annotation Counts")

print(report["num_annotations"].value_counts().sort_index())

print()

print("Saved Report")

print(REPORT_FILE)

print("=" * 70)