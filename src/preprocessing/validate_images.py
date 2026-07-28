"""
=========================================================
Project:
Privacy-Preserving Federated Transfer Learning
for Mammography Classification

Script:
validate_images.py

Purpose:
Validate all mammogram images before preprocessing.

Author:
Harshit Garg
=========================================================
"""

from pathlib import Path
import cv2
import pandas as pd
from tqdm import tqdm

# -------------------------------------------------------
# Paths
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MASTER_FILE = PROCESSED_DIR / "master_labels.csv"

OUTPUT_REPORT = PROCESSED_DIR / "image_validation_report.csv"

# -------------------------------------------------------
# Load
# -------------------------------------------------------

print("=" * 70)
print("VALIDATING IMAGES")
print("=" * 70)

df = pd.read_csv(MASTER_FILE)

results = []

# -------------------------------------------------------
# Validation Loop
# -------------------------------------------------------

for _, row in tqdm(df.iterrows(), total=len(df), desc="Validating"):

    image_path = row["image_path"]

    exists = Path(image_path).exists()

    valid = False
    height = None
    width = None
    dtype = None
    min_pixel = None
    max_pixel = None

    if exists:

        img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

        if img is not None:

            valid = True

            height, width = img.shape

            dtype = str(img.dtype)

            min_pixel = int(img.min())

            max_pixel = int(img.max())

    results.append(
        {
            "patient_id": row["patient_id"],
            "image_path": image_path,
            "exists": exists,
            "valid": valid,
            "height": height,
            "width": width,
            "dtype": dtype,
            "min_pixel": min_pixel,
            "max_pixel": max_pixel,
        }
    )

# -------------------------------------------------------
# Save
# -------------------------------------------------------

report = pd.DataFrame(results)

report.to_csv(OUTPUT_REPORT, index=False)

print("\n" + "=" * 70)

print("IMAGE VALIDATION SUMMARY")

print("=" * 70)

print(f"Total Images : {len(report)}")

print(f"Existing     : {report['exists'].sum()}")

print(f"Readable     : {report['valid'].sum()}")

print(f"Corrupted    : {~report['valid'].sum()}")

print(f"\nSaved Report : {OUTPUT_REPORT}")

print("=" * 70)