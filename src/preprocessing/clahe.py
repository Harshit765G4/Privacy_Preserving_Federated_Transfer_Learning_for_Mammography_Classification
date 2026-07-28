"""
=========================================================
Project:
Privacy-Preserving Federated Transfer Learning
for Mammography Classification

Script:
clahe.py

Purpose:
Apply CLAHE preprocessing to all mammograms.

Author:
Harshit Garg
=========================================================
"""

from pathlib import Path
import time

import cv2
import pandas as pd
from tqdm import tqdm

# --------------------------------------------------------
# Paths
# --------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MASTER_FILE = PROCESSED_DIR / "master_labels.csv"

OUTPUT_IMAGE_DIR = PROCESSED_DIR / "images_clahe"

OUTPUT_METADATA = PROCESSED_DIR / "master_labels_clahe.csv"

OUTPUT_REPORT = PROCESSED_DIR / "clahe_processing_report.csv"

OUTPUT_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

# --------------------------------------------------------
# Load metadata
# --------------------------------------------------------

print("=" * 70)
print("CLAHE PREPROCESSING")
print("=" * 70)

df = pd.read_csv(MASTER_FILE)

processing_log = []

processed_paths = []

clahe = cv2.createCLAHE(
    clipLimit=3.0,
    tileGridSize=(8, 8)
)

start_time = time.time()

# --------------------------------------------------------
# Process Images
# --------------------------------------------------------

for _, row in tqdm(df.iterrows(), total=len(df), desc="Processing"):

    image_path = row["image_path"]

    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:

        processed_paths.append(None)

        processing_log.append({
            "patient_id": row["patient_id"],
            "status": "FAILED"
        })

        continue

    t0 = time.time()

    original_h, original_w = img.shape

    mean_before = float(img.mean())

    std_before = float(img.std())

    # Resize
    img = cv2.resize(
        img,
        (512, 512),
        interpolation=cv2.INTER_AREA
    )

    # CLAHE
    img = clahe.apply(img)

    mean_after = float(img.mean())

    std_after = float(img.std())

    filename = (
        f"{row['patient_id']}_"
        f"{row['breast_side']}_"
        f"{row['image_view']}.png"
    )

    output_path = OUTPUT_IMAGE_DIR / filename

    cv2.imwrite(str(output_path), img)

    processed_paths.append(str(output_path))

    processing_log.append({

        "patient_id": row["patient_id"],

        "original_height": original_h,

        "original_width": original_w,

        "mean_before": mean_before,

        "mean_after": mean_after,

        "std_before": std_before,

        "std_after": std_after,

        "processing_time_ms": round(
            (time.time() - t0) * 1000,
            2
        ),

        "status": "SUCCESS"

    })

# --------------------------------------------------------
# Save metadata
# --------------------------------------------------------

df["processed_image_path"] = processed_paths

df.to_csv(
    OUTPUT_METADATA,
    index=False
)

# --------------------------------------------------------
# Save report
# --------------------------------------------------------

report = pd.DataFrame(processing_log)

report.to_csv(
    OUTPUT_REPORT,
    index=False
)

elapsed = time.time() - start_time

# --------------------------------------------------------
# Summary
# --------------------------------------------------------

print()

print("=" * 70)

print("CLAHE SUMMARY")

print("=" * 70)

print(f"Total Images        : {len(df)}")

print(f"Processed Images    : {(report['status']=='SUCCESS').sum()}")

print(f"Failed Images       : {(report['status']=='FAILED').sum()}")

print(f"Average Time/Image  : {report['processing_time_ms'].mean():.2f} ms")

print(f"Total Time          : {elapsed:.2f} sec")

print()

print("Images Saved To")

print(OUTPUT_IMAGE_DIR)

print()

print("Metadata")

print(OUTPUT_METADATA)

print()

print("Report")

print(OUTPUT_REPORT)

print("=" * 70)
