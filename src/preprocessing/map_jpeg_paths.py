"""
=========================================================
Project:
Privacy-Preserving Federated Transfer Learning
for Mammography Classification

Script:
map_jpeg_paths.py

Purpose:
Map verified mammogram metadata to JPEG image paths.

Author:
Harshit Garg
=========================================================
"""

from pathlib import Path
import pandas as pd


# -------------------------------------------------------
# Paths
# -------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "csv"
JPEG_DIR = PROJECT_ROOT / "data" / "raw" / "jpeg"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

INPUT_FILE = PROCESSED_DIR / "verified_full_images.csv"
DICOM_INFO = RAW_DIR / "dicom_info.csv"

OUTPUT_FILE = PROCESSED_DIR / "mapped_jpeg_images.csv"


# -------------------------------------------------------
# Load
# -------------------------------------------------------

print("=" * 60)
print("MAPPING JPEG PATHS")
print("=" * 60)

master = pd.read_csv(INPUT_FILE)
dicom = pd.read_csv(DICOM_INFO)

print(f"Verified records : {len(master)}")
print(f"DICOM records    : {len(dicom)}")


# -------------------------------------------------------
# Keep useful columns
# -------------------------------------------------------

dicom = dicom[
    [
        "SeriesInstanceUID",
        "image_path"
    ]
].drop_duplicates()


# -------------------------------------------------------
# Merge
# -------------------------------------------------------

master = master.merge(
    dicom,
    left_on="full_image_series_uid",
    right_on="SeriesInstanceUID",
    how="left"
)


# -------------------------------------------------------
# Build Local JPEG Path
# -------------------------------------------------------

def build_local_path(path):

    if pd.isna(path):
        return None

    path = path.replace("CBIS-DDSM/jpeg/", "")

    return str(JPEG_DIR / path)


master["local_jpg_path"] = master["image_path"].apply(build_local_path)


# -------------------------------------------------------
# Verify image exists
# -------------------------------------------------------

master["image_exists"] = master["local_jpg_path"].apply(
    lambda x: Path(x).exists() if pd.notna(x) else False
)


# -------------------------------------------------------
# Report
# -------------------------------------------------------

print("\nImage Verification")

print(master["image_exists"].value_counts())


# -------------------------------------------------------
# Save
# -------------------------------------------------------

master.to_csv(OUTPUT_FILE, index=False)

print("\nSaved")

print(OUTPUT_FILE)

print("=" * 60)