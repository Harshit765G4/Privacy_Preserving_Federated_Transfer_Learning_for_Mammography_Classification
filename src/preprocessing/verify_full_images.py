"""
=========================================================
Project:
Privacy-Preserving Federated Transfer Learning
for Mammography Classification

Script:
verify_full_images.py

Purpose:
Verify that every case points to a FULL mammogram image
using meta.csv.

Author:
Harshit Garg
=========================================================
"""

from pathlib import Path
import pandas as pd


# ---------------------------------------------------------
# Paths
# ---------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "raw" / "csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MERGED_FILE = PROCESSED_DIR / "merged_case_descriptions.csv"
META_FILE = RAW_DIR / "meta.csv"

OUTPUT_FILE = PROCESSED_DIR / "verified_full_images.csv"


# ---------------------------------------------------------
# Load Files
# ---------------------------------------------------------

print("=" * 60)
print("VERIFYING FULL MAMMOGRAM IMAGES")
print("=" * 60)

merged_df = pd.read_csv(MERGED_FILE)
meta_df = pd.read_csv(META_FILE)

print(f"Merged metadata : {len(merged_df)}")
print(f"Meta records    : {len(meta_df)}")


# ---------------------------------------------------------
# Extract Series UID
# ---------------------------------------------------------

def extract_series_uid(path):
    """
    Example

    Mass-Training_P_00001_LEFT_CC/
    UID1/
    UID2/
    000000.dcm

    Returns UID2
    """

    try:
        return path.strip().split("/")[-2]
    except Exception:
        return None


merged_df["full_image_series_uid"] = merged_df["image file path"].apply(extract_series_uid)


# ---------------------------------------------------------
# Keep only useful columns from meta.csv
# ---------------------------------------------------------

meta_df = meta_df[
    [
        "SeriesInstanceUID",
        "SeriesDescription"
    ]
].drop_duplicates()


# ---------------------------------------------------------
# Merge
# ---------------------------------------------------------

verified_df = merged_df.merge(
    meta_df,
    left_on="full_image_series_uid",
    right_on="SeriesInstanceUID",
    how="left"
)


# ---------------------------------------------------------
# Statistics BEFORE filtering
# ---------------------------------------------------------

print("\nSeries Description Distribution\n")

print(
    verified_df["SeriesDescription"]
    .fillna("Missing")
    .value_counts()
)


# ---------------------------------------------------------
# Keep ONLY Full Mammograms
# ---------------------------------------------------------

verified_df = verified_df[
    verified_df["SeriesDescription"] == "full mammogram images"
].copy()


# ---------------------------------------------------------
# Save
# ---------------------------------------------------------

verified_df.to_csv(OUTPUT_FILE, index=False)


# ---------------------------------------------------------
# Final Report
# ---------------------------------------------------------

print("\n" + "=" * 60)

print(f"Verified Full Mammograms : {len(verified_df)}")

print(f"Saved to :")

print(OUTPUT_FILE)

print("=" * 60)