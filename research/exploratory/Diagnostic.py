import pandas as pd
import os

RAW_ROOT = r"D:\Breast_Cancer\data\raw"
dicom_info = pd.read_csv(os.path.join(RAW_ROOT, "csv", "dicom_info.csv"))

# 1. Check for duplicate SeriesInstanceUID in dicom_info itself
dupe_count = dicom_info["SeriesInstanceUID"].duplicated().sum()
print("Duplicate SeriesInstanceUID rows in dicom_info:", dupe_count)

# 2. Check how many rows in dicom_info have a real image_path but NaN SeriesDescription
mask = dicom_info["image_path"].notna() & dicom_info["SeriesDescription"].isna()
print("dicom_info rows with image_path but NO SeriesDescription:", mask.sum())
print(dicom_info.loc[mask, ["file_path", "image_path", "SeriesInstanceUID", "SeriesDescription"]].head(10))

# 3. Pull out our 282 problem rows from merged and inspect a few of their series UIDs directly
merged = pd.read_csv(os.path.join(RAW_ROOT, "..", "processed_meta_check.csv"))
problem_rows = merged[merged["SeriesDescription"].isna()]
print("\nSample problem UIDs from our merged table:")
print(problem_rows[["patient_id", "abnormality_type", "full_image_series_uid", "local_jpg_path"]].head(5))

# 4. For one problem UID, show every matching row in dicom_info (not just the merge result)
sample_uid = problem_rows["full_image_series_uid"].iloc[0]
print(f"\nAll dicom_info rows matching UID {sample_uid}:")
print(dicom_info[dicom_info["SeriesInstanceUID"] == sample_uid])