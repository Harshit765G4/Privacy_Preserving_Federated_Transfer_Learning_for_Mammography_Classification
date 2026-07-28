import pandas as pd
import os

RAW_ROOT = r"D:\Breast_Cancer\data\raw"
dicom_info = pd.read_csv(os.path.join(RAW_ROOT, "csv", "dicom_info.csv"))
merged = pd.read_csv(os.path.join(RAW_ROOT, "..", "processed_meta_check.csv"))

# Baseline: typical Rows/Columns for CONFIRMED full mammogram images
confirmed_full = dicom_info[dicom_info["SeriesDescription"] == "full mammogram images"]
print("Confirmed full mammogram image dimensions:")
print(confirmed_full[["Rows", "Columns"]].describe())

# Our 282 problem rows: pull their actual Rows/Columns via SeriesInstanceUID
problem_uids = merged.loc[merged["SeriesDescription"].isna(), "full_image_series_uid"]
problem_dims = dicom_info[dicom_info["SeriesInstanceUID"].isin(problem_uids)]
print("\nProblem-row (NaN description) image dimensions:")
print(problem_dims[["SeriesInstanceUID", "Rows", "Columns"]].describe())

# Flag any problem row whose dimensions look crop/mask-sized (e.g. well below the 5th percentile of confirmed full images)
low_threshold = confirmed_full["Rows"].quantile(0.05)
suspicious = problem_dims[problem_dims["Rows"] < low_threshold]
print(f"\nSuspicious (small) rows out of {len(problem_dims)}:", len(suspicious))
print(suspicious[["SeriesInstanceUID", "Rows", "Columns"]])