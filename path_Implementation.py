import pandas as pd
import os

RAW_ROOT = r"D:\Breast_Cancer\data\raw"

def load_csv(name):
    return pd.read_csv(os.path.join(RAW_ROOT, "csv", name))

mass_train = load_csv("mass_case_description_train_set.csv")
mass_test  = load_csv("mass_case_description_test_set.csv")
calc_train = load_csv("calc_case_description_train_set.csv")
calc_test  = load_csv("calc_case_description_test_set.csv")
dicom_info = load_csv("dicom_info.csv")

# tag each frame with split/type before concatenating
for df, is_train, abn_type in [
    (mass_train, True, "mass"), (mass_test, False, "mass"),
    (calc_train, True, "calc"), (calc_test, False, "calc"),
]:
    df["source_split"] = "train" if is_train else "test"
    df["abnormality_type"] = abn_type

meta = pd.concat([mass_train, mass_test, calc_train, calc_test], ignore_index=True)

def extract_series_uid(path):
    """Pull the UID directory immediately before the filename."""
    parts = str(path).strip().split("/")
    return parts[-2] if len(parts) >= 2 else None

meta["full_image_series_uid"] = meta["image file path"].apply(extract_series_uid)

# lookup table: SeriesInstanceUID -> jpg path + description
dicom_lookup = dicom_info.set_index("SeriesInstanceUID")[["image_path", "SeriesDescription"]]

merged = meta.merge(dicom_lookup, left_on="full_image_series_uid", right_index=True, how="left")

# convert the CSV's virtual path prefix to our real local jpeg folder
merged["local_jpg_path"] = merged["image_path"].str.replace(
    "CBIS-DDSM/jpeg/", "", regex=False
).apply(lambda p: os.path.join(RAW_ROOT, "jpeg", p) if pd.notna(p) else None)

# --- verification ---
print("Total case rows:", len(merged))
print("Rows with NO matching jpg:", merged["image_path"].isna().sum())
print("\nSeriesDescription breakdown for our 'full image' join:")
print(merged["SeriesDescription"].value_counts(dropna=False))

print("\nFiles that don't actually exist on disk:",
      (~merged["local_jpg_path"].apply(lambda p: os.path.isfile(p) if p else False)).sum())

merged.to_csv(os.path.join(RAW_ROOT, "..", "processed_meta_check.csv"), index=False)