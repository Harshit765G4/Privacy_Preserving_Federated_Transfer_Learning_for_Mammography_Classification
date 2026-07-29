import pandas as pd

files = {
    "mass_train": r"D:\Breast_Cancer\data\raw\csv\mass_case_description_train_set.csv",
    "mass_test":  r"D:\Breast_Cancer\data\raw\csv\mass_case_description_test_set.csv",
    "calc_train": r"D:\Breast_Cancer\data\raw\csv\calc_case_description_train_set.csv",
    "calc_test":  r"D:\Breast_Cancer\data\raw\csv\calc_case_description_test_set.csv",
    "dicom_info": r"D:\Breast_Cancer\data\raw\csv\dicom_info.csv",
    "meta":       r"D:\Breast_Cancer\data\raw\csv\meta.csv",
}

for name, path in files.items():
    df = pd.read_csv(path)
    print(f"=== {name} === shape={df.shape}")
    print(df.columns.tolist())
    print(df.head(3).to_string())
    print("-" * 80)