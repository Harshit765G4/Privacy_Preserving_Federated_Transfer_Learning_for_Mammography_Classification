"""
PyTorch Dataset for CBIS-DDSM Mammograms
"""

from pathlib import Path

import pandas as pd
from PIL import Image
from torch.utils.data import Dataset

from config.config import DATASET_CSV
from src.utils.transforms import (
    get_train_transforms,
    get_valid_transforms,
)


class MammogramDataset(Dataset):

    def __init__(self, fold=0, mode="train"):

        self.df = pd.read_csv(DATASET_CSV)

        self.mode = mode

        if mode == "train":
            self.df = self.df[self.df.fold != fold].reset_index(drop=True)
            self.transforms = get_train_transforms()

        else:
            self.df = self.df[self.df.fold == fold].reset_index(drop=True)
            self.transforms = get_valid_transforms()

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image = Image.open(row["clahe_path"]).convert("RGB")

        image = self.transforms(image)

        label = int(row["label"])

        return image, label