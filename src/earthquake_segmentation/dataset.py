import os
import glob
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset
from albumentations import Compose
import albumentations as A
from albumentations.pytorch import ToTensorV2
from sklearn.model_selection import train_test_split


import json


def make_splits(cfg):
    """
    Create or load train/val splits stored as JSON in processed directory.
    """
    data_dir = cfg.data.dir
    split_file = os.path.join(data_dir, "splits.json")
    if os.path.exists(split_file):
        with open(split_file, "r") as f:
            data = json.load(f)
        train = data.get("train", [])
        val = data.get("val", [])
    else:
        labels_dir = os.path.join(data_dir, "labels")
        files = glob.glob(os.path.join(labels_dir, "*_label.tif"))
        prefixes = [os.path.basename(fp)[: -len("_label.tif")] for fp in files]
        train, val = train_test_split(
            prefixes,
            train_size=cfg.data.train_split,
            random_state=cfg.seed,
            shuffle=True,
        )
        os.makedirs(data_dir, exist_ok=True)
        with open(split_file, "w") as f:
            json.dump({"train": train, "val": val}, f, indent=2)
    return train, val


class EarthquakeDamageDataset(Dataset):
    """
    PyTorch Dataset for pre-disaster RGB images and pixel-wise labels.
    """

    def __init__(self, prefixes, cfg, mode="train"):
        self.prefixes = prefixes
        self.cfg = cfg
        self.mode = mode

        # optional metadata
        meta_path = os.path.join(cfg.data.dir, "metadata.csv")
        if os.path.exists(meta_path):
            df = pd.read_csv(meta_path)
            self.meta = {row["uid"]: row for _, row in df.iterrows()}
        else:
            self.meta = {}

        # build augmentations
        aug_list = cfg.augmentations[mode]
        ops = []
        for aug in aug_list:
            for name, params in aug.items():
                # dynamically fetch augmentation class from albumentations
                if not hasattr(A, name):
                    raise ValueError(f"Unknown augmentation: {name}")
                cls = getattr(A, name)
                ops.append(cls(**params))
        ops.append(ToTensorV2())
        self.transforms = Compose(ops)

        self.feature_cols = getattr(cfg.data, "feature_cols", None) or []

    def __len__(self):
        return len(self.prefixes)

    def __getitem__(self, idx):
        uid = self.prefixes[idx]
        # construct paths
        pre_path = os.path.join(
            self.cfg.data.dir, "pre-disaster", f"{uid}_pre_disaster.tif"
        )
        label_path = os.path.join(self.cfg.data.dir, "labels", f"{uid}_label.tif")

        # read image
        with rasterio.open(pre_path) as src:
            img = src.read([1, 2, 3]).transpose(1, 2, 0)
        # read label
        with rasterio.open(label_path) as src:
            mask = src.read(1)

        # apply transforms
        augmented = self.transforms(image=img, mask=mask)
        image, mask = augmented["image"], augmented["mask"]

        # load optional metadata params
        if self.feature_cols and uid in self.meta:
            row = self.meta[uid]
            vals = [row[col] for col in self.feature_cols]
            params = torch.tensor(vals, dtype=torch.float32)
        else:
            params = torch.zeros(0, dtype=torch.float32)

        return image, mask.long(), params
