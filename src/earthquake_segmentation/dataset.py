import os
import glob
from typing import Dict, List, Optional, Tuple

import albumentations as A
import pandas as pd
import rasterio
import torch
from albumentations import Compose
from albumentations.pytorch import ToTensorV2
from torch.utils.data import Dataset


def _collect_uids(labels_dir: str) -> List[str]:
    """Return sorted UID list for label rasters within ``labels_dir``."""

    if not os.path.isdir(labels_dir):
        return []

    label_paths = glob.glob(os.path.join(labels_dir, "*_label.tif"))
    prefixes = [os.path.basename(fp)[: -len("_label.tif")] for fp in label_paths]
    return sorted(prefixes)


def make_splits(cfg) -> Tuple[List[str], List[str]]:
    """Enumerate training/validation UIDs from the split sub-directories."""

    data_dir = cfg.data.dir
    train_labels = os.path.join(data_dir, "train", "labels")
    val_labels = os.path.join(data_dir, "val", "labels")

    train = _collect_uids(train_labels)
    val = _collect_uids(val_labels)

    return train, val


class EarthquakeDamageDataset(Dataset):
    """
    PyTorch Dataset for pre-disaster RGB images and pixel-wise labels.
    """

    def __init__(self, prefixes, cfg, mode="train"):
        self.prefixes = prefixes
        self.cfg = cfg
        self.mode = mode

        self.split_root = os.path.join(cfg.data.dir, mode)
        self.pre_dir = os.path.join(self.split_root, "pre-disaster")
        self.label_dir = os.path.join(self.split_root, "labels")
        self.building_dir = os.path.join(self.split_root, "buildings")

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

    def _build_split_path(self, directory: str, filename: str) -> str:
        return os.path.join(directory, filename)

    def _first_match(self, directory: str, pattern: str) -> Optional[str]:
        if not os.path.isdir(directory):
            return None
        matches = sorted(glob.glob(os.path.join(directory, pattern)))
        return matches[0] if matches else None

    def __getitem__(self, idx):
        uid = self.prefixes[idx]
        # construct paths rooted at the current split directory
        pre_path = self._build_split_path(
            self.pre_dir, f"{uid}_pre_disaster.tif"
        )
        label_path = self._build_split_path(self.label_dir, f"{uid}_label.tif")
        building_path = self._first_match(self.building_dir, f"{uid}*")

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

        extras: Dict[str, Optional[str]] = {
            "uid": uid,
            "pre_image_path": pre_path,
            "label_path": label_path,
            "building_path": building_path,
        }

        return image, mask.long(), params, extras
