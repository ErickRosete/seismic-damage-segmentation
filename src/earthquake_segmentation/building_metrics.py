"""Building-level metric helpers.

This module centralises the logic needed to aggregate per-building
predictions, compute derived statistics, and wire them into the
training/validation loop.
"""

from __future__ import annotations

import os
import warnings
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from omegaconf import DictConfig, OmegaConf
from sklearn.metrics import confusion_matrix, precision_recall_fscore_support
import torch

from .utils import accumulate_building_majority_labels


SPATIAL_MISALIGNMENT_KEYWORDS = {
    "crop",
    "flip",
    "rotate",
    "transpose",
    "perspective",
    "affine",
}


def _iter_transform_names(transforms_cfg: Iterable) -> Iterable[str]:
    """Yield augmentation names from a Hydra/OmegaConf list configuration."""

    if transforms_cfg is None:
        return []

    for item in transforms_cfg:
        container = item
        if OmegaConf.is_config(item):
            container = OmegaConf.to_container(item, resolve=True)
        if isinstance(container, dict):
            for name in container.keys():
                if name:
                    yield str(name)


def _find_spatially_incompatible_transform(cfg: DictConfig) -> Optional[str]:
    """Return the first validation transform that breaks building alignment, if any."""

    val_transforms = getattr(cfg.augmentations, "val", [])
    for name in _iter_transform_names(val_transforms):
        lower_name = name.lower()
        if any(keyword in lower_name for keyword in SPATIAL_MISALIGNMENT_KEYWORDS):
            return name
    return None


def resolve_building_metrics_settings(cfg: DictConfig) -> Tuple[bool, bool]:
    """Determine if building metrics should run and whether to raise on missing data."""

    evaluation_cfg = getattr(cfg, "evaluation", None)
    building_cfg = getattr(evaluation_cfg, "building_metrics", None)

    enabled = bool(getattr(building_cfg, "enabled", False))
    raise_on_missing = bool(getattr(building_cfg, "raise_on_missing", False))

    if enabled:
        incompatible = _find_spatially_incompatible_transform(cfg)
        if incompatible:
            warnings.warn(
                "Disabling building-level metrics because validation augmentations include "
                f"'{incompatible}', which changes spatial alignment relative to building polygons."
            )
            enabled = False

    return enabled, raise_on_missing


def collect_building_metrics_from_batch(
    extras: Sequence[Optional[Dict[str, Optional[str]]]],
    masks_cpu: torch.Tensor,
    preds_cpu: torch.Tensor,
    raise_on_missing: bool,
) -> Tuple[List[int], List[int]]:
    """Collect building-level targets/predictions for a batch."""

    batch_targets: List[int] = []
    batch_predictions: List[int] = []

    building_paths = extras.get("building_path", [])
    label_paths = extras.get("label_path", [])
    uids = extras.get("uid", [])

    for sample_idx in range(len(uids)):
        building_path = building_paths[sample_idx] if len(building_paths) > sample_idx else None
        label_path = label_paths[sample_idx] if len(label_paths) > sample_idx else None
        uid = uids[sample_idx] if len(uids) > sample_idx else "unknown"

        if not building_path or not os.path.exists(building_path):
            message = f"Missing building annotations for UID {uid}."
            if raise_on_missing:
                raise FileNotFoundError(message)
            warnings.warn(message)
            continue

        if not label_path or not os.path.exists(label_path):
            message = f"Missing label raster for UID {uid}."
            if raise_on_missing:
                raise FileNotFoundError(message)
            warnings.warn(message)
            continue

        try:
            pairs = accumulate_building_majority_labels(
                building_path,
                label_path,
                masks_cpu[sample_idx],
                preds_cpu[sample_idx],
            )
        except Exception as exc:  # pragma: no cover - defensive guard
            if raise_on_missing:
                raise
            warnings.warn(
                f"Failed to compute building metrics for UID {uid}: {exc}"
            )
            continue

        for target, prediction in pairs:
            batch_targets.append(int(target))
            batch_predictions.append(int(prediction))

    return batch_targets, batch_predictions


def compute_building_metric_statistics(
    targets: Sequence[int], predictions: Sequence[int], num_classes: int
) -> Optional[Dict[str, object]]:
    """Return precision/recall/F1/support/confusion for building predictions."""

    if not targets:
        return None

    labels = list(range(num_classes))
    precision, recall, f1, support = precision_recall_fscore_support(
        targets,
        predictions,
        labels=labels,
        zero_division=0,
    )
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        targets,
        predictions,
        labels=labels,
        zero_division=0,
        average="macro",
    )
    conf_mat = confusion_matrix(targets, predictions, labels=labels)

    return {
        "labels": labels,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "support": support,
        "precision_macro": precision_macro,
        "recall_macro": recall_macro,
        "f1_macro": f1_macro,
        "confusion": conf_mat,
    }


def update_log_with_building_metrics(
    log_payload: Dict[str, object], stats: Dict[str, object]
) -> None:
    """Populate the log payload with building metric scalars."""

    log_payload["building_precision_macro"] = stats["precision_macro"]
    log_payload["building_recall_macro"] = stats["recall_macro"]
    log_payload["building_f1_macro"] = stats["f1_macro"]

    labels: List[int] = stats["labels"]  # type: ignore[assignment]
    precision = stats["precision"]
    recall = stats["recall"]
    f1 = stats["f1"]
    support = stats["support"]
    conf_mat = stats["confusion"]

    for idx, label in enumerate(labels):
        log_payload[f"building_precision_class_{label}"] = float(precision[idx])
        log_payload[f"building_recall_class_{label}"] = float(recall[idx])
        log_payload[f"building_f1_class_{label}"] = float(f1[idx])
        log_payload[f"building_support_class_{label}"] = int(support[idx])
        for jdx, pred_label in enumerate(labels):
            log_payload[
                f"building_confusion/label_{label}_pred_{pred_label}"
            ] = int(
                conf_mat[idx, jdx]
            )


__all__ = [
    "collect_building_metrics_from_batch",
    "compute_building_metric_statistics",
    "resolve_building_metrics_settings",
    "update_log_with_building_metrics",
]

