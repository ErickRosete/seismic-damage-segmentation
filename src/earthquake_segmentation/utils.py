import json
import os
import shutil
import tempfile
import warnings
from typing import List, Optional, Sequence, Tuple

import numpy as np
import torch
import wandb
from rasterio import features
import rasterio
from rasterio.transform import Affine
from torchmetrics import JaccardIndex



def safe_torch_save(obj, path):
    """
    Save torch object safely:
    - Write to a temporary file first
    - Atomically replace the target file
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fd, tmp_path = tempfile.mkstemp(dir=os.path.dirname(path))
    os.close(fd)
    try:
        torch.save(obj, tmp_path)
        shutil.move(tmp_path, path)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def save_checkpoint(state, cfg, is_best):
    """
    Save model and optimizer state. Keep best checkpoint.
    """
    directory = cfg.training.checkpoint_dir
    epoch = state["epoch"]
    epoch_path = os.path.join(directory, f"epoch_{epoch}.pth")
    best_path = os.path.join(directory, "best.pth")

    try:
        if epoch % cfg.training.checkpoint_freq == 0:
            safe_torch_save(state, epoch_path)
        if is_best:
            safe_torch_save(state, best_path)
    except Exception as e:
        print(f"[✗] Failed to save checkpoint: {e}")


class MetricTracker:
    """
    Wrapper around Jaccard Index (IoU) metric for segmentation,
    automatically moved to the correct device.
    """

    def __init__(
        self,
        num_classes: int,
        device: torch.device,
        ignore_index: Optional[int] = None,
    ):
        # create the metric and immediately move it to device
        self.metric = JaccardIndex(
            task="multiclass",
            num_classes=num_classes,
            average="macro",
            ignore_index=ignore_index,
        ).to(device)

    def update(self, preds: torch.Tensor, targets: torch.Tensor):
        # preds and targets should already be on the same device
        self.metric.update(preds, targets)

    def compute(self) -> torch.Tensor:
        return self.metric.compute()

    def reset(self):
        self.metric.reset()


def denormalize_image(img: torch.Tensor) -> torch.Tensor:
    """
    Denormalize an image tensor from ImageNet stats.
    Converts from normalized [0,1] range to [0,255] uint8.
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=img.device).view(3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=img.device).view(3, 1, 1)
    img_denorm = img * std + mean
    img_denorm = (img_denorm * 255.0).clamp(0, 255).byte()
    return img_denorm


def log_filtered_imgs(
    inputs: list[torch.Tensor],
    masks: list[torch.Tensor],
    preds: list[torch.Tensor],
    epoch: int,
    max_samples: int = 10,
):
    """Logs filtered images to Weights & Biases."""

    filtered = [(img, m, p) for img, m, p in zip(inputs, masks, preds) if (m > 1).any()]
    if not filtered:
        return

    # Unzip and limit to max_samples
    imgs_f, masks_f, preds_f = zip(*filtered[:max_samples])

    # Delegate to your existing log_images
    log_images(list(imgs_f), list(masks_f), list(preds_f), epoch)


def log_images(
    inputs: list[torch.Tensor],
    masks: list[torch.Tensor],
    preds: list[torch.Tensor],
    epoch: int,
):
    """
    Log input images, ground truth, and predictions to Weights & Biases.
    Denormalizes inputs from ImageNet stats, scales to [0,255], and logs as uint8.
    """
    cmap = np.array(
        [
            [0, 0, 0],  # background
            [0, 255, 0],  # no-damaged
            [255, 255, 0],  # damaged
            [255, 0, 0],  # destroyed
        ],
        dtype=np.uint8,
    )

    for i in range(len(inputs)):
        img = denormalize_image(inputs[i])
        # Convert to HWC numpy array
        img_np = img.permute(1, 2, 0).cpu().numpy()

        # ground truth and preds as HxW arrays
        mask = masks[i].cpu().numpy().astype(np.int64)
        pred = preds[i].cpu().numpy().astype(np.int64)

        # map to RGB
        mask_rgb = cmap[mask]
        pred_rgb = cmap[pred]

        wandb.log(
            {
                f"input/{i}": wandb.Image(img_np, caption=f"Epoch {epoch} Input"),
                f"mask/{i}": wandb.Image(mask_rgb, caption=f"Epoch {epoch} GT"),
                f"pred/{i}": wandb.Image(pred_rgb, caption=f"Epoch {epoch} Pred"),
                "epoch": epoch,
            }
        )


def _load_building_geometries(building_json_path: str) -> Sequence[Tuple[dict, dict]]:
    """Return geometry/property pairs from a building GeoJSON file."""

    with open(building_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    if isinstance(data, dict) and "features" in data:
        features_iter = data["features"]
    elif isinstance(data, list):
        features_iter = data
    else:
        raise ValueError(
            f"Unsupported building annotation format in '{building_json_path}'."
        )

    geometries: List[Tuple[dict, dict]] = []
    for feature in features_iter:
        if not feature:
            continue
        geometry = feature.get("geometry") if isinstance(feature, dict) else feature
        properties = feature.get("properties", {}) if isinstance(feature, dict) else {}
        if geometry is None:
            continue
        geometries.append((geometry, properties))
    return geometries


def _load_building_centroids(building_json_path: str) -> List[Tuple[float, float]]:
    """Extract centroid column/row pairs from a building GeoJSON file."""

    centroids: List[Tuple[float, float]] = []
    for _geometry, properties in _load_building_geometries(building_json_path):
        if not properties:
            continue
        col = properties.get("centroid_col")
        row = properties.get("centroid_row")
        if col is None or row is None:
            continue
        try:
            centroids.append((float(col), float(row)))
        except (TypeError, ValueError):
            continue
    return centroids


def accumulate_building_majority_labels(
    building_json_path: str,
    label_raster_path: str,
    gt_mask: torch.Tensor,
    pred_mask: torch.Tensor,
) -> List[Tuple[int, int]]:
    """
    Rasterize building polygons and compute majority GT/pred labels per building.

    Returns a list of ``(target, prediction)`` tuples, one per building polygon.
    """

    if gt_mask.shape != pred_mask.shape:
        raise ValueError(
            "Ground truth and prediction masks must share the same spatial shape."
        )

    gt_np = gt_mask.detach().cpu().numpy()
    pred_np = pred_mask.detach().cpu().numpy()

    geometries = _load_building_geometries(building_json_path)
    if not geometries:
        warnings.warn(f"No building geometries found in '{building_json_path}'.")
        return []

    with rasterio.open(label_raster_path) as src:
        transform = src.transform
        out_shape = (src.height, src.width)

    # Ensure the rasterized map aligns with the tensors. If shapes diverge we
    # rescale the affine transform to the tensor's spatial shape.
    if out_shape != gt_np.shape:
        scale_y = out_shape[0] / gt_np.shape[0]
        scale_x = out_shape[1] / gt_np.shape[1]
        transform = transform * Affine.scale(scale_x, scale_y)
        out_shape = gt_np.shape

    shapes: List[Tuple[dict, int]] = []
    labels_and_properties: List[Tuple[int, dict]] = []
    for geometry, properties in geometries:
        if geometry is None:
            continue
        label = len(shapes) + 1
        shapes.append((geometry, label))
        labels_and_properties.append((label, properties or {}))
    if not shapes:
        return []

    building_index = features.rasterize(
        shapes,
        out_shape=out_shape,
        transform=transform,
        fill=0,
        dtype="int32",
        all_touched=True,
    )

    results: List[Tuple[int, int]] = []
    for label, _ in labels_and_properties:
        mask = building_index == label
        if not mask.any():
            continue
        gt_vals = gt_np[mask]
        pred_vals = pred_np[mask]
        if gt_vals.size == 0 or pred_vals.size == 0:
            continue
        gt_majority = int(np.bincount(gt_vals).argmax())
        pred_majority = int(np.bincount(pred_vals).argmax())
        results.append((gt_majority, pred_majority))

    return results


def _draw_cross(
    canvas: np.ndarray, row: int, col: int, color: Sequence[int], half_size: int = 2
) -> None:
    """Draw a small cross centred on ``(row, col)`` on ``canvas`` inplace."""

    if canvas.ndim != 3:
        raise ValueError("Canvas must be an HxWxC array.")

    height, width = canvas.shape[:2]
    if not (0 <= row < height and 0 <= col < width):
        return

    row_range = range(max(0, row - half_size), min(height, row + half_size + 1))
    col_range = range(max(0, col - half_size), min(width, col + half_size + 1))

    for r in row_range:
        canvas[r, col] = color
    for c in col_range:
        canvas[row, c] = color


def _build_centroid_overlay_panel(
    image: torch.Tensor,
    mask: torch.Tensor,
    prediction: torch.Tensor,
    centroids: Sequence[Tuple[float, float]],
) -> Optional[np.ndarray]:
    """Return an RGB panel visualising centroids on image/GT/prediction."""

    if not centroids:
        return None

    img = denormalize_image(image).permute(1, 2, 0).cpu().numpy()
    mask_np = mask.cpu().numpy().astype(np.int64)
    pred_np = prediction.cpu().numpy().astype(np.int64)

    cmap = np.array(
        [
            [0, 0, 0],
            [0, 255, 0],
            [255, 255, 0],
            [255, 0, 0],
        ],
        dtype=np.uint8,
    )

    mask_rgb = cmap[np.clip(mask_np, 0, len(cmap) - 1)]
    pred_rgb = cmap[np.clip(pred_np, 0, len(cmap) - 1)]

    overlay_color = np.array([255, 0, 255], dtype=np.uint8)

    image_overlay = img.copy()
    mask_overlay = mask_rgb.copy()
    pred_overlay = pred_rgb.copy()

    for row_f, col_f in centroids:
        row = int(round(row_f))
        col = int(round(col_f))
        _draw_cross(image_overlay, row, col, overlay_color)
        _draw_cross(mask_overlay, row, col, overlay_color)
        _draw_cross(pred_overlay, row, col, overlay_color)

    return np.concatenate([image_overlay, mask_overlay, pred_overlay], axis=1)


def log_building_centroid_overlays(
    samples: Sequence[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str]],
    epoch: int,
) -> None:
    """Log W&B images showing building centroids on images/masks/predictions."""

    if not samples:
        return

    for idx, (image, mask, prediction, building_json, uid) in enumerate(samples):
        if not os.path.exists(building_json):
            warnings.warn(
                f"Skipping centroid overlay for UID {uid}: '{building_json}' not found."
            )
            continue

        centroids = _load_building_centroids(building_json)
        panel = _build_centroid_overlay_panel(image, mask, prediction, centroids)
        if panel is None:
            warnings.warn(
                f"Skipping centroid overlay for UID {uid}: no centroid coordinates available."
            )
            continue

        caption = f"Epoch {epoch} UID {uid}"
        wandb.log(
            {
                f"building_centroids/{uid}_{idx}": wandb.Image(panel, caption=caption),
                "epoch": epoch,
            }
        )
