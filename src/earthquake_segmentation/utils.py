import os
import torch
import wandb
from torchmetrics import JaccardIndex
from typing import Optional
import numpy as np


def save_checkpoint(state, cfg, is_best):
    """
    Save model and optimizer state. Keep best checkpoint.
    """
    directory = cfg.training.checkpoint_dir
    os.makedirs(directory, exist_ok=True)
    path = os.path.join(directory, f"epoch_{state['epoch']}.pth")
    torch.save(state, path)
    if is_best:
        torch.save(state, os.path.join(directory, "best.pth"))


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

    filtered = [(img, m, p) for img, m, p in zip(inputs, masks, preds) if m.any()]
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
