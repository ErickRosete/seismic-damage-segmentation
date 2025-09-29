from typing import Callable, Optional

import torch
import torch.nn.functional as F


def dice_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    eps: float = 1e-6,
    ignore_index: Optional[int] = None,
) -> torch.Tensor:
    """Compute Dice loss for multi-class segmentation."""
    pred = torch.softmax(pred, dim=1)
    target_tensor = target
    if ignore_index is not None:
        valid_mask = target != ignore_index
        target_tensor = torch.where(
            valid_mask, target, torch.zeros_like(target, device=target.device)
        )
        pred = pred * valid_mask.unsqueeze(1)
    target_one_hot = (
        F.one_hot(target_tensor, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
    )
    if ignore_index is not None:
        target_one_hot = target_one_hot * valid_mask.unsqueeze(1)
    dims = (1, 2, 3)
    intersection = (pred * target_one_hot).sum(dims)
    union = pred.sum(dims) + target_one_hot.sum(dims)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def _build_pixel_loss(
    cfg, name: str, ignore_index: Optional[int]
) -> Callable[..., torch.Tensor]:
    if name == "cross_entropy":
        return lambda pred, target, **_: F.cross_entropy(
            pred, target, ignore_index=ignore_index
        )
    if name == "dice":
        return lambda pred, target, **_: dice_loss(
            pred, target, ignore_index=ignore_index
        )
    if name == "cross_entropy_dice":
        weight = cfg.loss.get("dice_weight", 1.0)

        def loss_fn(pred, target, **_):
            ce = F.cross_entropy(pred, target, ignore_index=ignore_index)
            d = dice_loss(pred, target, ignore_index=ignore_index)
            return ce + weight * d

        return loss_fn
    raise ValueError(f"Unknown pixel loss: {name}")


class BuildingPooledLoss:
    def __init__(
        self,
        pixel_loss: Callable[..., torch.Tensor],
        pixel_weight: float,
        building_weight: float,
        ignore_index: Optional[int],
        background_id: int,
        eps: float = 1e-12,
    ) -> None:
        self.pixel_loss = pixel_loss
        self.pixel_weight = float(pixel_weight)
        self.building_weight = float(building_weight)
        self.ignore_index = ignore_index
        self.background_id = background_id
        self.eps = eps

    def _building_component(
        self, pred: torch.Tensor, target: torch.Tensor, building_ids: torch.Tensor
    ) -> torch.Tensor:
        probs = torch.softmax(pred, dim=1)
        total_loss = pred.new_zeros(())
        building_count = 0
        num_classes = pred.shape[1]
        for batch_idx in range(pred.shape[0]):
            sample_buildings = torch.unique(building_ids[batch_idx])
            if sample_buildings.numel() == 0:
                continue
            if self.background_id is not None:
                sample_buildings = sample_buildings[
                    sample_buildings != self.background_id
                ]
            for building_id in sample_buildings:
                mask = building_ids[batch_idx] == building_id
                if self.ignore_index is not None:
                    mask = mask & (target[batch_idx] != self.ignore_index)
                if not torch.any(mask):
                    continue
                target_vals = target[batch_idx][mask]
                counts = torch.bincount(target_vals, minlength=num_classes)
                majority_class = torch.argmax(counts)
                building_probs = probs[batch_idx, :, mask].mean(dim=1)
                loss = -torch.log(
                    building_probs[majority_class].clamp_min(self.eps)
                )
                total_loss = total_loss + loss
                building_count += 1
        if building_count == 0:
            return pred.new_zeros(())
        return total_loss / building_count

    def __call__(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        *,
        building_ids: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        pixel_term = self.pixel_loss(pred, target)
        if self.building_weight == 0:
            return self.pixel_weight * pixel_term
        if building_ids is None:
            raise ValueError("building_ids must be provided for building-pooled loss")
        building_term = self._building_component(pred, target, building_ids)
        return self.pixel_weight * pixel_term + self.building_weight * building_term


def build_loss(cfg):
    """Return a loss function based on the config."""
    name = cfg.loss.name
    ignore_index = cfg.loss.get("ignore_index", None)
    if name in {"cross_entropy", "dice", "cross_entropy_dice"}:
        return _build_pixel_loss(cfg, name, ignore_index)
    if name == "building_pooled":
        pixel_loss_name = cfg.loss.get("pixel_loss", "cross_entropy")
        pixel_loss = _build_pixel_loss(cfg, pixel_loss_name, ignore_index)
        pixel_weight = cfg.loss.get("pixel_weight", 1.0)
        building_weight = cfg.loss.get("building_weight", 1.0)
        background_id = cfg.loss.get("building_background_id", 0)
        building_ignore_index = cfg.loss.get("building_ignore_index", ignore_index)
        return BuildingPooledLoss(
            pixel_loss=pixel_loss,
            pixel_weight=pixel_weight,
            building_weight=building_weight,
            ignore_index=building_ignore_index,
            background_id=background_id,
        )
    raise ValueError(f"Unknown loss: {name}")
