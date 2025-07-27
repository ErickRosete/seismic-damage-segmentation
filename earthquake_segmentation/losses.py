import torch
import torch.nn.functional as F


def dice_loss(
    pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6
) -> torch.Tensor:
    """Compute Dice loss for multi-class segmentation."""
    pred = torch.softmax(pred, dim=1)
    target_one_hot = (
        F.one_hot(target, num_classes=pred.shape[1]).permute(0, 3, 1, 2).float()
    )
    dims = (1, 2, 3)
    intersection = (pred * target_one_hot).sum(dims)
    union = pred.sum(dims) + target_one_hot.sum(dims)
    dice = (2 * intersection + eps) / (union + eps)
    return 1 - dice.mean()


def build_loss(cfg):
    """Return a loss function based on the config."""
    name = cfg.loss.name
    if name == "cross_entropy":
        return lambda pred, target: F.cross_entropy(pred, target)
    if name == "dice":
        return lambda pred, target: dice_loss(pred, target)
    if name == "cross_entropy_dice":
        weight = cfg.loss.get("dice_weight", 1.0)

        def loss_fn(pred, target):
            ce = F.cross_entropy(pred, target)
            d = dice_loss(pred, target)
            return ce + weight * d

        return loss_fn
    raise ValueError(f"Unknown loss: {name}")
