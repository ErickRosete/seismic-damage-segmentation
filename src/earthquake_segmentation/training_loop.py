"""End-to-end training loop orchestration."""

from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb

from .building_metrics import (
    collect_building_metrics_from_batch,
    compute_building_metric_statistics,
    resolve_building_metrics_settings,
    update_log_with_building_metrics,
)
from .dataset import EarthquakeDamageDataset, make_splits
from .losses import build_loss
from .models.build_model import build_model
from .utils import (
    MetricTracker,
    log_building_centroid_overlays,
    log_filtered_imgs,
    save_checkpoint,
)


def _initialise_wandb(cfg: DictConfig) -> None:
    wandb.init(
        project=cfg.logging.wandb_project,
        name=cfg.logging.wandb_name,
        config=OmegaConf.to_container(cfg, resolve=True),  # type: ignore[arg-type]
        dir=cfg.logging.wandb_dir,
    )


def _create_dataloaders(
    cfg: DictConfig, cuda: bool
) -> tuple[DataLoader, DataLoader]:
    train_ids, val_ids = make_splits(cfg)
    train_ds = EarthquakeDamageDataset(train_ids, cfg, mode="train")
    val_ds = EarthquakeDamageDataset(val_ids, cfg, mode="val")

    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.training.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=cuda,
        persistent_workers=cuda,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.training.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=cuda,
        persistent_workers=cuda,
    )
    return train_loader, val_loader


def _prepare_directories(cfg: DictConfig, run_dir: str) -> None:
    cfg.training.checkpoint_dir = os.path.join(run_dir, cfg.training.checkpoint_dir)
    cfg.logging.wandb_dir = os.path.join(run_dir, cfg.logging.wandb_dir)
    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.logging.wandb_dir, exist_ok=True)


def _collect_building_metrics_for_epoch(
    building_metrics_enabled: bool,
    raise_on_missing: bool,
    building_targets: List[int],
    building_predictions: List[int],
    extras: Sequence[Optional[Dict[str, Optional[str]]]],
    masks_cpu: torch.Tensor,
    preds_cpu: torch.Tensor,
) -> None:
    if not building_metrics_enabled:
        return

    batch_targets, batch_preds = collect_building_metrics_from_batch(
        extras,
        masks_cpu,
        preds_cpu,
        raise_on_missing,
    )
    building_targets.extend(batch_targets)
    building_predictions.extend(batch_preds)


def _move_batch_to_device(
    imgs: torch.Tensor,
    masks: torch.Tensor,
    building_ids: torch.Tensor,
    params: torch.Tensor,
    device: torch.device,
    cuda_available: bool,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Move tensors that make up a batch to the target device."""

    non_blocking = cuda_available
    imgs = imgs.to(device, non_blocking=non_blocking)
    masks = masks.to(device, non_blocking=non_blocking)
    building_ids = building_ids.to(device, non_blocking=non_blocking)
    params = params.to(device, non_blocking=non_blocking)
    return imgs, masks, building_ids, params


def _forward_model(
    model: torch.nn.Module,
    cfg: DictConfig,
    imgs: torch.Tensor,
    params: torch.Tensor,
) -> torch.Tensor:
    if cfg.model.is_conditional and cfg.data.feature_cols:
        return model(imgs, params)
    return model(imgs)


def run_training(cfg: DictConfig) -> None:
    torch.manual_seed(cfg.seed)
    run_dir = HydraConfig.get().runtime.output_dir
    OmegaConf.set_readonly(cfg, False)

    _prepare_directories(cfg, run_dir)
    _initialise_wandb(cfg)

    if cfg.data.feature_cols:
        cfg.model.vec_dim = len(cfg.data.feature_cols)

    cuda_available = torch.cuda.is_available() and cfg.training.device == "cuda"
    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True

    train_loader, val_loader = _create_dataloaders(cfg, cuda_available)

    model = build_model(cfg).to(device)
    loss_fn = build_loss(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )

    building_metrics_enabled_global, raise_on_missing_global = (
        resolve_building_metrics_settings(cfg)
    )
    building_cfg = getattr(getattr(cfg, "evaluation", {}), "building_metrics", {})
    log_centroid_overlays = bool(getattr(building_cfg, "log_centroid_overlays", False))
    max_overlay_samples = int(getattr(building_cfg, "max_overlay_samples", 4))

    best_iou = 0.0
    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, masks, building_ids, params, _extras in tqdm(train_loader, desc="Train"):
            imgs, masks, building_ids, params = _move_batch_to_device(
                imgs, masks, building_ids, params, device, cuda_available
            )

            optimizer.zero_grad()

            outputs = _forward_model(model, cfg, imgs, params)

            loss = loss_fn(outputs, masks, building_ids=building_ids)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        tracker = MetricTracker(cfg.model.num_classes, device)
        tracker.reset()

        val_loss = 0.0
        all_imgs, all_masks, all_preds = [], [], []
        building_targets: List[int] = []
        building_predictions: List[int] = []
        building_overlay_samples: List[
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, str, str]
        ] = []
        building_metrics_enabled = building_metrics_enabled_global
        raise_on_missing = raise_on_missing_global

        with torch.no_grad():
            for imgs, masks, building_ids, params, extras in tqdm(val_loader, desc="Val"):
                imgs, masks, building_ids, params = _move_batch_to_device(
                    imgs, masks, building_ids, params, device, cuda_available
                )

                outputs = _forward_model(model, cfg, imgs, params)

                val_loss += loss_fn(outputs, masks, building_ids=building_ids).item()
                preds = outputs.argmax(dim=1)
                tracker.update(preds, masks)

                imgs_cpu = imgs.cpu()
                masks_cpu = masks.cpu()
                preds_cpu = preds.cpu()
                all_imgs.extend(imgs_cpu)
                all_masks.extend(masks_cpu)
                all_preds.extend(preds_cpu)

                _collect_building_metrics_for_epoch(
                    building_metrics_enabled,
                    raise_on_missing,
                    building_targets,
                    building_predictions,
                    extras,
                    masks_cpu,
                    preds_cpu,
                )

                if (
                    log_centroid_overlays
                    and len(building_overlay_samples) < max_overlay_samples
                ):
                    for sample_idx, extra in enumerate(extras):
                        extra = extra or {}
                        building_path = extra.get("building_path")
                        if not building_path or not os.path.exists(building_path):
                            continue
                        uid = extra.get("uid", f"sample_{len(building_overlay_samples)}")
                        building_overlay_samples.append(
                            (
                                imgs_cpu[sample_idx],
                                masks_cpu[sample_idx],
                                preds_cpu[sample_idx],
                                building_path,
                                uid,
                            )
                        )
                        if len(building_overlay_samples) >= max_overlay_samples:
                            break

        val_loss /= len(val_loader)
        val_iou = tracker.compute().item()

        log_payload: Dict[str, object] = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_iou": val_iou,
        }

        building_stats = None
        if building_metrics_enabled:
            building_stats = compute_building_metric_statistics(
                building_targets,
                building_predictions,
                cfg.model.num_classes,
            )

        if building_stats:
            print(
                "[Val] Building metrics | "
                f"Precision (macro): {building_stats['precision_macro']:.4f} | "
                f"Recall (macro): {building_stats['recall_macro']:.4f} | "
                f"F1 (macro): {building_stats['f1_macro']:.4f}"
            )
            print("[Val] Building confusion matrix:\n", building_stats["confusion"])
            update_log_with_building_metrics(log_payload, building_stats)
        elif building_metrics_enabled:
            print("[Val] Building metrics requested but no predictions were collected.")

        wandb.log(log_payload)
        log_filtered_imgs(all_imgs, all_masks, all_preds, epoch)
        if log_centroid_overlays:
            log_building_centroid_overlays(building_overlay_samples, epoch)

        state = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optim": optimizer.state_dict(),
            "val_iou": val_iou,
        }
        is_best = val_iou > best_iou
        best_iou = max(val_iou, best_iou)
        save_checkpoint(state, cfg, is_best)

    wandb.finish()


__all__ = ["run_training"]

