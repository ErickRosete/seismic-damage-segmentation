import os
from hydra.core.hydra_config import HydraConfig
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
from omegaconf import DictConfig, OmegaConf
import hydra

from earthquake_segmentation.dataset import EarthquakeDamageDataset, make_splits
from earthquake_segmentation.model import build_model
from earthquake_segmentation.losses import build_loss
from earthquake_segmentation.utils import (
    MetricTracker,
    save_checkpoint,
    log_filtered_imgs,
)


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def main(cfg: DictConfig):
    torch.manual_seed(cfg.seed)

    run_dir = HydraConfig.get().runtime.output_dir
    OmegaConf.set_readonly(cfg, False)
    cfg.training.checkpoint_dir = os.path.join(run_dir, cfg.training.checkpoint_dir)
    cfg.logging.wandb_dir = os.path.join(run_dir, cfg.logging.wandb_dir)
    os.makedirs(cfg.training.checkpoint_dir, exist_ok=True)
    os.makedirs(cfg.logging.wandb_dir, exist_ok=True)
    wandb.init(
        project=cfg.logging.wandb_project,
        config=OmegaConf.to_container(cfg, resolve=True),  ## type: ignore
        dir=cfg.logging.wandb_dir,
    )

    train_ids, val_ids = make_splits(cfg)
    train_ds = EarthquakeDamageDataset(train_ids, cfg, mode="train")
    val_ds = EarthquakeDamageDataset(val_ids, cfg, mode="val")

    train_loader = DataLoader(
        train_ds, batch_size=cfg.training.batch_size, shuffle=True, num_workers=4
    )
    val_loader = DataLoader(
        val_ds, batch_size=cfg.training.batch_size, shuffle=False, num_workers=4
    )

    device = torch.device(cfg.training.device if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    loss_fn = build_loss(cfg)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.training.lr, weight_decay=cfg.training.weight_decay
    )

    best_iou = 0.0
    for epoch in range(1, cfg.training.epochs + 1):
        model.train()
        train_loss = 0.0
        for imgs, masks, _ in tqdm(train_loader, desc="Train"):
            imgs, masks = imgs.to(device), masks.to(device)
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = loss_fn(outputs, masks)
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        model.eval()
        tracker = MetricTracker(cfg.model.num_classes, device)
        tracker.reset()

        val_loss = 0.0
        all_imgs, all_masks, all_preds = [], [], []
        with torch.no_grad():
            for imgs, masks, _ in tqdm(val_loader, desc="Val"):
                imgs, masks = imgs.to(device), masks.to(device)
                outputs = model(imgs)
                val_loss += loss_fn(outputs, masks).item()
                preds = outputs.argmax(dim=1)
                tracker.update(preds, masks)
                all_imgs.extend(imgs.cpu())
                all_masks.extend(masks.cpu())
                all_preds.extend(preds.cpu())
        val_loss /= len(val_loader)
        val_iou = tracker.compute().item()

        wandb.log(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_iou": val_iou,
            }
        )

        log_filtered_imgs(all_imgs, all_masks, all_preds, epoch)

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


if __name__ == "__main__":
    main()
