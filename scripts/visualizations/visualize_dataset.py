import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
import hydra
from omegaconf import DictConfig

from earthquake_segmentation.dataset import EarthquakeDamageDataset, make_splits


@hydra.main(version_base=None, config_path="conf", config_name="config")
def visualize(cfg: DictConfig):
    """
    Visualize augmented images and masks for train or validation set.
    Use overrides via `visualize.stage` and `visualize.n` in Hydra.
    """
    # Retrieve visualization settings
    stage = cfg.visualize.stage  # 'train' or 'val'
    n = cfg.visualize.n  # number of samples

    # Determine prefixes for split
    train_ids, val_ids = make_splits(cfg)
    prefixes = train_ids if stage == "train" else val_ids

    # Load data
    ds = EarthquakeDamageDataset(prefixes, cfg, mode=stage)
    loader = DataLoader(ds, batch_size=n, shuffle=True)
    imgs, masks, _, extras = next(iter(loader))

    # Plot images and masks
    # Ensure axes is always a 2D array so indexing works when n == 1
    fig, axes = plt.subplots(n, 2, figsize=(6, 3 * n), squeeze=False)
    for i in range(n):
        img = imgs[i].permute(1, 2, 0).numpy()
        mask = masks[i].numpy()
        axes[i, 0].imshow(img)
        axes[i, 0].set_title(f"{stage} image")
        axes[i, 0].axis("off")
        axes[i, 1].imshow(mask, vmin=0, vmax=cfg.model.num_classes - 1, cmap="tab10")
        axes[i, 1].set_title(f"{stage} mask")
        axes[i, 1].axis("off")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    visualize()
