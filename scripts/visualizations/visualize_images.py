from pathlib import Path
from typing import Iterable

import rasterio
import matplotlib.pyplot as plt


def show_triplet(uid: str, base_dir: Path, split: str = "train", figsize=(9, 3)) -> None:
    """Visualize pre/post imagery and labels for a single UID."""

    split_dir = base_dir / split
    pre_path = split_dir / "pre-disaster" / f"{uid}_pre_disaster.tif"
    post_path = split_dir / "post-disaster" / f"{uid}_post_disaster.tif"
    label_path = split_dir / "labels" / f"{uid}_label.tif"

    with rasterio.open(pre_path) as src:
        pre = src.read([1, 2, 3])
    with rasterio.open(post_path) as src:
        post = src.read([1, 2, 3])
    with rasterio.open(label_path) as src:
        lbl = src.read(1)

    fig, axes = plt.subplots(1, 3, figsize=figsize)
    axes[0].imshow(pre.transpose(1, 2, 0))
    axes[0].set_title(f"{uid} — Pre‑disaster")
    axes[1].imshow(post.transpose(1, 2, 0))
    axes[1].set_title(f"{uid} — Post‑disaster")
    im = axes[2].imshow(lbl, cmap="tab20")
    axes[2].set_title(f"{uid} — Labels")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_axis_off()
    plt.tight_layout()
    plt.show()


def iter_uids(label_dir: Path) -> Iterable[str]:
    for tif_path in sorted(label_dir.glob("*_label.tif")):
        yield tif_path.name[: -len("_label.tif")]


if __name__ == "__main__":
    dataset_dir = Path("data/earthquake_dataset")
    max_examples = 3
    for split in ("train", "val"):
        labels_dir = dataset_dir / split / "labels"
        if not labels_dir.exists():
            continue
        for idx, uid in enumerate(iter_uids(labels_dir)):
            show_triplet(uid, dataset_dir, split=split)
            if idx + 1 >= max_examples:
                break
