from pathlib import Path
import pandas as pd


import rasterio
import matplotlib.pyplot as plt


# medbfs-afghanistan
def show_triplet(uid, base_dir=".", figsize=(9, 3)):
    # build paths
    pre_path = f"{base_dir}/pre-disaster/{uid}_pre_disaster.tif"
    post_path = f"{base_dir}/post-disaster/{uid}_post_disaster.tif"
    label_path = f"{base_dir}/labels/{uid}_label.tif"

    # read each
    with rasterio.open(pre_path) as src:
        pre = src.read([1, 2, 3])  # shape (3, H, W)
    with rasterio.open(post_path) as src:
        post = src.read([1, 2, 3])
    with rasterio.open(label_path) as src:
        lbl = src.read(1)  # labels are single‐band

    # plot
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    axes[0].imshow(pre.transpose(1, 2, 0))
    axes[0].set_title(f"{uid} — Pre‑disaster")
    axes[1].imshow(post.transpose(1, 2, 0))
    axes[1].set_title(f"{uid} — Post‑disaster")
    # for labels, use a colormap
    im = axes[2].imshow(lbl, cmap="tab20")
    axes[2].set_title(f"{uid} — Labels")
    fig.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)

    for ax in axes:
        ax.set_axis_off()
    plt.tight_layout()
    plt.show()


dataset_dir = "data/earthquake_dataset"
metadata_df = pd.read_csv(f"{dataset_dir}/metadata.csv")
# random order of metadata
metadata_df = metadata_df.sample(frac=1, random_state=42).reset_index(drop=True)

first_uids_df = metadata_df.drop_duplicates(subset="dataset", keep="first")[
    ["dataset", "uid"]
]

for uid in first_uids_df["uid"].tolist():
    show_triplet(uid, base_dir=dataset_dir)
