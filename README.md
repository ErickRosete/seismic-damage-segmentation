# Earthquake Damage Prediction

This repository contains code to load paired pre-disaster, post-disaster, and pixel-wise label TIFF images into PyTorch, plus optional earthquake parameters.

## Project Structure

. 
├── data/
│   └── earthquake_dataset/
│       ├── train/
│       │   ├── pre-disaster/     # ⟵ *.tif (RGB imagery)
│       │   ├── post-disaster/    # ⟵ *.tif (RGB imagery)
│       │   ├── labels/           # ⟵ *_label.tif (segmentation masks)
│       │   ├── overlays/         # ⟵ optional visualization overlays
│       │   └── buildings/        # ⟵ per-UID building metadata
│       └── val/
│           └── ... (same layout as train)
├── scripts/
│   └── visualizations/
│       ├── visualize_dataset.py
│       └── visualize_images.py
└── src/
    └── earthquake_segmentation/
        └── dataset.py            # PyTorch Dataset + helpers

## Image dimensions

All training and validation transforms operate on 512×512 imagery so that model
predictions line up with the provided building polygons. The training pipeline
uses a `RandomResizedCrop` (512×512 target) while validation applies a
deterministic `Resize`, both defined in `conf/config.yaml`.

## Installation

```bash
git clone <repo>
cd earthquake_segmentation_project
uv sync
uv run python main.py
```

### Configurable Loss

Training uses the loss specified in `conf/config.yaml`:

```yaml
loss:
  name: cross_entropy_dice # other options: cross_entropy or dice
  dice_weight: 1.0
```
