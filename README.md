# Earthquake Damage Prediction

This repository contains code to load paired pre-disaster, post-disaster, and pixel-wise label TIFF images into PyTorch, plus optional earthquake parameters.

## Project Structure

.
├── dataset/
│ ├── labels/ # ⟵ *.tif (pixel labels 0,1,2)
│ ├── pre-disaster/ # ⟵ *.tif (RGB)
│ └── post-disaster/ # ⟵ *.tif (RGB)
├── earthquake_dataset.py # ⟵ PyTorch Dataset + example
├── requirements.txt
└── README.md

## Installation

```bash
git clone <repo>
cd earthquake_segmentation_project

conda create -n building_damage python=3.12
conda activate building_damage
pip install -e .
```

### Configurable Loss

Training uses the loss specified in `conf/config.yaml`:

```yaml
loss:
  name: cross_entropy_dice  # other options: cross_entropy or dice
  dice_weight: 1.0
```
