# Proposed Experiment: Dice Loss for Building Damage Segmentation

This experiment evaluates whether combining Dice loss with cross entropy improves model performance on the earthquake damage dataset.

## Configuration

Update `conf/config.yaml`:

```yaml
loss:
  name: cross_entropy_dice
  dice_weight: 1.0
```

## Rationale

Cross entropy focuses on pixel-wise accuracy but may underperform on imbalanced classes. Dice loss directly optimizes overlap between predicted and ground truth regions, which helps segment damaged buildings that occupy small areas.

## Expected Outcome

We expect higher mean IoU compared to training with plain cross entropy. Results will be logged to Weights & Biases for comparison.

