import pytest

torch = pytest.importorskip("torch")
import torch.nn.functional as F
from omegaconf import OmegaConf

from earthquake_segmentation.losses import build_loss, dice_loss, focal_loss


def _make_loss(cfg_dict):
    cfg = OmegaConf.create({"loss": cfg_dict})
    return build_loss(cfg)


def _make_building_loss(cfg_dict):
    cfg = OmegaConf.create({"loss": cfg_dict})
    return build_loss(cfg)


def test_building_pooled_loss_combines_pixel_and_building_terms():
    probs = torch.tensor(
        [
            [
                [0.1, 0.2],
                [0.6, 0.1],
            ],
            [
                [0.8, 0.7],
                [0.2, 0.4],
            ],
            [
                [0.1, 0.1],
                [0.2, 0.5],
            ],
        ],
        dtype=torch.float32,
    )
    logits = torch.log(probs).unsqueeze(0)
    target = torch.tensor([[1, 1], [0, 2]], dtype=torch.long).unsqueeze(0)
    building_ids = torch.tensor([[1, 1], [0, 2]], dtype=torch.long).unsqueeze(0)

    loss_fn = _make_building_loss(
        {
            "name": "building_pooled",
            "pixel_loss": "cross_entropy",
            "pixel_weight": 1.0,
            "building_weight": 0.5,
            "building_background_id": 0,
        }
    )

    loss = loss_fn(logits, target, building_ids=building_ids)

    pixel_loss = torch.nn.functional.cross_entropy(logits, target)
    building_probs_majority = torch.tensor([
        torch.mean(torch.tensor([0.8, 0.7])),
        torch.tensor(0.5),
    ])
    expected_building_loss = -torch.log(building_probs_majority).mean()
    expected = pixel_loss + 0.5 * expected_building_loss
    assert torch.allclose(loss, expected, atol=1e-6)


def test_building_pooled_loss_skips_unlabeled_buildings():
    logits = torch.zeros((1, 2, 2, 2), dtype=torch.float32, requires_grad=True)
    target = torch.full((1, 2, 2), -1, dtype=torch.long)
    building_ids = torch.tensor([[1, 1], [2, 2]], dtype=torch.long).unsqueeze(0)

    loss_fn = _make_building_loss(
        {
            "name": "building_pooled",
            "pixel_loss": "cross_entropy",
            "pixel_weight": 0.0,
            "building_weight": 1.0,
            "building_background_id": 0,
            "building_ignore_index": -1,
        }
    )

    loss = loss_fn(logits, target, building_ids=building_ids)
    assert torch.allclose(loss, torch.tensor(0.0))


def test_building_pooled_loss_handles_batches_with_missing_buildings():
    logits = torch.zeros((2, 2, 2, 2), dtype=torch.float32, requires_grad=True)
    target = torch.zeros((2, 2, 2), dtype=torch.long)
    building_ids = torch.tensor(
        [
            [[0, 0], [1, 1]],
            [[0, 0], [0, 0]],
        ],
        dtype=torch.long,
    )

    loss_fn = _make_building_loss(
        {
            "name": "building_pooled",
            "pixel_loss": "cross_entropy",
            "pixel_weight": 1.0,
            "building_weight": 1.0,
            "building_background_id": 0,
        }
    )

    loss = loss_fn(logits, target, building_ids=building_ids)
    assert torch.isfinite(loss)


def test_focal_loss_matches_cross_entropy_when_gamma_zero():
    logits = torch.randn((2, 3, 4, 4), requires_grad=True)
    target = torch.randint(0, 3, (2, 4, 4))

    loss_fn = _make_loss({"name": "focal", "gamma": 0.0})
    ce = torch.nn.functional.cross_entropy(logits, target)
    focal = loss_fn(logits, target)

    assert torch.allclose(focal, ce, atol=1e-6)


def test_focal_loss_handles_all_ignored_pixels():
    logits = torch.zeros((1, 2, 2, 2), dtype=torch.float32)
    target = torch.full((1, 2, 2), -1, dtype=torch.long)

    loss_fn = _make_loss({"name": "focal", "ignore_index": -1})
    loss = loss_fn(logits, target)

    assert torch.allclose(loss, torch.tensor(0.0))


def test_cross_entropy_dice_combination_uses_weights():
    probs = torch.tensor(
        [
            [[0.9, 0.7], [0.6, 0.8]],
            [[0.1, 0.3], [0.4, 0.2]],
        ],
        dtype=torch.float32,
    )
    logits = torch.log(probs).unsqueeze(0)
    target = torch.tensor([[0, 1], [0, 1]], dtype=torch.long).unsqueeze(0)

    loss_fn = _make_loss(
        {
            "name": "cross_entropy_dice",
            "cross_entropy_weight": 2.0,
            "dice_weight": 0.5,
        }
    )

    loss = loss_fn(logits, target)
    ce = F.cross_entropy(logits, target)
    dice = dice_loss(logits, target)
    expected = 2.0 * ce + 0.5 * dice

    assert torch.allclose(loss, expected, atol=1e-6)


def test_focal_dice_combination_uses_weights():
    probs = torch.tensor(
        [
            [[0.9, 0.7], [0.6, 0.8]],
            [[0.1, 0.3], [0.4, 0.2]],
        ],
        dtype=torch.float32,
    )
    logits = torch.log(probs).unsqueeze(0)
    target = torch.tensor([[0, 1], [0, 1]], dtype=torch.long).unsqueeze(0)

    loss_fn = _make_loss(
        {
            "name": "focal_dice",
            "focal_weight": 2.0,
            "dice_weight": 0.5,
            "gamma": 0.0,
        }
    )

    loss = loss_fn(logits, target)
    focal = focal_loss(logits, target, gamma=0.0)
    dice = dice_loss(logits, target)
    expected = 2.0 * focal + 0.5 * dice

    assert torch.allclose(loss, expected, atol=1e-6)


def test_cross_entropy_focal_combination_uses_weights():
    probs = torch.tensor(
        [
            [[0.6, 0.2], [0.7, 0.4]],
            [[0.4, 0.8], [0.3, 0.6]],
        ],
        dtype=torch.float32,
    )
    logits = torch.log(probs).unsqueeze(0)
    target = torch.tensor([[0, 1], [0, 1]], dtype=torch.long).unsqueeze(0)

    loss_fn = _make_loss(
        {
            "name": "cross_entropy_focal",
            "cross_entropy_weight": 0.5,
            "focal_weight": 1.5,
            "gamma": 0.0,
        }
    )

    loss = loss_fn(logits, target)
    ce = F.cross_entropy(logits, target)
    focal = focal_loss(logits, target, gamma=0.0)
    expected = 0.5 * ce + 1.5 * focal

    assert torch.allclose(loss, expected, atol=1e-6)
