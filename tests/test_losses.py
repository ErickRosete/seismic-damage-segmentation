import pytest

torch = pytest.importorskip("torch")
from omegaconf import OmegaConf

from earthquake_segmentation.losses import build_loss


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
