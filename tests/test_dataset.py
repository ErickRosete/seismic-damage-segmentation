import json
from types import SimpleNamespace

import pytest

pytest.importorskip("numpy")
pytest.importorskip("rasterio")

import numpy as np
import rasterio
from rasterio.transform import from_origin

from earthquake_segmentation.dataset import EarthquakeDamageDataset


def _write_raster(path, array, transform):
    height, width = array.shape[1:]
    count = array.shape[0]
    dtype = array.dtype
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        height=height,
        width=width,
        count=count,
        dtype=dtype,
        transform=transform,
    ) as dst:
        for band in range(count):
            dst.write(array[band], band + 1)


def test_geojson_building_annotations(tmp_path):
    data_root = tmp_path / "data"
    train_root = data_root / "train"
    (train_root / "pre-disaster").mkdir(parents=True)
    (train_root / "labels").mkdir(parents=True)
    (train_root / "buildings").mkdir(parents=True)

    uid = "sample"
    transform = from_origin(0, 4, 1, 1)

    pre_image = np.stack(
        [np.full((4, 4), fill_value=idx, dtype=np.uint8) for idx in range(1, 4)]
    )
    _write_raster(train_root / "pre-disaster" / f"{uid}_pre_disaster.tif", pre_image, transform)

    label = np.zeros((1, 4, 4), dtype=np.uint8)
    _write_raster(train_root / "labels" / f"{uid}_label.tif", label, transform)

    building_geojson = train_root / "buildings" / f"{uid}_buildings.geojson"
    features = [
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[0, 4], [1, 4], [1, 3], [0, 3], [0, 4]],
                ],
            },
            "properties": {},
        },
        {
            "type": "Feature",
            "geometry": {
                "type": "Polygon",
                "coordinates": [
                    [[1, 3], [2, 3], [2, 2], [1, 2], [1, 3]],
                ],
            },
            "properties": {},
        },
    ]
    building_geojson.write_text(json.dumps({"type": "FeatureCollection", "features": features}))

    cfg = SimpleNamespace()
    cfg.data = SimpleNamespace(dir=str(data_root), feature_cols=[])
    cfg.augmentations = {"train": []}

    dataset = EarthquakeDamageDataset([uid], cfg, mode="train")

    _image, _mask, building, _params, extras = dataset[0]

    unique_ids = building.unique().tolist()
    assert any(val > 0 for val in unique_ids), "Expected non-zero building identifiers"
    assert extras["building_path"] == str(building_geojson)


def test_missing_building_annotations(tmp_path):
    data_root = tmp_path / "data"
    train_root = data_root / "train"
    (train_root / "pre-disaster").mkdir(parents=True)
    (train_root / "labels").mkdir(parents=True)
    (train_root / "buildings").mkdir(parents=True)

    uid = "sample"
    transform = from_origin(0, 4, 1, 1)

    pre_image = np.stack(
        [np.full((4, 4), fill_value=idx, dtype=np.uint8) for idx in range(1, 4)]
    )
    _write_raster(train_root / "pre-disaster" / f"{uid}_pre_disaster.tif", pre_image, transform)

    label = np.zeros((1, 4, 4), dtype=np.uint8)
    _write_raster(train_root / "labels" / f"{uid}_label.tif", label, transform)

    cfg = SimpleNamespace()
    cfg.data = SimpleNamespace(dir=str(data_root), feature_cols=[])
    cfg.augmentations = {"train": []}

    dataset = EarthquakeDamageDataset([uid], cfg, mode="train")

    _image, _mask, building, _params, extras = dataset[0]

    assert building.sum().item() == 0
    assert extras["building_path"] == ""
