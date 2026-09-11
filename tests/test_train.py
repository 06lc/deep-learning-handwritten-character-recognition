from __future__ import annotations

import struct
from pathlib import Path

import numpy as np

from config import TrainConfig
from train import prepare_indexes


def _write_one_record(path: Path, label: str) -> None:
    image = np.zeros((2, 2), dtype=np.uint8)
    encoded = label.encode("gbk")
    if len(encoded) == 1:
        encoded += b"\x00"
    with path.open("wb") as handle:
        handle.write(struct.pack("<I2sHH", 14, encoded, 2, 2))
        handle.write(image.tobytes())


def test_prepare_indexes_rebuilds_when_configured_roots_change(tmp_path: Path) -> None:
    first_train = tmp_path / "first-train"
    second_train = tmp_path / "second-train"
    test_root = tmp_path / "test"
    for directory in (first_train, second_train, test_root):
        directory.mkdir()
    _write_one_record(first_train / "1001-f.gnt", "A")
    _write_one_record(second_train / "1002-f.gnt", "A")
    _write_one_record(test_root / "1241-f.gnt", "A")

    cache_dir = tmp_path / "cache"
    first_config = TrainConfig(
        train_roots=(first_train,),
        test_root=test_root,
        cache_dir=cache_dir,
        expected_num_classes=1,
    )
    prepare_indexes(first_config)

    second_config = TrainConfig(
        train_roots=(second_train,),
        test_root=test_root,
        cache_dir=cache_dir,
        expected_num_classes=1,
    )
    train_index, _ = prepare_indexes(second_config)

    assert train_index.files == (second_train.resolve() / "1002-f.gnt",)
