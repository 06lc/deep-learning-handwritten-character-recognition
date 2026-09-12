from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from config import TrainConfig
from gnt_dataset import build_index
from model import create_model
from train import (
    ExponentialMovingAverage,
    _optimizer_and_scheduler,
    class_balanced_sample_weights,
    load_checkpoint,
    prepare_indexes,
)


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


def test_exponential_moving_average_updates_weights() -> None:
    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.data.fill_(1.0)
    ema = ExponentialMovingAverage(model, decay=0.5)
    model.weight.data.fill_(3.0)

    ema.update(model)

    assert float(ema.model.weight) == pytest.approx(2.0)


def test_warmup_cosine_scheduler_reaches_base_and_minimum_learning_rates() -> None:
    model = torch.nn.Linear(2, 2)
    config = TrainConfig(
        epochs=10,
        learning_rate=3e-4,
        warmup_epochs=2,
        min_learning_rate=1e-6,
    )
    optimizer, scheduler = _optimizer_and_scheduler(model, config)

    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-5)
    for _ in range(2):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(3e-4)
    for _ in range(8):
        optimizer.step()
        scheduler.step()
    assert optimizer.param_groups[0]["lr"] == pytest.approx(1e-6)


def test_old_checkpoint_without_preprocessing_remains_loadable(tmp_path: Path) -> None:
    # 使用旧轻量模型的真实结构，模拟缺少 preprocessing/EMA 字段的历史产物。
    model = create_model("cnn", 2)
    checkpoint = tmp_path / "old.pt"
    torch.save(
        {
            "model_name": "cnn",
            "class_names": ["A", "B"],
            "image_size": 16,
            "model_state": model.state_dict(),
        },
        checkpoint,
    )

    restored, payload = load_checkpoint(checkpoint)

    assert "preprocessing" not in payload
    assert restored(torch.randn(1, 1, 16, 16)).shape == (1, 2)


def test_class_balanced_weights_protect_rare_classes_with_cap(tmp_path: Path) -> None:
    root = tmp_path / "train"
    root.mkdir()
    _write_one_record(root / "rare.gnt", "A")
    for index in range(5):
        _write_one_record(root / f"common-{index}.gnt", "B")
    dataset_index = build_index(root)
    record_indices = list(range(len(dataset_index.records)))

    weights = class_balanced_sample_weights(dataset_index, record_indices)

    rare_label = dataset_index.class_names.index("A")
    rare_position = next(
        position
        for position, record_index in enumerate(record_indices)
        if dataset_index.records[record_index].label == rare_label
    )
    common_positions = [
        position
        for position, record_index in enumerate(record_indices)
        if dataset_index.records[record_index].label != rare_label
    ]
    assert float(weights.max()) <= 2.0
    assert all(
        float(weights[rare_position]) > float(weights[position])
        for position in common_positions
    )
