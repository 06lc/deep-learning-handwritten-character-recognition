from __future__ import annotations

import struct
from dataclasses import replace
from pathlib import Path

import numpy as np
from PIL import Image

from config import TrainConfig
from predict import predict_image
from train import evaluate_checkpoint, fit, load_checkpoint


def _write_gnt(path: Path, records: list[tuple[str, np.ndarray]]) -> None:
    with path.open("wb") as handle:
        for label, image in records:
            image = np.asarray(image, dtype=np.uint8)
            encoded = label.encode("gbk")
            if len(encoded) == 1:
                encoded += b"\x00"
            height, width = image.shape
            handle.write(struct.pack("<I2sHH", 10 + width * height, encoded, width, height))
            handle.write(image.tobytes())


def _make_fixture(root: Path) -> tuple[tuple[Path, Path], Path, Path]:
    train_part1 = root / "train-1"
    train_part2 = root / "train-2"
    test_root = root / "test"
    for directory in (train_part1, train_part2, test_root):
        directory.mkdir()
    images = {
        "A": np.array([[255, 0], [255, 0]], dtype=np.uint8),
        "B": np.array([[0, 255], [0, 255]], dtype=np.uint8),
    }
    for index, directory in enumerate((train_part1, train_part2), start=1001):
        _write_gnt(
            directory / f"{index}-f.gnt",
            [(label, image) for label, image in images.items()],
        )
    _write_gnt(test_root / "1241-f.gnt", [(label, image) for label, image in images.items()])
    input_image = root / "input.png"
    Image.fromarray(images["A"]).save(input_image)
    return (train_part1, train_part2), test_root, input_image


def test_fit_evaluate_and_predict_round_trip(tmp_path: Path) -> None:
    train_roots, test_root, test_file = _make_fixture(tmp_path)
    config = TrainConfig(
        train_roots=train_roots,
        test_root=test_root,
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "outputs",
        image_size=16,
        batch_size=2,
        epochs=1,
        val_fraction=0.5,
        num_workers=0,
        device="cpu",
        expected_num_classes=2,
        patience=1,
        max_train_batches=1,
        max_eval_batches=1,
    )

    result = fit(config)
    checkpoint = config.output_dir / "best.pt"
    assert checkpoint.is_file()
    assert result["class_names"] == ["A", "B"]

    metrics = evaluate_checkpoint(checkpoint, config)
    assert 0.0 <= metrics["top1"] <= 1.0
    assert 0.0 <= metrics["top5"] <= 1.0

    model, metadata = load_checkpoint(checkpoint, device="cpu")
    prediction = predict_image(
        model,
        test_file,
        metadata["class_names"],
        image_size=16,
        device="cpu",
        top_k=2,
    )
    assert prediction["prediction"] in {"A", "B"}
    assert len(prediction["top_k"]) == 2

    resumed_config = replace(
        config,
        output_dir=tmp_path / "resumed-outputs",
        epochs=2,
        resume=checkpoint,
    )
    resumed = fit(resumed_config)
    assert resumed["history"][0]["epoch"] == 2
