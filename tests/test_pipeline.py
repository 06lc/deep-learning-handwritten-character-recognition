from __future__ import annotations

import struct
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest
import torch
from PIL import Image

from compression import replace_conv_with_gslre
from config import TrainConfig
from error_analysis import analyze_checkpoint
from model import create_model
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

    analysis = analyze_checkpoint(
        checkpoint,
        config,
        output_dir=tmp_path / "analysis",
        max_errors=2,
    )
    assert analysis["split"] == "validation"
    assert analysis["samples"] == 2
    assert (tmp_path / "analysis" / "analysis.json").is_file()

    resumed_config = replace(
        config,
        output_dir=tmp_path / "resumed-outputs",
        epochs=2,
        learning_rate=9e-4,
        resume=checkpoint,
    )
    resumed = fit(resumed_config)
    assert resumed["history"][0]["epoch"] == 2

    source_payload = torch.load(checkpoint, map_location="cpu", weights_only=False)
    assert source_payload["primary_weight_source"] == "ema"
    assert source_payload["ema_model_state"] is not None
    assert source_payload["training_model_state"] is not None
    assert source_payload["validation_split"]["split_seed"] == 42
    resumed_payload = torch.load(
        resumed_config.output_dir / "last.pt", map_location="cpu", weights_only=False
    )
    source_optimizer = source_payload["optimizer_state"]
    resumed_optimizer = resumed_payload["optimizer_state"]
    assert resumed_optimizer["param_groups"][0]["initial_lr"] == pytest.approx(
        source_optimizer["param_groups"][0]["initial_lr"]
    )
    assert resumed_optimizer["param_groups"][0]["initial_lr"] != pytest.approx(
        resumed_config.learning_rate
    )
    assert resumed_payload["scheduler_state"]["T_max"] == source_payload["scheduler_state"][
        "T_max"
    ]


def test_finetune_starts_at_epoch_one_with_new_learning_rate(tmp_path: Path) -> None:
    train_roots, test_root, _ = _make_fixture(tmp_path)
    source_config = TrainConfig(
        train_roots=train_roots,
        test_root=test_root,
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "source-outputs",
        image_size=16,
        batch_size=2,
        epochs=1,
        val_fraction=0.5,
        learning_rate=3e-4,
        num_workers=0,
        device="cpu",
        expected_num_classes=2,
        patience=1,
        max_train_batches=1,
        max_eval_batches=1,
    )
    fit(source_config)
    source_checkpoint = source_config.output_dir / "best.pt"
    fine_tune_learning_rate = 1e-5
    fine_tune_config = replace(
        source_config,
        output_dir=tmp_path / "finetune-outputs",
        learning_rate=fine_tune_learning_rate,
        finetune_from=source_checkpoint,
    )

    result = fit(fine_tune_config)

    payload = torch.load(
        fine_tune_config.output_dir / "last.pt", map_location="cpu", weights_only=False
    )
    assert result["history"][0]["epoch"] == 1
    assert payload["epoch"] == 1
    assert payload["optimizer_state"]["param_groups"][0]["initial_lr"] == pytest.approx(
        fine_tune_learning_rate
    )
    assert payload["scheduler_state"]["base_lrs"] == pytest.approx(
        [fine_tune_learning_rate]
    )
    assert payload["finetune_source"] == {
        "checkpoint": str(source_checkpoint.resolve()),
        "epoch": 1,
    }


def test_warm_start_records_source_and_transfer_coverage(tmp_path: Path) -> None:
    train_roots, test_root, _ = _make_fixture(tmp_path)
    source_config = TrainConfig(
        train_roots=train_roots,
        test_root=test_root,
        cache_dir=tmp_path / "cache",
        output_dir=tmp_path / "source",
        model_name="hccr_cnn9",
        image_size=16,
        batch_size=2,
        epochs=1,
        warmup_epochs=0,
        val_fraction=0.5,
        num_workers=0,
        device="cpu",
        expected_num_classes=2,
        max_train_batches=1,
        max_eval_batches=1,
    )
    fit(source_config)
    source_checkpoint = source_config.output_dir / "best.pt"
    target_config = replace(
        source_config,
        output_dir=tmp_path / "warm-started",
        model_name="hccr_cnn9_ra",
        warm_start_from=source_checkpoint,
    )

    result = fit(target_config)
    payload = torch.load(
        target_config.output_dir / "last.pt", map_location="cpu", weights_only=False
    )

    expected_source = {"checkpoint": str(source_checkpoint.resolve()), "epoch": 1}
    assert result["warm_start_source"] == expected_source
    assert payload["warm_start_source"] == expected_source
    assert float(payload["warm_start_report"]["coverage"]) > 0.99


def test_residual_attention_gslre_checkpoint_round_trip(tmp_path: Path) -> None:
    model = create_model("hccr_cnn9_ra", 2)
    replaced = replace_conv_with_gslre(model, rank_ratio=0.5)
    checkpoint = tmp_path / "ra-gslre.pt"
    torch.save(
        {
            "format_version": 3,
            "model_name": "hccr_cnn9_ra_gslre",
            "gslre_rank_ratio": 0.5,
            "class_names": ["A", "B"],
            "class_to_idx": {"A": 0, "B": 1},
            "image_size": 16,
            "model_state": model.state_dict(),
        },
        checkpoint,
    )

    restored, metadata = load_checkpoint(checkpoint)
    restored.eval()

    assert replaced == 7
    assert metadata["model_name"] == "hccr_cnn9_ra_gslre"
    assert restored(torch.randn(1, 1, 16, 16)).shape == (1, 2)
