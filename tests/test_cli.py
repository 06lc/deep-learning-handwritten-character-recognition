from __future__ import annotations

from pathlib import Path

import pytest
import torch

from config import TrainConfig
from main import build_parser, main
from model import create_model
from train import load_checkpoint


def test_cli_exposes_index_train_evaluate_and_predict_commands() -> None:
    parser = build_parser()

    assert parser.parse_args(["index"]).command == "index"
    assert parser.parse_args(["train", "--epochs", "1"]).command == "train"
    assert parser.parse_args(["evaluate", "--checkpoint", "best.pt"]).command == "evaluate"
    assert parser.parse_args(["analyze", "--checkpoint", "best.pt"]).command == "analyze"
    assert (
        parser.parse_args(["predict", "--checkpoint", "best.pt", "--input", "sample.png"]).command
        == "predict"
    )
    assert parser.parse_args(["compress", "--checkpoint", "best.pt"]).command == "compress"
    assert parser.parse_args(["benchmark", "--checkpoint", "best.pt"]).command == "benchmark"
    assert parser.parse_args(["export", "--checkpoint", "best.pt"]).command == "export"


def test_resume_and_finetune_from_are_mutually_exclusive(
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser = build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(
            ["train", "--resume", "last.pt", "--finetune-from", "best.pt"]
        )
    error = capsys.readouterr().err
    assert "--finetune-from" in error
    assert "not allowed with argument --resume" in error
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["train", "--finetune-from", "best.pt", "--warm-start-from", "base.pt"]
        )
    with pytest.raises(ValueError, match="cannot be used together"):
        TrainConfig(resume=Path("last.pt"), finetune_from=Path("best.pt"))


def test_export_accepts_fp16_checkpoint(tmp_path: Path) -> None:
    model = create_model("cnn", 2).half()
    checkpoint = tmp_path / "source-fp16.pt"
    output_dir = tmp_path / "exported"
    torch.save(
        {
            "format_version": 3,
            "model_name": "cnn",
            "class_names": ["A", "B"],
            "class_to_idx": {"A": 0, "B": 1},
            "image_size": 16,
            "preprocessing": {
                "profile": "margin_v1",
                "augmentation_profile": "gentle_elastic",
            },
            "precision": "fp16",
            "model_state": model.state_dict(),
        },
        checkpoint,
    )

    exit_code = main(
        [
            "export",
            "--checkpoint",
            str(checkpoint),
            "--output-dir",
            str(output_dir),
        ]
    )
    _, exported = load_checkpoint(output_dir / "fp16.pt")

    assert exit_code == 0
    assert (output_dir / "model.ts").is_file()
    assert exported["preprocessing"]["profile"] == "margin_v1"
