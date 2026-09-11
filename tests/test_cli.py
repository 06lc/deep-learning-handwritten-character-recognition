from __future__ import annotations

from pathlib import Path

import pytest

from config import TrainConfig
from main import build_parser


def test_cli_exposes_index_train_evaluate_and_predict_commands() -> None:
    parser = build_parser()

    assert parser.parse_args(["index"]).command == "index"
    assert parser.parse_args(["train", "--epochs", "1"]).command == "train"
    assert parser.parse_args(["evaluate", "--checkpoint", "best.pt"]).command == "evaluate"
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
    with pytest.raises(ValueError, match="cannot be used together"):
        TrainConfig(resume=Path("last.pt"), finetune_from=Path("best.pt"))
