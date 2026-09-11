from __future__ import annotations

from main import build_parser


def test_cli_exposes_index_train_evaluate_and_predict_commands() -> None:
    parser = build_parser()

    assert parser.parse_args(["index"]).command == "index"
    assert parser.parse_args(["train", "--epochs", "1"]).command == "train"
    assert parser.parse_args(["evaluate", "--checkpoint", "best.pt"]).command == "evaluate"
    assert (
        parser.parse_args(["predict", "--checkpoint", "best.pt", "--input", "sample.png"])
        .command
        == "predict"
    )
