"""HWDB-1.1 命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from config import OUTPUT_ROOT, TEST_ROOT, TRAIN_ROOTS, TrainConfig
from predict import image_paths, predict_image
from train import evaluate_checkpoint, fit, load_checkpoint, prepare_indexes, resolve_device


def _add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-root", action="append", dest="train_roots", type=Path)
    parser.add_argument("--test-root", type=Path, default=TEST_ROOT)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=Path(r"D:\Python+Ai\03_机器学习\手写字符识别\cache"),
    )


def _add_train_arguments(parser: argparse.ArgumentParser) -> None:
    _add_dataset_arguments(parser)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--val-fraction", type=float, default=0.1)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--label-smoothing", type=float, default=0.05)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--no-amp", action="store_true")
    parser.add_argument("--expected-num-classes", type=int, default=3926)
    parser.add_argument("--rebuild-index", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-eval-batches", type=int)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="hwdb-recognizer",
        description="PyTorch CASIA-HWDB-1.1 handwritten character recognition",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="build and validate GNT indexes")
    _add_dataset_arguments(index_parser)
    index_parser.add_argument("--expected-num-classes", type=int, default=3926)
    index_parser.add_argument("--rebuild-index", action="store_true")

    train_parser = subparsers.add_parser("train", help="train the CNN")
    _add_train_arguments(train_parser)

    evaluate_parser = subparsers.add_parser("evaluate", help="evaluate a checkpoint on Test")
    evaluate_parser.add_argument("--checkpoint", type=Path, required=True)
    _add_dataset_arguments(evaluate_parser)
    evaluate_parser.add_argument("--batch-size", type=int, default=128)
    evaluate_parser.add_argument("--num-workers", type=int, default=0)
    evaluate_parser.add_argument("--device", default="auto")
    evaluate_parser.add_argument("--output-dir", type=Path)
    evaluate_parser.add_argument("--rebuild-index", action="store_true")
    evaluate_parser.add_argument("--max-eval-batches", type=int)

    predict_parser = subparsers.add_parser("predict", help="predict one image or a directory")
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    input_group = predict_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", type=Path)
    input_group.add_argument("--input-dir", type=Path)
    predict_parser.add_argument("--top-k", type=int, default=5)
    predict_parser.add_argument("--device", default="auto")
    return parser


def _dataset_roots(args: argparse.Namespace) -> tuple[tuple[Path, ...], Path, Path]:
    train_roots = tuple(args.train_roots) if args.train_roots else TRAIN_ROOTS
    return train_roots, args.test_root, args.cache_dir


def _run_index(args: argparse.Namespace) -> int:
    train_roots, test_root, cache_dir = _dataset_roots(args)
    config = TrainConfig(
        train_roots=train_roots,
        test_root=test_root,
        cache_dir=cache_dir,
        expected_num_classes=args.expected_num_classes,
        rebuild_index=args.rebuild_index,
    )
    train_index, test_index = prepare_indexes(config)
    print(
        json.dumps(
            {
                "train_files": len(train_index.files),
                "train_samples": len(train_index.records),
                "test_files": len(test_index.files),
                "test_samples": len(test_index.records),
                "num_classes": len(train_index.class_names),
                "cache_dir": str(cache_dir),
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _run_train(args: argparse.Namespace) -> int:
    train_roots, test_root, cache_dir = _dataset_roots(args)
    config = TrainConfig(
        train_roots=train_roots,
        test_root=test_root,
        cache_dir=cache_dir,
        output_dir=args.output_dir,
        image_size=args.image_size,
        batch_size=args.batch_size,
        epochs=args.epochs,
        val_fraction=args.val_fraction,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        label_smoothing=args.label_smoothing,
        patience=args.patience,
        num_workers=args.num_workers,
        seed=args.seed,
        device=args.device,
        amp=not args.no_amp,
        expected_num_classes=args.expected_num_classes,
        rebuild_index=args.rebuild_index,
        resume=args.resume,
        max_train_batches=args.max_train_batches,
        max_eval_batches=args.max_eval_batches,
    )
    print(json.dumps(fit(config), ensure_ascii=False, indent=2))
    return 0


def _run_evaluate(args: argparse.Namespace) -> int:
    train_roots, test_root, cache_dir = _dataset_roots(args)
    config = TrainConfig(
        train_roots=train_roots,
        test_root=test_root,
        cache_dir=cache_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        device=args.device,
        expected_num_classes=None,
        rebuild_index=args.rebuild_index,
        max_eval_batches=args.max_eval_batches,
    )
    print(
        json.dumps(
            evaluate_checkpoint(args.checkpoint, config, args.output_dir),
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


def _run_predict(args: argparse.Namespace) -> int:
    device = resolve_device(args.device)
    model, metadata = load_checkpoint(args.checkpoint, device)
    image_size = int(metadata.get("config", {}).get("image_size", 64))
    paths = [args.input] if args.input else image_paths(args.input_dir)
    results = [
        predict_image(model, path, metadata["class_names"], image_size, device, args.top_k)
        for path in paths
    ]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if args.command == "index":
            return _run_index(args)
        if args.command == "train":
            return _run_train(args)
        if args.command == "evaluate":
            return _run_evaluate(args)
        if args.command == "predict":
            return _run_predict(args)
        parser.error(f"unknown command: {args.command}")
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
