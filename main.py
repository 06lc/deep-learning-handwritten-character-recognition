"""HWDB-1.1 命令行入口。"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Sequence
from pathlib import Path

import torch

from compression import (
    apply_adw_pruning,
    compression_stats,
    finalize_pruning,
    model_parameter_count,
    quantized_payload,
    replace_conv_with_gslre,
)
from config import (
    CACHE_ROOT,
    OUTPUT_ROOT,
    TEST_ROOT,
    TRAIN_ROOTS,
    TrainConfig,
    validate_dataset_paths,
)
from predict import image_paths, predict_image
from train import (
    evaluate_checkpoint,
    fit,
    load_checkpoint,
    make_train_validation_loaders,
    prepare_indexes,
    resolve_device,
    run_epoch,
)


def _add_dataset_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--train-root", action="append", dest="train_roots", type=Path)
    parser.add_argument("--test-root", type=Path, default=TEST_ROOT)
    parser.add_argument("--cache-dir", type=Path, default=CACHE_ROOT)


def _add_train_arguments(parser: argparse.ArgumentParser) -> None:
    _add_dataset_arguments(parser)
    parser.add_argument("--output-dir", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--model", dest="model_name", default="hccr_cnn9")
    parser.add_argument("--recipe", choices=("modern", "paper"), default="modern")
    parser.add_argument("--image-size", type=int, default=96)
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
    parser = argparse.ArgumentParser(prog="hwdb-recognizer")
    subparsers = parser.add_subparsers(dest="command", required=True)

    index_parser = subparsers.add_parser("index", help="build and validate GNT indexes")
    _add_dataset_arguments(index_parser)
    index_parser.add_argument("--expected-num-classes", type=int, default=3926)
    index_parser.add_argument("--rebuild-index", action="store_true")

    train_parser = subparsers.add_parser("train", help="train HCCR-CNN9Layer")
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

    predict_parser = subparsers.add_parser("predict", help="predict one image or directory")
    predict_parser.add_argument("--checkpoint", type=Path, required=True)
    input_group = predict_parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument("--input", type=Path)
    input_group.add_argument("--input-dir", type=Path)
    predict_parser.add_argument("--top-k", type=int, default=5)
    predict_parser.add_argument("--device", default="auto")

    compress_parser = subparsers.add_parser("compress", help="apply GSLRE, ADW and quantization")
    compress_parser.add_argument("--checkpoint", type=Path, required=True)
    compress_parser.add_argument("--output-dir", type=Path)
    compress_parser.add_argument(
        "--stages",
        nargs="+",
        choices=("gslre", "adw", "quantize"),
        default=["gslre", "adw", "quantize"],
    )
    compress_parser.add_argument("--rank-ratio", type=float, default=0.5)
    compress_parser.add_argument("--sparsity", type=float, default=0.3)
    compress_parser.add_argument("--clusters", type=int, default=256)
    compress_parser.add_argument("--finetune-epochs", type=int, default=0)
    compress_parser.add_argument("--max-accuracy-drop", type=float, default=0.01)
    compress_parser.add_argument("--device", default="auto")

    benchmark_parser = subparsers.add_parser(
        "benchmark", help="measure model latency and throughput"
    )
    benchmark_parser.add_argument("--checkpoint", type=Path, required=True)
    benchmark_parser.add_argument("--device", default="auto")
    benchmark_parser.add_argument("--image-size", type=int)
    benchmark_parser.add_argument("--batch-size", type=int, default=32)
    benchmark_parser.add_argument("--warmup", type=int, default=5)
    benchmark_parser.add_argument("--iterations", type=int, default=20)
    benchmark_parser.add_argument("--output", type=Path)

    export_parser = subparsers.add_parser("export", help="export TorchScript and optional ONNX")
    export_parser.add_argument("--checkpoint", type=Path, required=True)
    export_parser.add_argument("--output-dir", type=Path)
    export_parser.add_argument("--onnx", action="store_true")
    export_parser.add_argument("--device", default="cpu")
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
    validate_dataset_paths(train_roots, test_root)
    train_index, test_index = prepare_indexes(config)
    print(
        json.dumps(
            {
                "train_files": len(train_index.files),
                "train_samples": len(train_index.records),
                "test_files": len(test_index.files),
                "test_samples": len(test_index.records),
                "num_classes": len(train_index.class_names),
                "train_roots": [str(p.resolve()) for p in train_roots],
                "test_root": str(test_root.resolve()),
                "cache_dir": str(cache_dir.resolve()),
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
        model_name=args.model_name,
        recipe=args.recipe,
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
    validate_dataset_paths(train_roots, test_root)
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
    validate_dataset_paths(train_roots, test_root)
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
    image_size = int(metadata.get("image_size", metadata.get("config", {}).get("image_size", 96)))
    paths = [args.input] if args.input else image_paths(args.input_dir)
    results = [
        predict_image(model, path, metadata["class_names"], image_size, device, args.top_k)
        for path in paths
    ]
    print(json.dumps(results, ensure_ascii=False, indent=2))
    return 0


def _run_compress(args: argparse.Namespace) -> int:
    device = resolve_device(args.device)
    model, payload = load_checkpoint(args.checkpoint, device)
    output_dir = args.output_dir or args.checkpoint.parent / "compressed"
    output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, object] = {"source": str(args.checkpoint), "stages": []}

    def finetune() -> dict[str, float] | None:
        if args.finetune_epochs <= 0:
            return None
        config = TrainConfig(
            train_roots=TRAIN_ROOTS,
            test_root=TEST_ROOT,
            cache_dir=CACHE_ROOT,
            output_dir=output_dir,
            model_name=str(payload.get("model_name", "hccr_cnn9")),
            image_size=int(payload.get("image_size", 96)),
            epochs=args.finetune_epochs,
            batch_size=128,
            num_workers=8,
            device=str(device),
            expected_num_classes=len(payload["class_names"]),
        )
        validate_dataset_paths(TRAIN_ROOTS, TEST_ROOT)
        train_index, _ = prepare_indexes(config)
        train_loader, validation_loader = make_train_validation_loaders(train_index, config, device)
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=1e-5)
        scaler = torch.amp.GradScaler("cuda", enabled=device.type == "cuda")
        criterion = torch.nn.CrossEntropyLoss(label_smoothing=0.02)
        metrics: dict[str, float] = {}
        for _ in range(args.finetune_epochs):
            run_epoch(model, train_loader, criterion, device, optimizer, scaler)
            with torch.inference_mode():
                metrics = run_epoch(model, validation_loader, criterion, device)
        baseline = float(payload.get("metrics", {}).get("top1", 0.0))
        if baseline > 0 and metrics.get("top1", 0.0) < baseline - args.max_accuracy_drop:
            raise ValueError(
                f"compression fine-tune dropped validation Top-1 from {baseline:.4f} "
                f"to {metrics['top1']:.4f}, beyond {args.max_accuracy_drop:.4f}"
            )
        return metrics

    for stage in args.stages:
        if stage == "gslre":
            replaced = replace_conv_with_gslre(model, args.rank_ratio)
            stage_payload = {
                "stage": stage,
                "replaced_layers": replaced,
                "rank_ratio": args.rank_ratio,
            }
            fine_tune_metrics = finetune()
            if fine_tune_metrics is not None:
                stage_payload["validation"] = fine_tune_metrics
            torch.save(
                {
                    **payload,
                    "model_name": f"{payload.get('model_name', 'hccr_cnn9')}_gslre",
                    "gslre_rank_ratio": args.rank_ratio,
                    "model_state": model.state_dict(),
                    "compression_stage": stage,
                },
                output_dir / "gslre.pt",
            )
            model.to(device)
        elif stage == "adw":
            stats = apply_adw_pruning(model, args.sparsity)
            stats = finalize_pruning(model)
            stage_payload = {"stage": stage, "sparsity": args.sparsity, **stats.as_dict()}
            fine_tune_metrics = finetune()
            if fine_tune_metrics is not None:
                stage_payload["validation"] = fine_tune_metrics
            torch.save(
                {
                    **payload,
                    "model_name": (
                        f"{payload.get('model_name', 'hccr_cnn9')}_gslre"
                        if any(
                            module.__class__.__name__ == "GSLREConv2d" for module in model.modules()
                        )
                        else payload.get("model_name", "hccr_cnn9")
                    ),
                    "gslre_rank_ratio": args.rank_ratio,
                    "model_state": model.state_dict(),
                    "compression_stage": stage,
                },
                output_dir / "adw.pt",
            )
            model.to(device)
        else:
            packed = quantized_payload(model.state_dict(), args.clusters)
            quantized_payload_data = {
                "format_version": 2,
                "model_name": (
                    f"{payload.get('model_name', 'hccr_cnn9')}_gslre"
                    if any(module.__class__.__name__ == "GSLREConv2d" for module in model.modules())
                    else payload.get("model_name", "hccr_cnn9")
                ),
                "gslre_rank_ratio": args.rank_ratio,
                "class_names": payload["class_names"],
                "class_to_idx": payload["class_to_idx"],
                "image_size": payload.get("image_size", 96),
                "config": payload.get("config", {}),
                "metrics": payload.get("metrics", {}),
                "epoch": payload.get("epoch", 0),
                **packed,
                "compression_stage": stage,
            }
            torch.save(quantized_payload_data, output_dir / "int8.pt")
            model.to(device)
            stage_payload = {
                "stage": stage,
                "clusters": args.clusters,
                "file": str(output_dir / "int8.pt"),
            }
        report["stages"].append(stage_payload)
    (output_dir / "compression_report.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


def _run_benchmark(args: argparse.Namespace) -> int:
    device = resolve_device(args.device)
    model, metadata = load_checkpoint(args.checkpoint, device)
    size = args.image_size or int(metadata.get("image_size", 96))
    model.eval()
    dtype = next(model.parameters()).dtype
    inputs = torch.randn(args.batch_size, 1, size, size, device=device, dtype=dtype)
    with torch.inference_mode():
        for _ in range(args.warmup):
            model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        for _ in range(args.iterations):
            model(inputs)
        if device.type == "cuda":
            torch.cuda.synchronize()
    elapsed = time.perf_counter() - start
    result = {
        "device": str(device),
        "batch_size": args.batch_size,
        "image_size": size,
        "iterations": args.iterations,
        "latency_ms": elapsed / args.iterations * 1000,
        "samples_per_second": args.batch_size * args.iterations / elapsed,
        "parameters": model_parameter_count(model),
        "sparsity": compression_stats(model).sparsity,
    }
    destination = args.output or args.checkpoint.parent / "benchmark.json"
    destination.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


def _run_export(args: argparse.Namespace) -> int:
    device = resolve_device(args.device)
    model, metadata = load_checkpoint(args.checkpoint, device)
    model.eval()
    output_dir = args.output_dir or args.checkpoint.parent / "exported"
    output_dir.mkdir(parents=True, exist_ok=True)
    size = int(metadata.get("image_size", 96))
    example = torch.randn(1, 1, size, size, device=device)
    traced = torch.jit.trace(model, example)
    traced.save(str(output_dir / "model.ts"))
    fp16_path = output_dir / "fp16.pt"
    fp16_state = {key: value.detach().half().cpu() for key, value in model.state_dict().items()}
    torch.save(
        {
            "format_version": 2,
            "model_name": metadata.get("model_name", "hccr_cnn9"),
            "class_names": metadata["class_names"],
            "class_to_idx": metadata["class_to_idx"],
            "image_size": size,
            "precision": "fp16",
            "model_state": fp16_state,
        },
        fp16_path,
    )
    result = {"torchscript": str(output_dir / "model.ts"), "fp16": str(fp16_path)}
    if args.onnx:
        onnx_path = output_dir / "model.onnx"
        torch.onnx.export(
            model,
            example,
            onnx_path,
            input_names=["image"],
            output_names=["logits"],
            opset_version=17,
        )
        result["onnx"] = str(onnx_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
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
        if args.command == "compress":
            return _run_compress(args)
        if args.command == "benchmark":
            return _run_benchmark(args)
        if args.command == "export":
            return _run_export(args)
        parser.error(f"unknown command: {args.command}")
    except (FileNotFoundError, RuntimeError, ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
