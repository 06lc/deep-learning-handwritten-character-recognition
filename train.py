"""HWDB-1.1 的训练、评估和 checkpoint 管理。"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader

from config import TrainConfig
from gnt_dataset import (
    GNTDataset,
    GNTIndex,
    build_index,
    gnt_files,
    load_index,
    save_index,
    split_record_indices_by_file,
)
from model import HandwrittenCNN


def resolve_device(requested: str = "auto") -> torch.device:
    normalized = requested.lower().strip()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if torch.backends.cudnn.is_available():
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def prepare_indexes(config: TrainConfig) -> tuple[GNTIndex, GNTIndex]:
    """加载缓存索引，或扫描训练/测试 GNT 文件后写入缓存。"""

    train_cache = config.cache_dir / "train.npz"
    test_cache = config.cache_dir / "test.npz"
    if not config.rebuild_index and train_cache.is_file() and test_cache.is_file():
        train_index = load_index(train_cache)
        test_index = load_index(test_cache)
        expected_train_files = gnt_files(config.train_roots)
        expected_test_files = gnt_files(config.test_root)
        cache_matches = (
            train_index.files == expected_train_files
            and test_index.files == expected_test_files
            and test_index.class_names == train_index.class_names
            and (
                config.expected_num_classes is None
                or len(train_index.class_names) == config.expected_num_classes
            )
        )
        if not cache_matches:
            train_index = build_index(
                config.train_roots,
                expected_num_classes=config.expected_num_classes,
            )
            test_index = build_index(config.test_root, class_names=train_index.class_names)
            save_index(train_index, train_cache)
            save_index(test_index, test_cache)
            return train_index, test_index
        if test_index.class_names != train_index.class_names:
            raise ValueError("cached train and test class mappings differ")
        return train_index, test_index

    train_index = build_index(
        config.train_roots,
        expected_num_classes=config.expected_num_classes,
    )
    test_index = build_index(config.test_root, class_names=train_index.class_names)
    save_index(train_index, train_cache)
    save_index(test_index, test_cache)
    return train_index, test_index


def make_loader(
    dataset: GNTDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    seed: int,
) -> DataLoader[tuple[Tensor, Tensor]]:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def _batch_metrics(logits: Tensor, targets: Tensor) -> tuple[int, int, int]:
    count = targets.numel()
    top1 = int(logits.argmax(dim=1).eq(targets).sum().item())
    top_k = min(5, logits.shape[1])
    candidates = logits.topk(top_k, dim=1).indices
    top5 = int(candidates.eq(targets[:, None]).any(dim=1).sum().item())
    return count, top1, top5


def run_epoch(
    model: nn.Module,
    loader: DataLoader[tuple[Tensor, Tensor]],
    criterion: nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None = None,
    scaler: torch.amp.GradScaler | None = None,
    max_batches: int | None = None,
) -> dict[str, float]:
    training = optimizer is not None
    model.train(training)
    amp_enabled = scaler is not None and scaler.is_enabled()
    total_loss = 0.0
    total_count = total_top1 = total_top5 = 0

    for batch_number, (images, targets) in enumerate(loader):
        if max_batches is not None and batch_number >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, targets)
            if training:
                if scaler is not None and amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
        count, top1, top5 = _batch_metrics(logits.detach(), targets)
        total_loss += float(loss.detach()) * count
        total_count += count
        total_top1 += top1
        total_top5 += top5

    if total_count == 0:
        raise RuntimeError("DataLoader produced no samples")
    return {
        "loss": total_loss / total_count,
        "top1": total_top1 / total_count,
        "top5": total_top5 / total_count,
    }


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    scaler: torch.amp.GradScaler,
    config: TrainConfig,
    class_names: tuple[str, ...],
    epoch: int,
    metrics: dict[str, float],
) -> None:
    payload: dict[str, Any] = {
        "format_version": 1,
        "model_name": "HandwrittenCNN",
        "class_names": list(class_names),
        "class_to_idx": {name: i for i, name in enumerate(class_names)},
        "config": config.as_dict(),
        "epoch": epoch,
        "metrics": metrics,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict(),
        "scaler_state": scaler.state_dict(),
    }
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)


def load_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[HandwrittenCNN, dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {source}")
    payload = torch.load(source, map_location=device, weights_only=False)
    class_names = payload.get("class_names")
    if not isinstance(class_names, list) or len(class_names) < 2:
        raise ValueError("checkpoint has no valid class_names")
    model = HandwrittenCNN(len(class_names))
    model.load_state_dict(payload["model_state"])
    model.to(device)
    return model, payload


def fit(config: TrainConfig) -> dict[str, object]:
    seed_everything(config.seed)
    device = resolve_device(config.device)
    train_index, _test_index = prepare_indexes(config)
    train_indices, validation_indices = split_record_indices_by_file(
        train_index, config.val_fraction, config.seed
    )
    train_set = GNTDataset(
        train_index,
        image_size=config.image_size,
        augment=True,
        record_indices=train_indices,
    )
    validation_set = GNTDataset(
        train_index,
        image_size=config.image_size,
        record_indices=validation_indices,
    )
    train_loader = make_loader(
        train_set, config.batch_size, True, config.num_workers, device, config.seed
    )
    validation_loader = make_loader(
        validation_set, config.batch_size, False, config.num_workers, device, config.seed
    )
    model = HandwrittenCNN(len(train_index.class_names)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, config.epochs))
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, object]] = []
    best_top1 = -1.0
    stale_epochs = 0
    start_epoch = 1

    if config.resume is not None:
        resume_payload = torch.load(config.resume, map_location=device, weights_only=False)
        if tuple(resume_payload.get("class_names", ())) != train_index.class_names:
            raise ValueError("resume checkpoint class mapping differs from current data")
        model.load_state_dict(resume_payload["model_state"])
        if "optimizer_state" in resume_payload:
            optimizer.load_state_dict(resume_payload["optimizer_state"])
        if "scheduler_state" in resume_payload:
            scheduler.load_state_dict(resume_payload["scheduler_state"])
        if "scaler_state" in resume_payload:
            scaler.load_state_dict(resume_payload["scaler_state"])
        start_epoch = int(resume_payload.get("epoch", 0)) + 1
        best_top1 = float(resume_payload.get("metrics", {}).get("top1", -1.0))

    for epoch in range(start_epoch, config.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler,
            config.max_train_batches,
        )
        with torch.inference_mode():
            validation_metrics = run_epoch(
                model,
                validation_loader,
                criterion,
                device,
                max_batches=config.max_eval_batches,
            )
        scheduler.step()
        epoch_result = {
            "epoch": epoch,
            "train": train_metrics,
            "validation": validation_metrics,
            "learning_rate": optimizer.param_groups[0]["lr"],
        }
        history.append(epoch_result)
        save_checkpoint(
            config.output_dir / "last.pt",
            model,
            optimizer,
            scheduler,
            scaler,
            config,
            train_index.class_names,
            epoch,
            validation_metrics,
        )
        if validation_metrics["top1"] > best_top1:
            best_top1 = validation_metrics["top1"]
            stale_epochs = 0
            save_checkpoint(
                config.output_dir / "best.pt",
                model,
                optimizer,
                scheduler,
                scaler,
                config,
                train_index.class_names,
                epoch,
                validation_metrics,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    result: dict[str, object] = {
        "device": str(device),
        "class_names": list(train_index.class_names),
        "train_samples": len(train_indices),
        "validation_samples": len(validation_indices),
        "best_validation_top1": best_top1,
        "history": history,
    }
    (config.output_dir / "history.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def evaluate_checkpoint(
    checkpoint: str | Path, config: TrainConfig, output_dir: str | Path | None = None
) -> dict[str, float]:
    device = resolve_device(config.device)
    model, payload = load_checkpoint(checkpoint, device)
    class_names = tuple(payload["class_names"])
    test_config = TrainConfig(
        train_roots=config.train_roots,
        test_root=config.test_root,
        cache_dir=config.cache_dir,
        output_dir=config.output_dir,
        image_size=int(payload.get("config", {}).get("image_size", config.image_size)),
        expected_num_classes=len(class_names),
        rebuild_index=config.rebuild_index,
    )
    _train_index, test_index = prepare_indexes(test_config)
    if test_index.class_names != class_names:
        raise ValueError("test labels do not match checkpoint class mapping")
    test_set = GNTDataset(test_index, image_size=test_config.image_size)
    loader = make_loader(test_set, config.batch_size, False, config.num_workers, device, 0)
    criterion = nn.CrossEntropyLoss()
    with torch.inference_mode():
        metrics = run_epoch(
            model,
            loader,
            criterion,
            device,
            max_batches=config.max_eval_batches,
        )
    destination = Path(output_dir) if output_dir else Path(checkpoint).parent / "evaluation"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics
