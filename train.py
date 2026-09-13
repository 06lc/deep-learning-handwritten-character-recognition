"""HWDB-1.1 的训练、评估和 checkpoint 管理。"""

from __future__ import annotations

import copy
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from compression import dequantize_state_dict, replace_conv_with_gslre
from config import TrainConfig
from gnt_dataset import (
    GNTDataset,
    GNTIndex,
    build_index,
    gnt_files,
    load_index,
    merge_indexes,
    save_index,
    split_record_indices_by_file,
)
from model import create_model


def resolve_device(requested: str = "auto") -> torch.device:
    """把命令行设备字符串解析成 PyTorch 设备对象。"""
    normalized = requested.lower().strip()
    if normalized == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(normalized)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return device


def seed_everything(seed: int) -> None:
    """固定随机数，让同一配置的实验尽量可以复现。"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


class ExponentialMovingAverage:
    """保存模型参数的指数滑动平均副本。

    训练权重每一步都会有小幅波动；EMA 用新旧权重的加权平均得到更平滑的
    模型，验证和保存 best.pt 时使用这份副本。
    """

    def __init__(self, model: nn.Module, decay: float) -> None:
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        self.model.requires_grad_(False)

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        current = model.state_dict()
        for name, averaged in self.model.state_dict().items():
            value = current[name].detach()
            if averaged.is_floating_point():
                averaged.mul_(self.decay).add_(value, alpha=1.0 - self.decay)
            else:
                averaged.copy_(value)

    def load_state_dict(self, state: dict[str, Tensor]) -> None:
        self.model.load_state_dict(state)


def _index_matches(
    index: GNTIndex,
    files: tuple[Path, ...],
    class_names: tuple[str, ...] | None = None,
    expected_num_classes: int | None = None,
) -> bool:
    return (
        index.files == files
        and (class_names is None or index.class_names == class_names)
        and (expected_num_classes is None or len(index.class_names) == expected_num_classes)
    )


def _prepare_primary_indexes(config: TrainConfig) -> tuple[GNTIndex, GNTIndex]:
    train_cache = config.cache_dir / "train.npz"
    test_cache = config.cache_dir / "test.npz"
    expected_train_files = gnt_files(config.train_roots)
    expected_test_files = gnt_files(config.test_root)
    rebuild_primary = config.rebuild_index and config.data_profile == "hwdb11"
    if not rebuild_primary and train_cache.is_file() and test_cache.is_file():
        train_index = load_index(train_cache)
        test_index = load_index(test_cache)
        cache_matches = (
            _index_matches(
                train_index,
                expected_train_files,
                expected_num_classes=config.expected_num_classes,
            )
            and _index_matches(test_index, expected_test_files, train_index.class_names)
        )
        if cache_matches:
            return train_index, test_index

    train_index = build_index(config.train_roots, expected_num_classes=config.expected_num_classes)
    test_index = build_index(config.test_root, class_names=train_index.class_names)
    save_index(train_index, train_cache)
    save_index(test_index, test_cache)
    return train_index, test_index


def _data_report(
    config: TrainConfig,
    primary: GNTIndex,
    train_index: GNTIndex,
    test_index: GNTIndex,
) -> dict[str, object]:
    additional_files = len(train_index.files) - len(primary.files)
    stats = train_index.filter_stats if config.data_profile == "hwdb10_11_shared" else None
    shared_class_names = set(stats.shared_class_names) if stats else set(primary.class_names)
    return {
        "data_profile": config.data_profile,
        "primary_train_roots": [str(path.resolve()) for path in config.train_roots],
        "additional_train_roots": (
            [str(path.resolve()) for path in config.additional_train_roots]
            if config.data_profile == "hwdb10_11_shared"
            else []
        ),
        "primary_train_files": len(primary.files),
        "additional_train_files": additional_files,
        "primary_samples": len(primary.records),
        "additional_scanned_samples": stats.scanned_samples if stats else 0,
        "additional_accepted_samples": stats.accepted_samples if stats else 0,
        "additional_excluded_samples": stats.excluded_samples if stats else 0,
        "shared_classes": len(stats.shared_class_names) if stats else len(primary.class_names),
        "primary_only_class_names": [
            name for name in primary.class_names if name not in shared_class_names
        ],
        "excluded_classes": len(stats.excluded_class_names) if stats else 0,
        "excluded_class_names": list(stats.excluded_class_names) if stats else [],
        "canonical_class_mapping_source": "HWDB1.1 training index",
        "num_classes": len(train_index.class_names),
        "test_samples": len(test_index.records),
    }


def _ensure_disjoint_sources(train_index: GNTIndex, test_index: GNTIndex) -> None:
    train_files = {path.resolve() for path in train_index.files}
    test_files = {path.resolve() for path in test_index.files}
    overlap = train_files.intersection(test_files)
    if overlap:
        raise ValueError(f"training and test indexes overlap: {sorted(overlap)!r}")


def prepare_indexes_with_report(
    config: TrainConfig,
) -> tuple[GNTIndex, GNTIndex, dict[str, object]]:
    """准备主数据或兼容扩充数据，并返回可写入 checkpoint 的来源报告。"""

    primary, test_index = _prepare_primary_indexes(config)
    if config.data_profile == "hwdb11":
        _ensure_disjoint_sources(primary, test_index)
        return primary, test_index, _data_report(config, primary, primary, test_index)

    additional_files = gnt_files(config.additional_train_roots)
    combined_files = primary.files + additional_files
    shared_cache = config.cache_dir / "train_hwdb10_11_shared.npz"
    train_index: GNTIndex | None = None
    if not config.rebuild_index and shared_cache.is_file():
        cached = load_index(shared_cache)
        if (
            _index_matches(cached, combined_files, primary.class_names)
            and cached.filter_stats is not None
        ):
            train_index = cached
    if train_index is None:
        additional = build_index(
            config.additional_train_roots,
            class_names=primary.class_names,
            expected_num_classes=len(primary.class_names),
            unknown_label="skip",
        )
        train_index = merge_indexes(primary, additional)
        save_index(train_index, shared_cache)
    _ensure_disjoint_sources(train_index, test_index)
    return train_index, test_index, _data_report(config, primary, train_index, test_index)


def prepare_indexes(config: TrainConfig) -> tuple[GNTIndex, GNTIndex]:
    """兼容旧调用方，只返回训练索引和 HWDB1.1 测试索引。"""

    train_index, test_index, _ = prepare_indexes_with_report(config)
    return train_index, test_index


def prepare_test_index(
    config: TrainConfig, class_names: tuple[str, ...]
) -> tuple[GNTIndex, dict[str, object]]:
    """按测试配置准备独立测试索引，不与训练或验证指标混合。"""

    if config.test_profile == "hwdb11":
        _, test_index = _prepare_primary_indexes(config)
        if test_index.class_names != class_names:
            raise ValueError("HWDB1.1 test labels do not match checkpoint class mapping")
    else:
        if config.test_profile == "hwdb10_shared":
            test_root = config.hwdb10_test_root
            cache_path = config.cache_dir / "test_hwdb10_shared.npz"
        elif config.test_profile == "icdar2013":
            test_root = config.competition_test_root
            cache_path = config.cache_dir / "test_icdar2013.npz"
        else:
            raise ValueError(f"unsupported test profile: {config.test_profile}")
        expected_files = gnt_files(test_root)
        test_index = None
        if not config.rebuild_index and cache_path.is_file():
            cached = load_index(cache_path)
            if (
                _index_matches(cached, expected_files, class_names)
                and cached.filter_stats is not None
            ):
                test_index = cached
        if test_index is None:
            test_index = build_index(
                test_root,
                class_names=class_names,
                expected_num_classes=len(class_names),
                unknown_label="skip",
            )
            save_index(test_index, cache_path)
    stats = test_index.filter_stats
    return test_index, {
        "test_profile": config.test_profile,
        "test_scanned_samples": stats.scanned_samples if stats else len(test_index.records),
        "test_accepted_samples": stats.accepted_samples if stats else len(test_index.records),
        "test_excluded_samples": stats.excluded_samples if stats else 0,
        "test_excluded_class_names": list(stats.excluded_class_names) if stats else [],
        "test_class_count": len(stats.shared_class_names) if stats else len(class_names),
    }


def make_loader(
    dataset: GNTDataset,
    batch_size: int,
    shuffle: bool,
    num_workers: int,
    device: torch.device,
    seed: int,
    sample_weights: Tensor | None = None,
) -> DataLoader[tuple[Tensor, Tensor]]:
    """把 Dataset 包装成按 batch 读取的 DataLoader。

    batch_size 决定一次送入 GPU 的样本数；num_workers 决定后台读取进程数；
    pin_memory 配合 CUDA 可以加快 CPU 到 GPU 的拷贝。
    """
    generator = torch.Generator().manual_seed(seed)
    sampler = None
    if sample_weights is not None:
        sampler = WeightedRandomSampler(
            sample_weights,
            num_samples=len(sample_weights),
            replacement=True,
            generator=generator,
        )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle and sampler is None,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=num_workers > 0,
        generator=generator,
    )


def class_balanced_sample_weights(
    index: GNTIndex,
    record_indices: list[int],
    max_multiplier: float = 2.0,
) -> Tensor:
    """按类别逆频率生成采样权重，并限制稀有类别的最大倍率。"""

    if max_multiplier < 1:
        raise ValueError("max_multiplier must be at least 1")
    counts = torch.zeros(len(index.class_names), dtype=torch.long)
    labels = torch.tensor([index.records[i].label for i in record_indices], dtype=torch.long)
    counts.scatter_add_(0, labels, torch.ones_like(labels))
    present = counts > 0
    if not bool(present.any()):
        raise ValueError("record_indices contains no samples")
    mean_count = counts[present].double().mean()
    class_weights = torch.ones(len(index.class_names), dtype=torch.double)
    class_weights[present] = (mean_count / counts[present].double()).clamp(max=max_multiplier)
    return class_weights[labels]


def make_train_validation_loaders(
    index: GNTIndex, config: TrainConfig, device: torch.device
) -> tuple[DataLoader[tuple[Tensor, Tensor]], DataLoader[tuple[Tensor, Tensor]]]:
    assert config.validation_manifest is not None
    train_indices, validation_indices = split_record_indices_by_file(
        index,
        config.val_fraction,
        config.split_seed,
        manifest_path=config.validation_manifest,
        roots=config.train_roots,
    )
    train_set = GNTDataset(
        index,
        config.image_size,
        augment=True,
        record_indices=train_indices,
        preprocess_profile=config.preprocess_profile,
        augmentation_profile=config.augmentation_profile,
    )
    validation_set = GNTDataset(
        index,
        config.image_size,
        record_indices=validation_indices,
        preprocess_profile=config.preprocess_profile,
    )
    sample_weights = None
    if config.sampling_strategy == "class-balanced":
        sample_weights = class_balanced_sample_weights(index, train_indices)
    return (
        make_loader(
            train_set,
            config.batch_size,
            True,
            config.num_workers,
            device,
            config.seed,
            sample_weights,
        ),
        make_loader(validation_set, config.batch_size, False, config.num_workers, device, 0),
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
    ema: ExponentialMovingAverage | None = None,
) -> dict[str, float]:
    """运行一个完整的数据轮次。

    optimizer 不为空表示训练模式：计算损失、反向传播并更新参数；optimizer
    为空表示验证模式：只前向推理和统计指标，不修改模型。
    """
    training = optimizer is not None
    model.train(training)
    amp_enabled = scaler is not None and scaler.is_enabled()
    total_loss = total_count = total_top1 = total_top5 = 0.0
    for batch_number, (images, targets) in enumerate(loader):
        if max_batches is not None and batch_number >= max_batches:
            break
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            # autocast 在 GPU 上使用半精度计算，减少显存占用；scaler 防止
            # 半精度下梯度太小而变成 0。CPU 或关闭 AMP 时走普通精度路径。
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=amp_enabled):
                logits = model(images)
                loss = criterion(logits, targets)
            if training:
                # loss.backward() 计算每个参数的梯度；optimizer.step() 按梯度
                # 和学习率更新参数；zero_grad() 会在下一个 batch 前清除旧梯度。
                if scaler is not None and amp_enabled:
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    loss.backward()
                    optimizer.step()
                if ema is not None:
                    ema.update(model)
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


def _optimizer_and_scheduler(
    model: nn.Module,
    config: TrainConfig,
    resume_scheduler_state: dict[str, Any] | None = None,
) -> tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LRScheduler]:
    # 优化器决定“如何根据梯度改参数”；调度器决定“每一轮使用多大的学习率”。
    # --finetune-from 必须创建全新的优化器，避免把旧实验的动量带入新实验。
    if config.finetune_from is not None or config.warm_start_from is not None:
        optimizer: torch.optim.Optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
    elif config.recipe.lower() == "paper":
        optimizer = torch.optim.SGD(
            model.parameters(),
            lr=0.01,
            momentum=0.9,
            nesterov=True,
            weight_decay=config.weight_decay,
        )
    elif config.recipe.lower() == "modern":
        # 现代默认方案从 3e-4 起步；微调应通过新建 optimizer 使用更小学习率。
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=config.learning_rate, weight_decay=config.weight_decay
        )
    else:
        raise ValueError("recipe must be 'modern' or 'paper'")
    if resume_scheduler_state and "T_max" in resume_scheduler_state:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, config.epochs)
        )
        return optimizer, scheduler

    warmup_epochs = min(config.warmup_epochs, max(0, config.epochs - 1))
    cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=max(1, config.epochs - warmup_epochs),
        eta_min=config.min_learning_rate,
    )
    if warmup_epochs == 0:
        scheduler = cosine
    else:
        warmup = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
        )
        scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer, schedulers=[warmup, cosine], milestones=[warmup_epochs]
        )
    return optimizer, scheduler


def warm_start_model(model: nn.Module, source_state: dict[str, Tensor]) -> dict[str, object]:
    """复制名称和形状都匹配的参数，并返回可审计的迁移报告。"""

    target_state = model.state_dict()
    matched = {
        name: value
        for name, value in source_state.items()
        if name in target_state and target_state[name].shape == value.shape
    }
    model.load_state_dict(matched, strict=False)
    copied_parameters = sum(target_state[name].numel() for name in matched)
    total_parameters = sum(value.numel() for value in target_state.values())
    return {
        "copied_tensors": len(matched),
        "total_tensors": len(target_state),
        "copied_parameters": copied_parameters,
        "total_parameters": total_parameters,
        "coverage": copied_parameters / total_parameters,
    }


def save_checkpoint(
    path: str | Path,
    model: nn.Module,
    optimizer: torch.optim.Optimizer | None,
    scheduler: torch.optim.lr_scheduler.LRScheduler | None,
    scaler: torch.amp.GradScaler | None,
    config: TrainConfig,
    class_names: tuple[str, ...],
    epoch: int,
    metrics: dict[str, float],
    model_name: str | None = None,
    extra: dict[str, Any] | None = None,
    ema: ExponentialMovingAverage | None = None,
    primary_is_ema: bool = False,
) -> None:
    """把一次实验恢复所需的状态集中保存到一个 .pt 文件。"""
    raw_state = model.state_dict()
    ema_state = ema.model.state_dict() if ema is not None else None
    validation_split = None
    if config.validation_manifest is not None and config.validation_manifest.is_file():
        validation_split = json.loads(config.validation_manifest.read_text(encoding="utf-8"))
    payload: dict[str, Any] = {
        "format_version": 3,
        "model_name": model_name or config.model_name,
        "class_names": list(class_names),
        "class_to_idx": {name: i for i, name in enumerate(class_names)},
        "image_size": config.image_size,
        "normalization": {"mean": [0.5], "std": [0.5]},
        "preprocessing": {
            "profile": config.preprocess_profile,
            "augmentation_profile": config.augmentation_profile,
        },
        "config": config.as_dict(),
        "dataset": {
            "train_roots": [str(p.resolve()) for p in config.train_roots],
            "test_root": str(config.test_root.resolve()),
        },
        "validation_split": validation_split,
        "epoch": epoch,
        "metrics": metrics,
        "model_state": ema_state if primary_is_ema and ema_state is not None else raw_state,
        "training_model_state": raw_state,
        "ema_model_state": ema_state,
        "primary_weight_source": "ema" if primary_is_ema and ema_state is not None else "raw",
        "scheduler_name": type(scheduler).__name__ if scheduler is not None else None,
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
    }
    if extra:
        payload.update(extra)
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, destination)


def load_checkpoint(
    path: str | Path, device: str | torch.device = "cpu"
) -> tuple[nn.Module, dict[str, Any]]:
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"checkpoint does not exist: {source}")
    payload = torch.load(source, map_location=device, weights_only=False)
    if isinstance(payload, dict) and payload.get("model_object") is not None:
        model = payload["model_object"].to(device)
        return model, payload
    class_names = payload.get("class_names")
    if not isinstance(class_names, list) or len(class_names) < 2:
        raise ValueError("checkpoint has no valid class_names")
    model_name = str(payload.get("model_name", "cnn"))
    base_name = model_name.removesuffix("_gslre")
    model = create_model(base_name, len(class_names))
    if model_name.endswith("_gslre"):
        replace_conv_with_gslre(model, float(payload.get("gslre_rank_ratio", 0.5)))
    state = payload.get("model_state", {})
    if payload.get("quantized_state"):
        state = dequantize_state_dict(state, payload["quantized_state"])
    model.load_state_dict(state)
    if payload.get("precision") == "fp16":
        model.half()
    model.to(device)
    return model, payload


def fit(config: TrainConfig) -> dict[str, object]:
    seed_everything(config.seed)
    device = resolve_device(config.device)
    train_index, _, data_report = prepare_indexes_with_report(config)
    if config.data_profile == "hwdb10_11_shared" and not (
        config.finetune_from or config.resume
    ):
        raise ValueError(
            "hwdb10_11_shared must start with --finetune-from; "
            "--resume is accepted only for its own interrupted run"
        )
    if config.data_profile == "hwdb10_11_shared" and config.warm_start_from is not None:
        raise ValueError("hwdb10_11_shared does not support --warm-start-from")
    train_loader, validation_loader = make_train_validation_loaders(train_index, config, device)
    model = create_model(config.model_name, len(train_index.class_names)).to(device)
    source_path = config.resume or config.finetune_from or config.warm_start_from
    source_payload: dict[str, Any] | None = None
    source_metadata: dict[str, object] | None = None
    if source_path is not None:
        resolved_source = source_path.resolve()
        if not resolved_source.is_file():
            raise FileNotFoundError(f"checkpoint does not exist: {resolved_source}")
        source_payload = torch.load(resolved_source, map_location=device, weights_only=False)
        if tuple(source_payload.get("class_names", ())) != train_index.class_names:
            raise ValueError("checkpoint class mapping differs from current data")
        if (
            config.data_profile == "hwdb10_11_shared"
            and config.resume is not None
            and source_payload.get("data_profile") != "hwdb10_11_shared"
        ):
            raise ValueError(
                "expanded training may resume only a hwdb10_11_shared checkpoint; "
                "use --finetune-from for the HWDB1.1 baseline"
            )
        source_metadata = {
            "checkpoint": str(resolved_source),
            "epoch": int(source_payload.get("epoch", 0)),
        }

    warm_start_report: dict[str, object] | None = None
    if config.resume is not None and source_payload is not None:
        model.load_state_dict(
            source_payload.get("training_model_state", source_payload["model_state"])
        )
    elif config.finetune_from is not None and source_payload is not None:
        source_model_name = str(source_payload.get("model_name", "cnn"))
        if source_model_name != config.model_name:
            raise ValueError(
                "fine-tune checkpoint model differs from configured model: "
                f"{source_model_name!r} != {config.model_name!r}"
            )
        model.load_state_dict(source_payload["model_state"])
    elif config.warm_start_from is not None and source_payload is not None:
        if config.model_name != "hccr_cnn9_ra":
            raise ValueError("warm-start is supported only for hccr_cnn9_ra")
        if str(source_payload.get("model_name")) != "hccr_cnn9":
            raise ValueError("warm-start source must be an hccr_cnn9 checkpoint")
        warm_start_report = warm_start_model(model, source_payload["model_state"])

    ema = ExponentialMovingAverage(model, config.ema_decay) if config.ema_decay > 0 else None
    resume_scheduler_state = (
        source_payload.get("scheduler_state")
        if config.resume is not None and source_payload is not None
        else None
    )
    optimizer, scheduler = _optimizer_and_scheduler(model, config, resume_scheduler_state)
    scaler = torch.amp.GradScaler("cuda", enabled=config.amp and device.type == "cuda")
    criterion = nn.CrossEntropyLoss(label_smoothing=config.label_smoothing)
    config.output_dir.mkdir(parents=True, exist_ok=True)
    history: list[dict[str, object]] = []
    best_top1 = -1.0
    stale_epochs = 0
    start_epoch = 1
    if config.resume is not None and source_payload is not None:
        if source_payload.get("ema_model_state") and ema is not None:
            ema.load_state_dict(source_payload["ema_model_state"])
        if source_payload.get("optimizer_state"):
            optimizer.load_state_dict(source_payload["optimizer_state"])
        if source_payload.get("scheduler_state"):
            scheduler.load_state_dict(source_payload["scheduler_state"])
        if source_payload.get("scaler_state"):
            scaler.load_state_dict(source_payload["scaler_state"])
        start_epoch = int(source_payload.get("epoch", 0)) + 1
        best_top1 = float(source_payload.get("metrics", {}).get("top1", -1.0))

    checkpoint_extra: dict[str, Any] = dict(data_report)
    if config.finetune_from is not None:
        checkpoint_extra["finetune_source"] = source_metadata
    if config.warm_start_from is not None:
        checkpoint_extra["warm_start_source"] = source_metadata
        checkpoint_extra["warm_start_report"] = warm_start_report

    for epoch in range(start_epoch, config.epochs + 1):
        train_metrics = run_epoch(
            model,
            train_loader,
            criterion,
            device,
            optimizer,
            scaler,
            config.max_train_batches,
            ema,
        )
        validation_model = ema.model if ema is not None else model
        with torch.inference_mode():
            validation_metrics = run_epoch(
                validation_model,
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
            extra=checkpoint_extra,
            ema=ema,
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
                extra=checkpoint_extra,
                ema=ema,
                primary_is_ema=True,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= config.patience:
                break

    assert config.validation_manifest is not None
    result: dict[str, object] = {
        "device": str(device),
        "model_name": config.model_name,
        "class_names": list(train_index.class_names),
        "train_samples": len(train_loader.dataset),
        "validation_samples": len(validation_loader.dataset),
        "best_validation_top1": best_top1,
        "validation_manifest": str(config.validation_manifest.resolve()),
        "validation_weight_source": "ema" if ema is not None else "raw",
        "history": history,
    }
    result.update(data_report)
    if config.finetune_from is not None:
        result["finetune_source"] = source_metadata
    if config.warm_start_from is not None:
        result["warm_start_source"] = source_metadata
        result["warm_start_report"] = warm_start_report
    (config.output_dir / "history.json").write_text(
        json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return result


def evaluate_checkpoint(
    checkpoint: str | Path, config: TrainConfig, output_dir: str | Path | None = None
) -> dict[str, object]:
    device = resolve_device(config.device)
    model, payload = load_checkpoint(checkpoint, device)
    class_names = tuple(payload["class_names"])
    checkpoint_config = payload.get("config", {})
    test_config = TrainConfig(
        train_roots=config.train_roots,
        test_root=config.test_root,
        hwdb10_test_root=config.hwdb10_test_root,
        competition_test_root=config.competition_test_root,
        cache_dir=config.cache_dir,
        output_dir=config.output_dir,
        model_name=str(payload.get("model_name", "cnn")),
        image_size=int(payload.get("image_size", checkpoint_config.get("image_size", 96))),
        preprocess_profile=str(
            payload.get("preprocessing", {}).get(
                "profile", checkpoint_config.get("preprocess_profile", "legacy")
            )
        ),
        expected_num_classes=len(class_names),
        rebuild_index=config.rebuild_index,
        test_profile=config.test_profile,
    )
    test_index, test_report = prepare_test_index(test_config, class_names)
    test_set = GNTDataset(
        test_index,
        image_size=test_config.image_size,
        preprocess_profile=test_config.preprocess_profile,
    )
    loader = make_loader(test_set, config.batch_size, False, config.num_workers, device, 0)
    criterion = nn.CrossEntropyLoss()
    with torch.inference_mode():
        metrics: dict[str, object] = run_epoch(
            model, loader, criterion, device, max_batches=config.max_eval_batches
        )
    metrics.update(test_report)
    destination = Path(output_dir) if output_dir else Path(checkpoint).parent / "evaluation"
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return metrics
