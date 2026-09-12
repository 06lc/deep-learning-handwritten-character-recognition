"""验证集和独立测试集的分类错误分析。"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import torch
from PIL import Image

from config import TrainConfig
from gnt_dataset import GNTDataset, split_record_indices_by_file
from train import load_checkpoint, make_loader, prepare_indexes, resolve_device


def _preprocess_profile(payload: dict[str, Any]) -> str:
    return str(
        payload.get("preprocessing", {}).get(
            "profile", payload.get("config", {}).get("preprocess_profile", "legacy")
        )
    )


def analyze_checkpoint(
    checkpoint: str | Path,
    config: TrainConfig,
    *,
    split: str = "validation",
    output_dir: str | Path | None = None,
    max_errors: int = 100,
) -> dict[str, object]:
    """生成类别召回率、混淆对和高置信错误样本。"""

    if split not in {"validation", "test"}:
        raise ValueError("split must be 'validation' or 'test'")
    if max_errors < 0:
        raise ValueError("max_errors must be non-negative")
    device = resolve_device(config.device)
    model, payload = load_checkpoint(checkpoint, device)
    class_names = tuple(payload["class_names"])
    image_size = int(payload.get("image_size", payload.get("config", {}).get("image_size", 96)))
    profile = _preprocess_profile(payload)
    train_index, test_index = prepare_indexes(config)
    if train_index.class_names != class_names or test_index.class_names != class_names:
        raise ValueError("dataset labels do not match checkpoint class mapping")
    if split == "validation":
        assert config.validation_manifest is not None
        _, record_indices = split_record_indices_by_file(
            train_index,
            config.val_fraction,
            config.split_seed,
            manifest_path=config.validation_manifest,
            roots=config.train_roots,
        )
        dataset = GNTDataset(
            train_index,
            image_size,
            record_indices=record_indices,
            preprocess_profile=profile,
        )
    else:
        dataset = GNTDataset(test_index, image_size, preprocess_profile=profile)
    loader = make_loader(dataset, config.batch_size, False, config.num_workers, device, 0)

    supports = [0] * len(class_names)
    correct = [0] * len(class_names)
    confusions: Counter[tuple[int, int]] = Counter()
    errors: list[dict[str, object]] = []
    total_loss = total_top1 = total_top5 = total = 0
    sample_offset = 0
    model.eval()
    dtype = next(model.parameters()).dtype
    with torch.inference_mode():
        for batch_number, (images, targets) in enumerate(loader):
            if config.max_eval_batches is not None and batch_number >= config.max_eval_batches:
                break
            images = images.to(device, dtype=dtype, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits = model(images)
            total_loss += float(torch.nn.functional.cross_entropy(logits, targets, reduction="sum"))
            probabilities = torch.softmax(logits, dim=1)
            confidence, predictions = probabilities.max(dim=1)
            top5 = logits.topk(min(5, logits.shape[1]), dim=1).indices
            total_top1 += int(predictions.eq(targets).sum())
            total_top5 += int(top5.eq(targets[:, None]).any(dim=1).sum())
            for local_index, (target, prediction, score) in enumerate(
                zip(targets.cpu(), predictions.cpu(), confidence.cpu())
            ):
                truth = int(target)
                predicted = int(prediction)
                supports[truth] += 1
                if truth == predicted:
                    correct[truth] += 1
                else:
                    confusions[(truth, predicted)] += 1
                    errors.append(
                        {
                            "dataset_index": sample_offset + local_index,
                            "true_index": truth,
                            "true_class": class_names[truth],
                            "predicted_index": predicted,
                            "predicted_class": class_names[predicted],
                            "confidence": float(score),
                        }
                    )
            batch_size = targets.numel()
            total += batch_size
            sample_offset += batch_size
    if total == 0:
        raise RuntimeError("analysis DataLoader produced no samples")

    destination = Path(output_dir) if output_dir else Path(checkpoint).parent / "analysis" / split
    error_dir = destination / "errors"
    error_dir.mkdir(parents=True, exist_ok=True)
    errors.sort(key=lambda item: float(item["confidence"]), reverse=True)
    exported_errors = errors[:max_errors]
    for rank, error in enumerate(exported_errors, start=1):
        filename = (
            f"{rank:04d}_true-{error['true_index']}_pred-{error['predicted_index']}.png"
        )
        Image.fromarray(dataset.raw_image(int(error["dataset_index"]))).save(error_dir / filename)
        error["image"] = str((error_dir / filename).resolve())

    per_class = [
        {
            "index": index,
            "class": class_names[index],
            "support": support,
            "correct": correct[index],
            "accuracy": correct[index] / support,
        }
        for index, support in enumerate(supports)
        if support > 0
    ]
    per_class.sort(key=lambda item: (float(item["accuracy"]), -int(item["support"])))
    top_confusions = [
        {
            "true_index": truth,
            "true_class": class_names[truth],
            "predicted_index": predicted,
            "predicted_class": class_names[predicted],
            "count": count,
        }
        for (truth, predicted), count in confusions.most_common(100)
    ]
    report: dict[str, object] = {
        "checkpoint": str(Path(checkpoint).resolve()),
        "split": split,
        "samples": total,
        "loss": total_loss / total,
        "top1": total_top1 / total,
        "top5": total_top5 / total,
        "preprocess_profile": profile,
        "hardest_classes": per_class[:100],
        "top_confusions": top_confusions,
        "high_confidence_errors": exported_errors,
    }
    destination.mkdir(parents=True, exist_ok=True)
    (destination / "analysis.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return report
