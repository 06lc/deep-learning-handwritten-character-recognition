"""单图和目录批量预测。"""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import torch
from PIL import Image

from gnt_dataset import preprocess_image

SUPPORTED_IMAGE_EXTENSIONS = {".bmp", ".jpeg", ".jpg", ".png", ".webp"}


def predict_image(
    model: torch.nn.Module,
    image_path: str | Path,
    class_names: Sequence[str],
    image_size: int,
    device: str | torch.device,
    top_k: int = 5,
    preprocess_profile: str = "legacy",
) -> dict[str, object]:
    """返回单张图片的预测字符、置信度和 Top-K 候选。"""

    path = Path(image_path)
    if not path.is_file():
        raise FileNotFoundError(f"input image does not exist: {path}")
    if top_k < 1:
        raise ValueError("top_k must be at least 1")
    with Image.open(path) as image:
        tensor = preprocess_image(
            image, image_size, preprocess_profile=preprocess_profile
        ).unsqueeze(0).to(device)
    try:
        tensor = tensor.to(dtype=next(model.parameters()).dtype)
    except StopIteration:
        pass
    model.eval()
    with torch.inference_mode(): # 推理模式下，关闭梯度计算
        probabilities = torch.softmax(model(tensor)[0], dim=0)
    count = min(top_k, len(class_names))
    values, indices = probabilities.topk(count)
    candidates = [
        {"class": class_names[int(index)], "confidence": float(value)}
        for value, index in zip(values.cpu(), indices.cpu())
    ]
    return {
        "input": str(path),
        "prediction": candidates[0]["class"],
        "confidence": candidates[0]["confidence"],
        "top_k": candidates,
    }


def image_paths(input_dir: str | Path) -> list[Path]:
    root = Path(input_dir)
    if not root.is_dir():
        raise FileNotFoundError(f"input directory does not exist: {root}")
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_IMAGE_EXTENSIONS
    )
    if not paths:
        raise ValueError(f"no supported images found under: {root}")
    return paths
