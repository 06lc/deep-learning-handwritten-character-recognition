"""项目配置。

路径始终相对于源码目录解析，因此同一份项目可以直接上传到 AutoDL；不会依赖
Windows 的盘符，也不会把训练目录和测试目录混用。
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
TRAIN_ROOTS = (
    PROJECT_ROOT / "Gnt1.1TrainPart1",
    PROJECT_ROOT / "Gnt1.1TrainPart2",
)
TEST_ROOT = PROJECT_ROOT / "Gnt1.1Test"
CACHE_ROOT = PROJECT_ROOT / "cache"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


def validate_dataset_paths(
    train_roots: tuple[Path, ...] = TRAIN_ROOTS,
    test_root: Path = TEST_ROOT,
) -> None:
    """检查项目内数据布局，提前阻止路径配置错误。"""

    if len(train_roots) != 2:
        raise ValueError("exactly two training directories are required")
    for path in (*train_roots, test_root):
        if not path.is_dir():
            raise FileNotFoundError(f"dataset directory does not exist: {path}")
        if not any(path.glob("*.gnt")):
            raise ValueError(f"dataset directory contains no .gnt files: {path}")


@dataclass(slots=True)
class TrainConfig:
    """训练、验证和 checkpoint 所需的全部参数。"""

    train_roots: tuple[Path, ...] = TRAIN_ROOTS
    test_root: Path = TEST_ROOT
    cache_dir: Path = CACHE_ROOT
    output_dir: Path = OUTPUT_ROOT
    model_name: str = "hccr_cnn9"
    recipe: str = "modern"
    image_size: int = 96
    batch_size: int = 128
    epochs: int = 30
    val_fraction: float = 0.1
    # AdamW 的初始学习率；Cosine 调度器会在每轮后逐步降低它。
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    label_smoothing: float = 0.05
    patience: int = 5
    num_workers: int = 0
    seed: int = 42
    device: str = "auto"
    amp: bool = True
    # HWDB-1.1 includes 3755 Chinese characters plus 171 digits/symbols.
    expected_num_classes: int | None = 3926
    rebuild_index: bool = False
    resume: Path | None = None
    finetune_from: Path | None = None
    max_train_batches: int | None = None
    max_eval_batches: int | None = None

    def __post_init__(self) -> None:
        if self.resume is not None and self.finetune_from is not None:
            raise ValueError("resume and finetune_from cannot be used together")

    def as_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["train_roots"] = [str(path) for path in self.train_roots]
        for key in ("test_root", "cache_dir", "output_dir"):
            values[key] = str(values[key])
        for key in ("resume", "finetune_from"):
            if values[key] is not None:
                values[key] = str(values[key])
        return values
