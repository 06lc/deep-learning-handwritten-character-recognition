"""训练配置和项目默认绝对路径。"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path

PROJECT_ROOT = Path(r"D:\Python+Ai\03_机器学习\手写字符识别")
TRAIN_ROOTS = (
    Path(r"F:\浏览器\Gnt1.1TrainPart1"),
    Path(r"F:\浏览器\Gnt1.1TrainPart2"),
)
TEST_ROOT = Path(r"F:\浏览器\Gnt1.1Test")
CACHE_ROOT = PROJECT_ROOT / "cache"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"


@dataclass(slots=True)
class TrainConfig:
    """训练、验证和 checkpoint 所需的全部参数。"""

    train_roots: tuple[Path, ...] = TRAIN_ROOTS
    test_root: Path = TEST_ROOT
    cache_dir: Path = CACHE_ROOT
    output_dir: Path = OUTPUT_ROOT
    image_size: int = 64
    batch_size: int = 128
    epochs: int = 30
    val_fraction: float = 0.1
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
    max_train_batches: int | None = None
    max_eval_batches: int | None = None

    def as_dict(self) -> dict[str, object]:
        values = asdict(self)
        values["train_roots"] = [str(path) for path in self.train_roots]
        for key in ("test_root", "cache_dir", "output_dir"):
            values[key] = str(values[key])
        if self.resume is not None:
            values["resume"] = str(self.resume)
        return values
