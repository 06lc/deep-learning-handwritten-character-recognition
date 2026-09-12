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
HWDB10_TRAIN_ROOTS = (
    PROJECT_ROOT / "Gnt1.0TrainPart1",
    PROJECT_ROOT / "Gnt1.0TrainPart2",
    PROJECT_ROOT / "Gnt1.0TrainPart3",
)
HWDB10_TEST_ROOT = PROJECT_ROOT / "Gnt1.0Test"
CACHE_ROOT = PROJECT_ROOT / "cache"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
VALIDATION_MANIFEST = CACHE_ROOT / "validation_split.json"
DATA_PROFILES = ("hwdb11", "hwdb10_11_shared")
TEST_PROFILES = ("hwdb11", "hwdb10_shared")
SAMPLING_STRATEGIES = ("natural", "class-balanced")


def validate_dataset_paths(
    train_roots: tuple[Path, ...] = TRAIN_ROOTS,
    test_root: Path = TEST_ROOT,
) -> None:
    """检查项目内数据布局，提前阻止路径配置错误。"""

    if len(train_roots) != 2:
        raise ValueError("exactly two training directories are required")
    resolved_train = tuple(path.resolve() for path in train_roots)
    if len(set(resolved_train)) != len(resolved_train):
        raise ValueError("training directories must be distinct")
    if test_root.resolve() in resolved_train:
        raise ValueError("test directory must be separate from training directories")
    for path in (*train_roots, test_root):
        if not path.is_dir():
            raise FileNotFoundError(f"dataset directory does not exist: {path}")
        if not any(path.glob("*.gnt")):
            raise ValueError(f"dataset directory contains no .gnt files: {path}")


def validate_config_paths(config: TrainConfig) -> None:
    """校验当前数据配置涉及的目录，并确保训练源与测试源互斥。"""

    validate_dataset_paths(config.train_roots, config.test_root)
    extra_paths: tuple[Path, ...] = ()
    if config.data_profile == "hwdb10_11_shared":
        if len(config.additional_train_roots) != 3:
            raise ValueError("hwdb10_11_shared requires three HWDB1.0 training directories")
        extra_paths = config.additional_train_roots
    if config.test_profile == "hwdb10_shared":
        extra_paths += (config.hwdb10_test_root,)
    for path in extra_paths:
        if not path.is_dir():
            raise FileNotFoundError(f"dataset directory does not exist: {path}")
        if not any(path.glob("*.gnt")):
            raise ValueError(f"dataset directory contains no .gnt files: {path}")
    train_paths = {path.resolve() for path in config.train_roots}
    if config.data_profile == "hwdb10_11_shared":
        train_paths.update(path.resolve() for path in config.additional_train_roots)
    test_paths = {config.test_root.resolve(), config.hwdb10_test_root.resolve()}
    if train_paths.intersection(test_paths):
        raise ValueError("training and test directories must be disjoint")


@dataclass(slots=True)
class TrainConfig:
    """训练、验证和 checkpoint 所需的全部参数。"""

    train_roots: tuple[Path, ...] = TRAIN_ROOTS
    test_root: Path = TEST_ROOT
    additional_train_roots: tuple[Path, ...] = HWDB10_TRAIN_ROOTS
    hwdb10_test_root: Path = HWDB10_TEST_ROOT
    cache_dir: Path = CACHE_ROOT
    output_dir: Path = OUTPUT_ROOT
    data_profile: str = "hwdb11"
    test_profile: str = "hwdb11"
    sampling_strategy: str = "natural"
    model_name: str = "hccr_cnn9"
    recipe: str = "modern"
    image_size: int = 96
    batch_size: int = 128
    epochs: int = 60
    val_fraction: float = 0.1
    split_seed: int = 42
    validation_manifest: Path | None = None
    # AdamW 的初始学习率；Cosine 调度器会在每轮后逐步降低它。
    learning_rate: float = 3e-4
    warmup_epochs: int = 5
    min_learning_rate: float = 1e-6
    ema_decay: float = 0.9999
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
    preprocess_profile: str = "legacy"
    augmentation_profile: str = "legacy"
    resume: Path | None = None
    finetune_from: Path | None = None
    warm_start_from: Path | None = None
    max_train_batches: int | None = None
    max_eval_batches: int | None = None

    def __post_init__(self) -> None:
        if self.validation_manifest is None:
            self.validation_manifest = self.cache_dir / "validation_split.json"
        sources = (self.resume, self.finetune_from, self.warm_start_from)
        if sum(source is not None for source in sources) > 1:
            raise ValueError("resume, finetune_from and warm_start_from cannot be used together")
        if self.epochs < 1 or self.warmup_epochs < 0:
            raise ValueError("epochs must be positive and warmup_epochs must be non-negative")

        if not 0 <= self.min_learning_rate <= self.learning_rate:
            raise ValueError("min_learning_rate must be in [0, learning_rate]")
        if not 0 <= self.ema_decay < 1:
            raise ValueError("ema_decay must be in [0, 1)")
        if self.preprocess_profile not in {"legacy", "margin_v1"}:
            raise ValueError("preprocess_profile must be 'legacy' or 'margin_v1'")
        if self.augmentation_profile not in {"legacy", "gentle_elastic"}:
            raise ValueError("augmentation_profile must be 'legacy' or 'gentle_elastic'")
        if self.data_profile not in DATA_PROFILES:
            raise ValueError(f"data_profile must be one of {DATA_PROFILES}")
        if self.test_profile not in TEST_PROFILES:
            raise ValueError(f"test_profile must be one of {TEST_PROFILES}")
        if self.sampling_strategy not in SAMPLING_STRATEGIES:
            raise ValueError(f"sampling_strategy must be one of {SAMPLING_STRATEGIES}")

    def as_dict(self) -> dict[str, Any]:
        values = asdict(self)
        for key in ("train_roots", "additional_train_roots"):
            values[key] = [str(path) for path in values[key]]
        for key in (
            "test_root",
            "hwdb10_test_root",
            "cache_dir",
            "output_dir",
            "validation_manifest",
        ):
            values[key] = str(values[key])
        for key in ("resume", "finetune_from", "warm_start_from"):
            if values[key] is not None:
                values[key] = str(values[key])
        return values
