from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parent
# 所有默认路径都从项目目录推导。这样把整个项目上传到 AutoDL 后，代码仍然可用。
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
COMPETITION_TEST_ROOT = PROJECT_ROOT / "competition-gnt"
CACHE_ROOT = PROJECT_ROOT / "cache"
OUTPUT_ROOT = PROJECT_ROOT / "outputs"
VALIDATION_MANIFEST = CACHE_ROOT / "validation_split.json"
DATA_PROFILES = ("hwdb11", "hwdb10_11_shared")
TEST_PROFILES = ("hwdb11", "hwdb10_shared", "icdar2013")
SAMPLING_STRATEGIES = ("natural", "class-balanced")

# 检查数据配置是否合理。
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
    if config.test_profile == "icdar2013":
        extra_paths += (config.competition_test_root,)
    for path in extra_paths:
        if not path.is_dir():
            raise FileNotFoundError(f"dataset directory does not exist: {path}")
        if not any(path.glob("*.gnt")):
            raise ValueError(f"dataset directory contains no .gnt files: {path}")
    train_paths = {path.resolve() for path in config.train_roots}
    if config.data_profile == "hwdb10_11_shared":
        train_paths.update(path.resolve() for path in config.additional_train_roots)
    test_paths = {
        config.test_root.resolve(),
        config.hwdb10_test_root.resolve(),
        config.competition_test_root.resolve(),
    }
    if train_paths.intersection(test_paths):
        raise ValueError("training and test directories must be disjoint")


@dataclass(slots=True)
class TrainConfig:
    """
    训练、验证和 checkpoint 所需的全部参数。

    """

    train_roots: tuple[Path, ...] = TRAIN_ROOTS  # HWDB1.1 训练目录。
    test_root: Path = TEST_ROOT  # HWDB1.1 官方测试目录，只用于最终评估。
    additional_train_roots: tuple[Path, ...] = HWDB10_TRAIN_ROOTS  # 可选的 HWDB1.0 训练目录。
    hwdb10_test_root: Path = HWDB10_TEST_ROOT  # HWDB1.0 测试目录，用于独立对照评估。
    competition_test_root: Path = COMPETITION_TEST_ROOT  # ICDAR-2013 比赛测试目录。
    cache_dir: Path = CACHE_ROOT  # GNT 索引和固定验证清单的保存目录。
    output_dir: Path = OUTPUT_ROOT  # best.pt、last.pt、history.json 等输出目录。
    data_profile: str = "hwdb11"  # 训练数据方案：仅 HWDB1.1，或合并兼容的 HWDB1.0。
    test_profile: str = "hwdb11"  # 评估数据方案：HWDB1.1 Test 或 HWDB1.0 共有类别 Test。
    sampling_strategy: str = "natural"  # 采样方式：按原始比例，或对类别做受限均衡采样。
    model_name: str = "hccr_cnn9"  # 模型工厂名称；当前默认是论文九层网络。
    recipe: str = "modern"  # 优化方案：modern 使用 AdamW，paper 使用论文风格 SGD。
    image_size: int = 96  # 模型输入的高和宽，所有图片会变成 image_size x image_size。
    batch_size: int = 128  # 一次参数更新使用多少张图片；越大越占显存。
    epochs: int = 60  # 完整遍历训练集的次数。
    val_fraction: float = 0.1  # 从 HWDB1.1 训练文件中留作验证的比例。
    split_seed: int = 42  # 划分训练/验证文件的随机种子，保证验证集固定。
    validation_manifest: Path | None = None  # 验证文件清单；None 时自动放在 cache_dir。
    # AdamW 的初始学习率；Cosine 调度器会在每轮后逐步降低它。
    learning_rate: float = 3e-4  # 每次参数更新的步长大小。
    warmup_epochs: int = 5  # 前几轮从较小学习率逐步升到 learning_rate。
    min_learning_rate: float = 1e-6  # 余弦调度结束时保留的最低学习率。
    ema_decay: float = 0.9999  # 模型参数滑动平均的旧值权重，越接近 1 越平滑。
    weight_decay: float = 1e-4  # 权重衰减（L2 正则化），抑制参数过大和过拟合。
    label_smoothing: float = 0.05  # 标签平滑，降低模型对单个硬标签的过度自信。
    patience: int = 5  # 验证集多少轮不提升后提前停止；0 表示不提前停止。
    num_workers: int = 0  # DataLoader 后台读取进程数；GPU 训练通常可设为 4 到 8。
    seed: int = 42  # 训练随机种子，控制初始化、抽样和数据顺序。
    device: str = "auto"  # 运行设备：auto 自动选择 CUDA，否则使用 cpu。
    amp: bool = True  # 是否启用自动混合精度，以减少 GPU 显存和提高吞吐。
    # HWDB-1.1 includes 3755 Chinese characters plus 171 digits/symbols.
    expected_num_classes: int | None = 3926  # 期望的类别总数；None 表示不强制检查。
    rebuild_index: bool = False  # 是否忽略已有缓存，重新扫描全部 GNT 文件。
    preprocess_profile: str = "legacy"  # 图片预处理方案：原始兼容版或带边距的 margin_v1。
    augmentation_profile: str = "legacy"  # 训练增强方案；推理阶段始终不做随机增强。
    resume: Path | None = None  # 中断恢复：加载模型、优化器、调度器、AMP 和轮次。
    finetune_from: Path | None = None  # 微调起点：只加载模型权重，重新创建优化器。
    warm_start_from: Path | None = None  # 结构迁移起点：将旧模型匹配权重复制到增强模型。
    max_train_batches: int | None = None  # 每轮最多训练多少个 batch；None 表示完整训练。
    max_eval_batches: int | None = None  # 每次验证最多评估多少个 batch；None 表示完整验证。

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
        # checkpoint 需要保存可序列化的字符串路径，而不是 Windows/Linux
        # 平台相关的 Path 对象。
        values = asdict(self)
        for key in ("train_roots", "additional_train_roots"):
            values[key] = [str(path) for path in values[key]]
        for key in (
            "test_root",
            "hwdb10_test_root",
            "competition_test_root",
            "cache_dir",
            "output_dir",
            "validation_manifest",
        ):
            values[key] = str(values[key])
        for key in ("resume", "finetune_from", "warm_start_from"):
            if values[key] is not None:
                values[key] = str(values[key])
        return values
