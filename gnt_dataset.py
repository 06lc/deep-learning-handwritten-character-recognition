"""按需读取 CASIA-HWDB-1.1 GNT 文件的索引和 PyTorch 数据集。"""

from __future__ import annotations

import json
import random
import struct
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import BinaryIO

import numpy as np
from PIL import Image
from torch import Tensor
from torch.utils.data import Dataset
from torchvision import transforms

# GNT 每条记录的前 10 个字节：记录总长度、两字节标签、图像宽度和高度。
# < 表示小端字节序，I/2s/H/H 分别对应 uint32、2 字节、uint16、uint16。
GNT_HEADER = struct.Struct("<I2sHH")
NORMALIZE_MEAN = (0.5,)
NORMALIZE_STD = (0.5,)


class DatasetFormatError(ValueError):
    """GNT 文件或类别映射不符合预期。"""


@dataclass(frozen=True, slots=True)
class GNTRecord:
    """一个 GNT 样本在源文件中的位置和元数据。"""

    file_id: int
    offset: int
    sample_size: int
    width: int
    height: int
    label: int


@dataclass(frozen=True, slots=True)
class GNTIndexStats:
    """一次索引扫描的接纳与标签过滤统计。"""

    scanned_samples: int
    accepted_samples: int
    excluded_samples: int
    shared_class_names: tuple[str, ...]
    excluded_class_names: tuple[str, ...]

    def as_dict(self) -> dict[str, object]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class GNTIndex:
    """GNT 文件路径、类别名称和所有样本记录。"""

    files: tuple[Path, ...]
    records: tuple[GNTRecord, ...]
    class_names: tuple[str, ...]
    filter_stats: GNTIndexStats | None = None


def decode_label(raw_label: bytes) -> str:
    """把 GNT 的两个字节 GBK 标签转成一个 Unicode 字符。

    GNT 标签不是整数类别编号，而是字符编码。例如两个 GBK 字节解码后可能
    得到“坐”。模型训练时再把这个字符映射成 0 到 3925 的整数类别编号。
    """

    try:
        label = raw_label.decode("gbk").rstrip("\x00")
    except UnicodeDecodeError as exc:
        raise DatasetFormatError(f"invalid GBK label bytes: {raw_label!r}") from exc
    if not label:
        raise DatasetFormatError(f"empty GNT label bytes: {raw_label!r}")
    return label[0]


def _label_sort_key(label: str) -> bytes:
    """优先按数据集使用的 GBK 字节顺序稳定排序。"""

    try:
        return label.encode("gbk")
    except UnicodeEncodeError:
        return label.encode("utf-8")


def gnt_files(roots: Sequence[str | Path]) -> tuple[Path, ...]:
    """收集目录下的 GNT 文件，并按路径排序保证顺序稳定。"""

    if isinstance(roots, (str, Path)):
        roots = (roots,)
    files: list[Path] = []
    for root_value in roots:
        root = Path(root_value).expanduser()
        if not root.is_dir():
            raise FileNotFoundError(f"GNT directory does not exist: {root}")
        files.extend(sorted(path.resolve() for path in root.glob("*.gnt")))
    if not files:
        raise DatasetFormatError("no .gnt files found in the configured directories")
    return tuple(files)


def build_index(
    roots: Sequence[str | Path],
    class_names: Sequence[str] | None = None,
    expected_num_classes: int | None = None,
    *,
    unknown_label: str = "error",
) -> GNTIndex:
    """扫描 GNT 记录头并建立可复用索引，不读取像素正文。

    索引只记录“图像在哪个文件、从哪个字节开始、尺寸是多少、标签是什么”。
    训练时再按偏移读取像素，因此避免把百万张图片转换成独立文件。
    """

    if unknown_label not in {"error", "skip"}:
        raise ValueError("unknown_label must be 'error' or 'skip'")
    files = gnt_files(roots)
    raw_records: list[tuple[int, int, int, int, int, str]] = []
    discovered_labels: set[str] = set()
    accepted_labels: set[str] = set()
    excluded_labels: set[str] = set()
    excluded_samples = 0
    ordered_classes = tuple(class_names) if class_names is not None else None
    if ordered_classes is not None and len(set(ordered_classes)) != len(ordered_classes):
        raise DatasetFormatError("class_names contains duplicate labels")
    class_to_idx = (
        {name: index for index, name in enumerate(ordered_classes)}
        if ordered_classes is not None
        else None
    )

    for file_id, path in enumerate(files):
        # 一个 GNT 文件由许多条连续记录组成；handle.tell() 是当前记录的起点。
        file_size = path.stat().st_size
        with path.open("rb") as handle:
            while handle.tell() < file_size:
                offset = handle.tell()
                header = handle.read(GNT_HEADER.size)
                if len(header) != GNT_HEADER.size:
                    raise DatasetFormatError(f"truncated GNT header: {path} at {offset}")
                sample_size, raw_label, width, height = GNT_HEADER.unpack(header)
                pixel_size = width * height
                # 正常记录的总长度必须等于“头部 10 字节 + 像素字节数”。
                if sample_size != GNT_HEADER.size + pixel_size:
                    raise DatasetFormatError(
                        f"invalid sample size in {path} at {offset}: "
                        f"header={sample_size}, expected={GNT_HEADER.size + pixel_size}"
                    )
                end_offset = offset + sample_size
                if end_offset > file_size:
                    raise DatasetFormatError(f"truncated GNT sample: {path} at {offset}")
                label = decode_label(raw_label)
                discovered_labels.add(label)
                if class_to_idx is not None and label not in class_to_idx:
                    if unknown_label == "error":
                        raise DatasetFormatError(f"unknown label {label!r} in GNT index")
                    excluded_samples += 1
                    excluded_labels.add(label)
                else:
                    raw_records.append((file_id, offset, sample_size, width, height, label))
                    accepted_labels.add(label)
                handle.seek(pixel_size, 1)

    if ordered_classes is None:
        ordered_classes = tuple(sorted(discovered_labels, key=_label_sort_key))
        accepted_labels = discovered_labels

    if expected_num_classes is not None and len(ordered_classes) != expected_num_classes:
        raise DatasetFormatError(
            f"expected {expected_num_classes} classes, found {len(ordered_classes)}"
        )

    class_to_idx = {name: index for index, name in enumerate(ordered_classes)}
    records: list[GNTRecord] = []
    for file_id, offset, sample_size, width, height, label in raw_records:
        records.append(GNTRecord(file_id, offset, sample_size, width, height, class_to_idx[label]))
    stats = GNTIndexStats(
        scanned_samples=len(raw_records) + excluded_samples,
        accepted_samples=len(records),
        excluded_samples=excluded_samples,
        shared_class_names=tuple(sorted(accepted_labels, key=_label_sort_key)),
        excluded_class_names=tuple(sorted(excluded_labels, key=_label_sort_key)),
    )
    return GNTIndex(files, tuple(records), ordered_classes, stats)


def merge_indexes(primary: GNTIndex, additional: GNTIndex) -> GNTIndex:
    """合并相同类别映射的索引，并重排附加记录的文件编号。

    附加索引的 file_id 从 0 开始，所以合并时必须整体加上主索引文件数，
    否则训练会把一条记录指向错误的 GNT 文件。
    """

    if primary.class_names != additional.class_names:
        raise DatasetFormatError("cannot merge GNT indexes with different class mappings")
    primary_files = {path.resolve() for path in primary.files}
    if primary_files.intersection(path.resolve() for path in additional.files):
        raise DatasetFormatError("cannot merge GNT indexes containing the same source file")
    file_offset = len(primary.files)
    records = primary.records + tuple(
        GNTRecord(
            record.file_id + file_offset,
            record.offset,
            record.sample_size,
            record.width,
            record.height,
            record.label,
        )
        for record in additional.records
    )
    return GNTIndex(
        primary.files + additional.files,
        records,
        primary.class_names,
        additional.filter_stats,
    )


def save_index(index: GNTIndex, path: str | Path) -> None:
    """把索引保存为不含 pickle 的 NumPy 压缩文件。"""

    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    records = np.asarray(
        [
            [
                record.file_id,
                record.offset,
                record.sample_size,
                record.width,
                record.height,
                record.label,
            ]
            for record in index.records
        ],
        dtype=np.int64,
    )
    stats_json = ""
    if index.filter_stats is not None:
        stats_json = json.dumps(index.filter_stats.as_dict(), ensure_ascii=False)
    np.savez_compressed(
        destination,
        files=np.asarray([str(path) for path in index.files], dtype=np.str_),
        class_names=np.asarray(index.class_names, dtype=np.str_),
        records=records,
        filter_stats=np.asarray(stats_json, dtype=np.str_),
    )


def load_index(path: str | Path) -> GNTIndex:
    """读取由 :func:`save_index` 生成的索引。"""

    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"index cache does not exist: {source}")
    with np.load(source, allow_pickle=False) as payload:
        files = tuple(Path(value) for value in payload["files"].tolist())
        class_names = tuple(str(value) for value in payload["class_names"].tolist())
        raw_records = np.asarray(payload["records"], dtype=np.int64)
        stats_json = str(payload["filter_stats"].item()) if "filter_stats" in payload else ""
    if raw_records.ndim != 2 or raw_records.shape[1] != 6:
        raise DatasetFormatError(f"invalid index record array: {source}")
    records = tuple(GNTRecord(*(int(value) for value in row)) for row in raw_records)
    stats = None
    if stats_json:
        raw_stats = json.loads(stats_json)
        stats = GNTIndexStats(
            scanned_samples=int(raw_stats["scanned_samples"]),
            accepted_samples=int(raw_stats["accepted_samples"]),
            excluded_samples=int(raw_stats["excluded_samples"]),
            shared_class_names=tuple(raw_stats["shared_class_names"]),
            excluded_class_names=tuple(raw_stats["excluded_class_names"]),
        )
    return GNTIndex(files, records, class_names, stats)


def split_record_indices_by_file(
    index: GNTIndex,
    validation_fraction: float = 0.1,
    seed: int = 42,
    *,
    manifest_path: str | Path | None = None,
    roots: Sequence[str | Path] | None = None,
) -> tuple[list[int], list[int]]:
    """按 GNT 文件切分训练/验证样本，避免书写者风格泄漏。"""

    if not 0 < validation_fraction < 1:
        raise ValueError("validation_fraction must be between 0 and 1")
    file_ids = list(range(len(index.files)))
    if len(file_ids) < 2:
        raise DatasetFormatError("at least two GNT files are required for validation")
    rng = random.Random(seed)
    if manifest_path is None:
        validation_count = min(
            max(1, round(len(file_ids) * validation_fraction)), len(file_ids) - 1
        )
        rng.shuffle(file_ids)
        validation_files = set(file_ids[:validation_count])
    else:
        if roots is None:
            raise ValueError("roots are required when using a validation manifest")
        eligible_files = _file_ids_under_roots(index, roots)
        if len(eligible_files) < 2:
            raise DatasetFormatError("at least two primary files are required for validation")
        validation_count = min(
            max(1, round(len(eligible_files) * validation_fraction)), len(eligible_files) - 1
        )
        validation_files = _load_or_create_validation_files(
            index,
            roots,
            Path(manifest_path),
            validation_count,
            validation_fraction,
            seed,
            eligible_files,
        )
    train_indices = [
        i for i, record in enumerate(index.records) if record.file_id not in validation_files
    ]
    validation_indices = [
        i for i, record in enumerate(index.records) if record.file_id in validation_files
    ]
    rng.shuffle(train_indices)
    rng.shuffle(validation_indices)
    return train_indices, validation_indices


def _portable_file_entry(path: Path, roots: Sequence[Path]) -> dict[str, object]:
    for root_index, root in enumerate(roots):
        try:
            relative = path.resolve().relative_to(root)
        except ValueError:
            continue
        return {"root": root_index, "path": relative.as_posix()}
    raise DatasetFormatError(f"indexed file is outside configured training roots: {path}")


def _file_ids_under_roots(index: GNTIndex, roots: Sequence[str | Path]) -> list[int]:
    resolved_roots = tuple(Path(root).resolve() for root in roots)
    result: list[int] = []
    for file_id, path in enumerate(index.files):
        for root in resolved_roots:
            try:
                path.resolve().relative_to(root)
            except ValueError:
                continue
            result.append(file_id)
            break
    return result


def _load_or_create_validation_files(
    index: GNTIndex,
    roots: Sequence[str | Path],
    manifest_path: Path,
    validation_count: int,
    validation_fraction: float,
    seed: int,
    eligible_files: Sequence[int],
) -> set[int]:
    resolved_roots = tuple(Path(root).resolve() for root in roots)
    file_lookup = {path.resolve(): file_id for file_id, path in enumerate(index.files)}
    if manifest_path.is_file():
        payload = json.loads(manifest_path.read_text(encoding="utf-8"))
        if payload.get("version") != 1:
            raise DatasetFormatError(f"unsupported validation manifest: {manifest_path}")
        if int(payload.get("split_seed", -1)) != seed:
            raise DatasetFormatError("validation manifest split_seed differs from configuration")
        if abs(float(payload.get("validation_fraction", -1)) - validation_fraction) > 1e-12:
            raise DatasetFormatError(
                "validation manifest fraction differs from configuration"
            )
        validation_files: set[int] = set()
        for entry in payload.get("validation_files", []):
            root_index = int(entry["root"])
            if not 0 <= root_index < len(resolved_roots):
                raise DatasetFormatError("validation manifest contains an invalid root")
            source = (resolved_roots[root_index] / str(entry["path"])).resolve()
            if source not in file_lookup:
                raise DatasetFormatError(f"validation file is missing from index: {source}")
            validation_files.add(file_lookup[source])
        if not validation_files or len(validation_files) >= len(eligible_files):
            raise DatasetFormatError("validation manifest must leave files for training")
        return validation_files

    shuffled = list(eligible_files)
    random.Random(seed).shuffle(shuffled)
    selected = set(shuffled[:validation_count])
    payload = {
        "version": 1,
        "split_seed": seed,
        "validation_fraction": validation_fraction,
        "validation_files": [
            _portable_file_entry(index.files[file_id], resolved_roots)
            for file_id in sorted(selected)
        ],
    }
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    return selected


def _letterbox(
    image: np.ndarray, image_size: int, preprocess_profile: str = "legacy"
) -> Image.Image:
    """等比例缩放字符，并把它放到固定大小的白色画布中央。

    直接拉伸会改变汉字笔画比例；letterbox 只缩放较长边，另一边用白边补齐，
    因而既能统一输入尺寸，又尽量保留原始字形。
    """
    if image.ndim != 2:
        raise DatasetFormatError(f"expected a grayscale image, got shape {image.shape}")
    source = Image.fromarray(image)
    if preprocess_profile == "legacy":
        content_size = image_size
    elif preprocess_profile == "margin_v1":
        content_size = max(1, round(image_size * 11 / 12))
    else:
        raise ValueError(f"unknown preprocess profile: {preprocess_profile}")
    scale = min(content_size / source.width, content_size / source.height)
    width = max(1, round(source.width * scale))
    height = max(1, round(source.height * scale))
    resized = source.resize((width, height), Image.Resampling.BILINEAR)
    canvas = Image.new("L", (image_size, image_size), color=255)
    canvas.paste(resized, ((image_size - width) // 2, (image_size - height) // 2))
    return canvas


def build_image_transform(
    image_size: int,
    augment: bool = False,
    augmentation_profile: str = "legacy",
) -> transforms.Compose:
    """构建训练或推理阶段的图像变换。

    训练时可以加入轻微仿射扰动；推理时不做随机增强。最后 ToTensor 把像素
    从 HxW 变成 1xHxW，并把 [0, 255] 映射到浮点数后进行标准化。
    """

    if image_size < 8:
        raise ValueError("image_size must be at least 8")
    steps: list[object] = []
    if augmentation_profile not in {"legacy", "gentle_elastic"}:
        raise ValueError(f"unknown augmentation profile: {augmentation_profile}")
    if augment and augmentation_profile == "legacy":
        steps.append(
            transforms.RandomAffine(
                degrees=10,
                translate=(0.08, 0.08),
                scale=(0.9, 1.1),
                fill=255,
            )
        )
    elif augment:
        steps.extend(
            [
                transforms.RandomAffine(
                    degrees=7,
                    translate=(0.05, 0.05),
                    scale=(0.92, 1.08),
                    fill=255,
                ),
                transforms.RandomApply(
                    [transforms.ElasticTransform(alpha=8.0, sigma=4.0, fill=255)],
                    p=0.25,
                ),
            ]
        )
    steps.extend(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=NORMALIZE_MEAN, std=NORMALIZE_STD),
        ]
    )
    return transforms.Compose(steps)


def preprocess_image(
    image: Image.Image | np.ndarray,
    image_size: int,
    augment: bool = False,
    preprocess_profile: str = "legacy",
    augmentation_profile: str = "legacy",
) -> Tensor:
    """使用与训练集相同的 letterbox、归一化和可选增强处理图片。"""

    array = np.asarray(image.convert("L") if isinstance(image, Image.Image) else image)
    return build_image_transform(
        image_size, augment=augment, augmentation_profile=augmentation_profile
    )(_letterbox(array, image_size, preprocess_profile))


class GNTDataset(Dataset[tuple[Tensor, int]]):
    """从索引中按需读取单字符图像，避免生成中间图片文件。"""

    def __init__(
        self,
        index: GNTIndex,
        image_size: int = 96,
        augment: bool = False,
        record_indices: Sequence[int] | None = None,
        preprocess_profile: str = "legacy",
        augmentation_profile: str = "legacy",
    ) -> None:
        self.index = index
        self.image_size = image_size
        self.record_indices = tuple(
            range(len(index.records)) if record_indices is None else record_indices
        )
        if any(i < 0 or i >= len(index.records) for i in self.record_indices):
            raise IndexError("record_indices contains an invalid record number")
        self.preprocess_profile = preprocess_profile
        self.transform = build_image_transform(
            image_size, augment=augment, augmentation_profile=augmentation_profile
        )
        self._handles: dict[int, BinaryIO] = {}

    def __len__(self) -> int:
        return len(self.record_indices)

    def __getstate__(self) -> dict[str, object]:
        state = self.__dict__.copy()
        state["_handles"] = {}
        return state

    def __del__(self) -> None:
        for handle in getattr(self, "_handles", {}).values():
            handle.close()

    def _read_image(self, record: GNTRecord) -> np.ndarray:
        handle = self._handles.get(record.file_id)
        if handle is None:
            handle = self.index.files[record.file_id].open("rb")
            self._handles[record.file_id] = handle
        handle.seek(record.offset)
        header = handle.read(GNT_HEADER.size)
        sample_size, _raw_label, width, height = GNT_HEADER.unpack(header)
        if sample_size != record.sample_size or width != record.width or height != record.height:
            raise DatasetFormatError(
                f"GNT index changed while reading {self.index.files[record.file_id]}"
            )
        pixels = handle.read(width * height)
        if len(pixels) != width * height:
            raise DatasetFormatError(f"truncated GNT pixels: {self.index.files[record.file_id]}")
        return np.frombuffer(pixels, dtype=np.uint8).reshape(height, width).copy()

    def __getitem__(self, item: int) -> tuple[Tensor, int]:
        # PyTorch 的 DataLoader 会反复调用这里：item 是索引中的第几条记录，
        # 返回值必须是“模型输入张量 + 目标类别整数”。
        record = self.index.records[self.record_indices[item]]
        image = _letterbox(self._read_image(record), self.image_size, self.preprocess_profile)
        return self.transform(image), record.label

    def raw_image(self, item: int) -> np.ndarray:
        """读取指定数据集项的原始灰度图，用于错误样本分析。"""

        record = self.index.records[self.record_indices[item]]
        return self._read_image(record)
