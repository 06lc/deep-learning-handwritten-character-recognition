from __future__ import annotations

import struct
from pathlib import Path

import numpy as np
import pytest
import torch

from gnt_dataset import (
    DatasetFormatError,
    GNTDataset,
    build_index,
    load_index,
    merge_indexes,
    preprocess_image,
    save_index,
    split_record_indices_by_file,
)


def _write_gnt(path: Path, records: list[tuple[str, np.ndarray]]) -> None:
    with path.open("wb") as handle:
        for label, image in records:
            image = np.asarray(image, dtype=np.uint8)
            encoded = label.encode("gbk")
            if len(encoded) == 1:
                encoded += b"\x00"
            assert len(encoded) == 2
            height, width = image.shape
            sample_size = 10 + width * height
            handle.write(struct.pack("<I2sHH", sample_size, encoded, width, height))
            handle.write(image.tobytes())


def test_build_index_and_dataset_read_gnt_records(tmp_path: Path) -> None:
    root = tmp_path / "train"
    root.mkdir()
    image_a = np.array([[255, 0], [128, 64]], dtype=np.uint8)
    image_b = np.full((3, 1), 255, dtype=np.uint8)
    _write_gnt(root / "1001-f.gnt", [("A", image_a), ("你", image_b)])

    index = build_index((root,))
    assert len(index.records) == 2
    assert index.class_names == ("A", "你")

    dataset = GNTDataset(index, image_size=8)
    image, label = dataset[0]
    assert image.shape == (1, 8, 8)
    assert image.dtype == torch.float32
    assert label == 0
    assert float(image.max()) <= 1.0
    assert float(image.min()) >= -1.0


def test_index_round_trip_preserves_records_and_labels(tmp_path: Path) -> None:
    root = tmp_path / "train"
    root.mkdir()
    _write_gnt(root / "1001-f.gnt", [("A", np.zeros((2, 2), dtype=np.uint8))])
    index = build_index((root,))
    cache_path = tmp_path / "cache" / "train.npz"

    save_index(index, cache_path)
    restored = load_index(cache_path)

    assert restored.class_names == index.class_names
    assert restored.files == index.files
    assert restored.records == index.records


def test_legacy_index_without_filter_stats_still_loads(tmp_path: Path) -> None:
    cache_path = tmp_path / "legacy.npz"
    records = np.asarray([[0, 0, 14, 2, 2, 0]], dtype=np.int64)
    np.savez_compressed(
        cache_path,
        files=np.asarray([str(tmp_path / "1001-f.gnt")], dtype=np.str_),
        class_names=np.asarray(["A"], dtype=np.str_),
        records=records,
    )

    restored = load_index(cache_path)

    assert restored.class_names == ("A",)
    assert restored.records[0].label == 0
    assert restored.filter_stats is None


def test_test_label_missing_from_training_mapping_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "test"
    root.mkdir()
    _write_gnt(root / "1241-f.gnt", [("B", np.zeros((2, 2), dtype=np.uint8))])

    with pytest.raises(DatasetFormatError, match="unknown label"):
        build_index((root,), class_names=("A",))


def test_corrupt_gnt_sample_size_is_rejected(tmp_path: Path) -> None:
    root = tmp_path / "broken"
    root.mkdir()
    with (root / "1001-f.gnt").open("wb") as handle:
        handle.write(struct.pack("<I2sHH", 11, b"A\x00", 2, 2))
        handle.write(b"\x00" * 4)

    with pytest.raises(DatasetFormatError, match="invalid sample size"):
        build_index((root,))


def test_validation_split_keeps_files_disjoint(tmp_path: Path) -> None:
    root = tmp_path / "train"
    root.mkdir()
    for writer in range(4):
        _write_gnt(
            root / f"{1001 + writer}-f.gnt",
            [("A", np.zeros((2, 2), dtype=np.uint8))],
        )
    index = build_index((root,))

    train_indices, validation_indices = split_record_indices_by_file(
        index, validation_fraction=0.5, seed=42
    )
    train_files = {index.records[i].file_id for i in train_indices}
    validation_files = {index.records[i].file_id for i in validation_indices}
    assert train_files.isdisjoint(validation_files)
    assert train_indices and validation_indices


def test_validation_manifest_is_portable_and_stable(tmp_path: Path) -> None:
    roots = (tmp_path / "first", tmp_path / "second")
    for root_index, root in enumerate(roots):
        root.mkdir()
        for writer in range(2):
            _write_gnt(
                root / f"{root_index}-{writer}.gnt",
                [("A", np.zeros((2, 2), dtype=np.uint8))],
            )
    manifest = tmp_path / "validation.json"
    index = build_index(roots)
    _, first_validation = split_record_indices_by_file(
        index,
        validation_fraction=0.5,
        seed=42,
        manifest_path=manifest,
        roots=roots,
    )
    selected_names = {index.files[index.records[i].file_id].name for i in first_validation}

    relocated = (tmp_path / "moved-first", tmp_path / "moved-second")
    for root_index, root in enumerate(relocated):
        root.mkdir()
        for writer in range(2):
            _write_gnt(
                root / f"{root_index}-{writer}.gnt",
                [("A", np.zeros((2, 2), dtype=np.uint8))],
            )
    for writer in range(2):
        _write_gnt(
            relocated[0] / f"extra-{writer}.gnt",
            [("A", np.zeros((2, 2), dtype=np.uint8))],
        )
    relocated_index = build_index(relocated)
    _, second_validation = split_record_indices_by_file(
        relocated_index,
        validation_fraction=0.5,
        seed=42,
        manifest_path=manifest,
        roots=relocated,
    )
    relocated_names = {
        relocated_index.files[relocated_index.records[i].file_id].name
        for i in second_validation
    }

    assert selected_names == relocated_names


def test_margin_preprocessing_leaves_safe_border() -> None:
    tensor = preprocess_image(
        np.zeros((12, 12), dtype=np.uint8), 96, preprocess_profile="margin_v1"
    )
    rows, columns = torch.where(tensor[0] < 0.0)

    assert int(rows.min()) >= 4
    assert int(columns.min()) >= 4
    assert int(rows.max()) <= 91
    assert int(columns.max()) <= 91


def test_gentle_elastic_augmentation_preserves_shape() -> None:
    tensor = preprocess_image(
        np.zeros((12, 8), dtype=np.uint8),
        96,
        augment=True,
        preprocess_profile="margin_v1",
        augmentation_profile="gentle_elastic",
    )

    assert tensor.shape == (1, 96, 96)


def test_skip_unknown_labels_and_merge_indexes(tmp_path: Path) -> None:
    primary_root = tmp_path / "primary"
    additional_root = tmp_path / "additional"
    primary_root.mkdir()
    additional_root.mkdir()
    image_a = np.full((2, 2), 10, dtype=np.uint8)
    image_b = np.full((2, 2), 20, dtype=np.uint8)
    image_c = np.full((2, 2), 30, dtype=np.uint8)
    _write_gnt(primary_root / "1001-f.gnt", [("A", image_a), ("B", image_b)])
    _write_gnt(additional_root / "0001-f.gnt", [("B", image_b), ("C", image_c)])

    primary = build_index(primary_root)
    additional = build_index(
        additional_root,
        class_names=primary.class_names,
        unknown_label="skip",
    )
    merged = merge_indexes(primary, additional)

    assert additional.filter_stats is not None
    assert additional.filter_stats.scanned_samples == 2
    assert additional.filter_stats.accepted_samples == 1
    assert additional.filter_stats.excluded_samples == 1
    assert additional.filter_stats.shared_class_names == ("B",)
    assert additional.filter_stats.excluded_class_names == ("C",)
    assert merged.class_names == ("A", "B")
    assert merged.records[-1].file_id == 1
    assert merged.records[-1].offset == 0
    dataset = GNTDataset(merged, image_size=8)
    assert dataset.raw_image(len(dataset) - 1).tolist() == image_b.tolist()
    assert dataset[len(dataset) - 1][1] == 1


def test_merged_index_round_trip_preserves_filter_stats(tmp_path: Path) -> None:
    primary_root = tmp_path / "primary"
    additional_root = tmp_path / "additional"
    primary_root.mkdir()
    additional_root.mkdir()
    image = np.zeros((2, 2), dtype=np.uint8)
    _write_gnt(primary_root / "1001-f.gnt", [("A", image), ("B", image)])
    _write_gnt(additional_root / "0001-f.gnt", [("B", image), ("C", image)])
    primary = build_index(primary_root)
    additional = build_index(
        additional_root, class_names=primary.class_names, unknown_label="skip"
    )
    merged = merge_indexes(primary, additional)
    cache = tmp_path / "merged.npz"

    save_index(merged, cache)
    restored = load_index(cache)

    assert restored.records == merged.records
    assert restored.filter_stats == additional.filter_stats


def test_additional_files_never_enter_fixed_validation(tmp_path: Path) -> None:
    primary_roots = (tmp_path / "primary-1", tmp_path / "primary-2")
    additional_root = tmp_path / "additional"
    for root in (*primary_roots, additional_root):
        root.mkdir()
    image = np.zeros((2, 2), dtype=np.uint8)
    for root_number, root in enumerate(primary_roots):
        for writer in range(2):
            _write_gnt(root / f"{root_number}-{writer}.gnt", [("A", image)])
    _write_gnt(additional_root / "extra.gnt", [("A", image)])
    primary = build_index(primary_roots)
    additional = build_index(additional_root, class_names=primary.class_names)
    merged = merge_indexes(primary, additional)

    train_indices, validation_indices = split_record_indices_by_file(
        merged,
        validation_fraction=0.5,
        seed=42,
        manifest_path=tmp_path / "validation.json",
        roots=primary_roots,
    )

    validation_file_ids = {merged.records[i].file_id for i in validation_indices}
    training_file_ids = {merged.records[i].file_id for i in train_indices}
    assert all(file_id < len(primary.files) for file_id in validation_file_ids)
    assert len(primary.files) in training_file_ids
