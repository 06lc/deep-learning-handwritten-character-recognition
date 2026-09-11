from config import TrainConfig


def test_hwdb_default_class_count_includes_symbols_and_digits() -> None:
    assert TrainConfig().expected_num_classes == 3926
