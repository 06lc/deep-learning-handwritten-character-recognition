from config import TrainConfig


def test_hwdb_default_class_count_includes_symbols_and_digits() -> None:
    assert TrainConfig().expected_num_classes == 3926


def test_icdar2013_profile_keeps_a_separate_test_root(tmp_path) -> None:
    competition_root = tmp_path / "competition-gnt"

    config = TrainConfig(
        test_profile="icdar2013",
        competition_test_root=competition_root,
    )

    assert config.competition_test_root == competition_root
