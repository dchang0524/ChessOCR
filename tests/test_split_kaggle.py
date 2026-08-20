import importlib.util
from pathlib import Path

from PIL import Image

SCRIPT_PATH = Path(__file__).resolve().parents[1] / "scripts" / "split_kaggle.py"
SPEC = importlib.util.spec_from_file_location("split_kaggle", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
build_split_rows = MODULE.build_split_rows
build_all_90_10_rows = MODULE.build_all_90_10_rows


def _write_board(directory: Path, fen_stem: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (8, 8), color="white").save(directory / f"{fen_stem}.jpeg")


def test_build_split_rows_preserves_train_and_halves_holdout(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    for index in range(8):
        _write_board(root / "train", f"8-8-8-8-8-8-8-{index + 1}")
    for index in range(4):
        _write_board(root / "test", f"8-8-8-8-8-8-{index + 1}-8")

    first = build_split_rows(root, seed=42)
    second = build_split_rows(root, seed=42)

    assert first == second
    assert sum(row["split"] == "train" for row in first) == 8
    assert sum(row["split"] == "val" for row in first) == 2
    assert sum(row["split"] == "test" for row in first) == 2
    assert {row["image_path"].split("/")[0] for row in first if row["split"] == "train"} == {
        "train"
    }
    assert all("/" in row["board_fen"] for row in first)


def test_build_all_90_10_rows_combines_source_folders(tmp_path: Path) -> None:
    root = tmp_path / "dataset"
    for index in range(90):
        _write_board(root / "train", f"8-8-8-8-8-8-8-{index + 1}")
    for index in range(10):
        _write_board(root / "test", f"8-8-8-8-8-8-{index + 101}-8")

    first = build_all_90_10_rows(root, seed=7)
    second = build_all_90_10_rows(root, seed=7)

    assert first == second
    assert len(first) == 100
    assert sum(row["split"] == "train" for row in first) == 90
    assert sum(row["split"] == "test" for row in first) == 10
    source_folders = {row["image_path"].split("/")[0] for row in first}
    assert source_folders == {"train", "test"}
