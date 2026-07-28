"""CLI for generating a labelled dataset of chessboard squares.

Example:
    python scripts/generate_dataset.py --positions 300 --output-dir data/processed/synthetic_v1
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chess_ocr.data.dataset_generator import (  # noqa: E402
    BoardTheme,
    DatasetGenerator,
    GenerationConfig,
    ImageAssetBoardTheme,
    SyntheticBoardTheme,
    class_coverage,
    generate_random_board_fens,
)

STARTING_BOARD_FEN = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR"

BUILTIN_THEMES = {
    "synthetic_classic": SyntheticBoardTheme(
        name="synthetic_classic",
        light_rgb=(240, 217, 181),
        dark_rgb=(181, 136, 99),
    ),
    "synthetic_green": SyntheticBoardTheme(
        name="synthetic_green",
        light_rgb=(238, 238, 210),
        dark_rgb=(118, 150, 86),
    ),
    "synthetic_blue": SyntheticBoardTheme(
        name="synthetic_blue",
        light_rgb=(222, 227, 230),
        dark_rgb=(140, 162, 173),
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate labelled chessboard square images.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("data/processed/synthetic_v1"),
        help="Directory receiving squares/ and metadata.csv",
    )
    parser.add_argument("--positions", type=int, default=200, help="Number of random positions")
    parser.add_argument(
        "--themes",
        nargs="+",
        default=["synthetic_classic", "synthetic_green"],
        choices=sorted(BUILTIN_THEMES),
        help="Built-in synthetic themes to render",
    )
    parser.add_argument(
        "--asset-theme",
        nargs=2,
        metavar=("NAME", "DIR"),
        action="append",
        default=None,
        help="Additional theme from your own piece sprites: NAME DIRECTORY",
    )
    parser.add_argument("--board-size", type=int, default=512, help="Rendered board size")
    parser.add_argument("--square-size", type=int, default=64, help="Saved square image size")
    parser.add_argument("--train-fraction", type=float, default=0.7)
    parser.add_argument("--val-fraction", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--min-pieces", type=int, default=4, help="Minimum pieces per position (kings included)"
    )
    parser.add_argument("--max-pieces", type=int, default=30, help="Maximum pieces per position")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    themes: list[BoardTheme] = [BUILTIN_THEMES[name] for name in args.themes]
    for name, directory in args.asset_theme or []:
        themes.append(ImageAssetBoardTheme(name=name, asset_dir=Path(directory)))

    board_fens = [STARTING_BOARD_FEN] + generate_random_board_fens(
        count=max(1, args.positions - 1),
        seed=args.seed,
        min_pieces=args.min_pieces,
        max_pieces=args.max_pieces,
    )

    coverage = class_coverage(board_fens)
    print("Class coverage across positions:")
    for class_name, count in coverage.items():
        print(f"  {class_name:<14}{count:>8}")
    missing = [name for name, count in coverage.items() if count == 0]
    if missing:
        print(f"WARNING: no examples for {missing}; increase --positions or --max-pieces")

    config = GenerationConfig(
        output_dir=args.output_dir,
        board_size=args.board_size,
        square_size=args.square_size,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        seed=args.seed,
    )
    metadata_path = DatasetGenerator(themes=themes, config=config).generate(board_fens)
    print(f"Done. Metadata written to {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
