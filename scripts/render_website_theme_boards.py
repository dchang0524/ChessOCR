"""Render full boards matching the website preview for grouped evaluation."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from chess_ocr.data.dataset_generator import (  # noqa: E402
    WebsitePreviewBoardTheme,
    generate_random_board_fens,
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--boards", type=int, default=200)
    parser.add_argument("--board-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20_260_821)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.boards <= 0 or args.board_size <= 0:
        raise ValueError("boards and board-size must be positive")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    theme = WebsitePreviewBoardTheme()
    positions = generate_random_board_fens(args.boards, seed=args.seed)
    for index, board_fen in enumerate(positions, start=1):
        path = args.output_dir / f"{board_fen.replace('/', '-')}.png"
        theme.render_board(board_fen, args.board_size).save(path)
        if index % 25 == 0 or index == len(positions):
            print(f"Rendered {index}/{len(positions)} website-preview boards", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
