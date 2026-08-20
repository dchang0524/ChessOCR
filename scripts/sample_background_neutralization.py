"""Visualize the retained experimental neutralizer (not used by either CNN)."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

from chess_ocr.data.kaggle_board_dataset import (  # noqa: E402
    KaggleBoardDataset,
    SUPPORTED_SUFFIXES,
)
from chess_ocr.models.background_normalizer import SquareBackgroundNormalizer  # noqa: E402

BOARD_SIDE = 8


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Sample boards and visualize the unused experimental neutralizer."
    )
    parser.add_argument(
        "--image-dir",
        type=Path,
        default=Path("data/raw/kaggle_chess_positions/test"),
    )
    parser.add_argument("--count", type=int, default=6)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--input-size", type=int, default=64)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/background_neutralization")
    )
    return parser.parse_args(argv)


def squares_to_board(squares: torch.Tensor) -> Image.Image:
    """Convert normalized ``(64, 3, H, W)`` squares back into one RGB board."""
    input_size = squares.shape[-1]
    board = (
        squares.reshape(BOARD_SIDE, BOARD_SIDE, 3, input_size, input_size)
        .permute(2, 0, 3, 1, 4)
        .reshape(3, BOARD_SIDE * input_size, BOARD_SIDE * input_size)
    )
    pixels = ((board.clamp(-1, 1) + 1) * 127.5).round().byte()
    return Image.fromarray(pixels.permute(1, 2, 0).numpy(), mode="RGB")


def build_montage(rows: list[tuple[str, Image.Image, Image.Image]]) -> Image.Image:
    """Lay out original and neutralized boards side-by-side."""
    preview_size = 320
    header_height = 42
    row_height = preview_size + header_height
    montage = Image.new("RGB", (preview_size * 2, row_height * len(rows)), "white")
    draw = ImageDraw.Draw(montage)
    for row, (name, original, neutralized) in enumerate(rows):
        top = row * row_height
        draw.text((8, top + 4), f"{row + 1}. original", fill="black")
        draw.text((preview_size + 8, top + 4), "model input: neutralized", fill="black")
        draw.text((8, top + 20), name[:92], fill=(70, 70, 70))
        original_preview = original.resize((preview_size, preview_size), Image.Resampling.LANCZOS)
        neutral_preview = neutralized.resize(
            (preview_size, preview_size), Image.Resampling.LANCZOS
        )
        montage.paste(original_preview, (0, top + header_height))
        montage.paste(neutral_preview, (preview_size, top + header_height))
    return montage


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.count <= 0:
        raise ValueError("--count must be positive")
    paths = sorted(
        path
        for path in args.image_dir.iterdir()
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    if len(paths) < args.count:
        raise ValueError(f"Requested {args.count} boards, but only found {len(paths)}")
    selected = random.Random(args.seed).sample(paths, args.count)
    dataset = KaggleBoardDataset(image_paths=selected, input_size=args.input_size)
    normalizer = SquareBackgroundNormalizer().eval()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[tuple[str, Image.Image, Image.Image]] = []
    with torch.no_grad():
        for index, path in enumerate(selected, start=1):
            squares, _, _ = dataset[index - 1]
            neutralized = squares_to_board(normalizer(squares))
            with Image.open(path) as source:
                original = source.convert("RGB").resize(
                    (args.input_size * BOARD_SIDE, args.input_size * BOARD_SIDE),
                    Image.Resampling.BICUBIC,
                )
            neutralized.save(args.output_dir / f"{index:02d}_neutralized.png")
            rows.append((path.name, original, neutralized))

    montage_path = args.output_dir / "montage.png"
    build_montage(rows).save(montage_path)
    print(f"Wrote {len(rows)} neutralized boards and montage to {args.output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
