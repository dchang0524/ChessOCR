"""Render examples of generator-side RGB and board-offset augmentation."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from PIL import Image, ImageDraw  # noqa: E402

from chess_ocr.data.dataset_generator import (  # noqa: E402
    ImageAssetBoardTheme,
    generate_random_board_fens,
    jitter_board_crop,
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SPRITE_SETS = {
    name: PROJECT_ROOT / "assets" / "themes" / name
    for name in (
        "chessnut",
        "fantasy",
        "spatial",
        "celtic",
        "rhosgfx",
        "kiwen-suwi",
    )
}
PALETTES = {
    "classic": ((240, 217, 181), (181, 136, 99)),
    "green": ((238, 238, 210), (118, 150, 86)),
    "blue": ((222, 227, 230), (104, 139, 164)),
}
VARIANT_NAMES = ("base", "rgb", "offset", "rgb_offset")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Render the same training position without augmentation, with background-only "
            "RGB variation, with crop offset, and with both augmentations."
        )
    )
    parser.add_argument("--count", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--theme", choices=sorted(SPRITE_SETS), default="chessnut")
    parser.add_argument("--palette", choices=sorted(PALETTES), default="classic")
    parser.add_argument("--board-size", type=int, default=512)
    parser.add_argument("--background-strength", type=float, default=0.75)
    parser.add_argument("--max-offset-pixels", type=int, default=6)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/training_augmentation_samples"),
    )
    return parser.parse_args(argv)


def render_variants(
    theme: ImageAssetBoardTheme,
    board_fen: str,
    board_size: int,
    background_strength: float,
    max_offset_pixels: int,
    seed: int,
) -> dict[str, Image.Image]:
    """Render four aligned views of one position using the training augmenters."""
    base = theme.render_board(board_fen, board_size)
    rgb = theme.render_board(
        board_fen,
        board_size,
        background_rng=random.Random(seed),
        background_variation_strength=background_strength,
    )
    return {
        "base": base,
        "rgb": rgb,
        "offset": jitter_board_crop(base, max_offset_pixels, random.Random(seed + 1)),
        "rgb_offset": jitter_board_crop(
            rgb, max_offset_pixels, random.Random(seed + 2)
        ),
    }


def build_montage(samples: list[dict[str, Image.Image]]) -> Image.Image:
    """Lay out samples as rows and augmentation variants as columns."""
    preview_size = 256
    header_height = 38
    row_label_width = 34
    montage = Image.new(
        "RGB",
        (row_label_width + preview_size * len(VARIANT_NAMES), header_height + preview_size * len(samples)),
        "white",
    )
    draw = ImageDraw.Draw(montage)
    labels = {
        "base": "original",
        "rgb": "RGB background only",
        "offset": "crop offset only",
        "rgb_offset": "RGB + crop offset",
    }
    for column, variant in enumerate(VARIANT_NAMES):
        draw.text(
            (row_label_width + column * preview_size + 8, 12),
            labels[variant],
            fill="black",
        )
    for row, variants in enumerate(samples):
        draw.text((8, header_height + row * preview_size + 8), str(row + 1), fill="black")
        for column, variant in enumerate(VARIANT_NAMES):
            preview = variants[variant].resize(
                (preview_size, preview_size), Image.Resampling.LANCZOS
            )
            montage.paste(
                preview,
                (row_label_width + column * preview_size, header_height + row * preview_size),
            )
    return montage


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.count <= 0:
        raise ValueError("--count must be positive")
    if args.board_size <= 0:
        raise ValueError("--board-size must be positive")
    if not 0 <= args.background_strength <= 1:
        raise ValueError("--background-strength must be in [0, 1]")
    if args.max_offset_pixels < 0:
        raise ValueError("--max-offset-pixels must be non-negative")

    light_rgb, dark_rgb = PALETTES[args.palette]
    theme = ImageAssetBoardTheme(
        name=f"{args.theme}_{args.palette}",
        asset_dir=SPRITE_SETS[args.theme],
        light_rgb=light_rgb,
        dark_rgb=dark_rgb,
    )
    positions = generate_random_board_fens(
        args.count, seed=args.seed, min_pieces=10, max_pieces=28
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    samples = []
    for sample_index, board_fen in enumerate(positions, start=1):
        variants = render_variants(
            theme,
            board_fen,
            args.board_size,
            args.background_strength,
            args.max_offset_pixels,
            args.seed + sample_index * 10,
        )
        samples.append(variants)
        for variant, image in variants.items():
            image.save(args.output_dir / f"sample_{sample_index:02d}_{variant}.png")

    montage_path = args.output_dir / "montage.png"
    build_montage(samples).save(montage_path)
    print(f"Wrote {len(samples)} four-way augmentation samples to {args.output_dir}")
    print(f"Montage: {montage_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
