"""Import permissively licensed Lichess piece sets as transparent PNGs.

The source SVGs are fetched from a pinned Lichess commit and rasterised to the
``wP.png`` / ``bK.png`` naming convention expected by ``ImageAssetBoardTheme``.
Only sets with clear permissive licences are included.
"""

from __future__ import annotations

import argparse
import re
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import resvg_py
from PIL import Image

LILA_COMMIT = "cf0c9f3e45415135f3e2261b4ef8c9b2cf5631c1"
LILA_RAW_BASE = f"https://raw.githubusercontent.com/lichess-org/lila/{LILA_COMMIT}"
PIECE_NAMES = tuple(f"{colour}{piece}" for colour in "wb" for piece in "PNBRQK")


@dataclass(frozen=True)
class SpriteSet:
    """Metadata needed to import and attribute one piece set."""

    source_directory: str
    author: str
    license_name: str
    license_url: str
    license_file_url: str | None = None


SPRITE_SETS = {
    "chessnut": SpriteSet(
        source_directory="chessnut",
        author="Alexis Luengas",
        license_name="Apache License 2.0",
        license_url="https://www.apache.org/licenses/LICENSE-2.0",
        license_file_url=(
            "https://raw.githubusercontent.com/LexLuengas/chessnut-pieces/master/LICENSE.txt"
        ),
    ),
    "fantasy": SpriteSet(
        source_directory="fantasy",
        author="Maurizio Monge",
        license_name="MIT License",
        license_url="https://opensource.org/license/mit",
        license_file_url="https://raw.githubusercontent.com/maurimo/chess-art/main/LICENSE",
    ),
    "spatial": SpriteSet(
        source_directory="spatial",
        author="Maurizio Monge",
        license_name="MIT License",
        license_url="https://opensource.org/license/mit",
        license_file_url="https://raw.githubusercontent.com/maurimo/chess-art/main/LICENSE",
    ),
    "celtic": SpriteSet(
        source_directory="celtic",
        author="Maurizio Monge",
        license_name="MIT License",
        license_url="https://opensource.org/license/mit",
        license_file_url="https://raw.githubusercontent.com/maurimo/chess-art/main/LICENSE",
    ),
    "rhosgfx": SpriteSet(
        source_directory="rhosgfx",
        author="RhosGFX",
        license_name="CC0 1.0 Universal",
        license_url="https://creativecommons.org/publicdomain/zero/1.0/",
    ),
}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sets",
        nargs="+",
        choices=sorted(SPRITE_SETS),
        default=sorted(SPRITE_SETS),
        help="Piece sets to import (defaults to every supported set)",
    )
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("assets/themes"),
        help="Directory receiving one folder per set",
    )
    parser.add_argument("--size", type=int, default=256, help="PNG width and height")
    return parser.parse_args(argv)


def download(url: str) -> bytes:
    """Download and return ``url`` with a bounded timeout."""
    request = urllib.request.Request(url, headers={"User-Agent": "ChessOCR sprite importer"})
    with urllib.request.urlopen(request, timeout=30) as response:
        return response.read()


def import_set(name: str, sprite_set: SpriteSet, output_root: Path, size: int) -> None:
    """Download, rasterise, and validate one complete 12-piece set."""
    output_dir = output_root / name
    output_dir.mkdir(parents=True, exist_ok=True)

    for piece_name in PIECE_NAMES:
        source_url = f"{LILA_RAW_BASE}/public/piece/{sprite_set.source_directory}/{piece_name}.svg"
        svg = download(source_url).decode("utf-8")
        root_end = svg.index(">")
        root = re.sub(r'\s(?:width|height)="[^"]*"', "", svg[:root_end])
        svg = f'{root} width="{size}" height="{size}"{svg[root_end:]}'
        output_path = output_dir / f"{piece_name}.png"
        output_path.write_bytes(
            resvg_py.svg_to_bytes(
                svg_string=svg,
                width=size,
                height=size,
            )
        )
        with Image.open(output_path) as image:
            if image.size != (size, size) or image.mode != "RGBA":
                raise ValueError(
                    f"Unexpected raster for {output_path}: size={image.size}, mode={image.mode}"
                )
            if image.getchannel("A").getbbox() is None:
                raise ValueError(f"Sprite has no visible pixels: {output_path}")

    if sprite_set.license_file_url is not None:
        (output_dir / "LICENSE.txt").write_bytes(download(sprite_set.license_file_url))
    print(f"Imported {name}: {len(PIECE_NAMES)} sprites -> {output_dir}")


def main(argv: list[str] | None = None) -> int:
    """Import the requested sprite sets."""
    args = parse_args(argv)
    if args.size <= 0:
        raise ValueError(f"size must be positive, got {args.size}")
    for name in args.sets:
        import_set(name, SPRITE_SETS[name], args.output_root, args.size)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
