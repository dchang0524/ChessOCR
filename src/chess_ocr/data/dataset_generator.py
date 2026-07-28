"""Generation of labelled square images from known positions.

The generator turns a collection of FEN positions into a directory of 64x64
square PNGs plus a metadata CSV that :class:`~chess_ocr.data.square_dataset.SquareDataset`
can read.

Two board themes are supported:

* :class:`SyntheticBoardTheme` draws a board procedurally. It uses Unicode chess
  glyphs when a suitable system font is available and falls back to simple
  geometric glyphs otherwise. Nothing is downloaded and no third-party artwork
  is bundled.
* :class:`ImageAssetBoardTheme` loads piece sprites you supply yourself (one
  transparent PNG per piece, named ``wP.png`` ... ``bK.png``). Point it at a
  piece set you are licensed to use; this project never scrapes assets from
  chess websites.

A synthetic board is *not* a substitute for real screenshots of the board themes
you intend to support. It exercises the whole pipeline end to end and gives the
model something to learn, but the honest path to a usable model is to add real
rendered boards from the themes you care about.
"""

from __future__ import annotations

import csv
import random
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import chess
from PIL import Image, ImageDraw, ImageFont

from chess_ocr.chess.fen_builder import board_fen_to_class_ids
from chess_ocr.data.labels import (
    CLASS_ID_TO_FEN,
    CLASS_NAMES,
    SQUARE_NAMES,
    class_id_to_name,
    square_color,
)
from chess_ocr.preprocessing.board_splitter import BoardSplitter

BOARD_SIDE = 8
DEFAULT_BOARD_SIZE = 512
DEFAULT_SQUARE_SIZE = 64

METADATA_FIELDS = (
    "image_path",
    "label",
    "label_id",
    "position_id",
    "square",
    "theme",
    "square_color",
    "split",
)

#: Unicode code points for the solid ("black") chess glyphs, keyed by piece type.
_GLYPHS = {
    "p": "\u265f",
    "n": "\u265e",
    "b": "\u265d",
    "r": "\u265c",
    "q": "\u265b",
    "k": "\u265a",
}

#: Fonts that are commonly present and contain the chess glyph block.
_FONT_CANDIDATES = (
    "/System/Library/Fonts/Apple Symbols.ttf",
    "/Library/Fonts/Arial Unicode.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/usr/share/fonts/truetype/freefont/FreeSerif.ttf",
    "C:/Windows/Fonts/seguisym.ttf",
)


def find_chess_font(size: int) -> ImageFont.FreeTypeFont | None:
    """Return a font containing chess glyphs at ``size`` pixels, if one exists.

    Args:
        size: Requested font size in pixels.

    Returns:
        A loaded font, or ``None`` when no candidate font could be used.
    """
    candidates = list(_FONT_CANDIDATES)
    try:  # matplotlib ships DejaVuSans, which covers the chess block.
        import matplotlib

        candidates.append(
            str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / "DejaVuSans.ttf")
        )
    except Exception:  # pragma: no cover - matplotlib is optional here
        pass

    for path in candidates:
        if not Path(path).is_file():
            continue
        try:
            font = ImageFont.truetype(path, size)
        except OSError:  # pragma: no cover - unreadable font file
            continue
        if font.getmask(_GLYPHS["k"]).getbbox() is not None:
            return font
    return None


class BoardTheme(Protocol):
    """A renderer that turns a board-FEN into a full board image."""

    name: str

    def render_board(self, board_fen: str, size: int) -> Image.Image:
        """Render ``board_fen`` as an RGB image of ``size`` x ``size`` pixels."""
        ...


@dataclass
class SyntheticBoardTheme:
    """Procedurally drawn board theme.

    Attributes:
        name: Theme identifier written to the metadata CSV.
        light_rgb: Colour of light squares.
        dark_rgb: Colour of dark squares.
        white_piece_rgb: Fill colour of white pieces.
        black_piece_rgb: Fill colour of black pieces.
        piece_scale: Glyph size as a fraction of the square size.
    """

    name: str = "synthetic_classic"
    light_rgb: tuple[int, int, int] = (240, 217, 181)
    dark_rgb: tuple[int, int, int] = (181, 136, 99)
    white_piece_rgb: tuple[int, int, int] = (250, 250, 250)
    black_piece_rgb: tuple[int, int, int] = (30, 30, 30)
    piece_scale: float = 0.78

    def render_board(self, board_fen: str, size: int = DEFAULT_BOARD_SIZE) -> Image.Image:
        """Render ``board_fen`` with White at the bottom.

        Args:
            board_fen: Board-placement field of a FEN string.
            size: Side length of the rendered board in pixels.

        Returns:
            An RGB board image.
        """
        board = Image.new("RGB", (size, size))
        draw = ImageDraw.Draw(board)
        edges = [round(index * size / BOARD_SIDE) for index in range(BOARD_SIDE + 1)]
        class_ids = board_fen_to_class_ids(board_fen)
        square_pixels = max(1, size // BOARD_SIDE)
        font = find_chess_font(int(square_pixels * self.piece_scale))

        for index, class_id in enumerate(class_ids):
            row, column = divmod(index, BOARD_SIDE)
            box = (edges[column], edges[row], edges[column + 1], edges[row + 1])
            is_light = square_color(index) == "light"
            draw.rectangle(box, fill=self.light_rgb if is_light else self.dark_rgb)

            symbol = CLASS_ID_TO_FEN[class_id]
            if not symbol:
                continue
            self._draw_piece(draw, box, symbol, font)
        return board

    def _draw_piece(
        self,
        draw: ImageDraw.ImageDraw,
        box: tuple[int, int, int, int],
        symbol: str,
        font: ImageFont.FreeTypeFont | None,
    ) -> None:
        """Draw one piece centred inside ``box``."""
        is_white = symbol.isupper()
        fill = self.white_piece_rgb if is_white else self.black_piece_rgb
        stroke = self.black_piece_rgb if is_white else self.white_piece_rgb
        centre = ((box[0] + box[2]) / 2, (box[1] + box[3]) / 2)

        if font is not None:
            draw.text(
                centre,
                _GLYPHS[symbol.lower()],
                font=font,
                fill=fill,
                anchor="mm",
                stroke_width=max(1, int((box[2] - box[0]) * 0.03)),
                stroke_fill=stroke,
            )
            return
        self._draw_geometric_piece(draw, box, symbol.lower(), fill, stroke)

    @staticmethod
    def _draw_geometric_piece(
        draw: ImageDraw.ImageDraw,
        box: tuple[int, int, int, int],
        piece_type: str,
        fill: tuple[int, int, int],
        stroke: tuple[int, int, int],
    ) -> None:
        """Fallback glyphs: one distinct shape per piece type."""
        left, top, right, bottom = box
        width = right - left
        pad = width * 0.16
        inner = (left + pad, top + pad, right - pad, bottom - pad)
        cx, cy = (left + right) / 2, (top + bottom) / 2
        radius = (width - 2 * pad) / 2
        outline_width = max(1, int(width * 0.04))

        if piece_type == "p":
            small = radius * 0.62
            draw.ellipse(
                (cx - small, cy - small, cx + small, cy + small),
                fill=fill,
                outline=stroke,
                width=outline_width,
            )
        elif piece_type == "n":
            draw.polygon(
                [(cx, inner[1]), (inner[2], inner[3]), (inner[0], inner[3])],
                fill=fill,
                outline=stroke,
            )
        elif piece_type == "b":
            draw.polygon(
                [(cx, inner[1]), (inner[2], cy), (cx, inner[3]), (inner[0], cy)],
                fill=fill,
                outline=stroke,
            )
        elif piece_type == "r":
            draw.rectangle(inner, fill=fill, outline=stroke, width=outline_width)
        elif piece_type == "q":
            draw.ellipse(inner, fill=fill, outline=stroke, width=outline_width)
            draw.polygon(
                [
                    (inner[0], cy),
                    (cx - radius * 0.4, inner[1]),
                    (cx, cy),
                    (cx + radius * 0.4, inner[1]),
                    (inner[2], cy),
                ],
                fill=stroke,
            )
        else:  # king
            draw.rectangle(inner, fill=fill, outline=stroke, width=outline_width)
            draw.line((cx, inner[1], cx, inner[3]), fill=stroke, width=outline_width * 2)
            draw.line((inner[0], cy, inner[2], cy), fill=stroke, width=outline_width * 2)


@dataclass
class ImageAssetBoardTheme:
    """Board theme backed by piece sprites you supply.

    Place one transparent PNG per piece in ``asset_dir`` using the standard
    naming convention ``wP.png``, ``wN.png``, ..., ``bK.png``. Only use artwork
    you have the right to use.

    Attributes:
        name: Theme identifier written to the metadata CSV.
        asset_dir: Directory containing the piece sprites.
        light_rgb: Colour of light squares.
        dark_rgb: Colour of dark squares.
        piece_scale: Sprite size as a fraction of the square size.
    """

    name: str
    asset_dir: Path
    light_rgb: tuple[int, int, int] = (238, 238, 210)
    dark_rgb: tuple[int, int, int] = (118, 150, 86)
    piece_scale: float = 0.92
    _cache: dict[str, Image.Image] = field(default_factory=dict, repr=False)

    def render_board(self, board_fen: str, size: int = DEFAULT_BOARD_SIZE) -> Image.Image:
        """Render ``board_fen`` with White at the bottom using the sprite set.

        Args:
            board_fen: Board-placement field of a FEN string.
            size: Side length of the rendered board in pixels.

        Returns:
            An RGB board image.

        Raises:
            FileNotFoundError: If a required sprite is missing.
        """
        board = Image.new("RGB", (size, size))
        draw = ImageDraw.Draw(board)
        edges = [round(index * size / BOARD_SIDE) for index in range(BOARD_SIDE + 1)]
        class_ids = board_fen_to_class_ids(board_fen)

        for index, class_id in enumerate(class_ids):
            row, column = divmod(index, BOARD_SIDE)
            box = (edges[column], edges[row], edges[column + 1], edges[row + 1])
            is_light = square_color(index) == "light"
            draw.rectangle(box, fill=self.light_rgb if is_light else self.dark_rgb)

            symbol = CLASS_ID_TO_FEN[class_id]
            if not symbol:
                continue
            square_size = box[2] - box[0]
            sprite_size = max(1, int(square_size * self.piece_scale))
            sprite = self._load_sprite(symbol).resize(
                (sprite_size, sprite_size), Image.Resampling.LANCZOS
            )
            offset = (
                box[0] + (square_size - sprite_size) // 2,
                box[1] + (box[3] - box[1] - sprite_size) // 2,
            )
            board.paste(sprite, offset, sprite)
        return board

    def _load_sprite(self, symbol: str) -> Image.Image:
        """Load and cache the RGBA sprite for a FEN piece symbol."""
        if symbol in self._cache:
            return self._cache[symbol]
        colour = "w" if symbol.isupper() else "b"
        path = Path(self.asset_dir) / f"{colour}{symbol.upper()}.png"
        if not path.is_file():
            raise FileNotFoundError(f"Missing piece sprite for {symbol!r}: expected {path}")
        with Image.open(path) as image:
            sprite = image.convert("RGBA")
        self._cache[symbol] = sprite
        return sprite


@dataclass
class GenerationConfig:
    """Settings for a dataset generation run.

    Attributes:
        output_dir: Directory that receives ``squares/`` and ``metadata.csv``.
        board_size: Side length of each rendered board in pixels.
        square_size: Side length of each saved square image in pixels.
        train_fraction: Fraction of positions assigned to the training split.
        val_fraction: Fraction of positions assigned to the validation split.
        seed: Seed controlling the position-level split.
    """

    output_dir: Path
    board_size: int = DEFAULT_BOARD_SIZE
    square_size: int = DEFAULT_SQUARE_SIZE
    train_fraction: float = 0.7
    val_fraction: float = 0.15
    seed: int = 0

    def __post_init__(self) -> None:
        if not 0 < self.train_fraction < 1:
            raise ValueError("train_fraction must be in (0, 1)")
        if not 0 <= self.val_fraction < 1:
            raise ValueError("val_fraction must be in [0, 1)")
        if self.train_fraction + self.val_fraction >= 1:
            raise ValueError("train_fraction + val_fraction must leave room for a test split")


class DatasetGenerator:
    """Render positions, split them into squares, and write labelled metadata."""

    def __init__(
        self,
        themes: Sequence[BoardTheme],
        config: GenerationConfig,
        splitter: BoardSplitter | None = None,
    ) -> None:
        """Initialise the generator.

        Args:
            themes: One or more board themes to render each position with.
            config: Generation settings.
            splitter: Splitter used to cut boards into squares. Defaults to a
                plain :class:`BoardSplitter`.

        Raises:
            ValueError: If ``themes`` is empty.
        """
        if not themes:
            raise ValueError("At least one board theme is required")
        self.themes = list(themes)
        self.config = config
        self.splitter = splitter or BoardSplitter()

    def generate(self, board_fens: Iterable[str], verbose: bool = True) -> Path:
        """Generate the dataset and return the path to the metadata CSV.

        Splits are assigned per position, so all 64 squares of a board land in
        the same split and no board leaks between train, validation and test.

        Args:
            board_fens: Board-placement FEN fields to render.
            verbose: Print progress while rendering.

        Returns:
            Path to the written ``metadata.csv``.

        Raises:
            ValueError: If ``board_fens`` is empty.
        """
        positions = list(board_fens)
        if not positions:
            raise ValueError("No positions supplied")

        output_dir = Path(self.config.output_dir)
        squares_dir = output_dir / "squares"
        squares_dir.mkdir(parents=True, exist_ok=True)

        splits = self._assign_splits(len(positions))
        rows: list[dict[str, object]] = []

        for position_index, board_fen in enumerate(positions):
            position_id = f"pos_{position_index:05d}"
            split = splits[position_index]
            class_ids = board_fen_to_class_ids(board_fen)

            for theme in self.themes:
                board = theme.render_board(board_fen, self.config.board_size)
                squares = self.splitter.split(board, white_at_bottom=True)
                for index, (square_image, class_id) in enumerate(
                    zip(squares, class_ids, strict=True)
                ):
                    label = class_id_to_name(class_id)
                    relative_path = (
                        Path("squares")
                        / label
                        / (f"{position_id}_{theme.name}_{SQUARE_NAMES[index]}.png")
                    )
                    absolute_path = output_dir / relative_path
                    absolute_path.parent.mkdir(parents=True, exist_ok=True)
                    square_image.resize(
                        (self.config.square_size, self.config.square_size),
                        Image.Resampling.LANCZOS,
                    ).save(absolute_path)
                    rows.append(
                        {
                            "image_path": relative_path.as_posix(),
                            "label": label,
                            "label_id": class_id,
                            "position_id": position_id,
                            "square": SQUARE_NAMES[index],
                            "theme": theme.name,
                            "square_color": square_color(index),
                            "split": split,
                        }
                    )
            if verbose and (position_index + 1) % 25 == 0:
                print(f"Rendered {position_index + 1}/{len(positions)} positions")

        metadata_path = output_dir / "metadata.csv"
        with metadata_path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(METADATA_FIELDS))
            writer.writeheader()
            writer.writerows(rows)

        if verbose:
            print(f"Wrote {len(rows)} square images to {output_dir}")
            print(f"Metadata: {metadata_path}")
        return metadata_path

    def _assign_splits(self, position_count: int) -> list[str]:
        """Assign a split name to every position id."""
        indices = list(range(position_count))
        random.Random(self.config.seed).shuffle(indices)
        train_end = int(position_count * self.config.train_fraction)
        val_end = train_end + int(position_count * self.config.val_fraction)

        splits = ["test"] * position_count
        for rank, position_index in enumerate(indices):
            if rank < train_end:
                splits[position_index] = "train"
            elif rank < val_end:
                splits[position_index] = "val"
        return splits


def generate_random_board_fens(
    count: int,
    seed: int = 0,
    min_pieces: int = 4,
    max_pieces: int = 30,
) -> list[str]:
    """Create pseudo-random legal-looking positions for dataset generation.

    Each position contains exactly one king per side plus a random selection of
    other pieces, with pawns kept off the first and eighth ranks. Positions are
    not guaranteed to be reachable in a real game; they exist to give the
    classifier broad coverage of every class on both light and dark squares.

    Args:
        count: Number of positions to create.
        seed: Random seed.
        min_pieces: Minimum total pieces including both kings.
        max_pieces: Maximum total pieces including both kings.

    Returns:
        A list of board-placement FEN fields.

    Raises:
        ValueError: If the piece-count bounds are inconsistent.
    """
    if count <= 0:
        raise ValueError("count must be positive")
    if not 2 <= min_pieces <= max_pieces <= 64:
        raise ValueError("Require 2 <= min_pieces <= max_pieces <= 64")

    rng = random.Random(seed)
    non_king_symbols = "PNBRQpnbrq"
    positions: list[str] = []

    for _ in range(count):
        board = chess.Board(None)
        free_squares = list(chess.SQUARES)
        rng.shuffle(free_squares)

        for symbol in ("K", "k"):
            square = free_squares.pop()
            board.set_piece_at(square, chess.Piece.from_symbol(symbol))

        target = rng.randint(min_pieces, max_pieces) - 2
        for _ in range(target):
            if not free_squares:
                break
            symbol = rng.choice(non_king_symbols)
            square = free_squares.pop()
            if symbol.lower() == "p" and chess.square_rank(square) in (0, 7):
                symbol = rng.choice("NBRQ") if symbol.isupper() else rng.choice("nbrq")
            board.set_piece_at(square, chess.Piece.from_symbol(symbol))

        positions.append(board.board_fen())
    return positions


def class_coverage(board_fens: Sequence[str]) -> dict[str, int]:
    """Count how often each class appears across ``board_fens``.

    Args:
        board_fens: Board-placement FEN fields.

    Returns:
        A mapping from class name to occurrence count, covering all 13 classes.
    """
    counts = {name: 0 for name in CLASS_NAMES}
    for board_fen in board_fens:
        for class_id in board_fen_to_class_ids(board_fen):
            counts[class_id_to_name(class_id)] += 1
    return counts
