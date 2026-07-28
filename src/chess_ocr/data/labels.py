"""Authoritative label definitions for the square classifier.

This module is the single source of truth for:

* the class ordering used by the neural network,
* the mapping between class names and FEN piece symbols,
* the mapping between board squares and their index in FEN order.

Nothing else in the code base should hard-code the class ordering.
"""

from __future__ import annotations

CLASS_NAMES: list[str] = [
    "empty",
    "white_pawn",
    "white_knight",
    "white_bishop",
    "white_rook",
    "white_queen",
    "white_king",
    "black_pawn",
    "black_knight",
    "black_bishop",
    "black_rook",
    "black_queen",
    "black_king",
]

NUM_CLASSES: int = len(CLASS_NAMES)

#: Class name -> FEN symbol. The empty square maps to the empty string.
CLASS_TO_FEN: dict[str, str] = {
    "empty": "",
    "white_pawn": "P",
    "white_knight": "N",
    "white_bishop": "B",
    "white_rook": "R",
    "white_queen": "Q",
    "white_king": "K",
    "black_pawn": "p",
    "black_knight": "n",
    "black_bishop": "b",
    "black_rook": "r",
    "black_queen": "q",
    "black_king": "k",
}

#: FEN symbol -> class name (inverse of :data:`CLASS_TO_FEN`).
FEN_TO_CLASS: dict[str, str] = {symbol: name for name, symbol in CLASS_TO_FEN.items()}

#: Class name -> integer class id used by the model.
CLASS_NAME_TO_ID: dict[str, int] = {name: index for index, name in enumerate(CLASS_NAMES)}

#: Class id -> FEN symbol, indexed positionally for fast lookup.
CLASS_ID_TO_FEN: list[str] = [CLASS_TO_FEN[name] for name in CLASS_NAMES]

FILES: str = "abcdefgh"

#: Square names in FEN order: a8, b8, ..., h8, a7, ..., h1.
SQUARE_NAMES: list[str] = [f"{FILES[index % 8]}{8 - index // 8}" for index in range(64)]

#: Square name -> index in FEN order.
SQUARE_NAME_TO_INDEX: dict[str, int] = {name: index for index, name in enumerate(SQUARE_NAMES)}


def class_id_to_name(class_id: int) -> str:
    """Return the class name for ``class_id``.

    Args:
        class_id: Integer class id produced by the model.

    Returns:
        The corresponding class name.

    Raises:
        ValueError: If ``class_id`` is outside the valid range.
    """
    if not 0 <= class_id < NUM_CLASSES:
        raise ValueError(f"class_id must be in [0, {NUM_CLASSES - 1}], got {class_id}")
    return CLASS_NAMES[class_id]


def class_id_to_fen(class_id: int) -> str:
    """Return the FEN symbol for ``class_id`` (empty string for empty squares).

    Args:
        class_id: Integer class id produced by the model.

    Returns:
        The FEN piece symbol, or ``""`` for an empty square.

    Raises:
        ValueError: If ``class_id`` is outside the valid range.
    """
    if not 0 <= class_id < NUM_CLASSES:
        raise ValueError(f"class_id must be in [0, {NUM_CLASSES - 1}], got {class_id}")
    return CLASS_ID_TO_FEN[class_id]


def fen_to_class_id(symbol: str) -> int:
    """Return the class id for a FEN piece symbol.

    Args:
        symbol: A FEN piece symbol such as ``"P"`` or ``"k"``. The empty string
            denotes an empty square.

    Returns:
        The integer class id.

    Raises:
        ValueError: If ``symbol`` is not a recognised FEN piece symbol.
    """
    try:
        return CLASS_NAME_TO_ID[FEN_TO_CLASS[symbol]]
    except KeyError as error:
        raise ValueError(f"Unknown FEN symbol: {symbol!r}") from error


def square_name(index: int) -> str:
    """Return the algebraic name of the square at ``index`` in FEN order.

    Args:
        index: Position in FEN order, where ``0`` is ``a8`` and ``63`` is ``h1``.

    Returns:
        The algebraic square name, for example ``"e4"``.

    Raises:
        ValueError: If ``index`` is outside ``[0, 63]``.
    """
    if not 0 <= index < 64:
        raise ValueError(f"index must be in [0, 63], got {index}")
    return SQUARE_NAMES[index]


def square_color(index: int) -> str:
    """Return ``"light"`` or ``"dark"`` for the square at ``index`` in FEN order.

    Args:
        index: Position in FEN order, where ``0`` is ``a8``.

    Returns:
        ``"light"`` for light squares and ``"dark"`` for dark squares.
    """
    if not 0 <= index < 64:
        raise ValueError(f"index must be in [0, 63], got {index}")
    file_index = index % 8
    rank_index = index // 8
    return "light" if (file_index + rank_index) % 2 == 0 else "dark"
