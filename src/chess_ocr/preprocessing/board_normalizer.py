"""Normalisation of an already-cropped chessboard image."""

from __future__ import annotations

from PIL import Image

DEFAULT_OUTPUT_SIZE = 512


class BoardNormalizer:
    """Turn a cropped board image into a fixed-size RGB square.

    The normaliser deliberately knows nothing about *how* the board was
    cropped. In the MVP the crop comes from the interactive cropper in the
    Streamlit UI; a future perspective-aware normaliser can replace this class
    without touching the splitter, the model, or the FEN logic, as long as it
    keeps the ``normalize(image) -> Image.Image`` contract.
    """

    def __init__(self, output_size: int = DEFAULT_OUTPUT_SIZE) -> None:
        """Initialise the normaliser.

        Args:
            output_size: Side length in pixels of the normalised board. Must be
                positive; using a multiple of 8 avoids uneven square sizes.

        Raises:
            ValueError: If ``output_size`` is not positive.
        """
        if output_size <= 0:
            raise ValueError(f"output_size must be positive, got {output_size}")
        self.output_size = output_size

    def normalize(self, image: Image.Image) -> Image.Image:
        """Convert ``image`` to an RGB square of ``self.output_size`` pixels.

        Args:
            image: An already-cropped Pillow image of a chessboard. It does not
                need to be square; a non-square crop is stretched to fit.

        Returns:
            A new RGB Pillow image of size ``output_size x output_size``.

        Raises:
            TypeError: If ``image`` is not a Pillow image.
            ValueError: If the image has a zero width or height.
        """
        if not isinstance(image, Image.Image):
            raise TypeError(f"Expected a PIL.Image.Image, got {type(image)!r}")

        width, height = image.size
        if width <= 0 or height <= 0:
            raise ValueError(f"Image must have nonzero dimensions, got {width}x{height}")

        rgb_image = image.convert("RGB")
        return rgb_image.resize(
            (self.output_size, self.output_size), resample=Image.Resampling.BICUBIC
        )
