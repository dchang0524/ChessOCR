"""Neural network definitions."""

from chess_ocr.models.background_normalizer import SquareBackgroundNormalizer
from chess_ocr.models.dino_joint_classifier import DinoJointClassifier
from chess_ocr.models.similarity_classifier import SimilarityClassifier
from chess_ocr.models.square_classifier import SquareClassifier

__all__ = [
    "DinoJointClassifier",
    "SimilarityClassifier",
    "SquareBackgroundNormalizer",
    "SquareClassifier",
]
