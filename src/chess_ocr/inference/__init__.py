"""Inference orchestration."""

from chess_ocr.inference.board_predictor import BoardPredictor
from chess_ocr.inference.prediction_result import BoardPrediction, SquarePrediction

__all__ = ["BoardPredictor", "BoardPrediction", "SquarePrediction"]
