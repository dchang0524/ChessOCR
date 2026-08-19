"""Compare independent and grouped OCR on evaluation-only Kaggle boards."""

from __future__ import annotations

import argparse
import json
import sys
import time
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from chess_ocr.data.kaggle_board_dataset import KaggleBoardDataset  # noqa: E402
from chess_ocr.data.labels import CLASS_NAMES, CLASS_NAME_TO_ID  # noqa: E402
from chess_ocr.inference.board_predictor import resolve_device  # noqa: E402
from chess_ocr.inference.group_label_assigner import GroupLabelAssigner  # noqa: E402
from chess_ocr.inference.piece_clusterer import PieceClusterer  # noqa: E402
from chess_ocr.models.similarity_classifier import SimilarityClassifier  # noqa: E402
from chess_ocr.models.square_classifier import SquareClassifier  # noqa: E402


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate grouped OCR on Kaggle boards.")
    parser.add_argument("--classifier-checkpoint", type=Path, required=True)
    parser.add_argument("--similarity-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--image-dir", type=Path, default=Path("data/raw/kaggle_chess_positions/test")
    )
    parser.add_argument("--max-boards", type=int, default=None)
    parser.add_argument(
        "--skip-boards",
        type=int,
        default=0,
        help="Skip a deterministic sorted prefix, e.g. a threshold-calibration slice",
    )
    parser.add_argument("--board-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--duplicate-penalty", type=float, default=1.5)
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=None,
        help="Optional clustering-threshold override for evaluation sweeps",
    )
    parser.add_argument(
        "--output", type=Path, default=Path("outputs/grouped_kaggle_evaluation.json")
    )
    return parser.parse_args()


def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def main() -> int:
    args = parse_args()
    device = resolve_device(args.device)
    classifier_checkpoint = torch.load(
        args.classifier_checkpoint, map_location=device, weights_only=False
    )
    similarity_checkpoint = torch.load(
        args.similarity_checkpoint, map_location=device, weights_only=False
    )
    class_names = list(classifier_checkpoint.get("class_names", CLASS_NAMES))
    classifier = SquareClassifier(len(class_names)).to(device)
    classifier.load_state_dict(classifier_checkpoint["model_state_dict"])
    classifier.eval()
    similarity = SimilarityClassifier(int(similarity_checkpoint.get("embedding_size", 64))).to(
        device
    )
    similarity.load_state_dict(similarity_checkpoint["model_state_dict"])
    similarity.eval()

    if args.skip_boards < 0:
        raise ValueError("--skip-boards must be non-negative")
    loaded_maximum = (
        args.skip_boards + args.max_boards if args.max_boards is not None else None
    )
    dataset = KaggleBoardDataset(
        image_dir=args.image_dir,
        max_boards=loaded_maximum,
        input_size=int(classifier_checkpoint.get("input_size", 64)),
    )
    if args.skip_boards:
        dataset.paths = dataset.paths[args.skip_boards :]
        dataset.board_fens = dataset.board_fens[args.skip_boards :]
    loader = DataLoader(
        dataset,
        batch_size=args.board_batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )
    clusterer = PieceClusterer(
        float(similarity_checkpoint["similarity_threshold"])
        if args.similarity_threshold is None
        else args.similarity_threshold
    )
    assigner = GroupLabelAssigner(args.duplicate_penalty, class_names)
    empty_id = CLASS_NAME_TO_ID["empty"]

    square_count = 0
    occupied_count = 0
    baseline_correct = 0
    grouped_correct = 0
    baseline_occupied_correct = 0
    grouped_occupied_correct = 0
    baseline_exact_boards = 0
    grouped_exact_boards = 0
    false_merge_pairs = 0
    predicted_merged_pairs = 0
    false_split_pairs = 0
    true_same_pairs = 0
    started = time.perf_counter()

    with torch.no_grad():
        for squares, labels, _ in loader:
            batch_size = squares.shape[0]
            flat_squares = squares.flatten(0, 1).to(device)
            logits = classifier(flat_squares).reshape(batch_size, 64, -1).cpu()
            embeddings = similarity.encode(flat_squares).reshape(batch_size, 64, -1).cpu()
            labels = labels.cpu()

            for board_index in range(batch_size):
                board_logits = logits[board_index]
                board_labels = labels[board_index]
                baseline = board_logits.argmax(dim=1)
                square_indices = list(range(64))
                clustering = clusterer.cluster(embeddings[board_index], square_indices)
                assignments = assigner.assign(board_logits, clustering.clusters)
                grouped = torch.tensor(
                    assigner.apply(
                        [int(value) for value in baseline], clustering.clusters, assignments
                    )
                )

                occupied = board_labels != empty_id
                baseline_hits = baseline == board_labels
                grouped_hits = grouped == board_labels
                square_count += 64
                occupied_count += int(occupied.sum())
                baseline_correct += int(baseline_hits.sum())
                grouped_correct += int(grouped_hits.sum())
                baseline_occupied_correct += int((baseline_hits & occupied).sum())
                grouped_occupied_correct += int((grouped_hits & occupied).sum())
                baseline_exact_boards += int(bool(baseline_hits.all()))
                grouped_exact_boards += int(bool(grouped_hits.all()))
                group_by_square = {
                    square: cluster.group_id
                    for cluster in clustering.clusters
                    for square in cluster.square_indices
                }
                for first, second in combinations(square_indices, 2):
                    predicted_same = group_by_square[first] == group_by_square[second]
                    true_same = int(board_labels[first]) == int(board_labels[second])
                    if predicted_same:
                        predicted_merged_pairs += 1
                        false_merge_pairs += int(not true_same)
                    if true_same:
                        true_same_pairs += 1
                        false_split_pairs += int(not predicted_same)

    elapsed = time.perf_counter() - started
    board_count = len(dataset)
    payload = {
        "classifier_checkpoint": str(args.classifier_checkpoint),
        "similarity_checkpoint": str(args.similarity_checkpoint),
        "training_source": classifier_checkpoint.get("training_metadata"),
        "evaluation_source": str(args.image_dir),
        "skip_boards": args.skip_boards,
        "board_count": board_count,
        "elapsed_seconds": elapsed,
        "similarity_threshold": clusterer.similarity_threshold,
        "duplicate_penalty": args.duplicate_penalty,
        "baseline": {
            "square_accuracy": safe_ratio(baseline_correct, square_count),
            "occupied_accuracy": safe_ratio(baseline_occupied_correct, occupied_count),
            "exact_board_accuracy": safe_ratio(baseline_exact_boards, board_count),
        },
        "grouped": {
            "square_accuracy": safe_ratio(grouped_correct, square_count),
            "occupied_accuracy": safe_ratio(grouped_occupied_correct, occupied_count),
            "exact_board_accuracy": safe_ratio(grouped_exact_boards, board_count),
        },
        "clustering": {
            "scope": "all_64_squares_including_empty",
            "false_merge_rate": safe_ratio(false_merge_pairs, predicted_merged_pairs),
            "false_split_rate": safe_ratio(false_split_pairs, true_same_pairs),
            "predicted_merged_pairs": predicted_merged_pairs,
            "true_same_pairs": true_same_pairs,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
