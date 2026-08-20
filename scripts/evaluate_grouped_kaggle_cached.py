"""Evaluate joint DINO OCR with exact caching of byte-identical square inputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
import time
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from PIL import Image  # noqa: E402
from torchvision.transforms import functional as transform_functional  # noqa: E402

from chess_ocr.chess.fen_builder import board_fen_to_class_ids  # noqa: E402
from chess_ocr.data.kaggle_board_dataset import fen_from_kaggle_filename  # noqa: E402
from chess_ocr.data.labels import CLASS_NAME_TO_ID, CLASS_NAMES  # noqa: E402
from chess_ocr.data.square_dataset import NORMALIZATION_MEAN, NORMALIZATION_STD  # noqa: E402
from chess_ocr.inference.board_predictor import resolve_device  # noqa: E402
from chess_ocr.inference.group_label_assigner import GroupLabelAssigner  # noqa: E402
from chess_ocr.inference.piece_clusterer import PieceClusterer  # noqa: E402
from chess_ocr.models.dino_joint_classifier import (  # noqa: E402
    DINO_ARCHITECTURE,
    dino_joint_model_from_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data/raw/kaggle_chess_positions"),
    )
    parser.add_argument("--split", default="test")
    parser.add_argument("--max-boards", type=int, default=None)
    parser.add_argument(
        "--skip-boards",
        type=int,
        default=0,
        help="Skip a deterministic manifest prefix before evaluation.",
    )
    parser.add_argument("--board-batch-size", type=int, default=8)
    parser.add_argument("--square-batch-size", type=int, default=256)
    parser.add_argument("--duplicate-penalty", type=float, default=1.5)
    parser.add_argument(
        "--similarity-thresholds",
        type=float,
        nargs="*",
        default=None,
        help=(
            "Optional thresholds to evaluate in the same model pass. The checkpoint "
            "threshold is still reported as the primary grouped result."
        ),
    )
    parser.add_argument("--progress-every", type=int, default=250)
    parser.add_argument("--device", choices=("cpu", "cuda", "mps"), default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/grouped_kaggle_joint_dino_kaggle90_reserved10.json"),
    )
    return parser.parse_args()


def safe_ratio(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def manifest_paths(
    manifest: Path,
    data_root: Path,
    split: str,
    maximum: int | None,
    skip: int = 0,
) -> list[Path]:
    with manifest.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = {"image_path", "split"}
        if reader.fieldnames is None or not required.issubset(reader.fieldnames):
            raise ValueError(f"Manifest must contain columns {sorted(required)}")
        paths = [data_root / row["image_path"] for row in reader if row["split"] == split]
    paths = paths[skip : skip + maximum if maximum is not None else None]
    if not paths:
        raise ValueError(f"Manifest split {split!r} contains no boards")
    return paths


def square_tensor(square: Image.Image) -> torch.Tensor:
    tensor = transform_functional.pil_to_tensor(square).float().div_(255.0)
    return transform_functional.normalize(tensor, NORMALIZATION_MEAN, NORMALIZATION_STD)


def main() -> int:
    args = parse_args()
    if args.board_batch_size <= 0 or args.square_batch_size <= 0 or args.progress_every <= 0:
        raise ValueError("Batch sizes and progress-every must be positive")
    if args.skip_boards < 0:
        raise ValueError("skip-boards must be non-negative")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != DINO_ARCHITECTURE:
        raise ValueError("Cached evaluator requires a joint DINO checkpoint")
    device = resolve_device(args.device)
    model = dino_joint_model_from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device).eval()
    input_size = int(checkpoint["input_size"])
    class_names = list(checkpoint.get("class_names", CLASS_NAMES))
    clusterer = PieceClusterer(float(checkpoint["similarity_threshold"]))
    sweep_thresholds = list(dict.fromkeys(args.similarity_thresholds or []))
    sweep_clusterers = {
        threshold: PieceClusterer(threshold) for threshold in sweep_thresholds
    }
    sweep_metrics = {
        threshold: {
            "square_correct": 0,
            "occupied_correct": 0,
            "exact_boards": 0,
            "cluster_count": 0,
        }
        for threshold in sweep_thresholds
    }
    assigner = GroupLabelAssigner(args.duplicate_penalty, class_names)
    empty_id = CLASS_NAME_TO_ID["empty"]
    paths = manifest_paths(
        args.manifest,
        args.data_root,
        args.split,
        args.max_boards,
        args.skip_boards,
    )

    # The key hashes the exact RGB bytes after whole-board resize and split.
    # Therefore cache hits skip only model calls whose input tensors are equal.
    output_cache: dict[bytes, tuple[torch.Tensor, torch.Tensor]] = {}
    square_count = occupied_count = 0
    baseline_correct = grouped_correct = 0
    baseline_occupied_correct = grouped_occupied_correct = 0
    baseline_exact_boards = grouped_exact_boards = 0
    false_merge_pairs = predicted_merged_pairs = 0
    false_split_pairs = true_same_pairs = 0
    cross_background_false_merge_pairs = cross_background_predicted_merged_pairs = 0
    cross_background_false_split_pairs = cross_background_true_same_pairs = 0
    cache_requests = cache_misses = 0
    started = time.perf_counter()

    with torch.no_grad():
        for batch_start in range(0, len(paths), args.board_batch_size):
            batch_paths = paths[batch_start : batch_start + args.board_batch_size]
            board_keys: list[list[bytes]] = []
            board_labels: list[torch.Tensor] = []
            missing: dict[bytes, torch.Tensor] = {}
            for path in batch_paths:
                with Image.open(path) as image:
                    board = image.convert("RGB").resize(
                        (8 * input_size, 8 * input_size), Image.Resampling.BICUBIC
                    )
                keys: list[bytes] = []
                for row in range(8):
                    for column in range(8):
                        square = board.crop(
                            (
                                column * input_size,
                                row * input_size,
                                (column + 1) * input_size,
                                (row + 1) * input_size,
                            )
                        )
                        key = hashlib.blake2b(square.tobytes(), digest_size=16).digest()
                        keys.append(key)
                        cache_requests += 1
                        if key not in output_cache and key not in missing:
                            missing[key] = square_tensor(square)
                            cache_misses += 1
                board_keys.append(keys)
                board_labels.append(
                    torch.tensor(
                        board_fen_to_class_ids(fen_from_kaggle_filename(path)),
                        dtype=torch.long,
                    )
                )

            if missing:
                missing_keys = list(missing)
                inputs = torch.stack([missing[key] for key in missing_keys])
                outputs = [
                    model.classify_and_encode(chunk.to(device))
                    for chunk in inputs.split(args.square_batch_size)
                ]
                logits = torch.cat([output[0].cpu() for output in outputs])
                embeddings = torch.cat([output[1].cpu() for output in outputs])
                output_cache.update(
                    {
                        key: (logits[index], embeddings[index])
                        for index, key in enumerate(missing_keys)
                    }
                )

            for keys, labels in zip(board_keys, board_labels, strict=True):
                logits = torch.stack([output_cache[key][0] for key in keys])
                embeddings = torch.stack([output_cache[key][1] for key in keys])
                baseline = logits.argmax(dim=1)
                square_indices = list(range(64))
                clustering = clusterer.cluster(embeddings, square_indices)
                assignments = assigner.assign(logits, clustering.clusters)
                grouped = torch.tensor(
                    assigner.apply(
                        [int(value) for value in baseline], clustering.clusters, assignments
                    )
                )
                occupied = labels != empty_id
                baseline_hits = baseline == labels
                grouped_hits = grouped == labels
                square_count += 64
                occupied_count += int(occupied.sum())
                baseline_correct += int(baseline_hits.sum())
                grouped_correct += int(grouped_hits.sum())
                baseline_occupied_correct += int((baseline_hits & occupied).sum())
                grouped_occupied_correct += int((grouped_hits & occupied).sum())
                baseline_exact_boards += int(bool(baseline_hits.all()))
                grouped_exact_boards += int(bool(grouped_hits.all()))
                for threshold, sweep_clusterer in sweep_clusterers.items():
                    sweep_clustering = sweep_clusterer.cluster(
                        embeddings, square_indices
                    )
                    sweep_assignments = assigner.assign(
                        logits, sweep_clustering.clusters
                    )
                    sweep_grouped = torch.tensor(
                        assigner.apply(
                            [int(value) for value in baseline],
                            sweep_clustering.clusters,
                            sweep_assignments,
                        )
                    )
                    sweep_hits = sweep_grouped == labels
                    metrics = sweep_metrics[threshold]
                    metrics["square_correct"] += int(sweep_hits.sum())
                    metrics["occupied_correct"] += int(
                        (sweep_hits & occupied).sum()
                    )
                    metrics["exact_boards"] += int(bool(sweep_hits.all()))
                    metrics["cluster_count"] += len(sweep_clustering.clusters)
                group_by_square = {
                    square: cluster.group_id
                    for cluster in clustering.clusters
                    for square in cluster.square_indices
                }
                for first, second in combinations(square_indices, 2):
                    predicted_same = group_by_square[first] == group_by_square[second]
                    true_same = int(labels[first]) == int(labels[second])
                    cross_background = (
                        (first // 8 + first % 8) % 2
                        != (second // 8 + second % 8) % 2
                    )
                    if predicted_same:
                        predicted_merged_pairs += 1
                        false_merge_pairs += int(not true_same)
                        if cross_background:
                            cross_background_predicted_merged_pairs += 1
                            cross_background_false_merge_pairs += int(not true_same)
                    if true_same:
                        true_same_pairs += 1
                        false_split_pairs += int(not predicted_same)
                        if cross_background:
                            cross_background_true_same_pairs += 1
                            cross_background_false_split_pairs += int(not predicted_same)

            completed = min(batch_start + len(batch_paths), len(paths))
            if completed == len(paths) or completed % args.progress_every < len(batch_paths):
                elapsed = time.perf_counter() - started
                print(
                    f"evaluate: {completed:,}/{len(paths):,} boards "
                    f"({completed / elapsed:.2f} boards/s), "
                    f"cache hit {1.0 - cache_misses / cache_requests:.1%}",
                    flush=True,
                )

    elapsed = time.perf_counter() - started
    board_count = len(paths)
    payload = {
        "checkpoint": str(args.checkpoint),
        "evaluation_source": f"{args.manifest}:{args.split}",
        "skip_boards": args.skip_boards,
        "board_count": board_count,
        "elapsed_seconds": elapsed,
        "preprocessing": "whole board resized to 8 * model input, then split",
        "exact_identical_input_cache": {
            "requests": cache_requests,
            "unique_inputs": cache_misses,
            "hit_rate": 1.0 - cache_misses / cache_requests,
        },
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
            "cross_background_false_merge_rate": safe_ratio(
                cross_background_false_merge_pairs,
                cross_background_predicted_merged_pairs,
            ),
            "cross_background_false_split_rate": safe_ratio(
                cross_background_false_split_pairs,
                cross_background_true_same_pairs,
            ),
            "cross_background_predicted_merged_pairs": (
                cross_background_predicted_merged_pairs
            ),
            "cross_background_true_same_pairs": cross_background_true_same_pairs,
        },
        "threshold_sweep": [
            {
                "similarity_threshold": threshold,
                "square_accuracy": safe_ratio(
                    sweep_metrics[threshold]["square_correct"], square_count
                ),
                "occupied_accuracy": safe_ratio(
                    sweep_metrics[threshold]["occupied_correct"], occupied_count
                ),
                "exact_board_accuracy": safe_ratio(
                    sweep_metrics[threshold]["exact_boards"], board_count
                ),
                "mean_group_count": safe_ratio(
                    sweep_metrics[threshold]["cluster_count"], board_count
                ),
            }
            for threshold in sweep_thresholds
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
