"""Export false-merge examples from Kaggle similarity clustering."""

from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import torch  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

from chess_ocr.data.dataset_generator import SQUARE_NAMES  # noqa: E402
from chess_ocr.data.kaggle_board_dataset import KaggleBoardDataset  # noqa: E402
from chess_ocr.data.labels import CLASS_NAMES  # noqa: E402
from chess_ocr.inference.board_predictor import resolve_device  # noqa: E402
from chess_ocr.inference.piece_clusterer import PieceClusterer  # noqa: E402
from chess_ocr.models.similarity_classifier import (  # noqa: E402
    SimilarityClassifier,
    similarity_model_from_checkpoint,
)


@dataclass(frozen=True)
class FalseMerge:
    """One differently labelled square pair placed in the same cluster."""

    board_path: Path
    board_offset: int
    first: int
    second: int
    first_label: int
    second_label: int
    similarity: float
    group_id: int
    cluster_members: tuple[int, ...]
    cluster_labels: tuple[int, ...]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--similarity-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--image-dir", type=Path, default=Path("data/raw/kaggle_chess_positions/test")
    )
    parser.add_argument("--skip-boards", type=int, default=0)
    parser.add_argument("--max-boards", type=int, default=None)
    parser.add_argument("--board-batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--device", choices=["cpu", "cuda", "mps"], default=None)
    parser.add_argument("--similarity-threshold", type=float, default=None)
    parser.add_argument("--montage-examples", type=int, default=16)
    parser.add_argument(
        "--output-dir", type=Path, default=Path("outputs/kaggle_false_merges")
    )
    return parser.parse_args()


def collect_false_merges(
    dataset: KaggleBoardDataset,
    model: SimilarityClassifier,
    threshold: float,
    device: torch.device,
    board_batch_size: int,
    num_workers: int,
) -> list[FalseMerge]:
    """Run clustering and return every differently labelled merged pair."""
    loader = DataLoader(
        dataset,
        batch_size=board_batch_size,
        shuffle=False,
        num_workers=num_workers,
    )
    clusterer = PieceClusterer(threshold)
    failures: list[FalseMerge] = []
    board_offset = 0
    with torch.no_grad():
        for squares, labels, _ in loader:
            batch_size = squares.shape[0]
            embeddings = model.encode(squares.flatten(0, 1).to(device))
            embeddings = embeddings.reshape(batch_size, 64, -1).cpu()
            labels = labels.cpu()
            for local_index in range(batch_size):
                board_labels = labels[local_index]
                clustering = clusterer.cluster(embeddings[local_index], list(range(64)))
                for cluster in clustering.clusters:
                    members = cluster.square_indices
                    cluster_labels = tuple(int(board_labels[index]) for index in members)
                    for first, second in combinations(members, 2):
                        first_label = int(board_labels[first])
                        second_label = int(board_labels[second])
                        if first_label == second_label:
                            continue
                        failures.append(
                            FalseMerge(
                                board_path=dataset.paths[board_offset + local_index],
                                board_offset=board_offset + local_index,
                                first=first,
                                second=second,
                                first_label=first_label,
                                second_label=second_label,
                                similarity=float(clustering.similarity_matrix[first, second]),
                                group_id=cluster.group_id,
                                cluster_members=members,
                                cluster_labels=cluster_labels,
                            )
                        )
            board_offset += batch_size
    return failures


def square_crop(board: Image.Image, index: int, size: int = 160) -> Image.Image:
    """Crop one indexed square from a board and enlarge it without smoothing."""
    board = board.resize((512, 512), Image.Resampling.BICUBIC)
    row, column = divmod(index, 8)
    crop = board.crop((column * 64, row * 64, (column + 1) * 64, (row + 1) * 64))
    return crop.resize((size, size), Image.Resampling.NEAREST)


def board_preview(board: Image.Image, first: int, second: int, size: int = 256) -> Image.Image:
    """Resize a board and outline the false-merged pair."""
    preview = board.convert("RGB").resize((size, size), Image.Resampling.BICUBIC)
    draw = ImageDraw.Draw(preview)
    square_size = size // 8
    for index, colour in ((first, "#ff2d55"), (second, "#00a7ff")):
        row, column = divmod(index, 8)
        box = (
            column * square_size + 1,
            row * square_size + 1,
            (column + 1) * square_size - 2,
            (row + 1) * square_size - 2,
        )
        draw.rectangle(box, outline=colour, width=4)
    return preview


def choose_montage_examples(
    failures: list[FalseMerge], maximum: int
) -> list[FalseMerge]:
    """Choose high-similarity examples without repeating a predicted cluster."""
    chosen: list[FalseMerge] = []
    seen_clusters: set[tuple[Path, int]] = set()
    for failure in sorted(failures, key=lambda item: item.similarity, reverse=True):
        key = (failure.board_path, failure.group_id)
        if key in seen_clusters:
            continue
        seen_clusters.add(key)
        chosen.append(failure)
        if len(chosen) >= maximum:
            break
    return chosen


def build_montage(failures: list[FalseMerge]) -> Image.Image:
    """Build a two-column card montage for selected false merges."""
    card_width = 640
    card_height = 338
    columns = 2
    rows = (len(failures) + columns - 1) // columns
    montage = Image.new("RGB", (card_width * columns, card_height * rows), "white")
    for example_index, failure in enumerate(failures):
        with Image.open(failure.board_path) as source:
            board = source.convert("RGB")
        card = Image.new("RGB", (card_width, card_height), "#f5f5f5")
        draw = ImageDraw.Draw(card)
        card.paste(board_preview(board, failure.first, failure.second), (12, 58))
        card.paste(square_crop(board, failure.first), (280, 58))
        card.paste(square_crop(board, failure.second), (452, 58))
        first_name = CLASS_NAMES[failure.first_label]
        second_name = CLASS_NAMES[failure.second_label]
        draw.text((12, 10), f"#{example_index + 1}  similarity={failure.similarity:.5f}", fill="black")
        draw.text((12, 30), failure.board_path.name[:76], fill="#333333")
        draw.text((280, 224), f"RED  {SQUARE_NAMES[failure.first]}", fill="#d3133a")
        draw.text((280, 244), first_name, fill="black")
        draw.text((452, 224), f"BLUE  {SQUARE_NAMES[failure.second]}", fill="#0079bd")
        draw.text((452, 244), second_name, fill="black")
        member_text = ", ".join(
            f"{SQUARE_NAMES[index]}:{CLASS_NAMES[label]}"
            for index, label in zip(
                failure.cluster_members, failure.cluster_labels, strict=True
            )
        )
        draw.text((280, 274), "Predicted cluster:", fill="#333333")
        draw.text((280, 294), member_text[:76], fill="black")
        if len(member_text) > 76:
            draw.text((280, 312), member_text[76:152], fill="black")
        montage.paste(card, ((example_index % columns) * card_width, (example_index // columns) * card_height))
    return montage


def write_csv(failures: list[FalseMerge], path: Path) -> None:
    """Write all false-merged pairs to a CSV."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "board_path",
                "board_offset",
                "first_square",
                "first_label",
                "second_square",
                "second_label",
                "cosine_similarity",
                "group_id",
                "cluster_members",
                "cluster_labels",
            ),
        )
        writer.writeheader()
        for failure in failures:
            writer.writerow(
                {
                    "board_path": failure.board_path,
                    "board_offset": failure.board_offset,
                    "first_square": SQUARE_NAMES[failure.first],
                    "first_label": CLASS_NAMES[failure.first_label],
                    "second_square": SQUARE_NAMES[failure.second],
                    "second_label": CLASS_NAMES[failure.second_label],
                    "cosine_similarity": failure.similarity,
                    "group_id": failure.group_id,
                    "cluster_members": " ".join(
                        SQUARE_NAMES[index] for index in failure.cluster_members
                    ),
                    "cluster_labels": " ".join(
                        CLASS_NAMES[label] for label in failure.cluster_labels
                    ),
                }
            )


def main() -> int:
    args = parse_args()
    if args.skip_boards < 0:
        raise ValueError("--skip-boards must be non-negative")
    if args.montage_examples <= 0:
        raise ValueError("--montage-examples must be positive")
    device = resolve_device(args.device)
    checkpoint = torch.load(
        args.similarity_checkpoint, map_location=device, weights_only=False
    )
    model = similarity_model_from_checkpoint(checkpoint).to(device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    loaded_maximum = (
        args.skip_boards + args.max_boards if args.max_boards is not None else None
    )
    dataset = KaggleBoardDataset(
        image_dir=args.image_dir,
        max_boards=loaded_maximum,
        input_size=int(checkpoint.get("input_size", 64)),
    )
    if args.skip_boards:
        dataset.paths = dataset.paths[args.skip_boards :]
        dataset.board_fens = dataset.board_fens[args.skip_boards :]
    threshold = (
        float(checkpoint["similarity_threshold"])
        if args.similarity_threshold is None
        else args.similarity_threshold
    )
    failures = collect_false_merges(
        dataset,
        model,
        threshold,
        device,
        args.board_batch_size,
        args.num_workers,
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "false_merges.csv"
    montage_path = args.output_dir / "montage.png"
    write_csv(failures, csv_path)
    selected = choose_montage_examples(failures, args.montage_examples)
    if selected:
        build_montage(selected).save(montage_path)
    print(
        f"Found {len(failures)} false-merged pairs across {len(dataset)} boards "
        f"at threshold {threshold:.4f}"
    )
    print(f"CSV: {csv_path}")
    if selected:
        print(f"Montage: {montage_path} ({len(selected)} distinct clusters)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
