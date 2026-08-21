"""Compare a joint DINO PyTorch checkpoint with its quantized ONNX export."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import numpy as np  # noqa: E402
import onnxruntime as ort  # noqa: E402
import torch  # noqa: E402

from chess_ocr.data.kaggle_board_dataset import KaggleBoardDataset  # noqa: E402
from chess_ocr.data.labels import CLASS_NAMES  # noqa: E402
from chess_ocr.inference.group_label_assigner import GroupLabelAssigner  # noqa: E402
from chess_ocr.inference.piece_clusterer import PieceClusterer  # noqa: E402
from chess_ocr.models.dino_joint_classifier import (  # noqa: E402
    dino_joint_model_from_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--max-boards", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--similarity-threshold", type=float, default=0.95)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def partition(clusters: object) -> list[list[int]]:
    return sorted(
        (sorted(cluster.square_indices) for cluster in clusters),
        key=lambda members: members[0],
    )


def main() -> int:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = dino_joint_model_from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    session = ort.InferenceSession(str(args.onnx), providers=["CPUExecutionProvider"])
    dataset = KaggleBoardDataset(
        image_dir=args.image_dir,
        input_size=int(checkpoint["input_size"]),
        max_boards=args.max_boards,
    )
    clusterer = PieceClusterer(args.similarity_threshold)
    assigner = GroupLabelAssigner(
        duplicate_penalty=1.5,
        class_names=list(checkpoint.get("class_names", CLASS_NAMES)),
    )
    label_matches = cluster_matches = square_count = 0
    torch_correct = onnx_correct = torch_grouped_correct = onnx_grouped_correct = 0
    torch_exact = onnx_exact = torch_grouped_exact = onnx_grouped_exact = 0
    maximum_logit_error = maximum_embedding_error = 0.0
    with torch.no_grad():
        for squares, labels, _ in dataset:
            torch_outputs = [
                model.classify_and_encode(chunk)
                for chunk in squares.split(args.batch_size)
            ]
            torch_logits = torch.cat([output[0] for output in torch_outputs])
            torch_embeddings = torch.cat([output[1] for output in torch_outputs])
            onnx_outputs = [
                session.run(None, {"squares": chunk.numpy()})
                for chunk in squares.split(args.batch_size)
            ]
            onnx_logits = torch.from_numpy(np.concatenate([output[0] for output in onnx_outputs]))
            onnx_embeddings = torch.from_numpy(
                np.concatenate([output[1] for output in onnx_outputs])
            )
            label_matches += int(
                (torch_logits.argmax(1) == onnx_logits.argmax(1)).sum()
            )
            square_count += len(squares)
            maximum_logit_error = max(
                maximum_logit_error,
                float((torch_logits - onnx_logits).abs().max()),
            )
            maximum_embedding_error = max(
                maximum_embedding_error,
                float((torch_embeddings - onnx_embeddings).abs().max()),
            )
            torch_groups = clusterer.cluster(torch_embeddings, list(range(64))).clusters
            onnx_groups = clusterer.cluster(onnx_embeddings, list(range(64))).clusters
            cluster_matches += int(partition(torch_groups) == partition(onnx_groups))
            torch_labels = torch_logits.argmax(1)
            onnx_labels = onnx_logits.argmax(1)
            torch_grouped = torch.tensor(
                assigner.apply(
                    [int(value) for value in torch_labels],
                    torch_groups,
                    assigner.assign(torch_logits, torch_groups),
                )
            )
            onnx_grouped = torch.tensor(
                assigner.apply(
                    [int(value) for value in onnx_labels],
                    onnx_groups,
                    assigner.assign(onnx_logits, onnx_groups),
                )
            )
            torch_hits = torch_labels == labels
            onnx_hits = onnx_labels == labels
            torch_grouped_hits = torch_grouped == labels
            onnx_grouped_hits = onnx_grouped == labels
            torch_correct += int(torch_hits.sum())
            onnx_correct += int(onnx_hits.sum())
            torch_grouped_correct += int(torch_grouped_hits.sum())
            onnx_grouped_correct += int(onnx_grouped_hits.sum())
            torch_exact += int(bool(torch_hits.all()))
            onnx_exact += int(bool(onnx_hits.all()))
            torch_grouped_exact += int(bool(torch_grouped_hits.all()))
            onnx_grouped_exact += int(bool(onnx_grouped_hits.all()))

    payload = {
        "checkpoint": str(args.checkpoint),
        "onnx": str(args.onnx),
        "image_dir": str(args.image_dir),
        "board_count": len(dataset),
        "square_label_agreement": label_matches / square_count,
        "cluster_partition_agreement": cluster_matches / len(dataset),
        "pytorch_square_accuracy": torch_correct / square_count,
        "onnx_square_accuracy": onnx_correct / square_count,
        "pytorch_grouped_square_accuracy": torch_grouped_correct / square_count,
        "onnx_grouped_square_accuracy": onnx_grouped_correct / square_count,
        "pytorch_exact_board_accuracy": torch_exact / len(dataset),
        "onnx_exact_board_accuracy": onnx_exact / len(dataset),
        "pytorch_grouped_exact_board_accuracy": torch_grouped_exact / len(dataset),
        "onnx_grouped_exact_board_accuracy": onnx_grouped_exact / len(dataset),
        "maximum_absolute_logit_error": maximum_logit_error,
        "maximum_absolute_embedding_error": maximum_embedding_error,
        "similarity_threshold": args.similarity_threshold,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
