"""Export the raw-RGB similarity encoder for browser inference."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import onnx  # noqa: E402
import torch  # noqa: E402

from chess_ocr.models.similarity_classifier import (  # noqa: E402
    SimilarityEncoderExport,
    similarity_model_from_checkpoint,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the similarity encoder to ONNX.")
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("models/similarity_generated.pt")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("web/model/similarity_encoder.onnx")
    )
    parser.add_argument("--metadata", type=Path, default=Path("web/model/model.json"))
    parser.add_argument("--similarity-threshold", type=float, default=None)
    parser.add_argument("--cross-background-similarity-threshold", type=float, default=None)
    parser.add_argument("--opset", type=int, default=18)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = similarity_model_from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    wrapper = SimilarityEncoderExport(model.eval()).eval()
    input_size = int(checkpoint.get("input_size", 64))
    sample = torch.zeros(1, 3, input_size, input_size)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.onnx.export(
        wrapper,
        sample,
        args.output,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["squares"],
        output_names=["embeddings"],
        dynamic_axes={"squares": {0: "batch"}, "embeddings": {0: "batch"}},
        external_data=False,
    )
    graph = onnx.load(args.output)
    onnx.checker.check_model(graph)
    threshold = (
        float(checkpoint["similarity_threshold"])
        if args.similarity_threshold is None
        else args.similarity_threshold
    )
    cross_background_threshold = (
        checkpoint.get("cross_background_similarity_threshold")
        if args.cross_background_similarity_threshold is None
        else args.cross_background_similarity_threshold
    )
    metadata = json.loads(args.metadata.read_text(encoding="utf-8"))
    metadata["similarity"] = {
        "model_path": args.output.name,
        "model_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "model_bytes": args.output.stat().st_size,
        "input_name": "squares",
        "output_name": "embeddings",
        "embedding_size": model.embedding_size,
        "architecture": model.architecture,
        "input_size": input_size,
        "similarity_threshold": threshold,
        "cross_background_similarity_threshold": cross_background_threshold,
        "duplicate_penalty": 1.5,
        "background_normalization": "disabled",
        "background_variation": "generator-only, before piece compositing",
        "includes_empty_class": bool(checkpoint.get("includes_empty_class", False)),
        "source_checkpoint": args.checkpoint.name,
        "source_epoch": checkpoint.get("epoch"),
        "source_validation_accuracy": checkpoint.get("validation_accuracy"),
    }
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Exported {args.output} ({args.output.stat().st_size:,} bytes)")
    print(f"Updated {args.metadata} with similarity threshold {threshold:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
