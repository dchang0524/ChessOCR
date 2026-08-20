"""Export a trained square classifier for ONNX Runtime Web.

Example:
    python scripts/export_onnx.py \
        --checkpoint models/square_classifier_2d.pt \
        --output web/model/square_classifier.onnx
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import onnx  # noqa: E402
import torch  # noqa: E402

from chess_ocr.data.labels import CLASS_NAMES, CLASS_TO_FEN  # noqa: E402
from chess_ocr.models.square_classifier import square_classifier_from_checkpoint  # noqa: E402


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Export a Chess OCR checkpoint to ONNX.")
    parser.add_argument(
        "--checkpoint", type=Path, default=Path("models/square_classifier_kaggle.pt")
    )
    parser.add_argument("--output", type=Path, default=Path("web/model/square_classifier.onnx"))
    parser.add_argument(
        "--metadata",
        type=Path,
        default=Path("web/model/model.json"),
        help="JSON metadata consumed by the browser application",
    )
    parser.add_argument("--opset", type=int, default=18)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    """Load a PyTorch checkpoint and export a validated ONNX graph."""
    args = parse_args(argv)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if "model_state_dict" not in checkpoint:
        raise KeyError(f"Checkpoint {args.checkpoint} has no model_state_dict")

    class_names = list(checkpoint.get("class_names", CLASS_NAMES))
    input_size = int(checkpoint.get("input_size", 64))
    model = square_classifier_from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample = torch.zeros(1, 3, input_size, input_size, dtype=torch.float32)
    torch.onnx.export(
        model,
        sample,
        args.output,
        export_params=True,
        opset_version=args.opset,
        do_constant_folding=True,
        input_names=["squares"],
        output_names=["logits"],
        dynamic_axes={"squares": {0: "batch"}, "logits": {0: "batch"}},
        external_data=False,
    )

    graph = onnx.load(args.output)
    onnx.checker.check_model(graph)
    actual_opset = max(imported.version for imported in graph.opset_import)
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    metadata = {
        "format": "onnx",
        "opset": actual_opset,
        "source_checkpoint": args.checkpoint.name,
        "source_epoch": checkpoint.get("epoch"),
        "source_validation_accuracy": checkpoint.get("validation_accuracy"),
        "model_path": args.output.name,
        "model_sha256": digest,
        "model_bytes": args.output.stat().st_size,
        "input_name": "squares",
        "output_name": "logits",
        "input_size": input_size,
        "architecture": model.architecture,
        "class_names": class_names,
        "fen_symbols": [CLASS_TO_FEN[name] for name in class_names],
        "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        "background_normalization": "disabled",
        "background_variation": "generator-only, before piece compositing",
    }
    args.metadata.parent.mkdir(parents=True, exist_ok=True)
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")

    print(f"Exported {args.output} ({args.output.stat().st_size:,} bytes)")
    print(f"Validated ONNX opset {actual_opset}; SHA-256 {digest}")
    print(f"Wrote browser metadata to {args.metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
