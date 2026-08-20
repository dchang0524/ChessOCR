"""Export and quantize the joint DINO classifier/embedding model for the browser."""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import onnx  # noqa: E402
import torch  # noqa: E402
from torch import nn  # noqa: E402

from chess_ocr.data.labels import CLASS_NAMES, CLASS_TO_FEN  # noqa: E402
from chess_ocr.models.dino_joint_classifier import (  # noqa: E402
    DINO_ARCHITECTURE,
    dino_joint_model_from_checkpoint,
)

MAX_STATIC_ASSET_BYTES = 25 * 1024 * 1024


class JointDinoExport(nn.Module):
    """Expose classification logits and grouping embeddings in one ONNX graph."""

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model

    def forward(self, squares: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return logits and embeddings while evaluating DINO only once."""
        return self.model.classify_and_encode(squares)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=Path("models/joint_dinov2_vits14_head.pt"),
    )
    parser.add_argument(
        "--output", type=Path, default=Path("web/model/joint_dinov2_vits14_int8.onnx")
    )
    parser.add_argument("--metadata", type=Path, default=Path("web/model/model.json"))
    parser.add_argument("--opset", type=int, default=18)
    parser.add_argument("--inference-batch-size", type=int, default=4)
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=None,
        help="Quantized-embedding cutoff; defaults to the float checkpoint cutoff",
    )
    return parser.parse_args(argv)


def _quantize_dynamic(source: Path, destination: Path) -> None:
    try:
        from onnxruntime.quantization import QuantType, quantize_dynamic
    except ImportError as error:
        raise RuntimeError(
            "Browser-sized DINO export requires onnxruntime; install the dev dependencies"
        ) from error
    quantize_dynamic(
        source,
        destination,
        weight_type=QuantType.QInt8,
        # ONNX Runtime Web's WASM backend does not implement ConvInteger.
        # Keeping DINO's one patch-embedding convolution in float32 costs less
        # than 1 MiB after specializing the positional grid below.
        op_types_to_quantize=["MatMul", "Gemm"],
    )


def _specialize_position_embedding(model: nn.Module, input_size: int) -> None:
    """Replace DINO's large pretraining grid with its exact fixed-input interpolation."""
    backbone = model.features
    patch_count = (input_size // int(backbone.patch_size)) ** 2
    dummy_tokens = torch.empty(1, patch_count + 1, model.feature_size)
    with torch.no_grad():
        position_embedding = backbone.interpolate_pos_encoding(
            dummy_tokens, input_size, input_size
        )
    backbone.pos_embed = nn.Parameter(position_embedding, requires_grad=False)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.inference_batch_size <= 0:
        raise ValueError("--inference-batch-size must be positive")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("architecture") != DINO_ARCHITECTURE:
        raise ValueError("Checkpoint is not a joint DINOv2 model")
    model = dino_joint_model_from_checkpoint(checkpoint)
    model.load_state_dict(checkpoint["model_state_dict"])
    _specialize_position_embedding(model, int(checkpoint["input_size"]))
    wrapper = JointDinoExport(model.eval()).eval()
    input_size = int(checkpoint["input_size"])
    embedding_size = int(checkpoint["embedding_size"])
    sample = torch.zeros(1, 3, input_size, input_size, dtype=torch.float32)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="chessocr-dino-export-") as temp_directory:
        float_model = Path(temp_directory) / "joint_dinov2_vits14_fp32.onnx"
        torch.onnx.export(
            wrapper,
            sample,
            float_model,
            export_params=True,
            opset_version=args.opset,
            do_constant_folding=True,
            input_names=["squares"],
            output_names=["logits", "embeddings"],
            dynamic_axes={
                "squares": {0: "batch"},
                "logits": {0: "batch"},
                "embeddings": {0: "batch"},
            },
            external_data=False,
            dynamo=False,
        )
        _quantize_dynamic(float_model, args.output)

    graph = onnx.load(args.output)
    onnx.checker.check_model(graph)
    model_bytes = args.output.stat().st_size
    if model_bytes > MAX_STATIC_ASSET_BYTES:
        raise ValueError(
            f"Quantized model is {model_bytes:,} bytes, above Cloudflare's "
            f"{MAX_STATIC_ASSET_BYTES:,}-byte static-asset limit"
        )
    class_names = list(checkpoint.get("class_names", CLASS_NAMES))
    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    actual_opset = max(imported.version for imported in graph.opset_import)
    metadata = {
        "format": "onnx",
        "opset": actual_opset,
        "architecture": DINO_ARCHITECTURE,
        "joint_model": True,
        "quantization": "dynamic-int8-weights",
        "source_checkpoint": args.checkpoint.name,
        "source_epoch": checkpoint.get("epoch"),
        "source_validation_accuracy": checkpoint.get("validation_accuracy"),
        "model_path": args.output.name,
        "model_sha256": digest,
        "model_bytes": model_bytes,
        "input_name": "squares",
        "output_name": "logits",
        "input_size": input_size,
        "inference_batch_size": args.inference_batch_size,
        "class_names": class_names,
        "fen_symbols": [CLASS_TO_FEN[name] for name in class_names],
        "normalization": {"mean": [0.5, 0.5, 0.5], "std": [0.5, 0.5, 0.5]},
        "background_normalization": "disabled",
        "background_variation": "generator-only, before piece compositing",
        "background_invariance": "paired cross-background consistency loss",
        "similarity": {
            "model_path": args.output.name,
            "model_sha256": digest,
            "model_bytes": model_bytes,
            "input_name": "squares",
            "output_name": "embeddings",
            "embedding_size": embedding_size,
            "architecture": DINO_ARCHITECTURE,
            "input_size": input_size,
            "similarity_threshold": (
                float(checkpoint["similarity_threshold"])
                if args.similarity_threshold is None
                else args.similarity_threshold
            ),
            "cross_background_similarity_threshold": checkpoint.get(
                "cross_background_similarity_threshold"
            ),
            "duplicate_penalty": 1.5,
            "includes_empty_class": True,
            "source_checkpoint": args.checkpoint.name,
            "source_epoch": checkpoint.get("epoch"),
            "source_validation_accuracy": checkpoint.get("validation_pair_accuracy"),
        },
    }
    args.metadata.write_text(json.dumps(metadata, indent=2) + "\n", encoding="utf-8")
    print(f"Exported joint INT8 model: {args.output} ({model_bytes:,} bytes)")
    print(f"Validated ONNX opset {actual_opset}; SHA-256 {digest}")
    print(f"Wrote joint browser metadata: {args.metadata}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
