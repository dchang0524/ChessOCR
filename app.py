"""Streamlit UI for Chess OCR.

Flow: upload a screenshot, crop the board by hand, choose the orientation, and
run recognition once on demand. This module owns the interactive cropper and
nothing else — normalisation, splitting, classification and FEN construction all
live in :mod:`chess_ocr`.
"""

from __future__ import annotations

import io
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import pandas as pd
import streamlit as st
import streamlit.components.v1 as components
from PIL import Image, UnidentifiedImageError
from streamlit_cropper import st_cropper

from chess_ocr.chess.board_renderer import BoardRenderer
from chess_ocr.chess.position_validator import PositionValidator
from chess_ocr.inference.board_predictor import BoardPredictor
from chess_ocr.inference.prediction_result import BoardPrediction

DEFAULT_CHECKPOINT = "models/square_classifier_generated.pt"
DEFAULT_SIMILARITY_CHECKPOINT = "models/similarity_generated.pt"
SUPPORTED_UPLOAD_TYPES = ["png", "jpg", "jpeg"]
ASSUMED_FIELD_NOTE = (
    "The classifier only reads piece placement. Castling rights, en passant "
    "target, halfmove clock and fullmove number are **assumed**, not detected."
)


@st.cache_resource(show_spinner="Loading model checkpoint...")
def load_predictor(
    checkpoint_path: str, similarity_checkpoint_path: str, device: str | None
) -> BoardPredictor:
    """Load and cache the predictor for a checkpoint.

    Args:
        checkpoint_path: Path to a checkpoint written by the trainer.
        similarity_checkpoint_path: Path to the Siamese similarity checkpoint.
        device: Device string, or ``None`` to auto-detect.

    Returns:
        A cached :class:`BoardPredictor`.
    """
    return BoardPredictor.from_checkpoints(
        checkpoint_path, similarity_checkpoint_path, device=device
    )


def render_svg(svg: str, height: int = 420) -> None:
    """Embed an SVG string in the page."""
    components.html(f'<div style="display:flex;justify-content:center">{svg}</div>', height=height)


def show_results(
    prediction: BoardPrediction,
    crop: Image.Image,
    white_at_bottom: bool,
    predictor: BoardPredictor,
) -> None:
    """Render every output panel for a completed prediction.

    Args:
        prediction: The board prediction to display.
        crop: The cropped board image that was recognised.
        white_at_bottom: Orientation used for the reconstructed board.
        predictor: Predictor used to apply and propagate group corrections.
    """
    left, right = st.columns(2)
    with left:
        st.subheader("Original crop")
        st.image(crop, use_container_width=True)
    with right:
        st.subheader("Predicted board")
        try:
            svg = BoardRenderer(size=400).render(
                prediction.board_fen,
                white_at_bottom=white_at_bottom,
                highlight_squares=prediction.low_confidence_squares,
            )
            render_svg(svg)
            st.caption("Highlighted squares are below the confidence threshold.")
        except ValueError as error:
            st.error(f"Could not render the predicted position: {error}")

    st.subheader("FEN")
    st.markdown("**Detected board FEN** (piece placement only)")
    st.code(prediction.board_fen, language="text")
    if prediction.full_fen is not None:
        st.markdown("**Assumed complete FEN**")
        st.code(prediction.full_fen, language="text")
        st.caption(ASSUMED_FIELD_NOTE)
    else:
        st.info("Select a side to move to also build a complete FEN.")

    st.subheader("Confidence")
    metric_columns = st.columns(3)
    metric_columns[0].metric("Mean confidence", f"{prediction.mean_confidence:.3f}")
    metric_columns[1].metric("Minimum confidence", f"{prediction.minimum_confidence:.3f}")
    metric_columns[2].metric("Low-confidence squares", str(len(prediction.low_confidence_squares)))
    if prediction.low_confidence_squares:
        st.warning(
            "Least confident squares (worst first): "
            + ", ".join(prediction.low_confidence_squares[:12])
        )
    else:
        st.success("Every square was predicted above the confidence threshold.")

    st.subheader("Position validation")
    validation = PositionValidator().validate(prediction.board_fen)
    if not validation.is_parseable:
        st.error(validation.warnings[0])
    elif validation.warnings:
        for warning in validation.warnings:
            st.warning(warning)
    else:
        st.success("No structural problems found in the predicted position.")

    with st.expander("Per-square predictions", expanded=False):
        frame = pd.DataFrame(prediction.to_rows())
        st.dataframe(frame, use_container_width=True, hide_index=True)
        st.download_button(
            "Download per-square CSV",
            frame.to_csv(index=False).encode("utf-8"),
            file_name="square_predictions.csv",
            mime="text/csv",
        )

    if prediction.groups:
        st.subheader("Correct an appearance group")
        group_by_label = {
            f"Group {group.group_id}: {', '.join(group.squares)} → {group.class_name}": group
            for group in prediction.groups
        }
        selected_label = st.selectbox("Appearance group", list(group_by_label))
        selected_group = group_by_label[selected_label]
        class_names = predictor.class_names
        selected_class_name = st.selectbox(
            "Correct class",
            class_names,
            index=(
                class_names.index(selected_group.class_name)
                if selected_group.class_name in class_names
                else 0
            ),
        )
        if st.button("Apply correction to whole group"):
            class_id = predictor.class_names.index(selected_class_name)
            predictor.apply_group_correction(prediction, selected_group.group_id, class_id)
            st.session_state["prediction"] = prediction
            st.rerun()


def main() -> None:
    """Run the Streamlit application."""
    st.set_page_config(page_title="Chess OCR", page_icon="♟", layout="wide")
    st.title("Chess OCR — board recognition from a screenshot")
    st.caption(
        "MVP scope: clean, axis-aligned digital chessboard screenshots that you crop by hand."
    )

    with st.sidebar:
        st.header("Settings")
        checkpoint_path = st.text_input("Checkpoint path", value=DEFAULT_CHECKPOINT)
        similarity_checkpoint_path = st.text_input(
            "Similarity checkpoint path", value=DEFAULT_SIMILARITY_CHECKPOINT
        )
        device_choice = st.selectbox("Device", ["auto", "cpu", "cuda", "mps"], index=0)
        threshold = st.slider(
            "Low-confidence threshold", min_value=0.0, max_value=1.0, value=0.80, step=0.01
        )
        st.markdown("---")
        st.markdown(
            "Train a checkpoint first:\n\n"
            "```bash\n"
            "python scripts/generate_dataset.py --positions 400\n"
            "python scripts/train_model.py "
            "--metadata data/processed/synthetic_v1/metadata.csv\n"
            "```"
        )

    uploaded = st.file_uploader("Upload a chessboard screenshot", type=SUPPORTED_UPLOAD_TYPES)
    if uploaded is None:
        st.info("Upload a PNG, JPG or JPEG screenshot to begin.")
        return

    try:
        image = Image.open(io.BytesIO(uploaded.getvalue())).convert("RGB")
    except (UnidentifiedImageError, OSError) as error:
        st.error(f"Could not read that image: {error}")
        return

    st.subheader("1. Crop the board")
    st.caption("Drag the box so it hugs the outer edge of the 8x8 grid.")
    crop_column, preview_column = st.columns([3, 2])
    with crop_column:
        crop = st_cropper(
            image,
            realtime_update=True,
            box_color="#e6a23c",
            aspect_ratio=(1, 1),
            should_resize_image=True,
        )
    with preview_column:
        st.markdown("**Crop preview**")
        st.image(crop, use_container_width=True)
        st.caption(f"Crop size: {crop.size[0]} x {crop.size[1]} px")

    st.subheader("2. Orientation and side to move")
    option_columns = st.columns(2)
    with option_columns[0]:
        orientation = st.radio(
            "Board orientation", ["White at bottom", "Black at bottom"], horizontal=True
        )
    with option_columns[1]:
        side_label = st.radio(
            "Side to move (assumed)",
            ["Not specified", "White", "Black"],
            horizontal=True,
            help="Used only to build the complete FEN; it is never detected from the image.",
        )
    white_at_bottom = orientation == "White at bottom"
    side_to_move = {"Not specified": None, "White": "w", "Black": "b"}[side_label]

    st.subheader("3. Recognize")
    if st.button("Recognize Position", type="primary"):
        try:
            predictor = load_predictor(
                checkpoint_path,
                similarity_checkpoint_path,
                None if device_choice == "auto" else device_choice,
            )
        except FileNotFoundError:
            st.error(
                f"No checkpoint at `{checkpoint_path}`. Train one first, or point the "
                "sidebar at an existing checkpoint."
            )
            return
        except (KeyError, RuntimeError) as error:
            st.error(f"Could not load the checkpoint: {error}")
            return

        predictor.low_confidence_threshold = threshold
        try:
            with st.spinner("Classifying 64 squares..."):
                prediction = predictor.predict(
                    crop, white_at_bottom=white_at_bottom, side_to_move=side_to_move
                )
        except (ValueError, TypeError, RuntimeError) as error:
            st.error(f"Recognition failed: {error}")
            return

        # Cache the result so moving the crop box does not re-run the model.
        st.session_state["prediction"] = prediction
        st.session_state["crop"] = crop
        st.session_state["white_at_bottom"] = white_at_bottom
        st.session_state["predictor"] = predictor

    if "prediction" in st.session_state:
        st.markdown("---")
        show_results(
            st.session_state["prediction"],
            st.session_state["crop"],
            st.session_state["white_at_bottom"],
            st.session_state["predictor"],
        )


if __name__ == "__main__":
    main()
