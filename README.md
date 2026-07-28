# Chess OCR

Recognise a chess position from a screenshot of a digital chessboard and turn it into FEN.

Upload a screenshot, crop the board by hand, and the app classifies all 64 squares in a single
batch, reconstructs the position, and shows how confident it was about every square.

---

## Current MVP scope

**In scope (Week 1 MVP)**

- Clean, axis-aligned **digital** chessboard screenshots
- **Manual** square cropping by the user
- One or two known board themes
- Board orientation chosen by the user (White at bottom / Black at bottom)
- Per-square classification into 13 classes (empty + 6 white + 6 black pieces)
- Board-FEN output, reconstructed board rendering, and per-square confidence

**Explicitly out of scope for now**

- Automatic board detection or four-corner detection
- Perspective correction / photographs of physical boards
- Arbitrary board themes and piece sets
- Engine analysis (Stockfish), move legality repair, or position search

The architecture keeps these boundaries clean so automatic detection can later replace the manual
crop without touching the classifier or the FEN logic.

### Honest status

The full pipeline — dataset generation, training, evaluation, inference and UI — is implemented
and exercised end to end. **No accuracy numbers are claimed here.** The bundled dataset generator
renders *synthetic* boards, which are useful for wiring and smoke tests but are not a substitute
for screenshots of the themes you actually want to support. See
[Remaining work](#remaining-work-before-real-predictions).

---

## Demo workflow

1. Upload a PNG/JPG/JPEG screenshot.
2. Drag the 1:1 crop box so it hugs the outer edge of the 8×8 grid.
3. Choose the board orientation.
4. Optionally choose the side to move (used only for the assumed complete FEN).
5. Click **Recognize Position**.
6. Read the detected board-FEN, the reconstructed board, and the confidence panels.

Inference runs only when the button is pressed — moving the crop box never triggers the model.

---

## Architecture

```text
        Streamlit UI (app.py)
        upload + interactive 1:1 crop + orientation
                    │  cropped PIL image
                    ▼
        BoardNormalizer          preprocessing/board_normalizer.py
        RGB + resize to 512x512
                    │  normalised square board
                    ▼
        BoardSplitter            preprocessing/board_splitter.py
        arithmetic split into 64 squares, FEN order
                    │  64 PIL squares
                    ▼
        BoardPredictor           inference/board_predictor.py
        transform -> stack -> single batched forward pass -> softmax
                    │  64 x (class id, confidence)
                    ▼
        SquareClassifier         models/square_classifier.py   (raw logits, no softmax)
                    │
                    ▼
        FenBuilder               chess/fen_builder.py          (no PyTorch dependency)
        board-FEN + assumed full FEN
                    │
                    ▼
        BoardRenderer            chess/board_renderer.py       (python-chess SVG)
        PositionValidator        chess/position_validator.py   (warnings only)
```

Supporting packages:

```text
src/chess_ocr/
├── data/          labels (single source of truth), dataset, synthetic generator
├── training/      Trainer (checkpointing) and Evaluator (detailed metrics)
└── inference/     BoardPredictor + prediction dataclasses
```

Rules that keep the boundaries usable:

- the cropper lives in the UI, never in the library;
- the normaliser receives an already-cropped board;
- the splitter assumes a normalised, top-down, square board;
- the network only ever sees square tensors and returns logits;
- the predictor orchestrates inference but contains no Streamlit code;
- `FenBuilder` has no PyTorch dependency;
- class ordering is defined once, in `src/chess_ocr/data/labels.py`.

---

## Installation

Requires Python 3.11+.

```bash
git clone <your-repo-url> chess-ocr
cd chess-ocr
python3 -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

Editable install (optional, puts `chess_ocr` on the path without any `sys.path` juggling):

```bash
pip install -e ".[app,dev]"
```

---

## Generating a dataset

```bash
python scripts/generate_dataset.py --positions 400 --themes synthetic_classic synthetic_green --output-dir data/processed/synthetic_v1
```

This renders each position with every selected theme, cuts each board into 64 squares, and writes
`squares/<label>/*.png` plus `metadata.csv` with columns
`image_path, label, label_id, position_id, square, theme, square_color, split`.

Splits are assigned **per position**, so all squares of a board stay in one split and no board
leaks between train, validation and test.

To use your own piece artwork (one transparent PNG per piece, named `wP.png` ... `bK.png`):

```bash
python scripts/generate_dataset.py --positions 400 --asset-theme my_theme assets/themes/my_theme
```

Only use artwork you are licensed to use — this project never scrapes assets from chess sites.

---

## Training

```bash
python scripts/train_model.py --metadata data/processed/synthetic_v1/metadata.csv --checkpoint models/square_classifier.pt --epochs 12 --batch-size 128
```

Defaults: AdamW (`lr=1e-3`, `weight_decay=1e-4`), `CrossEntropyLoss(label_smoothing=0.05)`,
batch size 128. The device is auto-detected (CUDA → Apple MPS → CPU) and can be forced with
`--device cpu|cuda|mps`.

The checkpoint with the best validation accuracy is saved as a dictionary:

```python
{
    "model_state_dict": ...,
    "class_names": [...],
    "input_size": 64,
    "epoch": ...,
    "validation_accuracy": ...,
}
```

---

## Evaluation

```bash
python scripts/evaluate_model.py --checkpoint models/square_classifier.pt --metadata data/processed/synthetic_v1/metadata.csv --split test
```

Reports overall / empty / occupied square accuracy, per-class precision, recall and one-vs-rest
accuracy, a confusion matrix (PNG + CSV in `outputs/confusion_matrices/`), and — because board
metadata is present — exact-board accuracy, mean incorrect squares per board, and the share of
boards with 0, 1, 2, or 3+ errors. Misclassified squares are written to `outputs/failure_cases/`.

Overall square accuracy alone is misleading: empty squares dominate a typical board, so always
read the occupied-square and board-level numbers next to it.

---

## Running the app

```bash
streamlit run app.py
```

Point the sidebar at your checkpoint (default `models/square_classifier.pt`), adjust the
low-confidence threshold (default `0.80`), then upload and crop.

---

## Tests

```bash
pytest -q
```

Covers the deterministic components (normaliser, splitter, FEN builder) and the predictor via a
mocked model — no trained checkpoint is needed to run the suite.

---

## What is detected and what is assumed

The classifier reads **piece placement only**. The complete FEN is built with assumed fields, and
the UI labels them as such:

| Field | Source |
| --- | --- |
| Piece placement | Detected by the model |
| Side to move | Chosen by the user |
| Castling rights | Assumed `-` |
| En passant target | Assumed `-` |
| Halfmove clock | Assumed `0` |
| Fullmove number | Assumed `1` |

---

## Known limitations

- Requires a manual, tight, axis-aligned crop; a sloppy crop shifts every square.
- Only the themes present in the training data are supported; an unseen piece set will degrade
  accuracy without necessarily lowering confidence.
- Confidence is raw softmax and is **not calibrated** — a confident mistake is possible.
- No perspective correction, so photographs of physical boards are out of scope.
- Coordinate labels, move arrows, last-move highlights and drag shadows in a screenshot are
  unmodelled noise.
- The position validator only warns; it never repairs a prediction.
- Synthetic training boards do not capture the anti-aliasing and shading of real board renderers.

---

## Remaining work before real predictions

1. Collect or render boards from the real themes you want to support (screenshots you are allowed
   to use, or your own licensed piece sets via `--asset-theme`).
2. Regenerate the dataset with those themes and check class coverage on both light and dark
   squares.
3. Train, then evaluate on a held-out **test** split of unseen positions.
4. Report the measured square, occupied-square and exact-board accuracy — from the evaluation run,
   never estimated.
5. Only then describe the model as working.

## Planned future work

- Automatic board detection and four-corner detection
- Perspective transformation for angled screenshots
- Physical 3D board support
- More board themes and piece sets
- Confidence calibration (temperature scaling) and abstention
- A user correction interface that feeds corrections back into training data
- Stockfish analysis of the recognised position
- React/Next.js frontend with a FastAPI backend

---

## License

MIT — see [LICENSE](LICENSE).
