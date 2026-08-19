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
        Browser UI (web/) or Streamlit UI (app.py)
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
                    │
          ┌─────────┴─────────┐
          ▼                   ▼
  SquareClassifier     SimilarityClassifier
  neutralise background in both -> 13 logits + 64-D embedding
          │                   │
          │            complete-linkage clustering
          └─────────┬─────────┘
                    ▼
        soft global group-label assignment
        duplicate penalties + fixed user corrections
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

The repository also includes five attributed, open-license 2D sprite sets. To generate a broader
digital-board dataset containing every set on three different board palettes:

```bash
python scripts/generate_dataset.py \
  --positions 400 \
  --no-synthetic-themes \
  --sprite-sets chessnut fantasy spatial celtic rhosgfx \
  --board-palettes classic green blue \
  --output-dir data/processed/sprites_v1
```

The resulting 15 set/palette combinations are a much better starting point for 2D screenshots
than the Unicode synthetic themes. They still do not model site UI overlays, move arrows,
coordinates, or every proprietary piece set.

By default, 80% of rendered boards also receive correlated crop jitter: each outer edge is
independently expanded or trimmed by up to 6 pixels before the board is resized and split. This
simulates a user crop that is slightly outside or inside the true grid. Use
`--crop-jitter-pixels 0` to generate perfectly aligned boards, or tune the amount and frequency
with `--crop-jitter-pixels` and `--crop-jitter-probability`.

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

### Theme-relative grouping experiment

Train both models from random initialization using generated sprite data only:

```bash
python scripts/train_model.py \
  --metadata data/processed/sprites_v1/metadata.csv \
  --checkpoint models/square_classifier_generated.pt \
  --epochs 1 --max-train-squares 80000 --max-val-squares 20000

python scripts/train_similarity.py \
  --metadata data/processed/sprites_v1/metadata.csv \
  --checkpoint models/similarity_generated.pt \
  --epochs 2 --pairs-per-epoch 20000 --validation-pairs 5000
```

Both CNNs estimate each square's board colour from its four corners, subtract that colour, and
soft-mask background-like pixels to neutral gray. Because this parameter-free module is inside
both model graphs, the same operation runs during augmented training, Python inference, and ONNX
browser inference. The Siamese sampler includes all 13 labels, including empty: same-label pairs
are positive, different-label pairs are negative, and opposite light/dark-square positives are
preferred. At inference all 64 squares are clustered, so empty is an appearance group rather than
a baseline occupancy filter. Empty may be assigned to multiple clusters without a repeat penalty
if conservative clustering splits the background.

Create an inspectable random Kaggle sample using the exact model preprocessing:

```bash
python scripts/sample_background_neutralization.py --count 6 --seed 42
```

This writes individual neutralized boards and a side-by-side montage under
`outputs/background_neutralization/`.

Evaluate independent classification and grouped assignment on evaluation-only Kaggle themes:

```bash
python scripts/evaluate_grouped_kaggle.py \
  --classifier-checkpoint models/square_classifier_generated.pt \
  --similarity-checkpoint models/similarity_generated.pt \
  --image-dir data/raw/kaggle_chess_positions/test \
  --max-boards 50
```

The empty-aware checkpoint calibrates a conservative `0.8935` cutoff on generated validation
pairs. A 50-board external smoke test improved square accuracy from 85.81% to 89.19% and occupied
accuracy from 43.65% to 53.04%; this small prefix is a pipeline check, not a final benchmark. The
evaluator reports baseline versus grouped square, occupied-square and exact-board accuracy plus
false-merge and false-split rates across all 64 squares.

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

To benchmark a checkpoint against the external Kaggle chess-positions test split without
extracting another 1.28 million square images:

```bash
python scripts/evaluate_kaggle.py \
  --checkpoint models/square_classifier_2d.pt \
  --image-dir data/raw/kaggle_chess_positions/test \
  --board-batch-size 16
```

The evaluator reads each full board once, applies the production resize/split/normalization path,
and writes detailed metrics to `outputs/kaggle_evaluation.json` plus a confusion matrix under
`outputs/confusion_matrices/`. Keep this dataset evaluation-only to preserve it as an external
generalization benchmark.

For Kaggle training, create a reproducible 80/10/10 manifest without copying the images, then
fine-tune the existing 2D checkpoint:

```bash
python scripts/split_kaggle.py --seed 42
python scripts/train_kaggle.py \
  --initial-checkpoint models/square_classifier_2d.pt \
  --checkpoint models/square_classifier_kaggle.pt \
  --epochs 1 \
  --board-batch-size 8 \
  --empty-samples 8
```

The split preserves the dataset's original 80,000-board training folder and deterministically
divides the original 20,000-board holdout into 10,000 validation and 10,000 final-test boards.
Training uses every occupied square plus eight randomly selected empty squares per board and
applies crop and color augmentation; validation and testing always use all 64 squares.

With split seed 42 and one fine-tuning epoch from `square_classifier_2d.pt`, the Kaggle-trained
checkpoint classified all 640,000 squares and all 10,000 boards in the final-test manifest
correctly. Treat this as **Kaggle in-distribution accuracy**, not a claim of perfect real-world
OCR: the split contains distinct positions but uses the same finite collection of synthetic
renderer styles across train, validation and test.

---

## Running the app

```bash
streamlit run app.py
```

Point the sidebar at your checkpoint (default `models/square_classifier.pt`), adjust the
low-confidence threshold (default `0.80`), then upload and crop.

### Browser inference

The static app in `web/` runs the same square classifier locally with ONNX Runtime Web. The
uploaded screenshot never goes to an inference server. Export the latest checkpoint whenever you
retrain:

```bash
python scripts/export_onnx.py \
  --checkpoint models/square_classifier_kaggle.pt \
  --output web/model/square_classifier.onnx
```

Serve the directory over HTTP (opening `index.html` directly will not allow the model fetch):

```bash
python -m http.server 4173 --directory web
```

Then open `http://localhost:4173`. The application resizes the selected crop to 512×512, packs
the 64 RGB squares into one `64×3×64×64` tensor, runs a single ONNX batch in WebAssembly, and
reconstructs the FEN entirely in JavaScript.

### Deploying to Cloudflare

The checked-in `wrangler.jsonc` serves `web/` as static Worker assets and attaches the Worker to
`chessocr.junyong.dev`. After authenticating Wrangler, deploy with:

```bash
npm install
npm run deploy
```

Cloudflare creates the custom-domain DNS record and certificate. The ONNX model is a static asset;
there is no Python server or GPU bill in this deployment.

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
