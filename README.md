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
  raw RGB -> 13 logits        raw RGB -> 64-D embedding
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
`image_path, label, label_id, position_id, square, theme, square_color, background_varied, split`.

Splits are assigned **per position**, so all squares of a board stay in one split and no board
leaks between train, validation and test.

To use your own piece artwork (one transparent PNG per piece, named `wP.png` ... `bK.png`):

```bash
python scripts/generate_dataset.py --positions 400 --asset-theme my_theme assets/themes/my_theme
```

Only use artwork you are licensed to use — this project never scrapes assets from chess sites.

The repository also includes six attributed, open-license 2D sprite sets. To generate a broader
digital-board dataset containing every set on three different board palettes:

```bash
python scripts/generate_dataset.py \
  --positions 400 \
  --no-synthetic-themes \
  --sprite-sets chessnut fantasy spatial celtic rhosgfx kiwen-suwi \
  --board-palettes classic green blue \
  --output-dir data/processed/sprites_v4_curated_themes
```

The resulting 18 set/palette combinations are a much better starting point for 2D screenshots
than the Unicode synthetic themes. They still do not model site UI overlays, move arrows,
coordinates, or every proprietary piece set.

The generated v4 corpus contains 460,800 squares from 400 positions: 322,560 training, 69,120
validation, and 69,120 test examples. Use
`data/processed/sprites_v4_curated_themes/metadata.csv` for the next clean training run. Existing
reported checkpoints and experiment histories remain tied to the v2 corpus for reproducibility.

Training boards receive background-only variation by default. The generator draws palette,
gradient, texture, and grid-border variation onto the board layer, renders each piece onto a
separate transparent RGBA layer, and composites the piece afterward. Opaque sprite pixels are
therefore unchanged; antialiased edge pixels blend naturally with the new background. Generated
validation and test boards retain their original palettes. Tune this with
`--background-variation-probability` and `--background-variation-strength`, or disable it with a
zero strength.

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
  --metadata data/processed/sprites_v2_background_aug/metadata.csv \
  --checkpoint models/square_classifier_background_aug.pt \
  --epochs 1 --max-train-squares 80000 --max-val-squares 20000

python scripts/train_similarity.py \
  --metadata data/processed/sprites_v2_background_aug/metadata.csv \
  --checkpoint models/similarity_background_aug.pt \
  --epochs 2 --pairs-per-epoch 20000 --validation-pairs 5000 \
  --cross-background-positive-weight 1.0
```

Continue an existing similarity encoder with newly sampled pairs while preserving the original
checkpoint:

```bash
python scripts/train_similarity.py \
  --metadata data/processed/sprites_v2_background_aug/metadata.csv \
  --initial-checkpoint models/similarity_background_aug.pt \
  --checkpoint models/similarity_background_aug_extended.pt \
  --epochs 4 --pairs-per-epoch 20000 --validation-pairs 5000 \
  --learning-rate 0.0003 \
  --cross-background-positive-weight 1.0 \
  --history-json outputs/training_history_similarity_background_aug_extended.json
```

`--cross-background-positive-weight` multiplies the loss for same-piece positive pairs drawn from
opposite light/dark square backgrounds. A value of `3.0` makes a false split on those pairs count
three times as much as an ordinary pair while leaving negative-pair loss unchanged.

To fine-tune the extended encoder with that weighted objective:

```bash
python scripts/train_similarity.py \
  --metadata data/processed/sprites_v2_background_aug/metadata.csv \
  --initial-checkpoint models/similarity_background_aug_extended.pt \
  --checkpoint models/similarity_background_aug_weighted.pt \
  --epochs 2 --pairs-per-epoch 20000 --validation-pairs 5000 \
  --learning-rate 0.0001 \
  --cross-background-positive-weight 3.0 \
  --history-json outputs/training_history_similarity_background_aug_weighted.json
```

Render four-way samples of the exact generator augmentations (original, background-only RGB,
crop offset, and both):

```bash
python scripts/sample_training_augmentations.py \
  --output-dir outputs/training_augmentation_samples
```

Both CNNs consume raw normalized RGB during training and inference. The experimental
`SquareBackgroundNormalizer` remains in the repository for ablations, but neither model calls it.
The training transform also avoids post-composite colour jitter, which would modify piece pixels.
Background invariance is instead learned from the generator-side compositing described above.

The Siamese sampler includes all 13 labels, including empty: same-label pairs are positive,
different-label pairs are negative, and opposite light/dark-square positives are preferred. It
also resamples a new deterministic pair set every epoch. At inference all 64 squares are
clustered, so empty is an appearance group rather than a baseline occupancy filter. Empty may be
assigned to multiple clusters without a repeat penalty if conservative clustering splits it.

Evaluate independent classification and grouped assignment on evaluation-only Kaggle themes:

```bash
python scripts/evaluate_grouped_kaggle.py \
  --classifier-checkpoint models/square_classifier_background_aug.pt \
  --similarity-checkpoint models/similarity_background_aug.pt \
  --image-dir data/raw/kaggle_chess_positions/test \
  --max-boards 50
```

After four additional epochs, generated validation pair accuracy improved from 86.14% to 90.76%.
At the same 0.5% generated-negative false-merge calibration target, positive-pair recall improved
from 0.08% to 56.8%. A `0.96` deployment cutoff was selected on the first 50 external boards, then
checked on a disjoint 200-board slice: grouped square accuracy improved from 89.09% with the old
encoder to 90.48%, occupied accuracy from 51.57% to 58.18%, false merges from 1.04% to 0.57%, and
false splits from 55.26% to 49.03%. These slices are development checks, not a final Kaggle test
benchmark. The evaluator reports baseline versus grouped square, occupied-square and exact-board
accuracy plus false-merge and false-split rates across all 64 squares.

The `3.0` weighted continuation improved generated validation pair accuracy to 92.16% and reduced
holdout false splits further to 47.49%, but it was not promoted to the browser model: holdout
grouped accuracy slipped from 90.48% to 90.13% and false merges rose slightly from 0.569% to
0.588%. The checkpoint is retained for further loss-weight experiments.

Four more weighted epochs at a higher `0.0003` learning rate continued from that checkpoint:

```bash
python scripts/train_similarity.py \
  --metadata data/processed/sprites_v2_background_aug/metadata.csv \
  --initial-checkpoint models/similarity_background_aug_weighted.pt \
  --checkpoint models/similarity_background_aug_weighted_high_lr.pt \
  --epochs 4 --pairs-per-epoch 20000 --validation-pairs 5000 \
  --learning-rate 0.0003 \
  --cross-background-positive-weight 3.0 \
  --history-json outputs/training_history_similarity_background_aug_weighted_high_lr.json
```

Epoch 10 was best at 96.74% generated validation pair accuracy and 85.8% calibrated positive
recall. A threshold sweep on a disjoint 1,000-board Kaggle slice (`--skip-boards 250`) compared
`0.80`, `0.85`, `0.90`, and `0.95`, selecting by final grouped occupied-square accuracy. The
winning `0.95` cutoff reached 56.79% occupied accuracy, 89.38% overall square accuracy, 1.40%
exact-board accuracy, 0.401% false merges, and 41.89% false splits. The independent one-shot
baseline on the same boards reached 50.92% occupied and 86.53% overall accuracy. Full sweep
results are stored in `outputs/grouped_kaggle_weighted_high_lr_threshold_sweep_1000.json`. The
deployed operating point remains `0.98`, which reached 57.97% occupied and 89.76% overall accuracy
on the same 1,000-board slice, outperforming every threshold in that requested sweep.

Four further weighted epochs continued from epoch 10 with the same `0.0003` learning rate. Epoch
12 was retained at 97.64% generated validation accuracy. At the fixed `0.98` operating point on
the same 1,000 external boards, it reached 58.09% occupied accuracy, 89.84% overall accuracy,
1.50% exact-board accuracy, 0.040% false merges, and 52.47% false splits, so it replaced the
epoch-10 encoder in the browser model. The previous checkpoint remains available for rollback.

Another four epochs continued at the lower `0.0001` learning rate. Epoch 16 was retained at
98.16% generated validation accuracy. A `0.94`/`0.96`/`0.98`/`0.99` threshold sweep on boards
0–249 selected `0.98` by occupied accuracy, with overall accuracy and false merges as tie-breakers.
On the disjoint 1,000-board slice, epoch 16 at `0.98` reached 58.15% occupied accuracy, 89.86%
overall accuracy, 1.30% exact-board accuracy, 0.039% false merges, and 55.86% false splits. It
slightly improved the primary occupied metric and was promoted, while epoch 12 remains available
for rollback. Sweep details are stored in
`outputs/grouped_kaggle_weighted_low_lr_extended_threshold_sweep_tuning250.json`.

A follow-up check on the disjoint 1,000-board slice compared the selected `0.98` threshold with
`0.97` and `0.96`. Lowering the threshold reduced false splits but hurt the primary label metrics:
occupied accuracy fell from 58.15% at `0.98` to 57.99% and 57.80%, while false merges rose from
0.039% to 0.111% and 0.215%. The general browser cutoff therefore remains at `0.98`.

Because the encoder still separated matching pieces on opposite light/dark backgrounds, clustering
now uses a separately calibrated cross-background cutoff while keeping all same-background pairs at
`0.98`. Sweeping cross-background cutoffs on boards 0–249 selected `0.80`, then a disjoint
1,000-board check reached 58.20% occupied accuracy, 89.81% overall accuracy, 0.160%
cross-background false merges, and 42.68% cross-background false splits. The previous uniform
`0.98` rule reached 58.15% occupied and 89.86% overall accuracy but split effectively every
matching cross-background pair on the tuning slice.

### 256px transfer-learning experiment

An experimental similarity path uses an ImageNet-pretrained MobileNetV3-Small backbone at
256×256, followed by a 576→256→128 projection head and L2-normalized cosine embeddings. The
high-resolution generator rendered 57,600 native 256px squares from 50 positions across the 18
curated sprite/palette themes. The backbone was first frozen while training the projection head,
then its final three blocks were fine-tuned for one epoch using 85% hard negatives that shared
piece colour or piece type.

The hard-negative checkpoint reached 97.0% generated validation accuracy and 99.0% positive
recall. On a 100-board Kaggle smoke slice at threshold 0.95, however, it reached only 54.08%
occupied and 88.56% overall labeling accuracy, with 0.51% cross-background false merges and
55.20% cross-background false splits. The deployed compact/parity-aware model reached 60.91%
occupied and 90.16% overall on the same boards, with 0.086% cross-background false merges and
44.56% cross-background false splits. CPU evaluation also took 171.5 seconds versus 23.5 seconds.
The transfer checkpoint is therefore experimental and is not exported to the browser. A future
iteration should jointly train a 13-class auxiliary head and the contrastive embedding head so the
backbone is explicitly rewarded for separating different piece identities.

A follow-up one-shot experiment replaced grouping entirely with a MobileNetV3-Small 13-class
classifier using the same native-256px generated corpus. Its 576→256→13 head was trained with
balanced class sampling while the backbone was frozen, then the final three MobileNet blocks were
fine-tuned at `0.0001`. Although generated validation accuracy reached 99.87% before fine-tuning
and 100% afterward, occupied accuracy on the same 100 Kaggle boards was only 27.34% and 26.64%,
respectively. The compact one-shot classifier reached 53.72% occupied accuracy and the deployed
grouped path reached 60.91% on that slice. The transfer classifier also required about 145.7 seconds
per 100 CPU boards. It remains an offline experiment; the result indicates a synthetic-theme domain
gap rather than insufficient model capacity.

### Joint DINOv2 shape model

The next experimental path shares one self-supervised
[DINOv2 ViT-S/14 backbone](https://github.com/facebookresearch/dinov2) between classification and
grouping. DINO returns a global class token plus one token per 14×14 image patch. A learned
six-head attention query selects locally useful piece details, combines them with the global token,
and feeds separate 13-class and 128-dimensional embedding heads. Training combines classification,
same/different-pair, and explicit opposite-background consistency losses. DINO is the shape
backbone; background invariance comes from the paired consistency objective rather than from DINO
alone.

Train the heads while freezing the pretrained backbone and holding out a complete sprite family:

```bash
python scripts/train_joint_dino.py \
  --metadata data/processed/sprites_transfer_256/metadata.csv \
  --checkpoint models/joint_dinov2_vits14_head.pt \
  --history-json outputs/training_history_joint_dinov2_vits14_head.json \
  --epochs 1 \
  --pairs-per-epoch 500 \
  --validation-pairs 125 \
  --batch-size 4 \
  --freeze-backbone \
  --validation-theme-families spatial
```

This bounded first-stage run reached 94.0% class accuracy on the entirely held-out `spatial`
family. On the same first 100 Kaggle boards used above, the joint checkpoint reached 91.76%
occupied, 93.95% overall, and 36% exact-board accuracy without grouping. At its automatically
calibrated `0.9242` grouping threshold it reached 90.01% occupied, 97.09% overall, and 57%
exact-board accuracy, with 0.094% false merges and 34.36% false splits. The pretrained shape
features therefore generalize much better than the previous MobileNet experiment, although the
grouping threshold still needs tuning because grouping currently lowers occupied-square accuracy.
The shared CPU evaluation took 181.3 seconds for 100 boards versus 23.5 seconds for the deployed
compact pipeline. This checkpoint remains experimental and has not been exported or deployed.

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

python scripts/export_similarity_onnx.py \
  --checkpoint models/similarity_background_aug_weighted_low_lr_extended.pt \
  --output web/model/similarity_encoder.onnx \
  --similarity-threshold 0.98 \
  --cross-background-similarity-threshold 0.80
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
