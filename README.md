# Chess OCR

Recognize a chess position from a screenshot of a digital chessboard and turn it into FEN.

Upload a screenshot, crop the board by hand, and the app classifies all 64 squares locally,
reconstructs the position, and shows how confident it was about every square.

---

## Current MVP scope

**In scope**

- Clean, axis-aligned **digital** chessboard screenshots
- **Manual** square cropping by the user
- Clean 2D themes, including experimental generalization to unseen piece sets
- Board orientation chosen by the user (White at bottom / Black at bottom)
- Per-square classification into 13 classes (empty + 6 white + 6 black pieces)
- Board-FEN output, reconstructed board rendering, and per-square confidence

**Explicitly out of scope for now**

- Automatic board detection or four-corner detection
- Perspective correction / photographs of physical boards
- Engine analysis (Stockfish), move legality repair, or position search

The architecture keeps these boundaries clean so automatic detection can later replace the manual
crop without touching the classifier or the FEN logic.

### Honest status

The full pipeline — generated-data creation, training, evaluation, inference, grouping and UI — is
implemented and exercised end to end. The current browser model is the production grouped DINOv2
model described below. Its chess-specific heads were initialized from the generated-only DINOv2
checkpoint and then trained on 90% of the Kaggle boards. The remaining 10% holds out board
positions, not piece themes: it contains the same renderer and theme distribution as training.
Production accuracy is therefore inflated as an estimate of performance on unseen themes, and the
production model has no demonstrated unseen-theme generalization benchmark.

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

## 2D Board Piece Classification

The classifier recognizes 13 square classes: empty plus the six white and six black piece types.
We have tested eight approaches, progressing from an independently applied compact CNN to a shared
DINOv2 model that classifies and groups squares. The first six methods were trained on generated
data only. The final two production methods continue training those DINOv2 heads on Kaggle data.

Reported **training** and **validation** accuracy comes from generated square or pair samples.
For the first six methods, reported **test** accuracy comes from inference-only evaluation on
rendered Kaggle boards with themes unseen during chess-specific training. For the two production
methods, the 10% Kaggle board holdout contains themes seen in the 90% training split. Empty squares
dominate most positions, so occupied-square and exact-board accuracy should be read alongside
overall square accuracy. “Pair accuracy” is same-piece versus different-piece accuracy; “false
merge” is the fraction of truly different pairs that clustering joined, while “false split” is the
fraction of truly matching pairs that clustering separated.

The earlier Kaggle results are useful for comparing experiments evaluated on the same slice, but
they are not untouched final-test estimates: repeated evaluation influenced later architecture and
operating-point decisions. The production results are even less representative of theme shift,
because Kaggle data enters backpropagation and only board identity—not theme identity—is held out.
DINOv2 and MobileNet also start from general-purpose pretrained weights, so “generated-only”
describes chess-specific training rather than their upstream ImageNet/LVD-142M pretraining.

### Splitting a board into squares

The user first selects a tight, axis-aligned 1:1 crop around the 8×8 grid and chooses whether White
or Black is at the bottom. `BoardNormalizer` converts the crop to RGB and resizes the whole board
to a fixed square image. `BoardSplitter` then uses arithmetic row and column boundaries to cut it
into 64 non-overlapping square images.

Squares are ordered in FEN scan order: left to right across the top displayed rank, followed by
each lower rank. For a White-at-bottom board this starts at `a8`; for Black-at-bottom, the square
coordinates are rotated while the output is still reconstructed in FEN order. The model never
sees the full board or coordinate labels—only individual square crops. This same normalization and
split path is used during evaluation and production inference.

```text
screenshot -> manual square crop -> RGB board resize -> 8 x 8 arithmetic split
           -> 64 square tensors -> classification/grouping -> FEN-order labels
```

Generated datasets are also split by complete chess position rather than by square. All 64 squares
from one generated board therefore stay in the same train, validation or test partition, avoiding
position leakage between partitions. Training boards receive background-only palette, gradient,
texture and crop-offset variation before the unmodified transparent piece sprite is composited.
The experimental background-neutralization module remains in the repository but is disabled in
the current training and inference paths.

### One-shot classifier

#### Architecture

The original one-shot model classifies every 64×64 square independently. It is a compact CNN
trained from random initialization:

```text
RGB square
-> [Conv 3x3 -> BatchNorm -> ReLU -> Conv 3x3 -> BatchNorm -> ReLU -> MaxPool] x 3
   channels: 3 -> 32 -> 64 -> 128
-> adaptive average pooling to 1 x 1
-> dropout(0.2)
-> linear 128 -> 13 logits
```

There is no communication between squares. The class with the largest softmax probability becomes
the predicted label.

#### Why we tried it

This was the simplest useful baseline: one square in, one semantic label out. It is small, fast in
the browser and easy to train, and it establishes how far per-square appearance alone can go. Its
main limitation is theme shift. If a new theme makes a bishop resemble the training pawns, the
model cannot use the fact that the board also contains a visibly different pawn group.

#### Training, validation, and test results

The recorded baseline used 80,000 generated training squares and 20,000 generated validation
squares from `sprites_v1`, AdamW, cross-entropy with 0.05 label smoothing, and one epoch.

| Split | Scope | Result |
| --- | --- | ---: |
| Generated training | 80,000 squares | 94.22% class accuracy |
| Generated validation | 20,000 squares | 97.87% class accuracy |
| Kaggle development test | boards 100–599 (500 boards) | 79.51% overall; 54.46% occupied; 3.40% exact-board |

No separate generated test evaluation was saved for this checkpoint. The Kaggle slice was
inference-only and did not train the model.

### Group classifier

#### Architecture

The compact grouped pipeline adds a Siamese CNN beside the one-shot classifier. “Siamese” means
that two squares pass through two copies of the same encoder with shared weights; in practice each
of the 64 squares is encoded once and pair similarity is calculated afterward.

The similarity encoder uses the same three compact convolution blocks as the one-shot model,
followed by adaptive average pooling and a linear `128 -> 64` projection. The resulting 64-value
embedding is L2-normalized, so the dot product between two embeddings is their cosine similarity.

At board inference:

1. Compute classifier logits and embeddings for all 64 squares, including empty squares.
2. Build the 64×64 cosine-similarity matrix.
3. Form appearance groups with complete-linkage agglomerative clustering. A merge is allowed only
   when the least-similar cross-pair clears the cutoff.
4. Average the one-shot logits inside each group.
5. Use a Hungarian maximum-score assignment to label all groups jointly. Repeated non-empty labels
   receive a soft penalty of 1.5, kings have one slot each, and empty may repeat without penalty.
6. Apply the selected label to every member of the group. A later user correction can therefore be
   propagated to all matching squares.

The final compact version uses a 0.98 same-background(e.g. two pieces on white squares) threshold and a 0.80 cross-background(e.g. a piece on a white square and a piece on a black square)
threshold.

#### Why we tried it

A new theme contains useful information about itself. Even when the semantic classifier confuses a
bishop with a pawn, a similarity model may still discover two distinct, internally consistent
shapes. Averaging evidence within each group reduces isolated mistakes, while global assignment
can raise the bishop probability for one group when another group already explains the pawns.
Conservative complete linkage was selected because a false merge is especially damaging: it can
give two different piece types the same label and propagate a user correction incorrectly.

#### Training, validation, and test results

The final compact classifier was trained for one epoch on 80,000 generated, background-augmented
squares. The similarity encoder was trained through epoch 16 on balanced positive/negative pairs.
Opposite-background positive pairs had 3× loss weight; later runs used hard negatives that shared
piece color or piece type. The threshold was tuned on Kaggle boards 0–249, then checked on the
disjoint boards 250–1,249.

| Split | Component | Result |
| --- | --- | ---: |
| Generated training | 13-class classifier | 95.56% class accuracy |
| Generated validation | 13-class classifier | 99.98% class accuracy |
| Generated training, epoch 16 | similarity encoder | 99.97% pair accuracy |
| Generated validation, epoch 16 | similarity encoder | 98.16% pair accuracy |
| Kaggle development test | one-shot baseline, 1,000 boards | 86.53% overall; 50.92% occupied; 0.20% exact-board |
| Kaggle development test | grouped, same 1,000 boards | 89.81% overall; 58.20% occupied; 1.00% exact-board |

On that 1,000-board grouped evaluation, false merges were 0.083% and false splits were 27.56%
across all squares. This compact grouped model improved occupied accuracy by 7.28 percentage points
over its accompanying one-shot classifier.

### Transfer-learning CNN one-shot classifier

#### Architecture

This experiment replaces the compact CNN with an ImageNet-pretrained MobileNetV3-Small and raises
the input resolution from 64×64 to a native 256×256. MobileNet’s convolutional backbone produces
576 features, followed by:

```text
linear 576 -> 256 -> LayerNorm -> Hardswish -> dropout(0.2) -> linear 256 -> 13
```

MobileNetV3 is still a CNN, but its pretrained depthwise-separable and inverted-residual blocks
have already learned general visual features such as edges, contours and object parts.

#### Why we tried it

The compact CNN often confused pieces whose identity depends on a small shape detail, such as a
bishop’s top marking(a cross). We tested whether four times the input width and an ImageNet-pretrained shape
backbone would preserve those details and generalize them to unseen chess themes.

#### Training, validation, and test results

The classifier head was trained for two epochs using balanced class sampling from 8,000 generated
training and 3,000 validation squares while the backbone was frozen. The final three MobileNet
feature blocks were then fine-tuned for one epoch at a lower learning rate.

| Split | Stage | Result |
| --- | --- | ---: |
| Generated training | frozen-head stage, epoch 2 | 98.96% class accuracy |
| Generated validation | frozen-head stage, epoch 2 | 99.87% class accuracy |
| Generated training | final-three-block fine-tuning | 99.78% class accuracy |
| Generated validation | final-three-block fine-tuning | 100.00% class accuracy |
| Kaggle development test | fine-tuned model, first 100 boards | 86.92% overall; 26.64% occupied; 0% exact-board |

The near-perfect generated validation result did not transfer to new themes. This indicates a
synthetic-theme domain gap, not merely a lack of model capacity. No separate generated test run was
saved.

### Transfer-learning CNN group classifier

#### Architecture

This pipeline keeps the compact 13-class classifier for group labeling but replaces the compact
similarity encoder with an ImageNet-pretrained MobileNetV3-Small at 256×256. Its projection head is:

```text
MobileNet features (576)
-> linear 576 -> 256
-> LayerNorm -> Hardswish -> dropout(0.1)
-> linear 256 -> 128
-> L2-normalized embedding
```

The resulting embeddings use the same cosine matrix, complete-linkage clustering, group-logit
averaging and global label assignment as the compact grouped method.

#### Why we tried it

This isolated the question of whether a higher-resolution pretrained CNN could learn theme-relative
piece shape better than the compact 64×64 similarity network. Hard-negative pairs were intended to
stop the network from solving similarity using only piece color or broad silhouette.

#### Training, validation, and test results

The projection head was trained with the MobileNet backbone frozen, followed by one epoch of
fine-tuning the final three feature blocks. Training used 256×256 generated squares, opposite-
background positives at 3× weight, and an 85% hard-negative probability.

| Split | Component | Result |
| --- | --- | ---: |
| Generated training, fine-tuning epoch | similarity encoder | 95.03% pair accuracy |
| Generated validation | similarity encoder | 97.00% pair accuracy; 99.00% calibrated positive recall |
| Kaggle development test | compact one-shot baseline, first 100 boards | 86.69% overall; 53.72% occupied; 0% exact-board |
| Kaggle development test | MobileNet grouped, same 100 boards | 88.56% overall; 54.08% occupied; 0% exact-board |

At threshold 0.95 the grouped test had 0.651% false merges and 30.45% false splits across all
squares. It improved only slightly over the compact one-shot baseline and took 171.5 seconds for
100 CPU boards, versus 23.5 seconds for the compact grouped pipeline. No separate generated test
run was saved.

### DINOv2 one-shot classifier

#### Architecture

The current model uses the self-supervised DINOv2 ViT-S/14 backbone at 224×224. A vision
transformer divides the square into a 16×16 grid of 14×14 patches, producing 256 local patch tokens
and one global class token, each with 384 values. Twelve transformer blocks with six-head
self-attention let every patch exchange information with every other patch.

A learned six-head attention query selects locally useful details from the 256 patch tokens. Its
384-value result is concatenated with the 384-value global token and passed through:

```text
LayerNorm(768) -> linear 768 -> 384 -> GELU -> dropout(0.1) -> LayerNorm(384)
-> linear 384 -> 13 logits
```

The DINOv2 backbone’s 22.1 million parameters are frozen. The learned shape combiner and heads
contain approximately 1.03 million trainable parameters.

#### Why we tried it

MobileNet’s ImageNet pretraining still transferred poorly to unseen chess themes. DINOv2’s
self-supervised representation preserves strong local and global shape information without being
trained around a fixed set of ImageNet labels. The extra learned-query attention explicitly gives
the chess head access to small local cues such as a bishop slit, rook battlements or crown points.

#### Training, validation, and test results

The DINO heads were trained for one bounded epoch on 500 generated pairs, or 1,000 square-image
occurrences. Training used 15 themes and held out the entire three-palette `spatial` sprite family
for validation. The joint loss included classification, pair similarity and opposite-background
consistency terms. The validation sample contained 125 pairs.

| Split | Scope | Result |
| --- | --- | ---: |
| Generated training | 500 sampled pairs | 65.00% class accuracy; 87.00% pair accuracy |
| Generated validation | 125 pairs from held-out `spatial` family | 94.00% class accuracy; 82.40% pair accuracy |
| Kaggle development test | one-shot, first 100 boards | 93.95% overall; 91.76% occupied; 36% exact-board |

The large training/validation difference is partly caused by the much stronger randomized training
augmentation and by the small validation sample. No separate generated test evaluation was saved.

### DINOv2 group classifier

#### Architecture

The DINOv2 grouped method uses the same backbone and learned shape representation as the DINOv2
one-shot classifier. A second head creates the similarity embedding:

```text
shared 384-value representation
-> linear 384 -> 256 -> LayerNorm -> GELU -> dropout(0.1)
-> linear 256 -> 128
-> L2-normalized embedding
```

Both outputs come from one shared forward pass. The 13-class logits feed group labeling, while the
128-value embeddings feed the cosine matrix and complete-linkage clustering. Training minimizes:

```text
classification loss + 0.5 * pair-similarity loss + 0.5 * background-consistency loss
```

The browser exports this joint network to ONNX with dynamically quantized matrix-multiplication
weights. Quantization changes embedding values slightly, so the browser cutoff was recalibrated on
500 generated validation pairs to 0.94895. Kaggle data was not used for that calibration.

#### Why we tried it

The earlier grouped pipelines used separate networks whose features were not directly rewarded for
both semantic identity and theme-relative consistency. Sharing DINO’s shape representation allows
class supervision to improve the embedding and pair supervision to improve the classifier. The
explicit consistency loss additionally penalizes changes in class probabilities and embeddings
when the same piece appears on opposite board colors.

#### Training, validation, and test results

Training and validation are shared with the DINOv2 one-shot method. Positive and negative pairs
were balanced, 75% of training negatives were hard negatives, and opposite-background positives
received 3× pair-loss weight. The full-precision checkpoint threshold of 0.92419 was calibrated
only from held-out generated negatives.

| Split | Scope | Result |
| --- | --- | ---: |
| Generated training | 500 sampled pairs | 65.00% class accuracy; 87.00% pair accuracy |
| Generated validation | 125 pairs from held-out `spatial` family | 94.00% class accuracy; 82.40% pair accuracy |
| Kaggle development test | grouped, first 100 boards | 97.09% overall; 90.01% occupied; 57% exact-board |

The full-precision grouping evaluation produced a 0.094% false-merge rate and a 34.36% false-split
rate. Relative to DINO one-shot inference on the same boards, grouping improved overall accuracy by
3.14 percentage points and exact-board accuracy by 21 points, but lowered occupied accuracy by
1.75 points. This means the groups are meaningful, while the clustering cutoff and global label
assignment still need improvement.

The complete 100-board measurement used the full-precision PyTorch checkpoint. The deployed
quantized ONNX model was checked for close agreement on one real board, but has not yet been run
through the complete 100-board evaluation. It is therefore not valid to claim the exact same
numbers for the deployed artifact.

### Production DINOv2 one-shot classifier (Kaggle fine-tuned)

#### Architecture

The production one-shot model keeps the frozen DINOv2 ViT-S/14 backbone, learned patch-attention
shape combiner and `384 -> 13` classification head described above. It initializes every weight
from the generated-only joint DINOv2 checkpoint, then continues training the chess-specific shape
combiner and heads while the 22.1-million-parameter backbone remains frozen. At one-shot inference,
each square simply uses the largest of its 13 logits; embeddings and grouping are ignored.

#### Why we tried it

The generated-only experiments measured theme transfer, but their classifier still made many
semantic mistakes on Kaggle. This continuation run creates a production-oriented model for the
known Kaggle distribution while preserving the earlier checkpoint as the clean generated-only
experiment. It answers a different question: how accurate can this architecture become after it
has seen examples from the deployment-style data distribution?

#### Training, validation, and test results

The 100,000 Kaggle boards were deterministically divided into 90,000 training boards and 10,000
board-holdout boards. One deterministic pair, or two square occurrences, was sampled from every
training board during the continuation epoch. The optimizer updated the shape combiner,
classifier, projection and pair-decision parameters with learning rate `1e-4`; DINOv2 remained
frozen. A 5,000-board subset of the training partition was used for checkpoint monitoring and did
not provide a theme holdout.

| Split | Scope | Result |
| --- | --- | ---: |
| Kaggle training | 90,000 board pairs | 99.77% class accuracy; 99.70% pair accuracy |
| Kaggle training calibration | 5,000 boards | 100.00% class accuracy; 100.00% pair accuracy |
| Kaggle board holdout | 10,000 boards | 99.9970% overall; 99.9810% occupied; 99.81% exact-board |

Only 19 of 640,000 holdout squares were misclassified by one-shot inference. These numbers are
inflated as an estimate of real generalization: training and holdout boards use the same Kaggle
themes and rendering pipeline. Holding out positions prevents exact-board leakage, but it does not
measure performance on a genuinely unseen piece set.

### Production DINOv2 group classifier (Kaggle fine-tuned)

#### Architecture

The production grouped model is the same joint checkpoint and shared forward pass as production
one-shot inference. Its normalized 128-value embeddings form a 64×64 cosine matrix. Complete-
linkage clustering uses a conservative `0.95` similarity threshold, after which group logits are
averaged and the Hungarian assignment applies the existing 1.5 duplicate-label penalty. The
browser packages both logits and embeddings in one dynamically quantized INT8-weight ONNX graph.

#### Why we tried it

Even a highly accurate one-shot model can make an isolated error that conflicts with repeated
evidence elsewhere on the same board. Grouping can correct that error and lets a user correction
propagate across every square assigned to the same appearance group. The initially calibrated
threshold of `0.14801` was much too permissive for board clustering: rare false merges propagated
errors across otherwise correct squares. We therefore swept stricter thresholds and selected
`0.95`, the center of the stable `0.90`–`0.98` range.

#### Training, validation, and test results

Training is shared with the production one-shot model. Thresholds were explored on the first 100
Kaggle holdout boards, then compared on the next 1,000 boards, which were excluded from that sweep.
The table below reports full-precision PyTorch inference; it does not claim that quantization leaves
every board prediction unchanged.

| Kaggle board split | Method | Overall | Occupied | Exact board |
| --- | --- | ---: | ---: | ---: |
| Threshold selection, first 100 | one-shot | 99.9844% | 99.8995% | 99.00% |
| Threshold selection, first 100 | grouped at 0.95 | 100.0000% | 100.0000% | 100.00% |
| Disjoint threshold validation, next 1,000 | one-shot | 99.9969% | 99.9800% | 99.80% |
| Disjoint threshold validation, next 1,000 | grouped at 0.95 | 99.9984% | 99.9900% | 99.90% |

On the disjoint 1,000-board slice, grouping reduced two wrong squares to one without introducing a
new error. This is a real observed improvement but very small in absolute terms, and it is based on
only three one-shot errors across the selection and validation slices. More importantly, neither
slice tests a new theme. The high grouped accuracy is inflated by same-theme train/holdout overlap
and must not be presented as evidence that the production model generalizes to unseen themes.

The exported quantized browser artifact was compared with PyTorch on 10 deterministic holdout
boards: all 640 one-shot labels, all 640 grouped labels and all 10 cluster partitions agreed. This
is an export smoke test, not a replacement for evaluating the quantized artifact on the full
holdout or on genuinely unseen themes.

### Current conclusion

| Method | Kaggle boards | Overall | Occupied | Exact board |
| --- | ---: | ---: | ---: | ---: |
| Compact CNN one-shot | 500 | 79.51% | 54.46% | 3.40% |
| Compact CNN grouped | 1,000 | 89.81% | 58.20% | 1.00% |
| MobileNetV3 one-shot | 100 | 86.92% | 26.64% | 0% |
| MobileNetV3 grouped | 100 | 88.56% | 54.08% | 0% |
| DINOv2 one-shot | 100 | 93.95% | **91.76%** | 36% |
| DINOv2 grouped | 100 | **97.09%** | 90.01% | **57%** |
| Production DINOv2 one-shot | 10,000 | 99.9970% | 99.9810% | 99.81% |
| Production DINOv2 grouped at 0.95 | 1,000 | 99.9984% | 99.9900% | 99.90% |

Rows with different board counts are not direct comparisons. On the shared 100-board slice,
DINOv2 is the clearest improvement among the generated-only experiments. The two production rows
are not comparable measures of unseen-theme generalization because their Kaggle training and
holdout partitions share themes. The production grouped DINOv2 model at threshold 0.95 is the model
currently packaged for the browser.

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

### Reproducing the grouped DINOv2 run

```bash
python scripts/train_joint_dino.py \
  --metadata data/processed/sprites_transfer_256/metadata.csv \
  --checkpoint models/joint_dinov2_vits14_head.pt \
  --history-json outputs/training_history_joint_dinov2_vits14_head.json \
  --epochs 1 --pairs-per-epoch 500 --validation-pairs 125 --batch-size 4 \
  --freeze-backbone --validation-theme-families spatial
```

Continue that generated-only checkpoint on the 90/10 Kaggle board split with:

```bash
python scripts/train_joint_dino_kaggle.py \
  --initial-checkpoint models/joint_dinov2_vits14_head.pt \
  --checkpoint models/joint_dinov2_vits14_kaggle90.pt \
  --manifest data/metadata/kaggle_all_90_10.csv \
  --epochs 1 --batch-size 16 --learning-rate 1e-4
```

This command does not create an unseen-theme validation set. Its 10% holdout contains different
boards rendered from the same Kaggle theme distribution as the 90% training partition.

Render examples of the exact generator-side background and crop-offset augmentations with:

```bash
python scripts/sample_training_augmentations.py \
  --output-dir outputs/training_augmentation_samples
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

To compare one-shot and grouped DINOv2 inference against the external Kaggle chess-position split
without extracting the squares first:

```bash
python scripts/evaluate_grouped_kaggle.py \
  --classifier-checkpoint models/joint_dinov2_vits14_head.pt \
  --similarity-checkpoint models/joint_dinov2_vits14_head.pt \
  --image-dir data/raw/kaggle_chess_positions/test \
  --max-boards 100 \
  --square-batch-size 4 \
  --output outputs/grouped_kaggle_joint_dinov2_vits14_head_smoke100.json
```

The evaluator reads each full board once, applies the production resize/split/normalization path,
and writes one-shot accuracy, grouped accuracy, false merges and false splits. The generated-only
checkpoint does not train on Kaggle; the production `kaggle90` checkpoint does. Use a completely
new theme source—not another random Kaggle board split—for a final generalization measurement.

---

## Running the app

```bash
streamlit run app.py
```

Point the sidebar at your checkpoint (default `models/square_classifier.pt`), adjust the
low-confidence threshold (default `0.80`), then upload and crop.

### Browser inference

The static app in `web/` runs the joint DINOv2 classifier and embedding model locally with ONNX
Runtime Web. The uploaded screenshot never goes to an inference server. Export and dynamically
quantize the latest checkpoint whenever you retrain:

```bash
python scripts/export_joint_dino_onnx.py \
  --checkpoint models/joint_dinov2_vits14_kaggle90.pt \
  --output web/model/joint_dinov2_vits14_int8.onnx \
  --similarity-threshold 0.95
```

Serve the directory over HTTP (opening `index.html` directly will not allow the model fetch):

```bash
python -m http.server 4173 --directory web
```

Then open `http://localhost:4173`. The application resizes the selected crop to 512×512, extracts
the 64 squares, resizes each model input to 224×224 and runs the joint ONNX graph in chunks of four
squares. Classification, clustering, global group-label assignment and FEN reconstruction all
happen in the browser.

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
- Unseen 2D themes are supported experimentally, but a sufficiently different piece set can still
  degrade accuracy without lowering confidence.
- Confidence is raw softmax and is **not calibrated** — a confident mistake is possible.
- No perspective correction, so photographs of physical boards are out of scope.
- Coordinate labels, move arrows, last-move highlights and drag shadows in a screenshot are
  unmodelled noise.
- The position validator only warns; it never repairs a prediction.
- Synthetic training boards do not capture the anti-aliasing and shading of real board renderers.

---

## Remaining work before a final generalization claim

1. Freeze the architecture and 0.95 similarity operating point before inspecting another external set.
2. Collect a never-inspected set of licensed 2D themes outside the Kaggle renderer distribution.
3. Evaluate the exact quantized browser artifact, not only the full-precision PyTorch checkpoint.
4. Report overall-square, occupied-square, exact-board, false-merge and false-split measurements
   together.

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
