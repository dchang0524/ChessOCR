# Piece sprite themes

Five open-license 2D sets are bundled: `chessnut`, `fantasy`, `spatial`, `celtic`, and
`rhosgfx`. Their provenance and licenses are recorded in [ATTRIBUTIONS.md](ATTRIBUTIONS.md).
Regenerate the PNGs from the pinned upstream SVGs with:

```bash
python scripts/import_piece_sprites.py
```

Render every bundled set on the classic, green, and blue board palettes:

```bash
python scripts/generate_dataset.py \
  --positions 400 \
  --no-synthetic-themes \
  --sprite-sets chessnut fantasy spatial celtic rhosgfx \
  --board-palettes classic green blue \
  --output-dir data/processed/sprites_v1
```

This produces 15 visual themes. Palette/set combinations are intentional: they prevent the
classifier from learning a false association between one piece style and one board colour.
The generator also applies up to 6 pixels of full-board crop jitter to 80% of renders by default,
so all 64 square boundaries shift together as they would after an imperfect manual crop.

## Adding your own set

Drop one directory per theme here, each containing twelve transparent PNG sprites:

```text
assets/themes/<theme_name>/
├── wP.png  wN.png  wB.png  wR.png  wQ.png  wK.png
└── bP.png  bN.png  bB.png  bR.png  bQ.png  bK.png
```

Then render a dataset with that theme:

```bash
python scripts/generate_dataset.py --positions 400 --asset-theme <theme_name> assets/themes/<theme_name>
```

Square colours can be tuned by constructing `ImageAssetBoardTheme` directly in a script.

Only add artwork you are licensed to use. This project deliberately does not download piece sets
from chess websites.
