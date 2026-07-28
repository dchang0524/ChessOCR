# Piece sprite themes

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
