#!/usr/bin/env python3
"""Download the Gooya Bozorg 1.5 native asset bundle (Q4 ONNX graphs + tokenizer).

Writes into desktop/data so the app can run offline.
"""

from __future__ import annotations

import sys
from pathlib import Path
from huggingface_hub import snapshot_download

REPO = "Reza2kn/gooya-bozorg-v1.5-native"
HERE = Path(__file__).resolve().parent
DEST = HERE.parent / "desktop" / "data"


def main() -> int:
    DEST.mkdir(parents=True, exist_ok=True)
    print(f":: downloading {REPO} -> {DEST}")
    snapshot_download(repo_id=REPO, local_dir=str(DEST), tqdm_class=None)
    bundle = DEST / "tract-bundle-b168"
    if not (bundle / "t3-prefill.onnx").exists():
        print(f":: ERROR: bundle missing at {bundle}", file=sys.stderr)
        return 1
    print(":: done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())