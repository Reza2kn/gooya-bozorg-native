#!/usr/bin/env python3
"""Download the pinned Gooya source and fail on any size or hash drift."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from huggingface_hub import snapshot_download


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "model" / "source.lock.json"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    lock = json.loads(LOCK.read_text(encoding="utf-8"))
    model = lock["model"]
    expected = lock["files"]
    output = args.output.resolve()
    snapshot_download(
        repo_id=model["repo_id"],
        revision=model["revision"],
        local_dir=output,
        allow_patterns=sorted(expected),
    )

    actual: dict[str, dict[str, object]] = {}
    failures: list[dict[str, object]] = []
    for relative, wanted in expected.items():
        path = output / relative
        if not path.is_file():
            failures.append({"file": relative, "error": "missing"})
            continue
        observed = {"size": path.stat().st_size, "sha256": sha256(path)}
        actual[relative] = observed
        if observed != wanted:
            failures.append(
                {"file": relative, "error": "mismatch", "expected": wanted, "actual": observed}
            )

    receipt = {
        "schema_version": "gooya.native.source-receipt/v1",
        "status": "pass" if not failures else "failed",
        "repo_id": model["repo_id"],
        "revision": model["revision"],
        "files": actual,
        "failures": failures,
    }
    receipt_path = output / "source-receipt.json"
    receipt_path.write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    if failures:
        raise RuntimeError(f"source verification failed; see {receipt_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

