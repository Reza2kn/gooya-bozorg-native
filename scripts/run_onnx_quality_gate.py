#!/usr/bin/env python3
"""Run the Gooya Native quality gate for one T3 Q4 candidate.

The contract in ``docs/RUNTIME_MATRIX.md`` does not require greedy-token
equality with FP32 (that proxy was shown unreachable even at Q8). It requires:

- waveform sanity and duration on the fixed prompt set;
- short-window Persian ASR anchor coverage of the complete utterance;
- the final word is a mandatory anchor.

This script synthesizes the fixed prompt set through the ONNX pipeline with the
given T3 graphs and a reference S3Gen/vocoder, transcribes each WAV with the
Shenava ASR service, and emits a gate receipt with WER/CER and final-word
anchors per prompt plus an overall ``status``.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import urllib.request
import uuid
import wave
from pathlib import Path


FIXED_PROMPTS = {
    "canonical": "سلام، حالت چطوره؟",
    "conversational": "امروز هوا خیلی خوب است و من قصد دارم قدم بزنم.",
    "punctuation": "از کجا آمدهای؟ به کجا میرویم؟ بمان! فقط یک لحظه.",
    "numeral": "قیمت این کتاب چهل و دو هزار تومان است.",
    "homograph": "او شیر تازه را به خانه آورد و شیر آب را باز کرد.",
    "mixed-script": "پیامک شما ارسال شد و لینک در تلگرام و واتساپ گذاشته شد.",
}

ORTHO = str.maketrans({"ي": "ی", "ك": "ک", "ۀ": "ه", "ة": "ه", "أ": "ا", "إ": "ا", "آ": "ا"})


def words(text: str) -> list[str]:
    text = text.translate(ORTHO).replace("\u200c", " ").lower()
    return re.findall(r"[\wآ-ی]+", text, flags=re.UNICODE)


def edit_distance(reference: list[str], hypothesis: list[str]) -> int:
    previous = list(range(len(hypothesis) + 1))
    for i, expected in enumerate(reference, 1):
        current = [i]
        for j, observed in enumerate(hypothesis, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[j] + 1,
                    previous[j - 1] + (expected != observed),
                )
            )
        previous = current
    return previous[-1]


def score(target: str, transcript: str) -> dict[str, object]:
    target_words = words(target)
    hypothesis_words = words(transcript)
    target_chars = list("".join(target_words))
    hypothesis_chars = list("".join(hypothesis_words))
    wer = edit_distance(target_words, hypothesis_words) / max(1, len(target_words))
    cer = edit_distance(target_chars, hypothesis_chars) / max(1, len(target_chars))
    final = bool(
        target_words and hypothesis_words and target_words[-1] == hypothesis_words[-1]
    )
    return {
        "wer": round(wer, 4),
        "cer": round(cer, 4),
        "final_word_exact": final,
        "target_words": target_words,
        "hypothesis_words": hypothesis_words,
    }


def transcribe(server: str, path: Path) -> dict:
    boundary = f"----shenava-{uuid.uuid4().hex}"
    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{path.name}"\r\n'
        "Content-Type: audio/wav\r\n\r\n"
    ).encode() + path.read_bytes() + f"\r\n--{boundary}--\r\n".encode()
    request = urllib.request.Request(
        f"{server}/transcribe",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=300) as response:
        return json.loads(response.read())


def duration_seconds(path: Path) -> float:
    with wave.open(str(path), "rb") as handle:
        return handle.getnframes() / handle.getframerate()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--t3", type=Path, required=True)
    parser.add_argument("--flow", type=Path, required=True)
    parser.add_argument("--vocoder", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--python", type=Path, default=Path(sys.executable))
    parser.add_argument("--server", default="http://100.96.172.113:3000")
    parser.add_argument("--bucket", type=int, default=168)
    parser.add_argument("--seed", type=int, default=20260816)
    args = parser.parse_args()

    runner = Path(__file__).resolve().parent / "run_onnx_pipeline.py"
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for name, text in FIXED_PROMPTS.items():
        stage = output / name
        stage.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            [
                str(args.python),
                str(runner),
                "--source", str(args.source.resolve()),
                "--t3", str(args.t3.resolve()),
                "--flow", str(args.flow.resolve()),
                "--vocoder", str(args.vocoder.resolve()),
                "--output", str(stage),
                "--text", text,
                "--bucket", str(args.bucket),
                "--seed", str(args.seed),
                "--cfg-weight", "0.5",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        wav = stage / "pipeline.wav"
        body = transcribe(args.server, wav)
        row = {
            "prompt": name,
            "target": text,
            "greedy": body.get("greedy") or "",
            "hotword": body.get("text") or "",
            "duration_seconds": duration_seconds(wav),
            "elapsed_ms": body.get("elapsed_ms"),
        }
        row["greedy_score"] = score(text, row["greedy"])
        row["hotword_score"] = score(text, row["hotword"])
        rows.append(row)
        g, h = row["greedy_score"], row["hotword_score"]
        print(
            f"[gate] {name:16s} dur={row['duration_seconds']:.2f}s "
            f"greedy WER={g['wer']:.3f} final={g['final_word_exact']} "
            f"hotword WER={h['wer']:.3f} final={h['final_word_exact']}",
            flush=True,
        )

    mean_wer = sum(row["hotword_score"]["wer"] for row in rows) / len(rows)
    mean_cer = sum(row["hotword_score"]["cer"] for row in rows) / len(rows)
    final_anchors = sum(row["hotword_score"]["final_word_exact"] for row in rows)
    waveform_ok = all(row["duration_seconds"] > 0.3 for row in rows)
    status = "PASS" if mean_wer <= 0.2 and final_anchors == len(rows) and waveform_ok else "FAIL"
    receipt = {
        "schema_version": "gooya.native.quality-gate/v1",
        "t3": str(args.t3.resolve()),
        "flow": str(args.flow.resolve()),
        "vocoder": str(args.vocoder.resolve()),
        "server": args.server,
        "prompts": len(rows),
        "hotword_summary": {
            "mean_wer": round(mean_wer, 4),
            "mean_cer": round(mean_cer, 4),
            "final_word_anchors_exact": final_anchors,
        },
        "waveform_all_duration_ok": waveform_ok,
        "status": status,
        "rows": rows,
    }
    (output / "quality-gate-receipt.json").write_text(
        json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(receipt, ensure_ascii=False, indent=2, sort_keys=True))
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())