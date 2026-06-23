#!/usr/bin/env python3
"""Download Phase-0 MeanAudio assets from Hugging Face."""

from __future__ import annotations

import argparse
from pathlib import Path


DEFAULT_FILES = [
    "fluxaudio_s_full.pth",
    "meanaudio_s_full.pth",
    "v1-16.pth",
    "best_netG.pt",
]


OPTIONAL_FILES = [
    "meanaudio_l_full.pth",
    "meanaudio_s_ac.pth",
    "empty_string_t5.pth",
    "empty_string_t5_c.pth",
    "empty_string_clap_c.pth",
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo-id", default="AndreasXi/MeanAudio")
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument("--include-optional", action="store_true")
    parser.add_argument("--token", default=None, help="Optional Hugging Face token.")
    args = parser.parse_args()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise SystemExit(
            "Missing dependency: huggingface_hub. Install it on the server with "
            "`pip install huggingface_hub` or `pip install -e .` if the project environment includes it."
        ) from exc

    args.weights_dir.mkdir(parents=True, exist_ok=True)
    allow_patterns = list(DEFAULT_FILES)
    if args.include_optional:
        allow_patterns.extend(OPTIONAL_FILES)

    print(f"[download] repo: {args.repo_id}")
    print(f"[download] target: {args.weights_dir.resolve()}")
    print(f"[download] files: {', '.join(allow_patterns)}")

    snapshot_download(
        repo_id=args.repo_id,
        local_dir=str(args.weights_dir),
        allow_patterns=allow_patterns,
        token=args.token,
    )

    missing = [name for name in DEFAULT_FILES if not (args.weights_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"Download finished but required files are missing: {missing}")

    print("[done] Phase-0 assets downloaded.")


if __name__ == "__main__":
    main()
