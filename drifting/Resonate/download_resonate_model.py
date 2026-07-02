#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

from huggingface_hub import hf_hub_download


REPOSITORY_ID = "AndreasXi/Resonate"
RELEASED_MODELS = ("Resonate_GRPO.pth", "Resonate_PT.pth")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Download one released Resonate checkpoint into the local weights directory."
    )
    parser.add_argument("--repo-id", default=REPOSITORY_ID)
    parser.add_argument(
        "--model-file",
        choices=RELEASED_MODELS,
        default="Resonate_GRPO.pth",
    )
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument("--revision", default="main")
    args = parser.parse_args()

    args.weights_dir.mkdir(parents=True, exist_ok=True)
    downloaded = Path(
        hf_hub_download(
            repo_id=args.repo_id,
            filename=args.model_file,
            revision=args.revision,
            local_dir=args.weights_dir,
        )
    )
    expected = args.weights_dir / args.model_file
    if not expected.is_file():
        raise FileNotFoundError(
            f"Hugging Face reported {downloaded}, but expected model file is missing: {expected}"
        )
    if expected.stat().st_size == 0:
        raise RuntimeError(f"Downloaded model file is empty: {expected}")
    print(f"[ok] Resonate model: {expected.resolve()}")
    print(f"[ok] Size: {expected.stat().st_size / (1024**3):.2f} GiB")


if __name__ == "__main__":
    main()
