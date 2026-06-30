#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import hf_hub_download
from transformers import BertTokenizer

from prepare_hf_ckpts import LAION_CLAP_FILE, MEANAUDIO_FILES, MSCLAP_FILE, MSCLAP_REPO


def main() -> None:
    parser = argparse.ArgumentParser(description="Verify drifting eval assets without network access.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--ckpt-root", type=Path, default=Path("drifting/ckpts"))
    args = parser.parse_args()

    if os.environ.get("HF_HUB_OFFLINE") != "1" or os.environ.get("TRANSFORMERS_OFFLINE") != "1":
        raise RuntimeError("Offline test requires HF_HUB_OFFLINE=1 and TRANSFORMERS_OFFLINE=1")

    repo_root = args.repo_root.resolve()
    ckpt_root = args.ckpt_root.resolve()
    hub_cache = ckpt_root / "huggingface" / "hub"

    tokenizer = BertTokenizer.from_pretrained("bert-base-uncased", local_files_only=True)
    token_ids = tokenizer.encode("offline checkpoint test", add_special_tokens=True)
    if not token_ids:
        raise RuntimeError("BERT tokenizer produced no token ids")
    print(f"[ok] bert-base-uncased tokenizer loaded locally: tokens={len(token_ids)}")

    msclap_path = Path(
        hf_hub_download(
            repo_id=MSCLAP_REPO,
            filename=MSCLAP_FILE,
            cache_dir=hub_cache,
            local_files_only=True,
        )
    )
    print(f"[ok] MS-CLAP resolved locally: {msclap_path}")

    for filename in MEANAUDIO_FILES:
        local_path = ckpt_root / "meanaudio" / filename
        if not local_path.exists():
            raise FileNotFoundError(f"Missing local MeanAudio asset: {local_path}")
        weights_path = repo_root / "weights" / filename
        if not weights_path.exists():
            raise FileNotFoundError(f"Missing MeanAudio weights link: {weights_path}")
        print(f"[ok] MeanAudio asset resolved locally: {local_path.resolve()}")

    laion_path = ckpt_root / "laion-clap" / LAION_CLAP_FILE
    if not laion_path.exists():
        raise FileNotFoundError(f"Missing local LAION-CLAP checkpoint: {laion_path}")
    print(f"[ok] LAION-CLAP resolved locally: {laion_path.resolve()}")

    av_benchmark_path = repo_root / "av-benchmark" / "weights" / LAION_CLAP_FILE
    if (repo_root / "av-benchmark").exists() and not av_benchmark_path.exists():
        raise FileNotFoundError(f"Missing av-benchmark LAION-CLAP link: {av_benchmark_path}")
    if av_benchmark_path.exists():
        print(f"[ok] av-benchmark checkpoint link: {av_benchmark_path.resolve()}")

    import laion_clap  # noqa: F401

    print("[ok] laion_clap imported with network disabled")
    print("[ok] all Hugging Face eval assets are available from drifting/ckpts")


if __name__ == "__main__":
    main()
