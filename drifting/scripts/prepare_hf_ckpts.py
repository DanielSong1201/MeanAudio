#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import logging
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download


log = logging.getLogger("drifting.prepare_hf_ckpts")

BERT_REPO = "bert-base-uncased"
BERT_FILES = (
    "config.json",
    "special_tokens_map.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.txt",
)
MEANAUDIO_REPO = "AndreasXi/MeanAudio"
MEANAUDIO_FILES = (
    "fluxaudio_s_full.pth",
    "v1-16.pth",
    "best_netG.pt",
    "empty_string_t5.pth",
    "empty_string_clap_c.pth",
    "music_speech_audioset_epoch_15_esc_89.98.pt",
)
MSCLAP_REPO = "microsoft/msclap"
MSCLAP_FILE = "CLAP_weights_2023.pth"
LAION_CLAP_FILE = "music_speech_audioset_epoch_15_esc_89.98.pt"


def replace_symlink(path: Path, target: Path) -> None:
    if path.is_symlink():
        if path.resolve() == target.resolve():
            return
        path.unlink()
    elif path.exists():
        log.info("Keeping existing local asset: %s", path)
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.symlink_to(os.path.relpath(target, path.parent))
    log.info("Linked %s -> %s", path, target)


def download_file(
    *,
    repo_id: str,
    filename: str,
    cache_dir: Path,
    local_path: Path,
    local_files_only: bool,
) -> Path:
    if local_path.exists():
        log.info("Found cached asset: %s", local_path)
        return local_path.resolve()
    downloaded = Path(
        hf_hub_download(
            repo_id=repo_id,
            filename=filename,
            cache_dir=cache_dir,
            local_files_only=local_files_only,
        )
    )
    replace_symlink(local_path, downloaded)
    return downloaded


def adopt_existing_file(local_path: Path, existing_path: Path) -> bool:
    if local_path.exists() or not existing_path.exists():
        return local_path.exists()
    local_path.parent.mkdir(parents=True, exist_ok=True)
    source = existing_path.resolve()
    try:
        os.link(source, local_path)
        log.info("Reused existing asset with a hard link: %s", existing_path)
    except OSError:
        shutil.copy2(source, local_path)
        log.info("Copied existing asset into checkpoint directory: %s", existing_path)
    return True


def prepare_assets(repo_root: Path, ckpt_root: Path, *, local_files_only: bool) -> None:
    hf_home = ckpt_root / "huggingface"
    hub_cache = hf_home / "hub"
    hub_cache.mkdir(parents=True, exist_ok=True)

    log.info("Checking Hugging Face assets under %s", ckpt_root)
    local_bert_path = ckpt_root / "bert-base-uncased"
    if local_bert_path.exists() and all((local_bert_path / name).exists() for name in BERT_FILES):
        bert_snapshot = local_bert_path.resolve()
        log.info("Found cached BERT tokenizer: %s", local_bert_path)
    else:
        bert_snapshot = Path(
            snapshot_download(
                repo_id=BERT_REPO,
                cache_dir=hub_cache,
                allow_patterns=list(BERT_FILES),
                local_files_only=local_files_only,
            )
        )
        replace_symlink(local_bert_path, bert_snapshot)
    log.info("BERT tokenizer ready: %s", bert_snapshot)

    meanaudio_paths: dict[str, Path] = {}
    for filename in MEANAUDIO_FILES:
        local_path = ckpt_root / "meanaudio" / filename
        adopt_existing_file(local_path, repo_root / "weights" / filename)
        path = download_file(
            repo_id=MEANAUDIO_REPO,
            filename=filename,
            cache_dir=hub_cache,
            local_path=local_path,
            local_files_only=local_files_only,
        )
        meanaudio_paths[filename] = path
        replace_symlink(repo_root / "weights" / filename, ckpt_root / "meanaudio" / filename)
        log.info("MeanAudio asset ready: %s", path)

    msclap_path = download_file(
        repo_id=MSCLAP_REPO,
        filename=MSCLAP_FILE,
        cache_dir=hub_cache,
        local_path=ckpt_root / "msclap" / MSCLAP_FILE,
        local_files_only=local_files_only,
    )
    log.info("MS-CLAP ready: %s", msclap_path)

    local_laion_path = ckpt_root / "laion-clap" / LAION_CLAP_FILE
    replace_symlink(local_laion_path, ckpt_root / "meanaudio" / LAION_CLAP_FILE)
    log.info("LAION-CLAP ready: %s", meanaudio_paths[LAION_CLAP_FILE])

    av_benchmark = repo_root / "av-benchmark"
    if av_benchmark.exists():
        replace_symlink(av_benchmark / "weights" / LAION_CLAP_FILE, local_laion_path)
    else:
        log.warning("av-benchmark is not present yet; skipping its LAION-CLAP link.")

    log.info("All required Hugging Face assets are ready.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare drifting Hugging Face assets once under drifting/ckpts.")
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--ckpt-root", type=Path, default=Path("drifting/ckpts"))
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | INFO | %(message)s")
    repo_root = args.repo_root.resolve()
    ckpt_root = args.ckpt_root.resolve()
    ckpt_root.mkdir(parents=True, exist_ok=True)

    lock_path = ckpt_root / ".prepare_hf_ckpts.lock"
    with lock_path.open("w") as lock_file:
        log.info("Waiting for Hugging Face asset lock: %s", lock_path)
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        log.info("Acquired Hugging Face asset lock.")
        prepare_assets(repo_root, ckpt_root, local_files_only=args.local_files_only)


if __name__ == "__main__":
    main()
