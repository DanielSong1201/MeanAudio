#!/usr/bin/env python3
"""Phase-0 asset and data sanity checks for FluxAudio distillation."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
import yaml


def require_path(path: Path, kind: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"Missing {kind}: {path}")
    print(f"[ok] {kind}: {path}")


def check_npz(npz_dir: Path) -> None:
    require_path(npz_dir, "npz directory")
    first = npz_dir / "0.npz"
    require_path(first, "sample npz")

    data = np.load(first)
    required = ["mean", "std", "text_features", "text_features_c"]
    for key in required:
        if key not in data:
            raise KeyError(f"{first} missing key: {key}")
        print(f"[ok] {first.name}:{key} shape={data[key].shape} dtype={data[key].dtype}")


def count_tsv(tsv_path: Path) -> int:
    require_path(tsv_path, "tsv")
    with tsv_path.open("r", newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")
        rows = list(reader)
    if not rows:
        raise ValueError(f"No rows in {tsv_path}")
    for key in ("id", "caption"):
        if key not in rows[0]:
            raise KeyError(f"{tsv_path} missing column: {key}")
    print(f"[ok] {tsv_path} rows={len(rows)} columns={reader.fieldnames}")
    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--data-config", type=Path, default=Path("config/data/t5_clap.yaml"))
    parser.add_argument("--teacher-weights", type=Path, default=Path("weights/fluxaudio_s_full.pth"))
    parser.add_argument("--baseline-weights", type=Path, default=Path("weights/meanaudio_s_full.pth"))
    parser.add_argument("--vae-weights", type=Path, default=Path("weights/v1-16.pth"))
    parser.add_argument("--vocoder-weights", type=Path, default=Path("weights/best_netG.pt"))
    args = parser.parse_args()

    root = args.root.resolve()
    print(f"[info] project root: {root}")

    for path, kind in [
        (args.teacher_weights, "FluxAudio-S-Full teacher weights"),
        (args.baseline_weights, "MeanAudio-S-Full baseline weights"),
        (args.vae_weights, "16k VAE weights"),
        (args.vocoder_weights, "BigVGAN vocoder weights"),
        (Path("sets/test-audiocaps.tsv"), "eval prompt tsv"),
    ]:
        require_path(root / path, kind)

    require_path(root / args.data_config, "data config")
    with (root / args.data_config).open("r") as f:
        data_cfg = yaml.safe_load(f)

    for split_name in ["AudioCaps_npz", "AudioCaps_val_npz", "AudioCaps_test_npz"]:
        split = data_cfg[split_name]
        print(f"\n[check] {split_name}")
        row_count = count_tsv(root / split["tsv"])
        check_npz(root / split["npz_dir"])
        npz_count = len(list((root / split["npz_dir"]).glob("*.npz")))
        print(f"[ok] npz count={npz_count}")
        if npz_count != row_count:
            raise ValueError(f"{split_name} row/npz mismatch: rows={row_count}, npz={npz_count}")
        if split.get("gt_cache"):
            require_path(root / split["gt_cache"], f"{split_name} gt cache")

    for stat_path, name in [(data_cfg["latent_mean"], "latent_mean"), (data_cfg["latent_std"], "latent_std")]:
        tensor = torch.load(root / stat_path, map_location="cpu", weights_only=True)
        print(f"[ok] {name}: {stat_path} shape={tuple(tensor.shape)} dtype={tensor.dtype}")

    print("\n[done] Phase-0 assets look ready.")


if __name__ == "__main__":
    main()
