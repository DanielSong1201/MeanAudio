#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

from drifting.flux.train import (
    AudioCapsNpzDataset,
    cleanup_distributed,
    freeze_module,
    load_data_config,
    load_torch,
    setup_distributed,
    setup_logger,
)
from meanaudio.model.mean_flow import MeanFlow
from meanaudio.model.networks import get_mean_audio


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary_path.open("wb") as f:
            np.savez(f, **arrays)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def atomic_write_json(path: Path, payload: dict[str, object]) -> None:
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary_path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build prompt-matched 25-step MeanAudio-L positives for AudioCaps."
    )
    parser.add_argument("--data-config", type=Path, default=Path("config/data/t5_clap.yaml"))
    parser.add_argument("--train-split", default="AudioCaps_npz")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--teacher-weights", type=Path, default=Path("weights/meanaudio_l_full.pth"))
    parser.add_argument("--target-latent-mean", type=Path, default=Path("sets/latent_mean.pt"))
    parser.add_argument("--target-latent-std", type=Path, default=Path("sets/latent_std.pt"))
    parser.add_argument("--positives-per-condition", type=int, default=3)
    parser.add_argument("--num-steps", type=int, default=25)
    parser.add_argument("--cfg-strength", type=float, default=6.0)
    parser.add_argument("--seed", type=int, default=20260507)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--use-rope", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    if args.positives_per_condition < 1:
        raise ValueError("--positives-per-condition must be >= 1")
    if args.num_steps < 1:
        raise ValueError("--num-steps must be >= 1")
    if not args.teacher_weights.exists():
        raise FileNotFoundError(
            f"Missing MeanAudio-L checkpoint: {args.teacher_weights}. Download it from "
            "https://huggingface.co/AndreasXi/MeanAudio/resolve/main/meanaudio_l_full.pth"
        )

    distributed, rank, local_rank, world_size = setup_distributed()
    is_main = rank == 0
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    device = torch.device(f"cuda:{local_rank}" if distributed else args.device)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(args.output_dir, enabled=is_main)

    data_cfg = load_data_config(args.data_config, args.train_split)
    split_cfg = data_cfg[args.train_split]
    dataset = AudioCapsNpzDataset(
        tsv_path=Path(split_cfg["tsv"]),
        npz_dir=Path(split_cfg["npz_dir"]),
    )

    use_amp = args.amp and device.type == "cuda"
    model_dtype = torch.bfloat16 if use_amp else torch.float32
    teacher = get_mean_audio(
        "meanaudio_l",
        use_rope=args.use_rope,
        text_c_dim=512,
    ).to(device=device, dtype=model_dtype)
    teacher.load_weights(load_torch(args.teacher_weights, "cpu"))
    freeze_module(teacher)
    target_latent_mean = load_torch(args.target_latent_mean, "cpu").to(
        device=device,
        dtype=model_dtype,
    )
    target_latent_std = load_torch(args.target_latent_std, "cpu").to(
        device=device,
        dtype=model_dtype,
    )
    autocast_dtype = torch.bfloat16 if use_amp else torch.float32
    mean_flow = MeanFlow(steps=args.num_steps)

    config = {
        "version": 2,
        "train_split": args.train_split,
        "tsv": str(split_cfg["tsv"]),
        "npz_dir": str(split_cfg["npz_dir"]),
        "num_items": len(dataset),
        "positives_per_condition": args.positives_per_condition,
        "teacher_weights": str(args.teacher_weights),
        "teacher_variant": "meanaudio_l",
        "sampler": "meanflow",
        "num_steps": args.num_steps,
        "cfg_strength": args.cfg_strength,
        "seed": args.seed,
        "normalized_latents": True,
        "target_latent_mean": str(args.target_latent_mean),
        "target_latent_std": str(args.target_latent_std),
        "use_rope": args.use_rope,
    }
    config_path = args.output_dir / "config.json"
    if config_path.exists() and not args.overwrite:
        existing_config = json.loads(config_path.read_text())
        if existing_config != config:
            raise ValueError(
                f"Existing bank config does not match this run: {config_path}. "
                "Use a different --output-dir, or pass --overwrite to rebuild it."
            )
    if is_main:
        atomic_write_json(config_path, config)
        (args.output_dir / "complete.json").unlink(missing_ok=True)
        logger.info("Teacher-positive bank config: %s", config)
    if distributed:
        dist.barrier()

    indices = range(rank, len(dataset), world_size)
    progress = tqdm(
        indices,
        total=(len(dataset) + world_size - 1 - rank) // world_size,
        desc=f"teacher-positives-rank{rank}",
        disable=not is_main,
        dynamic_ncols=True,
    )
    for idx in progress:
        output_path = args.output_dir / f"{idx}.npz"
        if output_path.exists() and not args.overwrite:
            continue

        item = dataset[idx]
        text_f = item["text_features"].unsqueeze(0).to(device=device)
        text_f_c = item["text_features_c"].unsqueeze(0).to(device=device)
        text_f = text_f.expand(args.positives_per_condition, -1, -1)
        text_f_c = text_f_c.expand(args.positives_per_condition, -1)

        generator = torch.Generator(device=device)
        generator.manual_seed(args.seed + idx)
        x_noise = torch.randn(
            args.positives_per_condition,
            teacher.latent_seq_len,
            teacher.latent_dim,
            device=device,
            dtype=torch.float32,
            generator=generator,
        )
        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=use_amp,
        ):
            conditions = teacher.preprocess_conditions(text_f, text_f_c)
            empty_conditions = teacher.get_empty_conditions(args.positives_per_condition)
            cfg_ode_wrapper = lambda t, r, x: teacher.ode_wrapper(
                t,
                r,
                x,
                conditions,
                empty_conditions,
                args.cfg_strength,
            )
            teacher_latents = mean_flow.to_data(cfg_ode_wrapper, x_noise)
            raw_latents = (
                teacher_latents * teacher.latent_std
                + teacher.latent_mean
            )
            teacher_latents = (
                raw_latents - target_latent_mean
            ) / target_latent_std

        latents_np = teacher_latents.detach().float().cpu().numpy().astype(np.float16)
        atomic_save_npz(
            output_path,
            latents_normalized=latents_np,
            item_index=np.asarray(idx, dtype=np.int64),
            item_id=np.asarray(item["id"]),
        )

    if distributed:
        dist.barrier()
    if is_main:
        missing = [idx for idx in range(len(dataset)) if not (args.output_dir / f"{idx}.npz").exists()]
        if missing:
            raise RuntimeError(
                f"Teacher-positive bank is incomplete: {len(missing)} files missing; "
                f"first missing indices={missing[:20]}"
            )
        atomic_write_json(args.output_dir / "complete.json", config)
        logger.info("Teacher-positive bank complete: %s (%d items)", args.output_dir, len(dataset))
    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
