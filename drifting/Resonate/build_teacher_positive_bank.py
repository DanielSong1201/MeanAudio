#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
from tqdm import tqdm

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drifting.Resonate.config import RESONATE_CONFIG
from drifting.Resonate.data import ResonateNpzDataset
from drifting.Resonate.model import build_resonate_model, freeze_resonate_teacher
from drifting.Resonate.progress import setup_tqdm_logger


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as file:
            np.savez(file, **arrays)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def setup_distributed(device_name: str) -> tuple[bool, int, int, int, torch.device]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    enabled = world_size > 1
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA positive generation requested, but torch.cuda.is_available() is False"
        )
    if enabled:
        if device_name != "cuda":
            raise ValueError("Distributed positive generation requires --device cuda")
        timeout = timedelta(minutes=int(os.environ.get("DDP_TIMEOUT_MINUTES", "180")))
        dist.init_process_group(backend="nccl", timeout=timeout)
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(device_name)
        if device.type == "cuda":
            torch.cuda.set_device(0)
    return enabled, rank, local_rank, world_size, device


def euler_sample(
    *,
    model,
    noise: torch.Tensor,
    conditions,
    empty_conditions,
    num_steps: int,
    cfg_strength: float,
) -> torch.Tensor:
    """Run Resonate's reverse-flow Euler sampler and return normalized latents."""

    if num_steps < 1:
        raise ValueError("num_steps must be positive")
    latents = noise
    timesteps = torch.linspace(
        1.0,
        0.0,
        num_steps + 1,
        device=noise.device,
        dtype=torch.float32,
    )
    for index, timestep in enumerate(timesteps[:-1]):
        flow = model.ode_wrapper(
            timestep,
            latents,
            conditions,
            empty_conditions,
            cfg_strength,
        )
        latents = latents + (timesteps[index + 1] - timestep) * flow
    return latents


def generate_teacher_positive_bank(args: argparse.Namespace) -> None:
    distributed, rank, _, world_size, device = setup_distributed(args.device)
    is_main = rank == 0
    logger = setup_tqdm_logger(
        "drifting.resonate.positives",
        enabled=is_main,
        level=args.log_level,
    )
    if args.positives_per_condition < 1:
        raise ValueError("--positives-per-condition must be positive")
    if args.num_steps < 1:
        raise ValueError("--num-steps must be positive")
    if not args.teacher_weights.is_file():
        raise FileNotFoundError(
            f"Missing Resonate checkpoint: {args.teacher_weights}. Run "
            "bash drifting/scripts/resonate/download_resonate_model.sh first."
        )

    dataset = ResonateNpzDataset(tsv_path=args.tsv, npz_dir=args.npz_dir)
    num_items = len(dataset)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        num_items = min(num_items, args.limit)

    data_complete = args.npz_dir / "complete.json"
    if not data_complete.is_file() and not args.allow_incomplete_data:
        raise FileNotFoundError(
            f"Processed dataset has no completion marker: {data_complete}. "
            "Finish preprocessing or pass --allow-incomplete-data for a deliberate partial run."
        )

    config = {
        "version": 1,
        "tsv": str(args.tsv),
        "npz_dir": str(args.npz_dir),
        "num_items": num_items,
        "teacher_weights": str(args.teacher_weights),
        "teacher_variant": RESONATE_CONFIG.model_name,
        "positives_per_condition": args.positives_per_condition,
        "sampler": "flow_matching_euler",
        "num_steps": args.num_steps,
        "cfg_strength": args.cfg_strength,
        "seed": args.seed,
        "normalized_latents": True,
        "storage_dtype": args.storage_dtype,
    }
    config_path = args.output_dir / "config.json"
    complete_path = args.output_dir / "complete.json"
    if config_path.exists() and not args.overwrite:
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config:
            raise ValueError(
                f"Existing positive-bank config differs: {config_path}. "
                "Use another --output-dir or pass --overwrite."
            )
    if is_main:
        args.output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(config_path, config)
        complete_path.unlink(missing_ok=True)
        logger.info("[positive-bank] output=%s", args.output_dir)
        logger.info(
            f"[positive-bank] items={num_items} positives={args.positives_per_condition} "
            f"steps={args.num_steps} cfg={args.cfg_strength}"
        )
    if distributed:
        dist.barrier()

    pending_indices = [
        index
        for index in range(rank, num_items, world_size)
        if args.overwrite or not (args.output_dir / f"{index}.npz").is_file()
    ]
    if is_main:
        remaining_total = sum(
            args.overwrite or not (args.output_dir / f"{index}.npz").is_file()
            for index in range(num_items)
        )
        logger.info(
            "[positive-bank] remaining=%d existing=%d total=%d",
            remaining_total,
            num_items - remaining_total,
            num_items,
        )
    if not args.overwrite and all(
        (args.output_dir / f"{index}.npz").is_file() for index in range(num_items)
    ):
        if is_main:
            atomic_write_json(complete_path, config)
            logger.info(
                "[ok] Resonate teacher-positive bank already complete: %s",
                args.output_dir,
            )
        if distributed:
            dist.barrier()
            dist.destroy_process_group()
        return

    model_dtype = torch.bfloat16 if args.amp and device.type == "cuda" else torch.float32
    teacher = build_resonate_model(
        weights_path=args.teacher_weights,
        map_location="cpu",
        device=device,
        dtype=model_dtype,
        use_rope=True,
    )
    freeze_resonate_teacher(teacher)
    storage_dtype = np.float16 if args.storage_dtype == "float16" else np.float32

    progress = tqdm(
        pending_indices,
        total=len(pending_indices),
        desc=f"resonate-positives-rank{rank}",
        disable=not is_main,
        dynamic_ncols=True,
        unit="prompt",
    )
    with torch.inference_mode():
        for index in progress:
            output_path = args.output_dir / f"{index}.npz"
            item = dataset[index]
            latent_seq_len = int(item["mean"].shape[0])
            if int(item["mean"].shape[1]) != RESONATE_CONFIG.latent_dim:
                raise ValueError(
                    f"Sample {index} latent shape {tuple(item['mean'].shape)} does not "
                    f"match Resonate latent_dim={RESONATE_CONFIG.latent_dim}"
                )
            teacher.update_seq_lengths(latent_seq_len)
            text_f = item["text_features"].unsqueeze(0).to(
                device=device,
                dtype=model_dtype,
            )
            text_f_c = item["text_features_c"].unsqueeze(0).to(
                device=device,
                dtype=model_dtype,
            )
            text_f = text_f.expand(args.positives_per_condition, -1, -1)
            text_f_c = text_f_c.expand(args.positives_per_condition, -1)
            conditions = teacher.preprocess_conditions(text_f, text_f_c)
            empty_conditions = teacher.get_empty_conditions(args.positives_per_condition)

            generator = torch.Generator(device=device)
            generator.manual_seed(args.seed + index)
            noise = torch.randn(
                args.positives_per_condition,
                latent_seq_len,
                RESONATE_CONFIG.latent_dim,
                device=device,
                dtype=model_dtype,
                generator=generator,
            )
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=args.amp and device.type == "cuda",
            ):
                positives = euler_sample(
                    model=teacher,
                    noise=noise,
                    conditions=conditions,
                    empty_conditions=empty_conditions,
                    num_steps=args.num_steps,
                    cfg_strength=args.cfg_strength,
                )
            atomic_save_npz(
                output_path,
                latents_normalized=positives.float()
                .cpu()
                .numpy()
                .astype(storage_dtype),
                item_index=np.asarray(index, dtype=np.int64),
                item_id=np.asarray(item["id"]),
            )

    if distributed:
        dist.barrier()
    if is_main:
        missing_outputs = [
            index
            for index in range(num_items)
            if not (args.output_dir / f"{index}.npz").is_file()
        ]
        if missing_outputs:
            raise RuntimeError(
                f"Positive bank incomplete: {len(missing_outputs)} files missing; "
                f"first indices={missing_outputs[:20]}"
            )
        atomic_write_json(complete_path, config)
        logger.info("[ok] Resonate teacher-positive bank complete: %s", args.output_dir)
    if distributed:
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate three prompt-matched 25-step Resonate teacher positives."
    )
    parser.add_argument(
        "--tsv",
        type=Path,
        default=Path("data/audiocaps_resonate/train.tsv"),
    )
    parser.add_argument(
        "--npz-dir",
        type=Path,
        default=Path("data/audiocaps_resonate/train-npz-flant5-44k"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/audiocaps_resonate/"
            "train-teacher-positives-resonate-grpo-25step-cfg4.5"
        ),
    )
    parser.add_argument(
        "--teacher-weights",
        type=Path,
        default=RESONATE_CONFIG.teacher_weights,
    )
    parser.add_argument("--positives-per-condition", type=int, default=3)
    parser.add_argument("--num-steps", type=int, default=25)
    parser.add_argument("--cfg-strength", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=20260702)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--storage-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--allow-incomplete-data", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


if __name__ == "__main__":
    generate_teacher_positive_bank(parse_args())
