#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
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


def setup_worker(device_name: str) -> tuple[bool, int, int, int, torch.device]:
    """Configure one independent torchrun worker without a process group.

    Positive-bank workers only write disjoint files. Initializing NCCL would
    add collectives solely for start/end barriers, which can time out when
    shards finish hours apart.
    """

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    multi_process = world_size > 1
    if device_name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA positive generation requested, but torch.cuda.is_available() is False"
        )
    if multi_process:
        if device_name != "cuda":
            raise ValueError("Multi-process positive generation requires --device cuda")
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(device_name)
        if device.type == "cuda":
            torch.cuda.set_device(0)
    return multi_process, rank, local_rank, world_size, device


def physical_gpu_label(local_rank: int) -> str:
    visible_devices = [
        item.strip()
        for item in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
        if item.strip()
    ]
    if local_rank < len(visible_devices):
        return visible_devices[local_rank]
    return str(local_rank)


def worker_run_id() -> str:
    raw = os.environ.get("POSITIVE_RUN_ID", "").strip()
    if not raw:
        raw = os.environ.get("TORCHELASTIC_RUN_ID", "").strip()
    if not raw or raw == "none":
        raw = (
            f"{os.environ.get('MASTER_ADDR', 'local')}_"
            f"{os.environ.get('MASTER_PORT', 'single')}_"
            f"{os.environ.get('WORLD_SIZE', '1')}"
        )
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", raw)


def count_missing_outputs(output_dir: Path, num_items: int) -> tuple[int, list[int]]:
    missing_count = 0
    first_missing: list[int] = []
    for index in range(num_items):
        if not (output_dir / f"{index}.npz").is_file():
            missing_count += 1
            if len(first_missing) < 20:
                first_missing.append(index)
    return missing_count, first_missing


def wait_for_worker_markers(
    *,
    run_dir: Path,
    world_size: int,
    poll_seconds: float,
    timeout_minutes: float,
    logger,
) -> None:
    start_time = time.monotonic()
    last_reported: tuple[int, ...] | None = None
    while True:
        missing_ranks = tuple(
            rank
            for rank in range(world_size)
            if not (run_dir / f"rank_{rank}.json").is_file()
        )
        if not missing_ranks:
            return
        elapsed = time.monotonic() - start_time
        if timeout_minutes > 0 and elapsed > timeout_minutes * 60:
            raise TimeoutError(
                f"Timed out waiting for positive-bank workers {missing_ranks} "
                f"after {elapsed / 60:.1f} minutes"
            )
        if missing_ranks != last_reported:
            logger.info(
                "[positive-bank] waiting for worker ranks=%s elapsed=%.1fmin",
                list(missing_ranks),
                elapsed / 60,
            )
            last_reported = missing_ranks
        time.sleep(poll_seconds)


def wait_for_file(
    path: Path,
    *,
    poll_seconds: float,
    timeout_minutes: float,
) -> None:
    start_time = time.monotonic()
    while not path.is_file():
        elapsed = time.monotonic() - start_time
        if timeout_minutes > 0 and elapsed > timeout_minutes * 60:
            raise TimeoutError(
                f"Timed out waiting for coordinator file {path} "
                f"after {elapsed / 60:.1f} minutes"
            )
        time.sleep(poll_seconds)


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
    multi_process, rank, local_rank, world_size, device = setup_worker(args.device)
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
    if args.completion_poll_seconds <= 0:
        raise ValueError("--completion-poll-seconds must be positive")
    if args.completion_timeout_minutes < 0:
        raise ValueError("--completion-timeout-minutes must be non-negative")
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
    run_dir = args.output_dir / ".run_state" / worker_run_id()
    work_path = run_dir / "work.json"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    run_dir.mkdir(parents=True, exist_ok=True)
    if config_path.exists() and not args.overwrite:
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config:
            raise ValueError(
                f"Existing positive-bank config differs: {config_path}. "
                "Use another --output-dir or pass --overwrite."
            )
    if is_main:
        atomic_write_json(config_path, config)
        complete_path.unlink(missing_ok=True)
        logger.info("[positive-bank] output=%s", args.output_dir)
        logger.info(
            f"[positive-bank] items={num_items} positives={args.positives_per_condition} "
            f"steps={args.num_steps} cfg={args.cfg_strength}"
        )

    if is_main:
        pending_all = [
            index
            for index in range(num_items)
            if args.overwrite or not (args.output_dir / f"{index}.npz").is_file()
        ]
        atomic_write_json(
            work_path,
            {
                "world_size": world_size,
                "overwrite": args.overwrite,
                "pending_indices": pending_all,
            },
        )
        remaining_total = len(pending_all)
        logger.info(
            "[positive-bank] remaining=%d existing=%d total=%d",
            remaining_total,
            num_items - remaining_total,
            num_items,
        )
        if remaining_total:
            logger.info(
                "[positive-bank] resume enabled: existing files are kept and only "
                "missing indices are generated"
            )
    else:
        wait_for_file(
            work_path,
            poll_seconds=args.completion_poll_seconds,
            timeout_minutes=args.completion_timeout_minutes,
        )
    work = json.loads(work_path.read_text(encoding="utf-8"))
    if work.get("world_size") != world_size:
        raise ValueError(
            f"Positive work manifest world_size={work.get('world_size')} "
            f"does not match worker world_size={world_size}"
        )
    pending_all = work.get("pending_indices")
    if not isinstance(pending_all, list) or not all(
        isinstance(index, int) for index in pending_all
    ):
        raise ValueError(f"Invalid pending index list in {work_path}")
    # Re-shard the current missing-file snapshot, rather than retaining the
    # original index modulo assignment. This keeps all GPUs busy when resuming
    # after only one rank was interrupted.
    pending_indices = pending_all[rank::world_size]

    progress = tqdm(
        pending_indices,
        total=len(pending_indices),
        desc=(
            f"resonate-positives-gpu{physical_gpu_label(local_rank)}"
            f"-rank{rank}"
        ),
        disable=not args.progress_all_ranks and not is_main,
        position=rank if args.progress_all_ranks else 0,
        leave=True,
        dynamic_ncols=True,
        unit="prompt",
    )
    if pending_indices:
        model_dtype = (
            torch.bfloat16 if args.amp and device.type == "cuda" else torch.float32
        )
        teacher = build_resonate_model(
            weights_path=args.teacher_weights,
            map_location="cpu",
            device=device,
            dtype=model_dtype,
            use_rope=True,
        )
        freeze_resonate_teacher(teacher)
        storage_dtype = (
            np.float16 if args.storage_dtype == "float16" else np.float32
        )

        with torch.inference_mode():
            for index in progress:
                output_path = args.output_dir / f"{index}.npz"
                item = dataset[index]
                latent_seq_len = int(item["mean"].shape[0])
                if int(item["mean"].shape[1]) != RESONATE_CONFIG.latent_dim:
                    raise ValueError(
                        f"Sample {index} latent shape {tuple(item['mean'].shape)} "
                        f"does not match Resonate "
                        f"latent_dim={RESONATE_CONFIG.latent_dim}"
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
                empty_conditions = teacher.get_empty_conditions(
                    args.positives_per_condition
                )

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
    progress.close()

    atomic_write_json(
        run_dir / f"rank_{rank}.json",
        {
            "rank": rank,
            "world_size": world_size,
            "generated": len(pending_indices),
            "pid": os.getpid(),
        },
    )
    if is_main:
        if multi_process:
            wait_for_worker_markers(
                run_dir=run_dir,
                world_size=world_size,
                poll_seconds=args.completion_poll_seconds,
                timeout_minutes=args.completion_timeout_minutes,
                logger=logger,
            )
        missing_count, first_missing = count_missing_outputs(
            args.output_dir,
            num_items,
        )
        if missing_count:
            raise RuntimeError(
                f"Positive bank incomplete: {missing_count} files missing; "
                f"first indices={first_missing}"
            )
        atomic_write_json(complete_path, config)
        logger.info("[ok] Resonate teacher-positive bank complete: %s", args.output_dir)
        shutil.rmtree(run_dir, ignore_errors=True)


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
        "--completion-poll-seconds",
        type=float,
        default=30.0,
        help="How often rank0 checks independent worker completion markers.",
    )
    parser.add_argument(
        "--completion-timeout-minutes",
        type=float,
        default=0.0,
        help="Rank0 file-wait timeout; 0 means no timeout.",
    )
    parser.add_argument(
        "--progress-all-ranks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Display one fixed terminal progress row for every GPU/rank.",
    )
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


if __name__ == "__main__":
    generate_teacher_positive_bank(parse_args())
