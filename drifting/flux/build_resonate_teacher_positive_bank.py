#!/usr/bin/env python3
"""Build a training-ready teacher-positive bank with official Resonate-GRPO.

For every prompt this script generates temporary 44.1 kHz FLAC files, encodes
them immediately with MeanAudio's 16 kHz VAE, atomically stores the normalized
latent tensor expected by drifting/flux/train.py, and removes the FLAC files.
Under torchrun, workers shard prompts by rank and rank 0 aggregates progress
through shared files without initializing torch.distributed or NCCL.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import threading
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from drifting.flux.build_resonate_positive_audio import (
    PromptRow,
    check_assets,
    generate_audio_batch,
    load_resonate_runtime,
    read_manifest,
    repository_revision,
    resolve_from,
    seed_for,
    select_rows,
    sha256_file,
)


LOG = logging.getLogger("resonate-teacher-positive-bank")
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class WorkerContext:
    rank: int
    local_rank: int
    world_size: int
    run_id: str

    @property
    def is_main(self) -> bool:
        return self.rank == 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate official Resonate-GRPO audio and immediately convert it "
            "to the normalized MeanAudio latents consumed by Flux training."
        )
    )
    parser.add_argument("--resonate-root", type=Path, default=Path("../Resonate"))
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/audiocaps/train-memmap.tsv"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/audiocaps/"
            "train-teacher-positives-resonate-grpo-25step-cfg4.5"
        ),
    )
    parser.add_argument("--checkpoint", type=Path, default=None)
    parser.add_argument(
        "--config-name",
        default="GRPO_flant5_44kMMVAE_fluxaudio_audiocaps_qwen25omni_semantic",
    )
    parser.add_argument("--target-vae-weights", type=Path, default=Path("weights/v1-16.pth"))
    parser.add_argument("--target-latent-mean", type=Path, default=Path("sets/latent_mean.pt"))
    parser.add_argument("--target-latent-std", type=Path, default=Path("sets/latent_std.pt"))
    parser.add_argument("--positives-per-condition", type=int, default=3)
    parser.add_argument("--num-steps", type=int, default=25)
    parser.add_argument("--cfg-strength", type=float, default=4.5)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--base-seed", type=int, default=20260802)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--download-if-missing", action="store_true")
    parser.add_argument("--full-precision", action="store_true")
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument("--end-index", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--progress-poll-interval", type=float, default=1.0)
    parser.add_argument("--coordination-timeout-minutes", type=float, default=720.0)
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO")
    return parser.parse_args()


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        temporary.write_text(text, encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def atomic_save_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    try:
        with temporary.open("wb") as file:
            np.savez(file, **arrays)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def load_torch(path: Path, torch: Any) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def require_file(path: Path, description: str) -> None:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"Missing {description}: {path}")


def get_worker_context() -> WorkerContext:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if world_size < 1:
        raise ValueError(f"Invalid WORLD_SIZE={world_size}")
    if not 0 <= rank < world_size:
        raise ValueError(f"Invalid RANK={rank} for WORLD_SIZE={world_size}")
    run_id = (
        os.environ.get("BANK_RUN_ID")
        or os.environ.get("TORCHELASTIC_RUN_ID")
        or f"single-{os.getpid()}"
    )
    run_id = re.sub(r"[^A-Za-z0-9_.-]", "_", run_id)
    if not run_id:
        raise ValueError("BANK_RUN_ID resolved to an empty value")
    return WorkerContext(
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        run_id=run_id,
    )


def wait_for_paths(
    paths: Sequence[Path],
    *,
    timeout_seconds: float,
    poll_interval: float,
    description: str,
) -> None:
    deadline = time.monotonic() + timeout_seconds
    while True:
        missing = [path for path in paths if not path.exists()]
        if not missing:
            return
        if time.monotonic() >= deadline:
            raise TimeoutError(
                f"Timed out waiting for {description}; missing paths: "
                + ", ".join(str(path) for path in missing)
            )
        time.sleep(poll_interval)


def initialize_coordination_dir(
    output_dir: Path,
    worker: WorkerContext,
    *,
    timeout_seconds: float,
    poll_interval: float,
) -> Path:
    coordination_dir = output_dir / ".bank-build-progress" / worker.run_id
    ready_path = coordination_dir / "initialized.json"
    if worker.is_main:
        coordination_dir.mkdir(parents=True, exist_ok=True)
        for rank in range(worker.world_size):
            for suffix in ("status.json", "count", "done.json"):
                (coordination_dir / f"rank-{rank}.{suffix}").unlink(missing_ok=True)
        (coordination_dir / "assets-ready.json").unlink(missing_ok=True)
        (coordination_dir / "config-ready.json").unlink(missing_ok=True)
        (coordination_dir / "generation-ready.json").unlink(missing_ok=True)
        ready_path.unlink(missing_ok=True)
        atomic_write_json(
            ready_path,
            {
                "run_id": worker.run_id,
                "world_size": worker.world_size,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "coordination": "shared-files-no-torch-distributed",
            },
        )
    else:
        wait_for_paths(
            [ready_path],
            timeout_seconds=timeout_seconds,
            poll_interval=poll_interval,
            description="rank-0 coordination initialization",
        )
        payload = json.loads(ready_path.read_text(encoding="utf-8"))
        if payload.get("world_size") != worker.world_size:
            raise ValueError(
                f"Coordination WORLD_SIZE mismatch: {payload.get('world_size')} "
                f"!= {worker.world_size}"
            )
    return coordination_dir


def read_worker_count(path: Path) -> int:
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except (FileNotFoundError, OSError, ValueError):
        return 0


def expected_config(
    *,
    args: argparse.Namespace,
    resonate_root: Path,
    manifest: Path,
    output_dir: Path,
    checkpoint: Path,
    target_vae_weights: Path,
    target_latent_mean: Path,
    target_latent_std: Path,
    num_items: int,
    target_num_audio_frames: int,
    latent_tokens: int,
    latent_dim: int,
) -> dict[str, Any]:
    return {
        "version": SCHEMA_VERSION,
        "generator": "official-resonate-grpo-to-meanaudio-16k-vae",
        "train_split": "AudioCaps_npz",
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "output_dir": str(output_dir),
        "num_items": num_items,
        "positives_per_condition": args.positives_per_condition,
        "normalized_latents": True,
        "latent_shape": [latent_tokens, latent_dim],
        "storage_dtype": "float16",
        "vae_posterior": "mean",
        "target_sample_rate": 16_000,
        "target_num_audio_frames": target_num_audio_frames,
        "target_vae_weights": str(target_vae_weights),
        "target_vae_sha256": sha256_file(target_vae_weights),
        "target_latent_mean": str(target_latent_mean),
        "target_latent_mean_sha256": sha256_file(target_latent_mean),
        "target_latent_std": str(target_latent_std),
        "target_latent_std_sha256": sha256_file(target_latent_std),
        "resonate_root": str(resonate_root),
        "resonate_revision": repository_revision(resonate_root),
        "resonate_config_name": args.config_name,
        "resonate_checkpoint": str(checkpoint),
        "resonate_checkpoint_sha256": sha256_file(checkpoint),
        "num_steps": args.num_steps,
        "cfg_strength": args.cfg_strength,
        "duration": args.duration,
        "negative_prompt": args.negative_prompt,
        "base_seed": args.base_seed,
        "generation_dtype": "float32" if args.full_precision else "bfloat16",
        "temporary_audio_format": "flac",
        "temporary_audio_deleted_after_encoding": True,
    }


def ensure_compatible_config(path: Path, config: dict[str, Any]) -> None:
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != config:
            differing = sorted(
                key
                for key in set(existing) | set(config)
                if existing.get(key) != config.get(key)
            )
            raise ValueError(
                f"Existing bank config differs in {differing}: {path}. "
                "Use a new OUTPUT_DIR for a different recipe."
            )
    else:
        atomic_write_json(path, config)


def load_target_encoder(
    *,
    vae_weights: Path,
    latent_mean_path: Path,
    latent_std_path: Path,
    device: Any,
    torch: Any,
) -> dict[str, Any]:
    from meanaudio.ext.autoencoder.vae import get_my_vae
    from meanaudio.ext.mel_converter import get_mel_converter
    from meanaudio.model.sequence_config import CONFIG_16K

    vae = get_my_vae("16k").eval()
    vae.load_state_dict(load_torch(vae_weights, torch), strict=True)
    vae.remove_weight_norm()
    vae.requires_grad_(False)
    # Only the encoder is needed; dropping the decoder saves GPU memory.
    del vae.decoder
    vae = vae.to(device)
    mel_converter = get_mel_converter("16k").eval().to(device)
    latent_mean = load_torch(latent_mean_path, torch).float().reshape(1, 1, -1).to(device)
    latent_std = load_torch(latent_std_path, torch).float().reshape(1, 1, -1).to(device)
    if not torch.isfinite(latent_mean).all() or not torch.isfinite(latent_std).all():
        raise ValueError("Target latent statistics contain non-finite values")
    if not torch.all(latent_std > 0):
        raise ValueError("Target latent standard deviations must be positive")
    return {
        "vae": vae,
        "mel_converter": mel_converter,
        "latent_mean": latent_mean,
        "latent_std": latent_std,
        "sample_rate": CONFIG_16K.sampling_rate,
        "num_audio_frames": CONFIG_16K.num_audio_frames,
        "latent_tokens": CONFIG_16K.latent_seq_len,
        "latent_dim": int(latent_mean.numel()),
    }


def normalize_waveform_length(waveform: Any, num_frames: int, torch: Any) -> Any:
    waveform = waveform.mean(dim=0)
    if waveform.numel() < num_frames:
        waveform = torch.nn.functional.pad(waveform, (0, num_frames - waveform.numel()))
    else:
        waveform = waveform[:num_frames]
    return waveform


def save_temporary_flacs(
    *,
    row: PromptRow,
    audio_samples: Any,
    temporary_dir: Path,
    sample_rate: int,
    target_frames: int,
    torch: Any,
    torchaudio: Any,
) -> list[Path]:
    paths: list[Path] = []
    temporary_dir.mkdir(parents=True, exist_ok=True)
    for positive_index, audio in enumerate(audio_samples):
        if audio.ndim == 1:
            audio = audio.unsqueeze(0)
        elif audio.ndim != 2:
            raise ValueError(f"Unexpected Resonate waveform shape: {tuple(audio.shape)}")
        if audio.shape[0] != 1:
            audio = audio.mean(dim=0, keepdim=True)
        if audio.shape[-1] < target_frames:
            audio = torch.nn.functional.pad(audio, (0, target_frames - audio.shape[-1]))
        else:
            audio = audio[..., :target_frames]
        if not torch.isfinite(audio).all():
            raise ValueError(f"Non-finite waveform generated for index {row.index}")
        path = temporary_dir / f"{row.index:08d}_{positive_index:02d}.flac"
        partial = path.with_name(f".{path.stem}.{os.getpid()}.tmp.flac")
        try:
            torchaudio.save(partial, audio.clamp(-1.0, 1.0), sample_rate, format="flac")
            os.replace(partial, path)
        finally:
            partial.unlink(missing_ok=True)
        paths.append(path)
    return paths


def encode_flacs(
    paths: Sequence[Path],
    *,
    target: dict[str, Any],
    device: Any,
    use_amp: bool,
    torch: Any,
    torchaudio: Any,
) -> np.ndarray:
    waveforms = []
    for path in paths:
        waveform, source_rate = torchaudio.load(path)
        if source_rate != target["sample_rate"]:
            waveform = torchaudio.functional.resample(
                waveform,
                source_rate,
                target["sample_rate"],
                lowpass_filter_width=64,
                rolloff=0.9475937167399596,
                resampling_method="sinc_interp_kaiser",
                beta=14.769656459379492,
            )
        waveforms.append(
            normalize_waveform_length(
                waveform,
                target["num_audio_frames"],
                torch,
            )
        )
    waveform_batch = torch.stack(waveforms).to(device=device, non_blocking=True)
    with torch.inference_mode(), torch.autocast(
        device_type=device.type,
        dtype=torch.bfloat16,
        enabled=use_amp,
    ):
        mel = target["mel_converter"](waveform_batch)
        posterior = target["vae"].encode(mel)
        raw_latents = posterior.mean.transpose(1, 2)
    normalized = (
        raw_latents.float() - target["latent_mean"]
    ) / target["latent_std"]
    expected_shape = (
        len(paths),
        target["latent_tokens"],
        target["latent_dim"],
    )
    if tuple(normalized.shape) != expected_shape:
        raise ValueError(
            f"Encoded latent shape {tuple(normalized.shape)} != {expected_shape}"
        )
    if not torch.isfinite(normalized).all():
        raise ValueError("Encoded teacher-positive latents contain non-finite values")
    return normalized.cpu().numpy().astype(np.float16)


def bank_item_is_valid(
    path: Path,
    *,
    row: PromptRow,
    positives_per_condition: int,
    latent_tokens: int,
    latent_dim: int,
) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            if "latents_normalized" not in data:
                return False
            latents = data["latents_normalized"]
            if latents.shape != (positives_per_condition, latent_tokens, latent_dim):
                return False
            if latents.dtype != np.float16 or not np.isfinite(latents).all():
                return False
            if "item_index" not in data or int(data["item_index"]) != row.index:
                return False
            if "item_id" not in data or str(data["item_id"]) != row.audio_id:
                return False
    except (OSError, ValueError, KeyError):
        return False
    return True


def main() -> None:
    args = parse_args()
    worker = get_worker_context()
    logging.basicConfig(
        level=getattr(logging, args.log_level) if worker.is_main else logging.WARNING,
        format=f"rank={worker.rank} | %(asctime)s | %(levelname)s | %(message)s",
    )
    if args.positives_per_condition < 1:
        raise ValueError("--positives-per-condition must be >= 1")
    if args.num_steps < 1:
        raise ValueError("--num-steps must be >= 1")
    if args.duration <= 0:
        raise ValueError("--duration must be positive")
    if args.progress_poll_interval <= 0:
        raise ValueError("--progress-poll-interval must be positive")
    if args.coordination_timeout_minutes <= 0:
        raise ValueError("--coordination-timeout-minutes must be positive")
    coordination_timeout_seconds = args.coordination_timeout_minutes * 60.0

    invocation_dir = Path.cwd()
    resonate_root = resolve_from(args.resonate_root, invocation_dir)
    manifest = resolve_from(args.manifest, invocation_dir)
    output_dir = resolve_from(args.output_dir, invocation_dir)
    checkpoint = (
        resolve_from(args.checkpoint, invocation_dir)
        if args.checkpoint is not None
        else resonate_root / "weights" / "Resonate_GRPO.pth"
    )
    target_vae_weights = resolve_from(args.target_vae_weights, invocation_dir)
    target_latent_mean = resolve_from(args.target_latent_mean, invocation_dir)
    target_latent_std = resolve_from(args.target_latent_std, invocation_dir)

    if not (resonate_root / "resonate" / "__init__.py").is_file():
        raise FileNotFoundError(f"Not an official Resonate checkout: {resonate_root}")
    rows = read_manifest(manifest)
    selected_rows = select_rows(rows, args)
    full_run = (
        len(selected_rows) == len(rows)
        and selected_rows[0].index == 0
        and selected_rows[-1].index == len(rows) - 1
    )
    worker_rows = selected_rows[worker.rank :: worker.world_size]
    if worker.is_main:
        LOG.info("Manifest: %s (%d prompts)", manifest, len(rows))
        LOG.info(
            "Selected: %d prompts, indices %d..%d",
            len(selected_rows),
            selected_rows[0].index,
            selected_rows[-1].index,
        )
        LOG.info(
            "Workers: %d GPUs; prompt assignment uses selected_rows[rank::world_size]",
            worker.world_size,
        )
        LOG.info("Training-ready output: %s", output_dir)
    if args.dry_run:
        if worker.is_main:
            print(
                json.dumps(
                    {
                        "resonate_root": str(resonate_root),
                        "manifest": str(manifest),
                        "output_dir": str(output_dir),
                        "num_manifest_rows": len(rows),
                        "num_selected_rows": len(selected_rows),
                        "full_dataset": full_run,
                        "world_size": worker.world_size,
                        "items_per_rank": [
                            len(selected_rows[rank :: worker.world_size])
                            for rank in range(worker.world_size)
                        ],
                        "uses_nccl": False,
                    },
                    ensure_ascii=False,
                    indent=2,
                )
            )
        return

    import torch
    import torchaudio
    from tqdm import tqdm

    if not args.device.startswith("cuda") or not torch.cuda.is_available():
        raise RuntimeError("This builder requires CUDA GPUs")
    if worker.world_size > torch.cuda.device_count():
        raise RuntimeError(
            f"WORLD_SIZE={worker.world_size}, but only {torch.cuda.device_count()} "
            "CUDA devices are visible"
        )
    device = torch.device(
        f"cuda:{worker.local_rank}" if worker.world_size > 1 else args.device
    )
    torch.cuda.set_device(device)
    output_dir.mkdir(parents=True, exist_ok=True)
    coordination_dir = initialize_coordination_dir(
        output_dir,
        worker,
        timeout_seconds=coordination_timeout_seconds,
        poll_interval=args.progress_poll_interval,
    )

    assets_ready_path = coordination_dir / "assets-ready.json"
    if worker.is_main:
        check_assets(
            resonate_root,
            checkpoint,
            download_if_missing=args.download_if_missing,
        )
        atomic_write_json(
            assets_ready_path,
            {"ready_at": datetime.now(timezone.utc).isoformat()},
        )
    else:
        wait_for_paths(
            [assets_ready_path],
            timeout_seconds=coordination_timeout_seconds,
            poll_interval=args.progress_poll_interval,
            description="rank-0 Resonate asset preparation",
        )
    require_file(target_vae_weights, "MeanAudio 16 kHz VAE checkpoint")
    require_file(target_latent_mean, "MeanAudio latent mean")
    require_file(target_latent_std, "MeanAudio latent std")

    LOG.info("Rank %d loading MeanAudio 16 kHz VAE encoder on %s", worker.rank, device)
    target = load_target_encoder(
        vae_weights=target_vae_weights,
        latent_mean_path=target_latent_mean,
        latent_std_path=target_latent_std,
        device=device,
        torch=torch,
    )
    config_ready_path = coordination_dir / "config-ready.json"
    config_path = output_dir / "config.json"
    config: dict[str, Any] | None = None
    if worker.is_main:
        config = expected_config(
            args=args,
            resonate_root=resonate_root,
            manifest=manifest,
            output_dir=output_dir,
            checkpoint=checkpoint,
            target_vae_weights=target_vae_weights,
            target_latent_mean=target_latent_mean,
            target_latent_std=target_latent_std,
            num_items=len(rows),
            target_num_audio_frames=target["num_audio_frames"],
            latent_tokens=target["latent_tokens"],
            latent_dim=target["latent_dim"],
        )
        ensure_compatible_config(config_path, config)
        atomic_write_json(
            config_ready_path,
            {"ready_at": datetime.now(timezone.utc).isoformat()},
        )
    else:
        wait_for_paths(
            [config_ready_path, config_path],
            timeout_seconds=coordination_timeout_seconds,
            poll_interval=args.progress_poll_interval,
            description="rank-0 bank configuration",
        )

    temporary_dir = output_dir / ".temporary-flac" / f"rank-{worker.rank}"
    if temporary_dir.exists():
        for stale_path in temporary_dir.glob("*.flac"):
            stale_path.unlink()

    valid_rows = [
        row
        for row in worker_rows
        if not args.overwrite
        and bank_item_is_valid(
            output_dir / f"{row.index}.npz",
            row=row,
            positives_per_condition=args.positives_per_condition,
            latent_tokens=target["latent_tokens"],
            latent_dim=target["latent_dim"],
        )
    ]
    valid_indices = {row.index for row in valid_rows}
    pending_rows = [row for row in worker_rows if row.index not in valid_indices]
    LOG.info(
        "Rank %d assignment: %d generated, %d pending",
        worker.rank,
        len(valid_rows),
        len(pending_rows),
    )

    status_path = coordination_dir / f"rank-{worker.rank}.status.json"
    count_path = coordination_dir / f"rank-{worker.rank}.count"
    done_path = coordination_dir / f"rank-{worker.rank}.done.json"
    atomic_write_text(count_path, "0\n")
    atomic_write_json(
        status_path,
        {
            "rank": worker.rank,
            "local_rank": worker.local_rank,
            "assigned": len(worker_rows),
            "initial_valid": len(valid_rows),
            "pending": len(pending_rows),
        },
    )

    progress = None
    progress_stop = threading.Event()
    progress_thread = None
    generation_ready_path = coordination_dir / "generation-ready.json"
    if worker.is_main:
        status_paths = [
            coordination_dir / f"rank-{rank}.status.json"
            for rank in range(worker.world_size)
        ]
        wait_for_paths(
            status_paths,
            timeout_seconds=coordination_timeout_seconds,
            poll_interval=args.progress_poll_interval,
            description="worker resume scans",
        )
        statuses = [json.loads(path.read_text(encoding="utf-8")) for path in status_paths]
        initial_valid = sum(int(status["initial_valid"]) for status in statuses)
        assigned = sum(int(status["assigned"]) for status in statuses)
        total_pending = sum(int(status["pending"]) for status in statuses)
        if assigned != len(selected_rows):
            raise RuntimeError(
                f"Worker assignments cover {assigned} items, expected {len(selected_rows)}"
            )
        progress = tqdm(
            total=len(selected_rows),
            initial=initial_valid,
            desc=f"teacher positives generated ({worker.world_size} GPUs)",
            unit="prompt",
            dynamic_ncols=True,
        )
        progress.set_postfix(generated=initial_valid, total=len(selected_rows))

        def monitor_worker_progress() -> None:
            displayed = initial_valid
            count_paths = [
                coordination_dir / f"rank-{rank}.count"
                for rank in range(worker.world_size)
            ]
            while not progress_stop.wait(args.progress_poll_interval):
                current = initial_valid + sum(read_worker_count(path) for path in count_paths)
                current = min(current, len(selected_rows))
                if current > displayed:
                    assert progress is not None
                    progress.update(current - displayed)
                    progress.set_postfix(generated=current, total=len(selected_rows))
                    displayed = current

        progress_thread = threading.Thread(
            target=monitor_worker_progress,
            name="bank-progress-monitor",
            daemon=True,
        )
        progress_thread.start()
        if total_pending > 0 or args.overwrite:
            (output_dir / "complete.json").unlink(missing_ok=True)
        atomic_write_json(
            generation_ready_path,
            {
                "initial_valid": initial_valid,
                "pending": total_pending,
                "ready_at": datetime.now(timezone.utc).isoformat(),
            },
        )
    else:
        wait_for_paths(
            [generation_ready_path],
            timeout_seconds=coordination_timeout_seconds,
            poll_interval=args.progress_poll_interval,
            description="rank-0 generation release",
        )

    resonate_runtime = None
    if pending_rows:
        LOG.info("Rank %d loading official Resonate-GRPO runtime", worker.rank)
        resonate_runtime = load_resonate_runtime(
            resonate_root=resonate_root,
            checkpoint=checkpoint,
            config_name=args.config_name,
            device_name=str(device),
            full_precision=args.full_precision,
            num_steps=args.num_steps,
            duration=args.duration,
        )

    generated_by_worker = 0
    for row in pending_rows:
        assert resonate_runtime is not None
        temporary_paths: list[Path] = []
        try:
            audio_samples, _ = generate_audio_batch(
                [row],
                runtime=resonate_runtime,
                positives_per_condition=args.positives_per_condition,
                negative_prompt=args.negative_prompt,
                cfg_strength=args.cfg_strength,
                base_seed=args.base_seed,
            )
            temporary_paths = save_temporary_flacs(
                row=row,
                audio_samples=audio_samples,
                temporary_dir=temporary_dir,
                sample_rate=resonate_runtime["sample_rate"],
                target_frames=round(args.duration * resonate_runtime["sample_rate"]),
                torch=torch,
                torchaudio=torchaudio,
            )
            latents = encode_flacs(
                temporary_paths,
                target=target,
                device=device,
                use_amp=not args.full_precision,
                torch=torch,
                torchaudio=torchaudio,
            )
            output_path = output_dir / f"{row.index}.npz"
            atomic_save_npz(
                output_path,
                latents_normalized=latents,
                item_index=np.asarray(row.index, dtype=np.int64),
                item_id=np.asarray(row.audio_id),
                seeds=np.asarray(
                    [
                        seed_for(
                            row.index,
                            positive_index,
                            args.positives_per_condition,
                            args.base_seed,
                        )
                        for positive_index in range(args.positives_per_condition)
                    ],
                    dtype=np.int64,
                ),
            )
            if not bank_item_is_valid(
                output_path,
                row=row,
                positives_per_condition=args.positives_per_condition,
                latent_tokens=target["latent_tokens"],
                latent_dim=target["latent_dim"],
            ):
                raise RuntimeError(f"Post-write validation failed: {output_path}")
        finally:
            for path in temporary_paths:
                path.unlink(missing_ok=True)
        generated_by_worker += 1
        atomic_write_text(count_path, f"{generated_by_worker}\n")
    try:
        temporary_dir.rmdir()
    except OSError:
        pass

    atomic_write_json(
        done_path,
        {
            "rank": worker.rank,
            "assigned": len(worker_rows),
            "initial_valid": len(valid_rows),
            "generated": generated_by_worker,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        },
    )

    if not worker.is_main:
        return

    done_paths = [
        coordination_dir / f"rank-{rank}.done.json"
        for rank in range(worker.world_size)
    ]
    wait_for_paths(
        done_paths,
        timeout_seconds=coordination_timeout_seconds,
        poll_interval=args.progress_poll_interval,
        description="all worker completion markers",
    )
    progress_stop.set()
    if progress_thread is not None:
        progress_thread.join(timeout=max(5.0, args.progress_poll_interval * 2))
    assert progress is not None
    done_payloads = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in done_paths
    ]
    for payload in done_payloads:
        completed_by_worker = int(payload["initial_valid"]) + int(payload["generated"])
        if completed_by_worker != int(payload["assigned"]):
            raise RuntimeError(
                f"Rank {payload['rank']} completed {completed_by_worker} items, "
                f"but was assigned {payload['assigned']}"
            )
    final_generated = sum(int(payload["generated"]) for payload in done_payloads)
    expected_progress = (
        sum(int(status["initial_valid"]) for status in statuses)
        + final_generated
    )
    if expected_progress != len(selected_rows):
        raise RuntimeError(
            f"Workers completed {expected_progress} prompts, expected {len(selected_rows)}"
        )
    if expected_progress > progress.n:
        progress.update(expected_progress - progress.n)
    progress.set_postfix(generated=progress.n, total=len(selected_rows))
    progress.close()

    if full_run:
        missing = [
            row.index
            for row in rows
            if not bank_item_is_valid(
                output_dir / f"{row.index}.npz",
                row=row,
                positives_per_condition=args.positives_per_condition,
                latent_tokens=target["latent_tokens"],
                latent_dim=target["latent_dim"],
            )
        ]
        if missing:
            raise RuntimeError(
                f"Teacher-positive bank incomplete: {len(missing)} invalid/missing items; "
                f"first indices={missing[:20]}"
            )
        assert config is not None
        completion = dict(config)
        completion["completed_at"] = datetime.now(timezone.utc).isoformat()
        atomic_write_json(output_dir / "complete.json", completion)
        LOG.info("Teacher-positive bank complete: %s (%d items)", output_dir, len(rows))
    else:
        subset_path = output_dir / (
            f"subset_complete_{selected_rows[0].index:08d}_"
            f"{selected_rows[-1].index:08d}.json"
        )
        atomic_write_json(
            subset_path,
            {
                "completed_at": datetime.now(timezone.utc).isoformat(),
                "num_selected_rows": len(selected_rows),
                "first_index": selected_rows[0].index,
                "last_index": selected_rows[-1].index,
                "full_dataset": False,
            },
        )
        LOG.info("Subset complete; run the full range before training: %s", subset_path)


if __name__ == "__main__":
    main()
