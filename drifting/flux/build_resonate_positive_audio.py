#!/usr/bin/env python3
"""Generate prompt-matched positive audio with the official Resonate-GRPO model.

This is stage 1 of the Resonate-audio -> Flux-latent positive-bank pipeline.
It deliberately writes waveforms only; conversion to MeanAudio's 16 kHz VAE
latent space belongs to stage 2.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import os
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterator, Sequence


LOG = logging.getLogger("resonate-positive-audio")
SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PromptRow:
    index: int
    audio_id: str
    caption: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Generate all prompt-matched positive waveforms with the official "
            "Resonate-GRPO model on one GPU."
        )
    )
    parser.add_argument(
        "--resonate-root",
        type=Path,
        default=Path("../Resonate"),
        help="Official Resonate checkout. Relative paths use the current working directory.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        default=Path("data/audiocaps/train-memmap.tsv"),
        help="Flux training TSV/JSONL; row order becomes the positive-bank index.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(
            "data/audiocaps/"
            "train-teacher-audio-resonate-grpo-25step-cfg4.5"
        ),
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Defaults to <resonate-root>/weights/Resonate_GRPO.pth.",
    )
    parser.add_argument(
        "--config-name",
        default="GRPO_flant5_44kMMVAE_fluxaudio_audiocaps_qwen25omni_semantic",
    )
    parser.add_argument("--positives-per-condition", type=int, default=3)
    parser.add_argument("--prompt-batch-size", type=int, default=1)
    parser.add_argument("--num-steps", type=int, default=25)
    parser.add_argument("--cfg-strength", type=float, default=4.5)
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--negative-prompt", default="")
    parser.add_argument("--base-seed", type=int, default=20260802)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--full-precision",
        action="store_true",
        help="Use float32 instead of the official demo's bfloat16 default.",
    )
    parser.add_argument(
        "--download-if-missing",
        action="store_true",
        help="Download AndreasXi/Resonate into <resonate-root>/weights when assets are missing.",
    )
    parser.add_argument("--start-index", type=int, default=0)
    parser.add_argument(
        "--end-index",
        type=int,
        default=None,
        help="Exclusive manifest index. Defaults to the complete manifest.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Generate at most this many prompts; intended for a separate smoke-test output dir.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Regenerate selected audio even when valid output already exists.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate paths/manifest and print the plan without importing torch or loading models.",
    )
    parser.add_argument("--log-level", choices=("DEBUG", "INFO", "WARNING"), default="INFO")
    return parser.parse_args()


def resolve_from(path: Path, base: Path) -> Path:
    path = path.expanduser()
    if not path.is_absolute():
        path = base / path
    return path.resolve()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(text, encoding="utf-8")
    temporary.replace(path)


def atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    atomic_write_text(path, json.dumps(payload, ensure_ascii=False, indent=2) + "\n")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def repository_revision(root: Path) -> str | None:
    try:
        result = subprocess.run(
            ["git", "-C", str(root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None
    return result.stdout.strip() or None


def read_manifest(path: Path) -> list[PromptRow]:
    if not path.is_file():
        raise FileNotFoundError(f"Training manifest does not exist: {path}")

    rows: list[PromptRow] = []
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as file:
            for line_number, raw_line in enumerate(file, start=1):
                if not raw_line.strip():
                    continue
                payload = json.loads(raw_line)
                caption = payload.get("caption", payload.get("prompt"))
                audio_id = payload.get("id", payload.get("audio_id"))
                if not isinstance(caption, str) or not caption.strip():
                    raise ValueError(f"Missing caption/prompt at {path}:{line_number}")
                if audio_id is None:
                    audio_id = str(len(rows))
                rows.append(PromptRow(len(rows), str(audio_id), caption.strip()))
    else:
        delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
        with path.open("r", encoding="utf-8", newline="") as file:
            reader = csv.DictReader(file, delimiter=delimiter)
            for line_number, payload in enumerate(reader, start=2):
                caption = payload.get("caption", payload.get("prompt"))
                audio_id = payload.get("id", payload.get("audio_id"))
                if not isinstance(caption, str) or not caption.strip():
                    raise ValueError(f"Missing caption/prompt at {path}:{line_number}")
                if audio_id is None:
                    audio_id = str(len(rows))
                rows.append(PromptRow(len(rows), str(audio_id), caption.strip()))

    if not rows:
        raise ValueError(f"No prompts found in {path}")
    return rows


def select_rows(rows: Sequence[PromptRow], args: argparse.Namespace) -> list[PromptRow]:
    if args.start_index < 0:
        raise ValueError("--start-index must be >= 0")
    if args.end_index is not None and args.end_index < args.start_index:
        raise ValueError("--end-index must be >= --start-index")
    end = len(rows) if args.end_index is None else min(args.end_index, len(rows))
    selected = list(rows[args.start_index:end])
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be >= 1")
        selected = selected[: args.limit]
    if not selected:
        raise ValueError("The selected manifest range is empty")
    return selected


def chunked(rows: Sequence[PromptRow], size: int) -> Iterator[Sequence[PromptRow]]:
    for start in range(0, len(rows), size):
        yield rows[start : start + size]


def audio_relative_path(index: int, positive_index: int) -> Path:
    bucket = index // 1000
    return Path("audio") / f"{bucket:05d}" / f"{index:08d}_{positive_index:02d}.flac"


def metadata_relative_path(index: int) -> Path:
    bucket = index // 1000
    return Path("metadata") / f"{bucket:05d}" / f"{index:08d}.json"


def seed_for(index: int, positive_index: int, positives_per_condition: int, base_seed: int) -> int:
    return base_seed + index * positives_per_condition + positive_index


def required_resonate_assets(resonate_root: Path, checkpoint: Path) -> tuple[Path, ...]:
    return (
        checkpoint,
        resonate_root / "weights" / "v1-44.pth",
        resonate_root / "weights" / "bigvgan_v2_44khz_128band_512x",
        resonate_root / "sets" / "latent_mean_44k.pt",
        resonate_root / "sets" / "latent_std_44k.pt",
    )


def download_assets(resonate_root: Path) -> None:
    from huggingface_hub import snapshot_download

    weights_dir = resonate_root / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    LOG.info("Downloading official Resonate assets into %s", weights_dir)
    snapshot_download(repo_id="AndreasXi/resonate", local_dir=weights_dir)


def check_assets(resonate_root: Path, checkpoint: Path, *, download_if_missing: bool) -> None:
    missing = [path for path in required_resonate_assets(resonate_root, checkpoint) if not path.exists()]
    if missing and download_if_missing:
        download_assets(resonate_root)
        missing = [path for path in required_resonate_assets(resonate_root, checkpoint) if not path.exists()]
    if missing:
        joined = "\n  ".join(str(path) for path in missing)
        raise FileNotFoundError(
            "Missing official Resonate assets:\n  "
            f"{joined}\nRun with --download-if-missing or prepare Resonate/weights first."
        )


def expected_run_config(
    *,
    args: argparse.Namespace,
    resonate_root: Path,
    manifest: Path,
    output_dir: Path,
    checkpoint: Path,
    num_manifest_rows: int,
) -> dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "generator": "official-resonate-grpo",
        "resonate_root": str(resonate_root),
        "resonate_revision": repository_revision(resonate_root),
        "config_name": args.config_name,
        "checkpoint": str(checkpoint),
        "checkpoint_sha256": None,
        "manifest": str(manifest),
        "manifest_sha256": sha256_file(manifest),
        "num_manifest_rows": num_manifest_rows,
        "output_dir": str(output_dir),
        "positives_per_condition": args.positives_per_condition,
        "num_steps": args.num_steps,
        "cfg_strength": args.cfg_strength,
        "duration": args.duration,
        "sample_rate": 44_100,
        "negative_prompt": args.negative_prompt,
        "base_seed": args.base_seed,
        "dtype": "float32" if args.full_precision else "bfloat16",
        "audio_format": "flac",
    }


def ensure_compatible_config(path: Path, expected: dict[str, Any]) -> None:
    if not path.exists():
        atomic_write_json(path, expected)
        return
    existing = json.loads(path.read_text(encoding="utf-8"))
    if existing != expected:
        differing = sorted(
            key
            for key in set(existing) | set(expected)
            if existing.get(key) != expected.get(key)
        )
        raise ValueError(
            f"Existing bank config does not match this run ({', '.join(differing)}). "
            "Use a new --output-dir for a different generation recipe."
        )


def audio_file_is_valid(path: Path, *, sample_rate: int, num_frames: int, torchaudio: Any) -> bool:
    if not path.is_file() or path.stat().st_size == 0:
        return False
    try:
        info = torchaudio.info(path)
    except Exception:
        return False
    return info.sample_rate == sample_rate and info.num_frames == num_frames and info.num_channels >= 1


def load_resonate_runtime(
    *,
    resonate_root: Path,
    checkpoint: Path,
    config_name: str,
    device_name: str,
    full_precision: bool,
    num_steps: int,
    duration: float,
) -> dict[str, Any]:
    sys.path.insert(0, str(resonate_root))
    os.chdir(resonate_root)

    import torch
    from hydra import compose, initialize_config_dir

    from resonate.model.flow_matching import FlowMatching
    from resonate.model.networks import get_model
    from resonate.model.sequence_config import CONFIG_44K
    from resonate.model.utils.features_utils import FeaturesUtils

    if not device_name.startswith("cuda"):
        raise ValueError("Stage-1 generation is intentionally CUDA-only; use --device cuda:0")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required but torch.cuda.is_available() is false")

    device = torch.device(device_name)
    torch.cuda.set_device(device)
    dtype = torch.float32 if full_precision else torch.bfloat16
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    with initialize_config_dir(version_base="1.3.2", config_dir=str(resonate_root / "config")):
        cfg = compose(config_name=config_name)
    if int(cfg.audio_sample_rate) != 44_100:
        raise ValueError(f"Expected a 44.1 kHz Resonate config, got {cfg.audio_sample_rate}")

    sequence_config = CONFIG_44K
    sequence_config.duration = duration
    model = get_model(
        cfg.model,
        use_rope=cfg.get("use_rope", True),
        text_dim=cfg.get("text_dim"),
        text_c_dim=cfg.get("text_c_dim"),
    ).to(device=device, dtype=dtype).eval()
    try:
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(checkpoint, map_location="cpu")
    model.load_weights(state)
    del state

    feature_utils = FeaturesUtils(
        tod_vae_ckpt=str(resonate_root / "weights" / "v1-44.pth"),
        enable_conditions=True,
        encoder_name=cfg.get("text_encoder_name", "flan-t5"),
        mode="44k",
        need_vae_encoder=False,
    ).to(device=device, dtype=dtype).eval()
    model.update_seq_lengths(sequence_config.latent_seq_len)

    return {
        "torch": torch,
        "model": model,
        "feature_utils": feature_utils,
        "flow_matching": FlowMatching(min_sigma=0, inference_mode="euler", num_steps=num_steps),
        "device": device,
        "dtype": dtype,
        "latent_seq_len": sequence_config.latent_seq_len,
        "sample_rate": sequence_config.sampling_rate,
    }


def generate_audio_batch(
    rows: Sequence[PromptRow],
    *,
    runtime: dict[str, Any],
    positives_per_condition: int,
    negative_prompt: str,
    cfg_strength: float,
    base_seed: int,
) -> tuple[Any, list[dict[str, Any]]]:
    torch = runtime["torch"]
    model = runtime["model"]
    feature_utils = runtime["feature_utils"]
    prompts: list[str] = []
    seeds: list[int] = []
    sample_records: list[dict[str, Any]] = []
    for row in rows:
        for positive_index in range(positives_per_condition):
            seed = seed_for(row.index, positive_index, positives_per_condition, base_seed)
            prompts.append(row.caption)
            seeds.append(seed)
            sample_records.append(
                {
                    "index": row.index,
                    "audio_id": row.audio_id,
                    "caption": row.caption,
                    "positive_index": positive_index,
                    "seed": seed,
                }
            )

    with torch.inference_mode():
        condition_features = feature_utils.encode_text([row.caption for row in rows])
        text_features = condition_features[0].repeat_interleave(
            positives_per_condition,
            dim=0,
        )
        text_features_c = condition_features[1].repeat_interleave(
            positives_per_condition,
            dim=0,
        )
        if runtime.get("negative_prompt") != negative_prompt:
            runtime["negative_prompt"] = negative_prompt
            runtime["negative_features"] = feature_utils.encode_text([negative_prompt])
        cached_negative = runtime["negative_features"]
        negative_features = (
            cached_negative[0].expand(len(prompts), -1, -1),
            cached_negative[1].expand(len(prompts), -1),
        )
        conditions = model.preprocess_conditions(text_features, text_features_c)
        empty_conditions = model.get_empty_conditions(
            len(prompts),
            negative_text_features=negative_features,
        )
        noise_parts = []
        for seed in seeds:
            generator = torch.Generator(device=runtime["device"])
            generator.manual_seed(seed)
            noise_parts.append(
                torch.randn(
                    1,
                    runtime["latent_seq_len"],
                    model.latent_dim,
                    device=runtime["device"],
                    dtype=runtime["dtype"],
                    generator=generator,
                )
            )
        noise = torch.cat(noise_parts, dim=0)
        ode = lambda timestep, latent: model.ode_wrapper(
            timestep,
            latent,
            conditions,
            empty_conditions,
            cfg_strength,
        )
        normalized_latent = runtime["flow_matching"].to_data(ode, noise)
        latent = model.unnormalize(normalized_latent)
        spectrogram = feature_utils.decode(latent)
        audio = feature_utils.vocode(spectrogram).float().cpu()
    return audio, sample_records


def save_prompt_outputs(
    *,
    output_dir: Path,
    row: PromptRow,
    audio_samples: Any,
    records: Sequence[dict[str, Any]],
    target_frames: int,
    sample_rate: int,
    overwrite: bool,
    torch: Any,
    torchaudio: Any,
) -> dict[str, Any]:
    output_records: list[dict[str, Any]] = []
    for audio, record in zip(audio_samples, records, strict=True):
        relative_path = audio_relative_path(row.index, int(record["positive_index"]))
        output_path = output_dir / relative_path
        output_path.parent.mkdir(parents=True, exist_ok=True)

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
            raise ValueError(f"Non-finite waveform generated for manifest index {row.index}")

        peak_before_clamp = float(audio.abs().max().item())
        rms = float(audio.float().square().mean().sqrt().item())
        clipping_fraction = float((audio.abs() > 1.0).float().mean().item())
        audio = audio.clamp(-1.0, 1.0)

        if overwrite or not audio_file_is_valid(
            output_path,
            sample_rate=sample_rate,
            num_frames=target_frames,
            torchaudio=torchaudio,
        ):
            temporary = output_path.with_name(f".{output_path.stem}.{os.getpid()}.tmp.flac")
            torchaudio.save(temporary, audio, sample_rate, format="flac")
            temporary.replace(output_path)

        output_record = dict(record)
        output_record.update(
            {
                "audio_path": str(relative_path),
                "sample_rate": sample_rate,
                "num_frames": target_frames,
                "peak_before_clamp": peak_before_clamp,
                "rms": rms,
                "clipping_fraction": clipping_fraction,
            }
        )
        output_records.append(output_record)

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "index": row.index,
        "audio_id": row.audio_id,
        "caption": row.caption,
        "positives": output_records,
    }
    atomic_write_json(output_dir / metadata_relative_path(row.index), metadata)
    return metadata


def existing_prompt_is_complete(
    *,
    output_dir: Path,
    row: PromptRow,
    positives_per_condition: int,
    sample_rate: int,
    target_frames: int,
    torchaudio: Any,
) -> bool:
    metadata_path = output_dir / metadata_relative_path(row.index)
    if not metadata_path.is_file():
        return False
    try:
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    if metadata.get("audio_id") != row.audio_id or metadata.get("caption") != row.caption:
        return False
    positives = metadata.get("positives")
    if not isinstance(positives, list) or len(positives) != positives_per_condition:
        return False
    return all(
        audio_file_is_valid(
            output_dir / str(record.get("audio_path", "")),
            sample_rate=sample_rate,
            num_frames=target_frames,
            torchaudio=torchaudio,
        )
        for record in positives
    )


def consolidate_manifest(output_dir: Path, rows: Sequence[PromptRow], manifest_path: Path) -> int:
    lines: list[str] = []
    for row in rows:
        path = output_dir / metadata_relative_path(row.index)
        if not path.is_file():
            continue
        metadata = json.loads(path.read_text(encoding="utf-8"))
        for record in metadata["positives"]:
            lines.append(json.dumps(record, ensure_ascii=False))
    atomic_write_text(manifest_path, "\n".join(lines) + ("\n" if lines else ""))
    return len(lines)


def main() -> None:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(message)s",
    )
    if args.positives_per_condition < 1:
        raise ValueError("--positives-per-condition must be >= 1")
    if args.prompt_batch_size < 1:
        raise ValueError("--prompt-batch-size must be >= 1")
    if args.num_steps < 1:
        raise ValueError("--num-steps must be >= 1")
    if args.duration <= 0:
        raise ValueError("--duration must be positive")

    invocation_dir = Path.cwd()
    resonate_root = resolve_from(args.resonate_root, invocation_dir)
    manifest = resolve_from(args.manifest, invocation_dir)
    output_dir = resolve_from(args.output_dir, invocation_dir)
    checkpoint = (
        resolve_from(args.checkpoint, invocation_dir)
        if args.checkpoint is not None
        else resonate_root / "weights" / "Resonate_GRPO.pth"
    )
    if not (resonate_root / "resonate" / "__init__.py").is_file():
        raise FileNotFoundError(f"Not an official Resonate checkout: {resonate_root}")

    all_rows = read_manifest(manifest)
    selected_rows = select_rows(all_rows, args)
    full_run = (
        args.start_index == 0
        and args.end_index is None
        and args.limit is None
        and len(selected_rows) == len(all_rows)
    )
    run_config = expected_run_config(
        args=args,
        resonate_root=resonate_root,
        manifest=manifest,
        output_dir=output_dir,
        checkpoint=checkpoint,
        num_manifest_rows=len(all_rows),
    )

    LOG.info("Official Resonate checkout: %s", resonate_root)
    LOG.info("Training manifest: %s (%d prompts)", manifest, len(all_rows))
    LOG.info(
        "Selected prompts: %d (indices %d..%d)",
        len(selected_rows),
        selected_rows[0].index,
        selected_rows[-1].index,
    )
    LOG.info(
        "Generation: %d positives/prompt, %d steps, CFG %.3f, duration %.3fs",
        args.positives_per_condition,
        args.num_steps,
        args.cfg_strength,
        args.duration,
    )
    LOG.info("Output directory: %s", output_dir)
    if args.dry_run:
        print(json.dumps(run_config, ensure_ascii=False, indent=2))
        return

    import torch
    import torchaudio
    from tqdm import tqdm

    check_assets(resonate_root, checkpoint, download_if_missing=args.download_if_missing)
    run_config["checkpoint_sha256"] = sha256_file(checkpoint)
    output_dir.mkdir(parents=True, exist_ok=True)
    ensure_compatible_config(output_dir / "config.json", run_config)
    full_completion_marker = output_dir / "complete.json"
    if full_run and full_completion_marker.exists():
        full_completion_marker.unlink()

    sample_rate = 44_100
    target_frames = round(args.duration * sample_rate)
    pending_rows = [
        row
        for row in selected_rows
        if args.overwrite
        or not existing_prompt_is_complete(
            output_dir=output_dir,
            row=row,
            positives_per_condition=args.positives_per_condition,
            sample_rate=sample_rate,
            target_frames=target_frames,
            torchaudio=torchaudio,
        )
    ]
    LOG.info("Resume scan: %d complete, %d pending", len(selected_rows) - len(pending_rows), len(pending_rows))

    runtime = None
    if pending_rows:
        runtime = load_resonate_runtime(
            resonate_root=resonate_root,
            checkpoint=checkpoint,
            config_name=args.config_name,
            device_name=args.device,
            full_precision=args.full_precision,
            num_steps=args.num_steps,
            duration=args.duration,
        )

    progress = tqdm(total=len(pending_rows), desc="resonate-positive-audio", unit="prompt", dynamic_ncols=True)
    for batch_rows in chunked(pending_rows, args.prompt_batch_size):
        assert runtime is not None
        audio_batch, records = generate_audio_batch(
            batch_rows,
            runtime=runtime,
            positives_per_condition=args.positives_per_condition,
            negative_prompt=args.negative_prompt,
            cfg_strength=args.cfg_strength,
            base_seed=args.base_seed,
        )
        for offset, row in enumerate(batch_rows):
            begin = offset * args.positives_per_condition
            end = begin + args.positives_per_condition
            save_prompt_outputs(
                output_dir=output_dir,
                row=row,
                audio_samples=audio_batch[begin:end],
                records=records[begin:end],
                target_frames=target_frames,
                sample_rate=sample_rate,
                overwrite=args.overwrite,
                torch=torch,
                torchaudio=torchaudio,
            )
            progress.update(1)
    progress.close()

    if full_run:
        generated_manifest_path = output_dir / "manifest.jsonl"
        completion_path = output_dir / "complete.json"
    else:
        range_tag = f"{selected_rows[0].index:08d}_{selected_rows[-1].index:08d}"
        generated_manifest_path = output_dir / f"manifest_subset_{range_tag}.jsonl"
        completion_path = output_dir / f"subset_complete_{range_tag}.json"
    manifest_records = consolidate_manifest(
        output_dir,
        selected_rows,
        generated_manifest_path,
    )
    expected_records = len(selected_rows) * args.positives_per_condition
    if manifest_records != expected_records:
        raise RuntimeError(
            f"Positive-audio bank is incomplete: expected {expected_records} manifest records, "
            f"found {manifest_records}"
        )

    completion = {
        "schema_version": SCHEMA_VERSION,
        "completed_at": datetime.now(timezone.utc).isoformat(),
        "full_dataset": full_run,
        "num_manifest_rows": len(all_rows),
        "num_selected_rows": len(selected_rows),
        "num_audio_files": manifest_records,
        "positives_per_condition": args.positives_per_condition,
        "first_index": selected_rows[0].index,
        "last_index": selected_rows[-1].index,
        "manifest": str(generated_manifest_path.relative_to(output_dir)),
        "elapsed_note": "Per-prompt metadata and audio files passed the resume validation.",
    }
    atomic_write_json(completion_path, completion)
    LOG.info("Generation complete: %s", completion_path)


if __name__ == "__main__":
    started_at = time.monotonic()
    try:
        main()
    finally:
        LOG.info("Elapsed time: %.1f minutes", (time.monotonic() - started_at) / 60.0)
