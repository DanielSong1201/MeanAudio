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
import torch.nn.functional as F
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import AutoTokenizer, T5EncoderModel

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drifting.Resonate.config import RESONATE_CONFIG
from drifting.Resonate.data import (
    AudioCapsRecord,
    resolve_audiocaps_records,
    write_processed_manifest,
)
from drifting.Resonate.progress import setup_tqdm_logger
from meanaudio.ext.autoencoder.vae import get_my_vae
from meanaudio.ext.mel_converter import get_mel_converter


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
        raise RuntimeError("CUDA preprocessing requested, but torch.cuda.is_available() is False")
    if enabled:
        if device_name != "cuda":
            raise ValueError("Distributed preprocessing currently requires --device cuda")
        timeout = timedelta(minutes=int(os.environ.get("DDP_TIMEOUT_MINUTES", "180")))
        dist.init_process_group(backend="nccl", timeout=timeout)
        torch.cuda.set_device(local_rank)
        device = torch.device(f"cuda:{local_rank}")
    else:
        device = torch.device(device_name)
        if device.type == "cuda":
            torch.cuda.set_device(0)
    return enabled, rank, local_rank, world_size, device


class RawAudioDataset(Dataset):
    def __init__(
        self,
        records: list[AudioCapsRecord],
        indices: list[int],
        *,
        sample_rate: int,
        num_samples: int,
    ) -> None:
        self.records = records
        self.indices = indices
        self.sample_rate = sample_rate
        self.num_samples = num_samples
        self.resamplers: dict[int, torchaudio.transforms.Resample] = {}

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, local_index: int) -> dict[str, Any]:
        index = self.indices[local_index]
        record = self.records[index]
        waveform, source_rate = torchaudio.load(record.audio_path)
        waveform = waveform.mean(dim=0)
        if source_rate != self.sample_rate:
            resampler = self.resamplers.get(source_rate)
            if resampler is None:
                resampler = torchaudio.transforms.Resample(
                    source_rate,
                    self.sample_rate,
                    lowpass_filter_width=64,
                    rolloff=0.9475937167399596,
                    resampling_method="sinc_interp_kaiser",
                    beta=14.769656459379492,
                )
                self.resamplers[source_rate] = resampler
            waveform = resampler(waveform)
        if waveform.numel() < self.num_samples:
            waveform = F.pad(waveform, (0, self.num_samples - waveform.numel()))
        else:
            waveform = waveform[: self.num_samples]
        peak = waveform.abs().max()
        if peak > 1e-6:
            waveform = waveform / peak * 0.95
        return {
            "index": index,
            "id": record.id,
            "caption": record.caption,
            "waveform": waveform,
        }


def load_resonate_vae(path: Path, device: torch.device) -> torch.nn.Module:
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing Resonate 44.1 kHz VAE: {path}. Download v1-44.pth from "
            "https://huggingface.co/AndreasXi/Resonate"
        )
    vae = get_my_vae("44k").eval()
    try:
        state = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        state = torch.load(path, map_location="cpu")
    vae.load_state_dict(state, strict=True)
    vae.remove_weight_norm()
    vae.requires_grad_(False)
    return vae.to(device)


def prepare_audiocaps(args: argparse.Namespace) -> None:
    distributed, rank, _, world_size, device = setup_distributed(args.device)
    is_main = rank == 0
    logger = setup_tqdm_logger(
        "drifting.resonate.prepare",
        enabled=is_main,
        level=args.log_level,
    )
    records, manifest, audio_dir, missing_audio = resolve_audiocaps_records(
        dataset_root=args.dataset_root,
        split=args.split,
        manifest_path=args.manifest,
        strict_missing_audio=args.strict_missing_audio,
    )
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be positive")
        records = records[: args.limit]

    output_dir = args.output_root / f"{args.split}-npz-flant5-44k"
    output_tsv = args.output_root / f"{args.split}.tsv"
    config_path = output_dir / "config.json"
    complete_path = output_dir / "complete.json"
    config = {
        "version": 1,
        "dataset_root": str(args.dataset_root),
        "split": args.split,
        "source_manifest": str(manifest),
        "source_audio_dir": str(audio_dir),
        "missing_audio_rows_skipped": missing_audio,
        "num_items": len(records),
        "sample_rate": RESONATE_CONFIG.sample_rate,
        "duration": args.duration,
        "num_samples": round(RESONATE_CONFIG.sample_rate * args.duration),
        "vae_weights": str(args.vae_weights),
        "text_encoder": args.text_encoder,
        "text_seq_len": RESONATE_CONFIG.text_seq_len,
        "text_dim": RESONATE_CONFIG.text_dim,
        "text_c_dim": RESONATE_CONFIG.text_c_dim,
        "latent_dim": RESONATE_CONFIG.latent_dim,
        "storage_dtype": args.storage_dtype,
    }

    if config_path.exists() and not args.overwrite:
        existing = json.loads(config_path.read_text(encoding="utf-8"))
        if existing != config:
            raise ValueError(
                f"Existing preprocessing config differs: {config_path}. "
                "Use a different --output-root or pass --overwrite."
            )
    if is_main:
        logger.info(
            f"[data] split={args.split} records={len(records)} "
            f"missing_audio_skipped={missing_audio}"
        )
        logger.info("[data] manifest=%s", manifest)
        logger.info("[data] audio_dir=%s", audio_dir)
        logger.info("[data] output=%s", output_dir)

    if args.dry_run:
        if is_main:
            logger.info("[ok] dry-run discovery completed; model extraction was skipped")
        if distributed:
            dist.destroy_process_group()
        return

    if is_main:
        output_dir.mkdir(parents=True, exist_ok=True)
        atomic_write_json(config_path, config)
        write_processed_manifest(output_tsv, records)
        complete_path.unlink(missing_ok=True)
    if distributed:
        dist.barrier()

    pending_indices = [
        index
        for index in range(rank, len(records), world_size)
        if args.overwrite or not (output_dir / f"{index}.npz").is_file()
    ]
    if is_main:
        remaining_total = sum(
            args.overwrite or not (output_dir / f"{index}.npz").is_file()
            for index in range(len(records))
        )
        logger.info(
            "[data] remaining=%d existing=%d total=%d",
            remaining_total,
            len(records) - remaining_total,
            len(records),
        )
    if not args.overwrite and all(
        (output_dir / f"{index}.npz").is_file() for index in range(len(records))
    ):
        if is_main:
            atomic_write_json(complete_path, config)
            logger.info(
                "[ok] Resonate AudioCaps preprocessing already complete: %s",
                output_dir,
            )
        if distributed:
            dist.barrier()
            dist.destroy_process_group()
        return
    dataset = RawAudioDataset(
        records,
        pending_indices,
        sample_rate=RESONATE_CONFIG.sample_rate,
        num_samples=config["num_samples"],
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=args.num_workers > 0,
    )

    tokenizer = AutoTokenizer.from_pretrained(args.text_encoder)
    text_encoder = T5EncoderModel.from_pretrained(args.text_encoder).eval().to(device)
    text_encoder.requires_grad_(False)
    vae = load_resonate_vae(args.vae_weights, device)
    mel_converter = get_mel_converter("44k").eval().to(device)
    use_amp = args.amp and device.type == "cuda"
    storage_dtype = np.float16 if args.storage_dtype == "float16" else np.float32

    progress = tqdm(
        loader,
        desc=f"resonate-preprocess-{args.split}-rank{rank}",
        disable=not is_main,
        dynamic_ncols=True,
        unit="batch",
    )
    with torch.inference_mode():
        for batch in progress:
            waveforms = batch["waveform"].to(device=device, non_blocking=True)
            tokens = tokenizer(
                list(batch["caption"]),
                max_length=RESONATE_CONFIG.text_seq_len,
                padding="max_length",
                truncation=True,
                return_tensors="pt",
            )
            input_ids = tokens.input_ids.to(device=device, non_blocking=True)
            attention_mask = tokens.attention_mask.to(device=device, non_blocking=True)
            with torch.autocast(
                device_type=device.type,
                dtype=torch.bfloat16,
                enabled=use_amp,
            ):
                text_features = text_encoder(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                ).last_hidden_state
                # Match the released Resonate implementation exactly.
                text_features_c = text_features.mean(dim=1)
                mel = mel_converter(waveforms)
                posterior = vae.encode(mel)
                means = posterior.mean.transpose(1, 2)
                stds = posterior.std.transpose(1, 2)

            for batch_index, global_index_tensor in enumerate(batch["index"]):
                global_index = int(global_index_tensor)
                atomic_save_npz(
                    output_dir / f"{global_index}.npz",
                    mean=means[batch_index].float().cpu().numpy().astype(storage_dtype),
                    std=stds[batch_index].float().cpu().numpy().astype(storage_dtype),
                    text_features=text_features[batch_index]
                    .float()
                    .cpu()
                    .numpy()
                    .astype(storage_dtype),
                    text_features_c=text_features_c[batch_index]
                    .float()
                    .cpu()
                    .numpy()
                    .astype(storage_dtype),
                )

    if distributed:
        dist.barrier()
    if is_main:
        missing_outputs = [
            index
            for index in range(len(records))
            if not (output_dir / f"{index}.npz").is_file()
        ]
        if missing_outputs:
            raise RuntimeError(
                f"Preprocessing incomplete: {len(missing_outputs)} NPZ files missing; "
                f"first indices={missing_outputs[:20]}"
            )
        atomic_write_json(complete_path, config)
        logger.info("[ok] Resonate AudioCaps preprocessing complete: %s", output_dir)
    if distributed:
        dist.destroy_process_group()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Extract Resonate 44.1 kHz VAE latents and Flan-T5 conditions."
    )
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=Path("drifting/data/AudioCaps_CVSSP"),
    )
    parser.add_argument("--split", choices=("train", "eval", "test"), required=True)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("data/audiocaps_resonate"),
    )
    parser.add_argument("--vae-weights", type=Path, default=RESONATE_CONFIG.vae_weights)
    parser.add_argument("--text-encoder", default="google/flan-t5-large")
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--storage-dtype", choices=("float16", "float32"), default="float16")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument(
        "--amp",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument("--strict-missing-audio", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--log-level",
        choices=("DEBUG", "INFO", "WARNING", "ERROR"),
        default="INFO",
    )
    return parser.parse_args()


if __name__ == "__main__":
    prepare_audiocaps(parse_args())
