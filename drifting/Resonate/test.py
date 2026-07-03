#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import logging
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch

from drifting.Resonate.build_teacher_positive_bank import euler_sample
from drifting.Resonate.config import RESONATE_CONFIG
from drifting.Resonate.model import build_resonate_model
from meanaudio.ext.autoencoder.vae import get_my_vae
from meanaudio.ext.bigvgan_v2.bigvgan import BigVGAN


log = logging.getLogger("drifting.resonate.eval")


def setup_eval_logger() -> None:
    if log.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
    )


def load_torch(path: Path, map_location: str | torch.device) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def read_eval_manifest(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as file:
        rows = list(csv.DictReader(file, delimiter="\t"))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    required = {"id", "caption"}
    missing = required - set(rows[0])
    if missing:
        raise KeyError(f"{path} is missing columns: {sorted(missing)}")
    return rows


def load_eval_condition(
    npz_dir: Path,
    index: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, torch.Tensor, int]:
    path = npz_dir / f"{index}.npz"
    if not path.is_file():
        raise FileNotFoundError(f"Missing eval condition: {path}")
    with np.load(path) as data:
        required = ("mean", "text_features", "text_features_c")
        missing = [key for key in required if key not in data]
        if missing:
            raise KeyError(f"{path} is missing arrays: {missing}")
        latent_seq_len = int(data["mean"].shape[0])
        text_f = torch.from_numpy(data["text_features"].copy()).unsqueeze(0)
        text_f_c = torch.from_numpy(data["text_features_c"].copy()).unsqueeze(0)
    return (
        text_f.to(device=device, dtype=dtype),
        text_f_c.to(device=device, dtype=dtype),
        latent_seq_len,
    )


def load_decoder(
    *,
    vae_weights: Path,
    vocoder_dir: Path,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[torch.nn.Module, torch.nn.Module]:
    if not vae_weights.is_file():
        raise FileNotFoundError(f"Missing Resonate 44.1 kHz VAE: {vae_weights}")
    if not vocoder_dir.is_dir():
        raise FileNotFoundError(
            f"Missing local BigVGAN-v2 directory: {vocoder_dir}. "
            "Download nvidia/bigvgan_v2_44khz_128band_512x into this path."
        )
    vae = get_my_vae("44k").eval()
    vae.load_state_dict(load_torch(vae_weights, "cpu"), strict=True)
    vae.remove_weight_norm()
    vae.requires_grad_(False)
    if hasattr(vae, "encoder"):
        del vae.encoder
    vae = vae.to(device=device, dtype=dtype)

    vocoder = BigVGAN.from_pretrained(
        str(vocoder_dir),
        use_cuda_kernel=False,
    ).eval()
    vocoder.remove_weight_norm()
    vocoder.requires_grad_(False)
    vocoder = vocoder.to(device=device, dtype=dtype)
    return vae, vocoder


@torch.inference_mode()
def generate_audio(
    args: argparse.Namespace,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    import torchaudio
    from tqdm import tqdm

    required_paths = (
        args.model_path,
        args.latent_mean,
        args.latent_std,
        args.eval_tsv,
        args.eval_npz_dir,
        args.vae_weights,
        args.vocoder_weights,
    )
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing eval asset: {path}")

    rows = read_eval_manifest(args.eval_tsv)
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError("--limit must be >= 1")
        rows = rows[: args.limit]
    audio_dir = args.output / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    latent_mean = load_torch(args.latent_mean, "cpu")
    latent_std = load_torch(args.latent_std, "cpu")
    net = build_resonate_model(
        weights_path=args.model_path,
        map_location="cpu",
        device=device,
        dtype=dtype,
        latent_mean=latent_mean,
        latent_std=latent_std,
        use_rope=args.use_rope,
    ).eval()
    vae, vocoder = load_decoder(
        vae_weights=args.vae_weights,
        vocoder_dir=args.vocoder_weights,
        device=device,
        dtype=dtype,
    )
    generator = torch.Generator(device=device)
    generator.manual_seed(args.seed)

    tqdm_position = int(
        os.environ.get("EVAL_TQDM_POSITION", os.environ.get("TQDM_POSITION", "0"))
    )
    tqdm_desc = os.environ.get("EVAL_TQDM_DESC", "resonate-eval")
    tqdm_leave = os.environ.get("EVAL_TQDM_LEAVE", "0") == "1"
    tqdm_disable = os.environ.get("EVAL_TQDM_DISABLE", "0") == "1"
    for index in tqdm(
        range(len(rows)),
        desc=tqdm_desc,
        dynamic_ncols=True,
        position=tqdm_position,
        leave=tqdm_leave,
        disable=tqdm_disable,
        unit="prompt",
    ):
        text_f, text_f_c, latent_seq_len = load_eval_condition(
            args.eval_npz_dir,
            index,
            device=device,
            dtype=dtype,
        )
        net.update_seq_lengths(latent_seq_len)
        conditions = net.preprocess_conditions(text_f, text_f_c)
        empty_conditions = net.get_empty_conditions(1)
        noise = torch.randn(
            1,
            latent_seq_len,
            RESONATE_CONFIG.latent_dim,
            device=device,
            dtype=dtype,
            generator=generator,
        )
        normalized_latent = euler_sample(
            model=net,
            noise=noise,
            conditions=conditions,
            empty_conditions=empty_conditions,
            num_steps=args.num_steps,
            cfg_strength=args.cfg_strength,
        )
        latent = net.unnormalize(normalized_latent)
        mel = vae.decode(latent.transpose(1, 2))
        audio = vocoder(mel).float().cpu()[0]
        target_samples = round(args.duration * RESONATE_CONFIG.sample_rate)
        audio = audio[..., :target_samples]
        torchaudio.save(
            audio_dir / f"{rows[index]['id']}.flac",
            audio,
            RESONATE_CONFIG.sample_rate,
        )


def run_and_tee(command: list[str], log_path: Path) -> None:
    with log_path.open("w", encoding="utf-8") as file:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            file.write(line)
            file.flush()
            print(line, end="")
        return_code = process.wait()
    if return_code != 0:
        raise subprocess.CalledProcessError(return_code, command)


def run_eval(args: argparse.Namespace) -> None:
    setup_eval_logger()
    args.output.mkdir(parents=True, exist_ok=True)
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    log.info("Preparing Resonate AV-Benchmark evaluation")
    log.info("model_path=%s", args.model_path)
    log.info("output=%s", args.output)
    log.info(
        "num_steps=%d cfg_strength=%s duration=%s limit=%s",
        args.num_steps,
        args.cfg_strength,
        args.duration,
        args.limit,
    )
    log.info("Step 1/2: generating 44.1 kHz audio")
    generate_audio(args, device=device, dtype=dtype)
    log.info("Step 1/2 complete: audio_dir=%s", args.output / "audio")

    if args.skip_av_benchmark:
        (args.output / "evaluate.log").write_text(
            "AV-Benchmark skipped by --skip-av-benchmark\n",
            encoding="utf-8",
        )
        log.info("Step 2/2 skipped by request")
        return
    benchmark_entrypoint = Path("av-benchmark/evaluate.py")
    if not benchmark_entrypoint.is_file():
        raise FileNotFoundError(
            f"Missing {benchmark_entrypoint}. Install av-benchmark at the repository root."
        )
    if not args.gt_audio.exists():
        raise FileNotFoundError(f"Missing AV-Benchmark ground-truth audio: {args.gt_audio}")
    command = [
        sys.executable,
        str(benchmark_entrypoint),
        "--gt_audio",
        str(args.gt_audio),
        "--gt_cache",
        str(args.gt_cache),
        "--pred_audio",
        str(args.output / "audio"),
        "--pred_cache",
        str(args.output / "cache"),
        f"--audio_length={args.duration:g}",
        "--recompute_pred_cache",
        "--skip_video_related",
    ]
    log.info("Step 2/2: computing AV-Benchmark metrics")
    run_and_tee(command, args.output / "evaluate.log")
    log.info("Step 2/2 complete: evaluate_log=%s", args.output / "evaluate.log")


def build_parser() -> argparse.ArgumentParser:
    env_limit = os.environ.get("EVAL_LIMIT", "").strip()
    parser = argparse.ArgumentParser(description="Resonate drifting evaluation")
    parser.add_argument("--mode", choices=("eval",), default="eval")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--model-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--gt-cache",
        type=Path,
        default=Path("data/audiocaps/test-features"),
    )
    parser.add_argument("--gt-audio", type=Path, default=Path("gt_audio"))
    parser.add_argument(
        "--eval-tsv",
        type=Path,
        default=Path("data/audiocaps_resonate/test.tsv"),
    )
    parser.add_argument(
        "--eval-npz-dir",
        type=Path,
        default=Path("data/audiocaps_resonate/test-npz-flant5-44k"),
    )
    parser.add_argument(
        "--vae-weights",
        type=Path,
        default=RESONATE_CONFIG.vae_weights,
    )
    parser.add_argument(
        "--vocoder-weights",
        type=Path,
        default=RESONATE_CONFIG.vocoder_dir,
    )
    parser.add_argument(
        "--latent-mean",
        type=Path,
        default=RESONATE_CONFIG.latent_mean,
    )
    parser.add_argument(
        "--latent-std",
        type=Path,
        default=RESONATE_CONFIG.latent_std,
    )
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--cfg-strength", type=float, default=4.5)
    parser.add_argument("--use-rope", action="store_true")
    parser.add_argument(
        "--limit",
        type=int,
        default=int(env_limit) if env_limit else None,
    )
    parser.add_argument(
        "--skip-av-benchmark",
        action="store_true",
        default=os.environ.get("EVAL_SKIP_AV_BENCHMARK", "0") == "1",
    )
    return parser


if __name__ == "__main__":
    run_eval(build_parser().parse_args())
