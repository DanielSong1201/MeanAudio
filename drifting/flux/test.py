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

import torch

from meanaudio.model.teacher_feature_drifting import TeacherFeatureDriftingLoss

log = logging.getLogger("drifting.flux.eval")


def setup_eval_logger() -> None:
    if log.handlers:
        return
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        stream=sys.stdout,
    )


def load_torch(path: Path, map_location: str | torch.device):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_data_config(path: Path, split: str) -> dict[str, Any]:
    import yaml

    with path.open("r") as f:
        data_cfg = yaml.safe_load(f)
    if split not in data_cfg:
        raise KeyError(f"Split {split!r} not found in {path}")
    return data_cfg


class AudioCapsNpzDataset(torch.utils.data.Dataset):
    def __init__(self, *, tsv_path: Path, npz_dir: Path) -> None:
        self.npz_dir = npz_dir
        with tsv_path.open("r", newline="") as f:
            self.rows = list(csv.DictReader(f, delimiter="\t"))
        if not self.rows:
            raise ValueError(f"No rows found in {tsv_path}")
        if not npz_dir.exists():
            raise FileNotFoundError(f"Missing npz directory: {npz_dir}")
        if not (npz_dir / "0.npz").exists():
            raise FileNotFoundError(f"Missing sample npz: {npz_dir / '0.npz'}")

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        import numpy as np

        data = np.load(self.npz_dir / f"{idx}.npz")
        return {
            "id": self.rows[idx]["id"],
            "caption": self.rows[idx]["caption"],
            "a_mean": torch.from_numpy(data["mean"]),
            "a_std": torch.from_numpy(data["std"]),
            "text_features": torch.from_numpy(data["text_features"]),
            "text_features_c": torch.from_numpy(data["text_features_c"]),
        }


def check_assets(args: argparse.Namespace) -> None:
    required = [
        args.teacher_weights,
        args.student_init,
        args.latent_mean,
        args.latent_std,
        args.data_config,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"Missing required path: {path}")
        print(f"[ok] {path}")
    data_cfg = load_data_config(args.data_config, args.train_split)
    split = data_cfg[args.train_split]
    dataset = AudioCapsNpzDataset(tsv_path=Path(split["tsv"]), npz_dir=Path(split["npz_dir"]))
    print(f"[ok] dataset size={len(dataset)} npz_dir={split['npz_dir']}")


def check_loss(device: torch.device, dtype: torch.dtype) -> None:
    layers = ("joint_3", "fused_3", "fused_7")
    generated = {
        layer: torch.randn(2, 32, 64, device=device, dtype=dtype, requires_grad=True)
        for layer in layers
    }
    positive = {
        layer: torch.randn(2, 32, 64, device=device, dtype=dtype)
        for layer in layers
    }
    criterion = TeacherFeatureDriftingLoss(pool_tokens=16)
    output = criterion(generated, positive)
    output.loss.backward()
    print(f"[ok] synthetic_loss={output.loss.detach().float().item():.6f}")
    for layer in layers:
        grad = generated[layer].grad
        if grad is None:
            raise AssertionError(f"{layer} did not receive gradients")
        print(f"[ok] {layer} grad_norm={grad.float().norm().item():.6f}")


def check_train_step(args: argparse.Namespace, device: torch.device) -> None:
    from drifting.flux.train import (
        build_flux_model,
        maybe_load_empty_features,
        one_step_flux_loss,
        parse_layers,
        parse_radii,
        setup_logger,
    )

    logger = setup_logger(args.output_dir)
    data_cfg = load_data_config(args.data_config, args.train_split)
    split = data_cfg[args.train_split]
    dataset = AudioCapsNpzDataset(tsv_path=Path(split["tsv"]), npz_dir=Path(split["npz_dir"]))
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False, drop_last=True)))

    latent_mean = load_torch(args.latent_mean, "cpu")
    latent_std = load_torch(args.latent_std, "cpu")
    empty_text, empty_text_c = maybe_load_empty_features(
        args.weights_dir,
        text_seq_len=77,
        text_dim=1024,
        text_c_dim=512,
        logger=logger,
    )
    student = build_flux_model(
        weights_path=args.student_init,
        device=device,
        latent_mean=latent_mean,
        latent_std=latent_std,
        empty_text=empty_text,
        empty_text_c=empty_text_c,
        use_rope=args.use_rope,
    )
    teacher = build_flux_model(
        weights_path=args.teacher_weights,
        device=device,
        latent_mean=latent_mean,
        latent_std=latent_std,
        empty_text=empty_text,
        empty_text_c=empty_text_c,
        use_rope=args.use_rope,
    )
    teacher.eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    student.train()

    criterion = TeacherFeatureDriftingLoss(
        radii=parse_radii(args.radii),
        pool_tokens=args.pool_tokens,
        lambda_tfd=args.lambda_tfd,
        lambda_anchor=args.lambda_anchor,
    )
    text_f = batch["text_features"].to(device)
    text_f_c = batch["text_features_c"].to(device)
    a_mean = batch["a_mean"].to(device)
    a_std = batch["a_std"].to(device)
    loss, parts = one_step_flux_loss(
        student=student,
        teacher=teacher,
        criterion=criterion,
        text_f=text_f,
        text_f_c=text_f_c,
        a_mean=a_mean,
        a_std=a_std,
        feature_layers=parse_layers(args.feature_layers),
        feature_noise=args.feature_noise,
        lambda_flow=args.lambda_flow,
    )
    loss.backward()
    grad_count = sum(1 for parameter in student.parameters() if parameter.grad is not None)
    teacher_grad_count = sum(1 for parameter in teacher.parameters() if parameter.grad is not None)
    print(f"[ok] train_step_total_loss={loss.detach().float().item():.6f}")
    print(f"[ok] flow_loss={parts['flow_loss'].float().item():.6f}")
    print(f"[ok] tfd_loss={parts['tfd_loss'].float().item():.6f}")
    print(f"[ok] student_grad_tensors={grad_count}")
    print(f"[ok] teacher_grad_tensors={teacher_grad_count}")
    if grad_count == 0:
        raise AssertionError("Student did not receive gradients.")
    if teacher_grad_count != 0:
        raise AssertionError("Frozen teacher unexpectedly received gradients.")


def _read_eval_manifest(tsv_path: Path) -> list[dict[str, str]]:
    with tsv_path.open("r", newline="") as f:
        rows = list(csv.DictReader(f, delimiter="\t"))
    if not rows:
        raise ValueError(f"No rows found in {tsv_path}")
    for key in ("id", "caption"):
        if key not in rows[0]:
            raise KeyError(f"Missing {key!r} column in {tsv_path}")
    return rows


def _load_eval_condition(npz_dir: Path, idx: int, *, device: torch.device, dtype: torch.dtype) -> tuple[torch.Tensor, torch.Tensor]:
    import numpy as np

    npz_path = npz_dir / f"{idx}.npz"
    if not npz_path.exists():
        raise FileNotFoundError(f"Missing eval condition npz: {npz_path}")
    with np.load(npz_path) as data:
        text_f = torch.from_numpy(data["text_features"]).unsqueeze(0).to(device=device, dtype=dtype)
        text_f_c = torch.from_numpy(data["text_features_c"]).unsqueeze(0).to(device=device, dtype=dtype)
    return text_f, text_f_c


@torch.inference_mode()
def generate_audio_from_precomputed_conditions(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> None:
    import torchaudio
    from tqdm import tqdm

    from drifting.flux.train import build_flux_model, maybe_load_empty_features, setup_logger
    from meanaudio.model.flow_matching import FlowMatching
    from meanaudio.model.sequence_config import CONFIG_16K
    from meanaudio.model.utils.features_utils import FeaturesUtils

    audio_dir = args.output / "audio"
    audio_dir.mkdir(parents=True, exist_ok=True)

    if not args.eval_tsv.exists():
        raise FileNotFoundError(f"Missing eval TSV: {args.eval_tsv}")
    if not args.eval_npz_dir.exists():
        raise FileNotFoundError(f"Missing eval npz directory: {args.eval_npz_dir}")
    if not args.vae_weights.exists():
        raise FileNotFoundError(f"Missing VAE weights: {args.vae_weights}")
    if not args.vocoder_weights.exists():
        raise FileNotFoundError(f"Missing vocoder weights: {args.vocoder_weights}")

    rows = _read_eval_manifest(args.eval_tsv)
    logger = setup_logger(args.output)
    latent_mean = load_torch(args.latent_mean, "cpu")
    latent_std = load_torch(args.latent_std, "cpu")
    empty_text, empty_text_c = maybe_load_empty_features(
        args.weights_dir,
        text_seq_len=77,
        text_dim=1024,
        text_c_dim=512,
        logger=logger,
    )
    net = build_flux_model(
        weights_path=args.model_path,
        device=device,
        latent_mean=latent_mean,
        latent_std=latent_std,
        empty_text=empty_text,
        empty_text_c=empty_text_c,
        use_rope=args.use_rope,
    ).to(device=device, dtype=dtype).eval()

    feature_utils = FeaturesUtils(
        tod_vae_ckpt=str(args.vae_weights),
        enable_conditions=False,
        encoder_name="t5_clap",
        mode="16k",
        bigvgan_vocoder_ckpt=str(args.vocoder_weights),
        need_vae_encoder=False,
    ).to(device=device, dtype=dtype).eval()

    seq_cfg = CONFIG_16K
    seq_cfg.duration = args.duration
    net.update_seq_lengths(seq_cfg.latent_seq_len)
    fm = FlowMatching(min_sigma=0, inference_mode="euler", num_steps=args.num_steps)
    rng = torch.Generator(device=device)
    rng.manual_seed(args.seed)

    tqdm_position = int(os.environ.get("EVAL_TQDM_POSITION", os.environ.get("TQDM_POSITION", "0")))
    tqdm_desc = os.environ.get("EVAL_TQDM_DESC", "generate-audio")
    tqdm_leave = os.environ.get("EVAL_TQDM_LEAVE", "0") == "1"
    for idx in tqdm(
        range(len(rows)),
        desc=tqdm_desc,
        dynamic_ncols=True,
        position=tqdm_position,
        leave=tqdm_leave,
    ):
        text_f, text_f_c = _load_eval_condition(args.eval_npz_dir, idx, device=device, dtype=dtype)
        x0 = torch.randn(
            1,
            net.latent_seq_len,
            net.latent_dim,
            device=device,
            dtype=dtype,
            generator=rng,
        )
        conditions = net.preprocess_conditions(text_f, text_f_c)
        empty_conditions = net.get_empty_conditions(1)
        cfg_ode_wrapper = lambda t, x: net.ode_wrapper(t, x, conditions, empty_conditions, args.cfg_strength)
        x1 = fm.to_data(cfg_ode_wrapper, x0)
        x1 = net.unnormalize(x1)
        spec = feature_utils.decode(x1)
        audio = feature_utils.vocode(spec).float().cpu()[0]
        torchaudio.save(audio_dir / f"{rows[idx]['id']}.flac", audio, seq_cfg.sampling_rate)


def _run_and_tee(cmd: list[str], log_path: Path) -> None:
    with log_path.open("w") as f:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            f.write(line)
            f.flush()
            print(line, end="")
        returncode = process.wait()
    if returncode != 0:
        raise subprocess.CalledProcessError(returncode, cmd)


def run_eval(args: argparse.Namespace) -> None:
    setup_eval_logger()
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    log.info("Preparing FluxAudio eval")
    log.info("model_path=%s", args.model_path)
    log.info("output=%s", output)
    log.info("num_steps=%s cfg_strength=%s use_rope=%s", args.num_steps, args.cfg_strength, args.use_rope)
    log.info("eval_tsv=%s", args.eval_tsv)
    log.info("eval_npz_dir=%s", args.eval_npz_dir)
    log.info("Step 1/2: generating audio from precomputed t5_clap conditions")
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    generate_audio_from_precomputed_conditions(args, device, dtype)
    log.info("Step 1/2 complete: audio_dir=%s", output / "audio")

    bench_cmd = [
        sys.executable,
        "av-benchmark/evaluate.py",
        "--gt_audio",
        "gt_audio",
        "--gt_cache",
        str(args.gt_cache),
        "--pred_audio",
        str(output / "audio"),
        "--pred_cache",
        str(output / "cache"),
        "--audio_length=10",
        "--recompute_pred_cache",
        "--skip_video_related",
    ]
    log.info("Step 2/2: computing metrics with av-benchmark/evaluate.py")
    _run_and_tee(bench_cmd, output / "evaluate.log")
    log.info("Step 2/2 complete: evaluate_log=%s", output / "evaluate.log")
    print(f"[ok] evaluation log written to {output / 'evaluate.log'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Pure FluxAudio drifting diagnostic entrypoint.")
    parser.add_argument("--mode", choices=["assets", "loss", "train-step", "eval"], default="loss")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    parser.add_argument("--output-dir", type=Path, default=Path("exps/drifting_flux/diagnostics"))
    parser.add_argument("--data-config", type=Path, default=Path("config/data/t5_clap.yaml"))
    parser.add_argument("--train-split", default="AudioCaps_npz")
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument("--teacher-weights", type=Path, default=Path("weights/fluxaudio_s_full.pth"))
    parser.add_argument("--student-init", type=Path, default=Path("weights/fluxaudio_s_full.pth"))
    parser.add_argument("--latent-mean", type=Path, default=Path("sets/latent_mean.pt"))
    parser.add_argument("--latent-std", type=Path, default=Path("sets/latent_std.pt"))
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--use-rope", action="store_true")
    parser.add_argument("--feature-layers", default="joint_3,fused_3,fused_7")
    parser.add_argument("--feature-noise", type=float, default=0.1)
    parser.add_argument("--pool-tokens", type=int, default=64)
    parser.add_argument("--radii", default="0.02,0.05,0.1,0.2")
    parser.add_argument("--lambda-flow", type=float, default=1.0)
    parser.add_argument("--lambda-tfd", type=float, default=0.2)
    parser.add_argument("--lambda-anchor", type=float, default=0.05)
    parser.add_argument("--model-path", type=Path, default=Path("exps/drifting_flux/flux_drifting_s_1x4090/flux_drifting_s_1x4090_last.pth"))
    parser.add_argument("--output", type=Path, default=Path("exps/drifting_flux_eval"))
    parser.add_argument("--gt-cache", type=Path, default=Path("data/audiocaps/test-features"))
    parser.add_argument("--eval-tsv", type=Path, default=Path("sets/test-audiocaps.tsv"))
    parser.add_argument("--eval-npz-dir", type=Path, default=Path("data/audiocaps/test-npz-t5-clap"))
    parser.add_argument("--vae-weights", type=Path, default=Path("weights/v1-16.pth"))
    parser.add_argument("--vocoder-weights", type=Path, default=Path("weights/best_netG.pt"))
    parser.add_argument("--duration", type=float, default=9.975)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--cfg-strength", type=float, default=4.5)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16

    if args.mode == "assets":
        check_assets(args)
    elif args.mode == "loss":
        check_loss(device, dtype)
    elif args.mode == "train-step":
        check_train_step(args, device)
    elif args.mode == "eval":
        run_eval(args)


if __name__ == "__main__":
    main()
