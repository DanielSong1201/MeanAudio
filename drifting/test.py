#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch

from meanaudio.model.teacher_feature_drifting import TeacherFeatureDriftingLoss


def load_torch(path: Path, map_location: str | torch.device):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


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
    print(f"[ok] loss={output.loss.detach().float().item():.6f}")
    for layer in layers:
        grad = generated[layer].grad
        if grad is None:
            raise AssertionError(f"{layer} did not receive gradients")
        print(f"[ok] {layer} grad_norm={grad.float().norm().item():.6f}")


def check_teacher(args: argparse.Namespace, device: torch.device, dtype: torch.dtype) -> None:
    from meanaudio.model.networks import get_mean_audio

    data = np.load(args.sample_npz)
    text_f = torch.from_numpy(data["text_features"]).unsqueeze(0).to(device=device, dtype=dtype)
    text_f_c = torch.from_numpy(data["text_features_c"]).unsqueeze(0).to(device=device, dtype=dtype)
    teacher = get_mean_audio("fluxaudio_s", use_rope=args.use_rope, text_c_dim=512)
    teacher = teacher.to(device=device, dtype=dtype).eval()
    teacher.load_weights(load_torch(args.teacher_weights, device))
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    conditions = teacher.preprocess_conditions(text_f, text_f_c)
    latent = torch.randn(1, teacher.latent_seq_len, teacher.latent_dim, device=device, dtype=dtype, requires_grad=True)
    t = torch.full((1,), 0.1, device=device, dtype=dtype)
    layers = tuple(layer.strip() for layer in args.layers.split(",") if layer.strip())
    features = teacher.extract_features(latent, t, conditions, layers=layers)
    loss = sum(value.float().mean() for value in features.values())
    loss.backward()
    if latent.grad is None:
        raise AssertionError("latent did not receive gradients through teacher features")
    for name, value in features.items():
        print(f"[ok] {name}: shape={tuple(value.shape)} dtype={value.dtype}")
    print(f"[ok] latent_grad_norm={latent.grad.float().norm().item():.6f}")


def run_eval(args: argparse.Namespace) -> None:
    output = args.output
    output.mkdir(parents=True, exist_ok=True)
    eval_cmd = [
        sys.executable,
        "eval.py",
        "--variant",
        "meanaudio_s",
        "--model_path",
        str(args.model_path),
        "--output",
        str(output / "audio"),
        "--cfg_strength",
        str(args.cfg_strength),
        "--encoder_name",
        "t5_clap",
        "--duration",
        "10",
        "--use_rope",
        "--text_c_dim",
        "512",
        "--num_steps",
        str(args.num_steps),
    ]
    if args.use_rope:
        eval_cmd.append("--use_rope")
    eval_cmd.extend(["--use_meanflow", "--full_precision"])
    subprocess.run(eval_cmd, check=True)

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
    with (output / "evaluate.log").open("w") as f:
        subprocess.run(bench_cmd, check=True, stdout=f, stderr=subprocess.STDOUT)
    print(f"[ok] evaluation log written to {output / 'evaluate.log'}")


def main() -> None:
    parser = argparse.ArgumentParser(description="DriftingAudio test entrypoint.")
    parser.add_argument("--mode", choices=["loss", "teacher", "eval"], default="loss")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--teacher-weights", type=Path, default=Path("weights/fluxaudio_s_full.pth"))
    parser.add_argument("--sample-npz", type=Path, default=Path("data/audiocaps/test-npz-t5-clap/0.npz"))
    parser.add_argument("--layers", default="joint_3,fused_3,fused_7")
    parser.add_argument("--use-rope", action="store_true")
    parser.add_argument("--model-path", type=Path, default=Path("exps/drifting/drifting_fluxaudio_s/drifting_fluxaudio_s_last.pth"))
    parser.add_argument("--output", type=Path, default=Path("exps/drifting_eval"))
    parser.add_argument("--gt-cache", type=Path, default=Path("data/audiocaps/test-features"))
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--cfg-strength", type=float, default=0.9)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16

    if args.mode == "loss":
        check_loss(device, dtype)
    elif args.mode == "teacher":
        check_teacher(args, device, dtype)
    elif args.mode == "eval":
        run_eval(args)


if __name__ == "__main__":
    main()
