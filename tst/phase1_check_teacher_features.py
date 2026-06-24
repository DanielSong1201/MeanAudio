#!/usr/bin/env python3
"""Validate FluxAudio teacher feature extraction for Phase 1."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from meanaudio.model.networks import get_mean_audio


def load_text_features(npz_path: Path, batch_size: int, device: torch.device, dtype: torch.dtype):
    data = np.load(npz_path)
    text_f = torch.from_numpy(data["text_features"]).unsqueeze(0).repeat(batch_size, 1, 1)
    text_f_c = torch.from_numpy(data["text_features_c"]).unsqueeze(0).repeat(batch_size, 1)
    return text_f.to(device=device, dtype=dtype), text_f_c.to(device=device, dtype=dtype)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher-weights", type=Path, default=Path("weights/fluxaudio_s_full.pth"))
    parser.add_argument("--sample-npz", type=Path, default=Path("data/audiocaps/test-npz-t5-clap/0.npz"))
    parser.add_argument("--layers", default="joint_3,fused_3,fused_7")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="bfloat16")
    parser.add_argument("--use-rope", action="store_true")
    parser.add_argument("--text-c-dim", type=int, default=512)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    layers = tuple(layer.strip() for layer in args.layers.split(",") if layer.strip())

    if not args.teacher_weights.exists():
        raise FileNotFoundError(f"Missing teacher weights: {args.teacher_weights}")
    if not args.sample_npz.exists():
        raise FileNotFoundError(f"Missing sample npz: {args.sample_npz}")

    teacher = get_mean_audio("fluxaudio_s", use_rope=args.use_rope, text_c_dim=args.text_c_dim)
    teacher = teacher.to(device=device, dtype=dtype).eval()
    state = torch.load(args.teacher_weights, map_location=device, weights_only=True)
    teacher.load_weights(state)

    for parameter in teacher.parameters():
        parameter.requires_grad_(False)

    text_f, text_f_c = load_text_features(args.sample_npz, args.batch_size, device, dtype)
    conditions = teacher.preprocess_conditions(text_f, text_f_c)

    latent = torch.randn(
        args.batch_size,
        teacher.latent_seq_len,
        teacher.latent_dim,
        device=device,
        dtype=dtype,
        requires_grad=True,
    )
    t = torch.full((args.batch_size,), 0.1, device=device, dtype=dtype)

    features = teacher.extract_features(latent, t, conditions, layers=layers)
    if set(features) != set(layers):
        raise AssertionError(f"Feature key mismatch: expected={layers}, got={tuple(features)}")

    print("[ok] extracted FluxAudio teacher features:")
    for name in layers:
        value = features[name]
        print(f"  {name}: shape={tuple(value.shape)} dtype={value.dtype} device={value.device}")
        expected_shape = (args.batch_size, teacher.latent_seq_len, teacher.hidden_dim)
        if tuple(value.shape) != expected_shape:
            raise AssertionError(f"{name} shape mismatch: expected={expected_shape}, got={tuple(value.shape)}")
        if not value.requires_grad:
            raise AssertionError(f"{name} does not require grad; student latent gradients would be blocked.")

    probe_loss = sum(value.float().mean() for value in features.values())
    probe_loss.backward()
    if latent.grad is None:
        raise AssertionError("Generated latent did not receive gradients through teacher features.")

    teacher_grads = [parameter.grad for parameter in teacher.parameters() if parameter.grad is not None]
    if teacher_grads:
        raise AssertionError(f"Teacher parameters unexpectedly received gradients: {len(teacher_grads)} tensors")

    grad_norm = latent.grad.float().norm().item()
    print(f"[ok] latent gradient norm: {grad_norm:.6f}")
    print("[done] Phase-1 teacher feature extraction check passed.")


if __name__ == "__main__":
    main()
