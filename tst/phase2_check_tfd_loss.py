#!/usr/bin/env python3
"""Synthetic gradient check for TeacherFeatureDriftingLoss."""

from __future__ import annotations

import argparse

import torch

from meanaudio.model.teacher_feature_drifting import TeacherFeatureDriftingLoss


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--tokens", type=int, default=32)
    parser.add_argument("--hidden-dim", type=int, default=64)
    parser.add_argument("--pool-tokens", type=int, default=16)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--dtype", choices=["float32", "bfloat16"], default="float32")
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is false.")
    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16

    layers = ("joint_3", "fused_3", "fused_7")
    generated = {
        layer: torch.randn(
            args.batch_size,
            args.tokens,
            args.hidden_dim,
            device=device,
            dtype=dtype,
            requires_grad=True,
        )
        for layer in layers
    }
    positive = {
        layer: torch.randn(args.batch_size, args.tokens, args.hidden_dim, device=device, dtype=dtype)
        for layer in layers
    }

    criterion = TeacherFeatureDriftingLoss(pool_tokens=args.pool_tokens)
    output = criterion(generated, positive)
    output.loss.backward()

    print(f"[ok] total_loss={output.loss.detach().float().item():.6f}")
    print(f"[ok] drifting_loss={output.drifting_loss.float().item():.6f}")
    print(f"[ok] anchor_loss={output.anchor_loss.float().item():.6f}")
    for layer in layers:
        grad = generated[layer].grad
        if grad is None:
            raise AssertionError(f"{layer} did not receive gradients")
        print(f"[ok] {layer} grad_norm={grad.float().norm().item():.6f}")
    print("[done] Phase-2 TFD loss check passed.")


if __name__ == "__main__":
    main()
