#!/usr/bin/env python3
from __future__ import annotations

import argparse
import gc
import subprocess
import sys
from dataclasses import replace
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from drifting.Resonate.config import RESONATE_CONFIG
from drifting.Resonate.model import (
    ResonateFluxAudio,
    freeze_resonate_teacher,
    load_resonate_checkpoint,
)


FLUX_BASELINE_COMMIT = "a7f5239"
FLUX_PROTECTED_PATHS = (
    "drifting/flux",
    "drifting/scripts/flux",
    "drifting/eval_helpers.py",
    "meanaudio/model/networks.py",
    "meanaudio/model/teacher_feature_drifting.py",
)


def check_flux_isolation(repo_root: Path, baseline: str) -> None:
    command = [
        "git",
        "diff",
        "--exit-code",
        baseline,
        "--",
        *FLUX_PROTECTED_PATHS,
    ]
    result = subprocess.run(
        command,
        cwd=repo_root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise AssertionError(
            "Flux-protected files differ from the Phase 0 baseline "
            f"{baseline}:\n{result.stdout}"
        )
    print(f"[ok] Flux-protected paths match baseline commit {baseline}")


def check_released_config() -> None:
    expected = {
        "model_name": "fluxaudio_m_44k",
        "source_revision": "b08fb6f7887e129623e0efae3e84653783be5c69",
        "sample_rate": 44_100,
        "latent_dim": 40,
        "latent_seq_len": 430,
        "text_dim": 1024,
        "text_c_dim": 1024,
        "hidden_dim": 448,
        "depth": 52,
        "fused_depth": 36,
        "num_heads": 7,
    }
    for name, value in expected.items():
        actual = getattr(RESONATE_CONFIG, name)
        if actual != value:
            raise AssertionError(f"{name}: expected {value!r}, got {actual!r}")
    if RESONATE_CONFIG.joint_depth != 16:
        raise AssertionError(
            f"joint_depth: expected 16, got {RESONATE_CONFIG.joint_depth}"
        )
    if RESONATE_CONFIG.latent_length_for_duration(10.0) != 431:
        raise AssertionError("44.1 kHz 10-second runtime length should be 431")
    print("[ok] released Resonate architecture and 44.1 kHz config")


def tiny_config():
    return replace(
        RESONATE_CONFIG,
        latent_dim=4,
        latent_seq_len=4,
        max_latent_seq_len=8,
        text_dim=6,
        text_c_dim=5,
        text_seq_len=3,
        hidden_dim=8,
        depth=3,
        fused_depth=2,
        num_heads=2,
        feature_layers=("joint_0", "fused_0", "fused_1"),
        use_rope=False,
    )


def check_tfd_model_contract(device: torch.device) -> None:
    config = tiny_config()
    teacher = ResonateFluxAudio(config=config, use_rope=False).to(device)
    student = ResonateFluxAudio(config=config, use_rope=False).to(device)
    student.load_state_dict(teacher.state_dict(), strict=True)
    teacher_parameter = next(teacher.parameters())
    student_parameter = next(student.parameters())
    if teacher_parameter.data_ptr() == student_parameter.data_ptr():
        raise AssertionError("Teacher and student unexpectedly share parameter storage")
    freeze_resonate_teacher(teacher)
    student.train()

    batch_size = 2
    latent = torch.randn(
        batch_size,
        config.latent_seq_len,
        config.latent_dim,
        device=device,
        requires_grad=True,
    )
    text_f = torch.randn(
        batch_size,
        config.text_seq_len,
        config.text_dim,
        device=device,
    )
    text_f_c = torch.randn(batch_size, config.text_c_dim, device=device)
    timestep = torch.full((batch_size,), 0.1, device=device)
    conditions = teacher.preprocess_conditions(text_f, text_f_c)

    generated = teacher.extract_features(latent, timestep, conditions)
    if tuple(generated) != config.feature_layers:
        raise AssertionError(
            f"Feature ordering mismatch: {tuple(generated)} vs {config.feature_layers}"
        )
    for name, feature in generated.items():
        expected_shape = (
            batch_size,
            config.latent_seq_len,
            config.hidden_dim,
        )
        if tuple(feature.shape) != expected_shape:
            raise AssertionError(
                f"{name}: expected shape {expected_shape}, got {tuple(feature.shape)}"
            )

    sum(feature.float().square().mean() for feature in generated.values()).backward()
    if latent.grad is None or not torch.isfinite(latent.grad).all():
        raise AssertionError("Generated teacher features did not backpropagate to latent")
    if latent.grad.abs().sum().item() == 0:
        raise AssertionError("Generated teacher-feature gradient is identically zero")
    if any(parameter.grad is not None for parameter in teacher.parameters()):
        raise AssertionError("Frozen teacher unexpectedly accumulated parameter gradients")

    with torch.no_grad():
        positive = teacher.extract_features(
            latent.detach(),
            timestep,
            conditions,
        )
    if any(feature.requires_grad for feature in positive.values()):
        raise AssertionError("Positive features unexpectedly require gradients")

    prediction = teacher.predict_flow(
        latent.detach(),
        timestep,
        conditions,
    )
    if prediction.shape != latent.shape:
        raise AssertionError(
            f"Flow shape mismatch: {tuple(prediction.shape)} vs {tuple(latent.shape)}"
        )

    try:
        teacher.extract_features(
            latent.detach(),
            timestep,
            conditions,
            layers=("not_a_layer",),
        )
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid feature-layer name was not rejected")

    print(
        "[ok] independent teacher/student construction and TFD feature contract: "
        "generated path differentiable, positive path detached, teacher frozen"
    )


def check_released_checkpoint(checkpoint: Path) -> None:
    if not checkpoint.exists():
        raise FileNotFoundError(
            f"Missing checkpoint: {checkpoint}. Download Resonate_GRPO.pth from "
            "https://huggingface.co/AndreasXi/Resonate"
        )

    print(f"[info] strictly loading released checkpoint: {checkpoint}")
    model = ResonateFluxAudio(config=RESONATE_CONFIG)
    load_resonate_checkpoint(model, checkpoint, map_location="cpu")

    if len(model.joint_blocks) != RESONATE_CONFIG.joint_depth:
        raise AssertionError("Released model has the wrong number of joint blocks")
    if len(model.fused_blocks) != RESONATE_CONFIG.fused_depth:
        raise AssertionError("Released model has the wrong number of fused blocks")
    if model.latent_dim != RESONATE_CONFIG.latent_dim:
        raise AssertionError("Released model has the wrong latent dimension")
    if not torch.isfinite(model.latent_mean).all():
        raise AssertionError("Checkpoint did not restore finite latent_mean")
    if not torch.isfinite(model.latent_std).all():
        raise AssertionError("Checkpoint did not restore finite latent_std")

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print(
        f"[ok] strict checkpoint load; parameters={parameter_count:,}, "
        f"joint={len(model.joint_blocks)}, fused={len(model.fused_blocks)}"
    )
    del model
    gc.collect()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Validate the Resonate TFD Phase 0-3 implementation."
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=RESONATE_CONFIG.teacher_weights,
        help="Released Resonate checkpoint to load with strict=True.",
    )
    parser.add_argument(
        "--flux-baseline",
        default=FLUX_BASELINE_COMMIT,
        help="Commit whose Flux-protected paths must remain unchanged.",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for the tiny differentiability contract check.",
    )
    parser.add_argument(
        "--skip-checkpoint",
        action="store_true",
        help="Run structural checks without loading the released checkpoint.",
    )
    args = parser.parse_args()

    check_flux_isolation(REPO_ROOT, args.flux_baseline)
    check_released_config()
    check_tfd_model_contract(torch.device(args.device))
    if args.skip_checkpoint:
        print("[skip] released checkpoint strict-load check")
    else:
        check_released_checkpoint(args.checkpoint)
    print("[success] Resonate TFD Phase 0-3 validation passed")


if __name__ == "__main__":
    main()
