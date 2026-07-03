#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import fcntl
import logging
import os
import random
import sys
from contextlib import contextmanager
from datetime import timedelta
from pathlib import Path
from typing import Any, Iterator, Mapping

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from drifting.Resonate.config import (
    RESONATE_CONFIG,
    missing_av_benchmark_gt_cache_files,
)
from drifting.Resonate.data import ResonateNpzDataset
from drifting.Resonate.model import build_resonate_model, freeze_resonate_teacher
from drifting.eval_helpers import (
    ExponentialMovingAverage,
    empty_cuda_cache,
    find_latest_weight_checkpoint,
    move_optimizer_state,
    run_checkpoint_evaluation,
)
from meanaudio.model.teacher_feature_drifting import TeacherFeatureDriftingLoss


def load_torch(path: Path, map_location: str | torch.device) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def load_training_state(
    path: Path,
    map_location: str | torch.device,
) -> dict[str, Any]:
    try:
        state = torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        state = torch.load(path, map_location=map_location)
    if not isinstance(state, dict) or "iteration" not in state:
        raise ValueError(f"Invalid training state: {path}")
    return state


def to_cpu_state(value: Any) -> Any:
    if torch.is_tensor(value):
        return value.detach().cpu()
    if isinstance(value, dict):
        return type(value)((key, to_cpu_state(item)) for key, item in value.items())
    if isinstance(value, list):
        return [to_cpu_state(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_cpu_state(item) for item in value)
    return value


def module_state_dict_to_cpu(module: nn.Module) -> dict[str, torch.Tensor]:
    state_dict = module.state_dict()
    cpu_state_dict = type(state_dict)(
        (key, value.detach().cpu()) for key, value in state_dict.items()
    )
    if hasattr(state_dict, "_metadata"):
        cpu_state_dict._metadata = state_dict._metadata
    return cpu_state_dict


def atomic_torch_save(payload: Any, path: Path) -> None:
    temporary_path = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        torch.save(payload, temporary_path)
        os.replace(temporary_path, path)
    finally:
        temporary_path.unlink(missing_ok=True)


@contextmanager
def checkpoint_io_lock(output_root: Path) -> Iterator[Path]:
    output_root.mkdir(parents=True, exist_ok=True)
    lock_path = output_root / ".checkpoint_io.lock"
    with lock_path.open("a+") as lock_file:
        fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        try:
            yield lock_path
        finally:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)


@contextmanager
def temporary_environment(values: Mapping[str, str]) -> Iterator[None]:
    previous = {key: os.environ.get(key) for key in values}
    os.environ.update(values)
    try:
        yield
    finally:
        for key, old_value in previous.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def save_training_state(
    path: Path,
    *,
    iteration: int,
    student: nn.Module,
    optimizer: torch.optim.Optimizer,
    ema: ExponentialMovingAverage | None,
    sampler_epoch: int,
) -> None:
    atomic_torch_save(
        {
            "version": 1,
            "iteration": iteration,
            "student": module_state_dict_to_cpu(student),
            "optimizer": to_cpu_state(optimizer.state_dict()),
            "ema": None if ema is None else ema.state_dict(),
            "sampler_epoch": sampler_epoch,
        },
        path,
    )


def setup_logger(output_dir: Path, *, enabled: bool = True) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("drifting.resonate.train")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    if not enabled:
        logger.addHandler(logging.NullHandler())
        return logger

    file_formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    log_prefix = os.environ.get("LOG_PREFIX", "").strip()
    stream_format = (
        f"{log_prefix} %(asctime)s | %(levelname)s | %(message)s"
        if log_prefix
        else "%(asctime)s | %(levelname)s | %(message)s"
    )
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(logging.Formatter(stream_format))
    file_handler = logging.FileHandler(output_dir / "train.log")
    file_handler.setFormatter(file_formatter)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def disable_console_logging(logger: logging.Logger) -> None:
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler,
            logging.FileHandler,
        ):
            handler.setLevel(logging.CRITICAL + 1)


def suspend_console_logging(
    logger: logging.Logger,
) -> list[tuple[logging.Handler, int]]:
    states: list[tuple[logging.Handler, int]] = []
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(
            handler,
            logging.FileHandler,
        ):
            states.append((handler, handler.level))
            handler.setLevel(logging.CRITICAL + 1)
    return states


def restore_console_logging(states: list[tuple[logging.Handler, int]]) -> None:
    for handler, level in states:
        handler.setLevel(level)


def append_metrics(path: Path, row: dict[str, Any]) -> None:
    write_header = not path.exists()
    with path.open("a", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(row))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_data_config(path: Path, split: str) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as file:
        data_config = yaml.safe_load(file)
    if split not in data_config:
        raise KeyError(f"Split {split!r} not found in {path}")
    return data_config


def parse_layers(text: str) -> tuple[str, ...]:
    layers = tuple(item.strip() for item in text.split(",") if item.strip())
    if not layers:
        raise ValueError("At least one feature layer is required")
    return layers


def parse_radii(text: str) -> tuple[float, ...]:
    radii = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not radii:
        raise ValueError("At least one radius is required")
    return radii


def setup_distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    timeout_minutes = int(os.environ.get("DDP_TIMEOUT_MINUTES", "180"))
    dist.init_process_group(
        backend="nccl",
        timeout=timedelta(minutes=timeout_minutes),
    )
    torch.cuda.set_device(local_rank)
    return True, rank, local_rank, world_size


def cleanup_distributed(enabled: bool) -> None:
    if enabled and dist.is_initialized():
        dist.destroy_process_group()


def reduce_scalar(
    value: torch.Tensor,
    *,
    distributed: bool,
    world_size: int,
) -> torch.Tensor:
    reduced = value.detach().float()
    if distributed:
        dist.all_reduce(reduced, op=dist.ReduceOp.SUM)
        reduced = reduced / world_size
    return reduced


def build_training_model(
    *,
    weights_path: Path,
    device: torch.device,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    use_rope: bool,
) -> nn.Module:
    return build_resonate_model(
        weights_path=weights_path,
        map_location="cpu",
        device=device,
        latent_mean=latent_mean,
        latent_std=latent_std,
        use_rope=use_rope,
    )


def one_step_resonate_loss(
    *,
    student: nn.Module,
    teacher: nn.Module,
    criterion: TeacherFeatureDriftingLoss,
    text_f: torch.Tensor,
    text_f_c: torch.Tensor,
    a_mean: torch.Tensor,
    a_std: torch.Tensor,
    feature_layers: tuple[str, ...],
    feature_noise: float,
    lambda_flow: float,
    samples_per_condition: int = 1,
    teacher_positives: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    if samples_per_condition < 1:
        raise ValueError("samples_per_condition must be >= 1")
    if a_mean.ndim != 3 or a_std.shape != a_mean.shape:
        raise ValueError(
            "a_mean and a_std must have shape [batch, latent_tokens, latent_dim]"
        )

    condition_batch_size = a_mean.shape[0]
    latent_seq_len = a_mean.shape[1]
    if a_mean.shape[-1] != RESONATE_CONFIG.latent_dim:
        raise ValueError(
            f"Expected latent_dim={RESONATE_CONFIG.latent_dim}, got {a_mean.shape[-1]}"
        )
    student.update_seq_lengths(latent_seq_len)
    teacher.update_seq_lengths(latent_seq_len)

    real_a_mean = a_mean
    real_a_std = a_std
    condition_text_f = text_f
    condition_text_f_c = text_f_c

    if samples_per_condition > 1:
        text_f = (
            text_f.unsqueeze(1)
            .expand(-1, samples_per_condition, *text_f.shape[1:])
            .reshape(condition_batch_size * samples_per_condition, *text_f.shape[1:])
        )
        text_f_c = (
            text_f_c.unsqueeze(1)
            .expand(-1, samples_per_condition, *text_f_c.shape[1:])
            .reshape(
                condition_batch_size * samples_per_condition,
                *text_f_c.shape[1:],
            )
        )
        a_mean = (
            a_mean.unsqueeze(1)
            .expand(-1, samples_per_condition, *a_mean.shape[1:])
            .reshape(condition_batch_size * samples_per_condition, *a_mean.shape[1:])
        )
        a_std = (
            a_std.unsqueeze(1)
            .expand(-1, samples_per_condition, *a_std.shape[1:])
            .reshape(condition_batch_size * samples_per_condition, *a_std.shape[1:])
        )

    x_real_for_flow = a_mean + a_std * torch.randn_like(a_mean)
    x_real_for_flow = student.normalize(x_real_for_flow)
    x_noise = torch.randn_like(x_real_for_flow)
    t_one = torch.ones(
        x_real_for_flow.shape[0],
        device=x_real_for_flow.device,
        dtype=x_real_for_flow.dtype,
    )
    student_conditions = student.preprocess_conditions(text_f, text_f_c)
    pred_flow = student.predict_flow(x_noise, t_one, student_conditions)
    target_flow = x_noise - x_real_for_flow
    flow_loss = (pred_flow - target_flow).pow(2).mean()
    x_student = x_noise - pred_flow

    if teacher_positives is None:
        positive_count = samples_per_condition
        positive_latents = real_a_mean.unsqueeze(1).expand(
            -1,
            positive_count,
            *real_a_mean.shape[1:],
        ) + real_a_std.unsqueeze(1).expand(
            -1,
            positive_count,
            *real_a_std.shape[1:],
        ) * torch.randn(
            condition_batch_size,
            positive_count,
            *real_a_std.shape[1:],
            device=real_a_std.device,
            dtype=real_a_std.dtype,
        )
        positive_shape = positive_latents.shape[2:]
        positive_latents = student.normalize(
            positive_latents.reshape(
                condition_batch_size * positive_count,
                *positive_shape,
            )
        ).reshape(condition_batch_size, positive_count, *positive_shape)
    else:
        if teacher_positives.ndim != 4:
            raise ValueError(
                "teacher_positives must have shape "
                "[conditions, positives, latent_tokens, latent_dim]"
            )
        if teacher_positives.shape[0] != condition_batch_size:
            raise ValueError(
                "Teacher-positive condition count mismatch: "
                f"{teacher_positives.shape[0]} vs {condition_batch_size}"
            )
        real_positive = real_a_mean + real_a_std * torch.randn_like(real_a_mean)
        real_positive = student.normalize(real_positive).unsqueeze(1)
        if teacher_positives.shape[2:] != real_positive.shape[2:]:
            raise ValueError(
                "Teacher-positive latent shape mismatch: "
                f"{tuple(teacher_positives.shape[2:])} vs "
                f"{tuple(real_positive.shape[2:])}"
            )
        # The offline bank already contains normalized latents. Do not normalize
        # these three entries a second time.
        positive_latents = torch.cat(
            [
                real_positive,
                teacher_positives.to(
                    device=real_positive.device,
                    dtype=real_positive.dtype,
                ),
            ],
            dim=1,
        )
        positive_count = positive_latents.shape[1]

    positive_latents_flat = positive_latents.reshape(
        condition_batch_size * positive_count,
        *positive_latents.shape[2:],
    )
    positive_text_f = (
        condition_text_f.unsqueeze(1)
        .expand(-1, positive_count, *condition_text_f.shape[1:])
        .reshape(
            condition_batch_size * positive_count,
            *condition_text_f.shape[1:],
        )
    )
    positive_text_f_c = (
        condition_text_f_c.unsqueeze(1)
        .expand(-1, positive_count, *condition_text_f_c.shape[1:])
        .reshape(
            condition_batch_size * positive_count,
            *condition_text_f_c.shape[1:],
        )
    )

    positive_sigma = torch.full(
        (positive_latents_flat.shape[0], 1, 1),
        feature_noise,
        device=positive_latents_flat.device,
        dtype=positive_latents_flat.dtype,
    )
    generated_sigma = torch.full(
        (x_student.shape[0], 1, 1),
        feature_noise,
        device=x_student.device,
        dtype=x_student.dtype,
    )
    positive_latents_feat = (
        (1.0 - positive_sigma) * positive_latents_flat.detach()
        + positive_sigma * torch.randn_like(positive_latents_flat)
    )
    x_student_feat = (
        (1.0 - generated_sigma) * x_student
        + generated_sigma * torch.randn_like(x_student)
    )
    positive_feature_t = torch.full(
        (positive_latents_flat.shape[0],),
        feature_noise,
        device=positive_latents_flat.device,
        dtype=positive_latents_flat.dtype,
    )
    generated_feature_t = torch.full(
        (x_student.shape[0],),
        feature_noise,
        device=x_student.device,
        dtype=x_student.dtype,
    )

    positive_teacher_conditions = teacher.preprocess_conditions(
        positive_text_f,
        positive_text_f_c,
    )
    generated_teacher_conditions = teacher.preprocess_conditions(text_f, text_f_c)
    with torch.no_grad():
        positive_features = teacher.extract_features(
            positive_latents_feat,
            positive_feature_t,
            positive_teacher_conditions,
            layers=feature_layers,
        )
    # Teacher parameters are frozen, but this call deliberately keeps autograd
    # enabled so the feature loss can reach x_student and the student model.
    generated_features = teacher.extract_features(
        x_student_feat,
        generated_feature_t,
        generated_teacher_conditions,
        layers=feature_layers,
    )
    positive_features = {
        name: feature.reshape(
            condition_batch_size,
            positive_count,
            *feature.shape[1:],
        )
        for name, feature in positive_features.items()
    }
    generated_features = {
        name: feature.reshape(
            condition_batch_size,
            samples_per_condition,
            *feature.shape[1:],
        )
        for name, feature in generated_features.items()
    }
    tfd_output = criterion(generated_features, positive_features)
    total_loss = lambda_flow * flow_loss + tfd_output.loss
    return total_loss, {
        "flow_loss": flow_loss.detach(),
        "tfd_loss": tfd_output.loss.detach(),
        "drifting_loss": tfd_output.drifting_loss,
        "anchor_loss": tfd_output.anchor_loss,
    }


class ResonateDriftingModule(nn.Module):
    def __init__(
        self,
        *,
        student: nn.Module,
        teacher: nn.Module,
        criterion: TeacherFeatureDriftingLoss,
        feature_layers: tuple[str, ...],
        feature_noise: float,
        lambda_flow: float,
        samples_per_condition: int,
    ) -> None:
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.criterion = criterion
        self.feature_layers = feature_layers
        self.feature_noise = feature_noise
        self.lambda_flow = lambda_flow
        self.samples_per_condition = samples_per_condition

    def forward(
        self,
        text_f: torch.Tensor,
        text_f_c: torch.Tensor,
        a_mean: torch.Tensor,
        a_std: torch.Tensor,
        teacher_positives: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return one_step_resonate_loss(
            student=self.student,
            teacher=self.teacher,
            criterion=self.criterion,
            text_f=text_f,
            text_f_c=text_f_c,
            a_mean=a_mean,
            a_std=a_std,
            feature_layers=self.feature_layers,
            feature_noise=self.feature_noise,
            lambda_flow=self.lambda_flow,
            samples_per_condition=self.samples_per_condition,
            teacher_positives=teacher_positives,
        )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Train a one-step Resonate student with frozen Resonate "
            "teacher-feature drifting."
        )
    )
    parser.add_argument("--exp-id", default="resonate_drifting_1step")
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("exps/drifting_resonate"),
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("config/data/resonate_flant5_44k.yaml"),
    )
    parser.add_argument("--train-split", default="AudioCaps_npz")
    parser.add_argument(
        "--teacher-positive-dir",
        type=Path,
        default=Path(
            "data/audiocaps_resonate/"
            "train-teacher-positives-resonate-grpo-25step-cfg4.5"
        ),
    )
    parser.add_argument("--teacher-positive-count", type=int, default=3)
    parser.add_argument(
        "--teacher-weights",
        type=Path,
        default=RESONATE_CONFIG.teacher_weights,
    )
    parser.add_argument(
        "--student-init",
        type=Path,
        default=RESONATE_CONFIG.student_init,
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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=200_000)
    parser.add_argument("--learning-rate", type=float, default=1e-6)
    parser.add_argument("--lr-warmup-steps", type=int, default=1_000)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=14_159_265)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--use-rope", action="store_true")
    parser.add_argument(
        "--feature-layers",
        default=",".join(RESONATE_CONFIG.feature_layers),
    )
    parser.add_argument("--feature-noise", type=float, default=0.1)
    parser.add_argument("--pool-tokens", type=int, default=64)
    parser.add_argument("--samples-per-condition", type=int, default=4)
    parser.add_argument("--radii", default="0.02,0.05,0.1,0.2")
    parser.add_argument("--lambda-flow", type=float, default=0.1)
    parser.add_argument("--lambda-tfd", type=float, default=100.0)
    parser.add_argument("--lambda-anchor", type=float, default=1.0)
    parser.add_argument("--anchor-margin-alpha", type=float, default=0.5)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=1_000)
    parser.add_argument("--eval-interval", type=int, default=10_000)
    parser.add_argument(
        "--eval-output-root",
        type=Path,
        default=Path("exps/drifting_resonate_eval"),
    )
    parser.add_argument(
        "--eval-gt-cache",
        type=Path,
        default=Path("data/audiocaps/test-features"),
    )
    parser.add_argument(
        "--eval-gt-audio",
        type=Path,
        default=Path("gt_audio"),
        help=(
            "Ground-truth audio used only when the AV-Benchmark GT cache "
            "must be rebuilt."
        ),
    )
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
        "--eval-vae-weights",
        type=Path,
        default=RESONATE_CONFIG.vae_weights,
    )
    parser.add_argument(
        "--eval-vocoder-dir",
        type=Path,
        default=RESONATE_CONFIG.vocoder_dir,
    )
    parser.add_argument("--eval-duration", type=float, default=10.0)
    parser.add_argument("--eval-num-steps", type=int, default=1)
    parser.add_argument("--eval-cfg-strength", type=float, default=4.5)
    parser.add_argument(
        "--eval-limit",
        type=int,
        help="Limit generated eval samples for a deliberate smoke test.",
    )
    parser.add_argument(
        "--eval-skip-av-benchmark",
        action="store_true",
        help="Generate eval audio but skip AV-Benchmark (smoke/debug only).",
    )
    parser.add_argument(
        "--no-eval-offload-train-state",
        dest="eval_offload_train_state",
        action="store_false",
    )
    parser.add_argument("--disable-ema", dest="ema", action="store_false")
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--ema-start", type=int, default=0)
    parser.add_argument("--ema-update-interval", type=int, default=1)
    parser.add_argument("--ema-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--eval-raw", dest="eval_use_ema", action="store_false")
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument(
        "--no-auto-resume",
        dest="auto_resume",
        action="store_false",
    )
    parser.set_defaults(
        eval_offload_train_state=True,
        ema=True,
        eval_use_ema=True,
        auto_resume=True,
    )
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.iterations < 1:
        raise ValueError("--iterations must be >= 1")
    if args.batch_size < 1:
        raise ValueError("--batch-size must be >= 1")
    if args.samples_per_condition < 1:
        raise ValueError("--samples-per-condition must be >= 1")
    if args.teacher_positive_count < 1:
        raise ValueError("--teacher-positive-count must be >= 1")
    if args.samples_per_condition != args.teacher_positive_count + 1:
        raise ValueError(
            "Hybrid positives require --samples-per-condition to equal "
            "1 real positive + --teacher-positive-count generated positives"
        )
    if args.lr_warmup_steps < 0:
        raise ValueError("--lr-warmup-steps must be >= 0")
    if args.save_interval < 1:
        raise ValueError("--save-interval must be >= 1")
    if args.eval_interval < 0:
        raise ValueError("--eval-interval must be >= 0")
    if args.ema and args.ema_update_interval < 1:
        raise ValueError("--ema-update-interval must be >= 1")
    if args.eval_limit is not None and args.eval_limit < 1:
        raise ValueError("--eval-limit must be >= 1")
    required_paths = (
        args.data_config,
        args.teacher_weights,
        args.student_init,
        args.latent_mean,
        args.latent_std,
        args.teacher_positive_dir / "complete.json",
    )
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"Missing required training asset: {path}")
    if args.eval_interval > 0:
        eval_paths = (
            args.eval_tsv,
            args.eval_npz_dir / "complete.json",
            args.eval_vae_weights,
            args.eval_vocoder_dir,
        )
        for path in eval_paths:
            if not path.exists():
                raise FileNotFoundError(f"Missing periodic-eval asset: {path}")
        if not args.eval_skip_av_benchmark:
            benchmark_entrypoint = Path("av-benchmark/evaluate.py")
            if not benchmark_entrypoint.is_file():
                raise FileNotFoundError(
                    f"Missing AV-Benchmark asset: {benchmark_entrypoint}. "
                    "Use --eval-skip-av-benchmark only for a generation smoke test."
                )
            missing_gt_cache = missing_av_benchmark_gt_cache_files(
                args.eval_gt_cache
            )
            if missing_gt_cache and not args.eval_gt_audio.is_dir():
                missing_text = ", ".join(str(path) for path in missing_gt_cache)
                raise FileNotFoundError(
                    "AV-Benchmark GT cache is incomplete and the fallback GT "
                    f"audio directory is missing: {args.eval_gt_audio}. "
                    f"Missing cache files: {missing_text}"
                )


def main() -> None:
    args = build_parser().parse_args()
    validate_args(args)
    distributed, rank, local_rank, world_size = setup_distributed()
    is_main = rank == 0
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    device = torch.device(f"cuda:{local_rank}" if distributed else args.device)
    if args.ema and args.ema_device == "cuda" and device.type != "cuda":
        raise ValueError("--ema-device cuda requires --device cuda")

    output_dir = args.output_root / args.exp_id
    logger = setup_logger(output_dir, enabled=is_main)
    metrics_path = output_dir / "metrics.csv"
    eval_metrics_path = output_dir / "eval_metrics.csv"
    logger.info("Writing logs to %s", output_dir / "train.log")
    logger.info("Writing metrics to %s", metrics_path)
    if args.eval_interval > 0:
        logger.info(
            "Writing eval metrics to %s every %d iterations",
            eval_metrics_path,
            args.eval_interval,
        )
    if distributed:
        logger.info("Distributed training enabled: world_size=%d", world_size)
    logger.info("Arguments: %s", vars(args))

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    resume_iteration = 0
    resume_path: Path | None = None
    resume_mode: str | None = None
    training_state_path = output_dir / f"{args.exp_id}_train_state_last.pth"
    if args.auto_resume:
        latest: tuple[int, Path, str] | None = None
        if is_main:
            latest_weights = find_latest_weight_checkpoint(output_dir, args.exp_id)
            if training_state_path.exists():
                try:
                    state_metadata = load_training_state(training_state_path, "cpu")
                    state_iteration = int(state_metadata["iteration"])
                    del state_metadata
                    latest = (state_iteration, training_state_path, "full")
                    if (
                        latest_weights is not None
                        and latest_weights[0] > state_iteration
                    ):
                        logger.warning(
                            "Ignoring newer orphan weights-only checkpoint at it=%d; "
                            "resuming the complete state at it=%d.",
                            latest_weights[0],
                            state_iteration,
                        )
                except Exception:
                    logger.exception(
                        "Could not read %s; falling back to weights-only resume.",
                        training_state_path,
                    )
            if latest is None and latest_weights is not None:
                latest = (latest_weights[0], latest_weights[1], "weights")
        if distributed:
            payload: list[tuple[int, str, str] | None] = [
                None if latest is None else (latest[0], str(latest[1]), latest[2])
            ]
            dist.broadcast_object_list(payload, src=0)
            latest_payload = payload[0]
            latest = (
                None
                if latest_payload is None
                else (
                    latest_payload[0],
                    Path(latest_payload[1]),
                    latest_payload[2],
                )
            )
        if latest is not None:
            resume_iteration, resume_path, resume_mode = latest
            logger.info(
                "Auto-resume found checkpoint: it=%d mode=%s path=%s",
                resume_iteration,
                resume_mode,
                resume_path,
            )
        else:
            logger.info("Auto-resume found no checkpoint in %s", output_dir)

    data_config = load_data_config(args.data_config, args.train_split)
    split_config = data_config[args.train_split]
    positive_dir = args.teacher_positive_dir
    configured_positive_dir = split_config.get("teacher_positive_dir")
    if configured_positive_dir and positive_dir is None:
        positive_dir = Path(configured_positive_dir)
    dataset = ResonateNpzDataset(
        tsv_path=Path(split_config["tsv"]),
        npz_dir=Path(split_config["npz_dir"]),
        teacher_positive_dir=positive_dir,
        teacher_positive_count=args.teacher_positive_count,
    )
    sampler = (
        DistributedSampler(
            dataset,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
            drop_last=True,
        )
        if distributed
        else None
    )
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=sampler is None,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
        drop_last=True,
        persistent_workers=args.num_workers > 0,
    )
    if len(loader) == 0:
        raise ValueError("DataLoader is empty; reduce batch size or check dataset paths")
    logger.info(
        "Loaded %d training items from %s with positives=%s",
        len(dataset),
        split_config["npz_dir"],
        positive_dir,
    )

    latent_mean = load_torch(args.latent_mean, "cpu")
    latent_std = load_torch(args.latent_std, "cpu")
    logger.info("Loading Resonate student init: %s", args.student_init)
    student = build_training_model(
        weights_path=args.student_init,
        device=device,
        latent_mean=latent_mean,
        latent_std=latent_std,
        use_rope=args.use_rope,
    )
    resume_state: dict[str, Any] | None = None
    if resume_path is not None:
        if resume_mode == "full":
            resume_state = load_training_state(resume_path, "cpu")
            student.load_state_dict(resume_state["student"], strict=True)
        else:
            student.load_weights(load_torch(resume_path, "cpu"))
    logger.info("Loading frozen Resonate teacher: %s", args.teacher_weights)
    teacher = build_training_model(
        weights_path=args.teacher_weights,
        device=device,
        latent_mean=latent_mean,
        latent_std=latent_std,
        use_rope=args.use_rope,
    )
    freeze_resonate_teacher(teacher)
    student.train()

    ema_device = torch.device(args.ema_device if args.ema_device == "cpu" else device)
    ema = (
        ExponentialMovingAverage(student, decay=args.ema_decay, device=ema_device)
        if args.ema and is_main
        else None
    )
    if ema is not None:
        if resume_state is not None and resume_state.get("ema") is not None:
            ema.load_state_dict(resume_state["ema"])
            logger.info("Restored EMA from full training checkpoint")
        elif resume_path is not None:
            ema_path = output_dir / f"{args.exp_id}_{resume_iteration}_ema.pth"
            if ema_path.exists():
                ema.load_state_dict(load_torch(ema_path, "cpu"))
            else:
                logger.warning("EMA checkpoint missing; initialized from resumed student")
        logger.info(
            "EMA enabled on rank0: decay=%.6f device=%s eval_use_ema=%s",
            args.ema_decay,
            ema_device,
            args.eval_use_ema,
        )

    optimizer = torch.optim.AdamW(
        student.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    if resume_state is not None:
        optimizer.load_state_dict(resume_state["optimizer"])
        move_optimizer_state(optimizer, device)
        logger.info("Restored optimizer state from iteration %d", resume_iteration)

    criterion = TeacherFeatureDriftingLoss(
        radii=parse_radii(args.radii),
        pool_tokens=args.pool_tokens,
        anchor_margin_alpha=args.anchor_margin_alpha,
        lambda_tfd=args.lambda_tfd,
        lambda_anchor=args.lambda_anchor,
    )
    feature_layers = parse_layers(args.feature_layers)
    use_amp = args.amp and device.type == "cuda"
    autocast_dtype = torch.bfloat16 if use_amp else torch.float32
    train_module: nn.Module = ResonateDriftingModule(
        student=student,
        teacher=teacher,
        criterion=criterion,
        feature_layers=feature_layers,
        feature_noise=args.feature_noise,
        lambda_flow=args.lambda_flow,
        samples_per_condition=args.samples_per_condition,
    ).to(device)
    if distributed:
        train_module = DistributedDataParallel(
            train_module,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    sampler_epoch = (
        int(resume_state.get("sampler_epoch", 0))
        if resume_state is not None
        else 0
    )
    resume_state = None
    if sampler is not None:
        sampler.set_epoch(sampler_epoch)
    data_iterator = iter(loader)
    start_iteration = resume_iteration + 1
    logger.info(
        "TRAIN_START mode=%s checkpoint=%s resume_iteration=%d "
        "start_iteration=%d target_iterations=%d output_dir=%s",
        "fresh" if resume_path is None else "resume",
        resume_path,
        resume_iteration,
        start_iteration,
        args.iterations,
        output_dir,
    )

    tqdm_position = int(os.environ.get("TQDM_POSITION", "0"))
    tqdm_desc = os.environ.get("TQDM_DESC", "resonate-drifting-train")
    eval_failure_fatal = os.environ.get("EVAL_FAILURE_FATAL", "0") == "1"
    if os.environ.get("QUIET_CONSOLE_AFTER_TQDM", "0") == "1":
        disable_console_logging(logger)
    progress = tqdm(
        total=args.iterations,
        initial=min(resume_iteration, args.iterations),
        desc=tqdm_desc,
        disable=not is_main,
        position=tqdm_position,
        leave=True,
        dynamic_ncols=True,
    )

    last_completed_iteration = resume_iteration
    for iteration in range(start_iteration, args.iterations + 1):
        try:
            batch = next(data_iterator)
        except StopIteration:
            sampler_epoch += 1
            if sampler is not None:
                sampler.set_epoch(sampler_epoch)
            data_iterator = iter(loader)
            batch = next(data_iterator)

        text_f = batch["text_features"].to(device=device, non_blocking=True)
        text_f_c = batch["text_features_c"].to(device=device, non_blocking=True)
        a_mean = batch["mean"].to(device=device, non_blocking=True)
        a_std = batch["std"].to(device=device, non_blocking=True)
        teacher_positives = batch["teacher_positives"].to(
            device=device,
            non_blocking=True,
        )

        with torch.autocast(
            device_type=device.type,
            dtype=autocast_dtype,
            enabled=use_amp,
        ):
            total_loss, loss_parts = train_module(
                text_f,
                text_f_c,
                a_mean,
                a_std,
                teacher_positives,
            )

        lr_scale = (
            min(iteration / args.lr_warmup_steps, 1.0)
            if args.lr_warmup_steps > 0
            else 1.0
        )
        for parameter_group in optimizer.param_groups:
            parameter_group["lr"] = args.learning_rate * lr_scale

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            student.parameters(),
            args.clip_grad_norm,
        )
        optimizer.step()
        if (
            ema is not None
            and iteration >= args.ema_start
            and iteration % args.ema_update_interval == 0
        ):
            ema.update(student)
        last_completed_iteration = iteration
        progress.update(1)

        should_log = iteration == 1 or iteration % args.log_interval == 0
        if should_log:
            reduced = {
                "total_loss": reduce_scalar(
                    total_loss,
                    distributed=distributed,
                    world_size=world_size,
                ),
                "flow_loss": reduce_scalar(
                    loss_parts["flow_loss"],
                    distributed=distributed,
                    world_size=world_size,
                ),
                "tfd_loss": reduce_scalar(
                    loss_parts["tfd_loss"],
                    distributed=distributed,
                    world_size=world_size,
                ),
                "drifting_loss": reduce_scalar(
                    loss_parts["drifting_loss"],
                    distributed=distributed,
                    world_size=world_size,
                ),
                "anchor_loss": reduce_scalar(
                    loss_parts["anchor_loss"],
                    distributed=distributed,
                    world_size=world_size,
                ),
            }
        if is_main and should_log:
            row = {
                "iteration": iteration,
                **{key: float(value.item()) for key, value in reduced.items()},
                "grad_norm": float(grad_norm.detach().float().item()),
                "lr": optimizer.param_groups[0]["lr"],
            }
            append_metrics(metrics_path, row)
            progress.set_postfix(
                {
                    "total": f"{row['total_loss']:.4f}",
                    "flow": f"{row['flow_loss']:.4f}",
                    "tfd": f"{row['tfd_loss']:.4f}",
                    "drift": f"{row['drifting_loss']:.4f}",
                    "anchor": f"{row['anchor_loss']:.4f}",
                    "grad": f"{row['grad_norm']:.3f}",
                    "lr": f"{row['lr']:.2e}",
                }
            )

        should_save = iteration % args.save_interval == 0
        if distributed and should_save:
            dist.barrier()
        if is_main and should_save:
            logger.info("CHECKPOINT_SAVE_START iteration=%d", iteration)
            with checkpoint_io_lock(args.output_root):
                atomic_torch_save(
                    module_state_dict_to_cpu(student),
                    output_dir / f"{args.exp_id}_{iteration}.pth",
                )
                if ema is not None:
                    atomic_torch_save(
                        ema.state_dict(),
                        output_dir / f"{args.exp_id}_{iteration}_ema.pth",
                    )
                save_training_state(
                    training_state_path,
                    iteration=iteration,
                    student=student,
                    optimizer=optimizer,
                    ema=ema,
                    sampler_epoch=sampler_epoch,
                )
            logger.info("CHECKPOINT_SAVE_DONE iteration=%d", iteration)
        if distributed and should_save:
            dist.barrier()

        should_eval = args.eval_interval > 0 and iteration % args.eval_interval == 0
        if distributed and should_eval:
            dist.barrier()
        if should_eval:
            console_states = suspend_console_logging(logger) if is_main else []
            eval_error: Exception | None = None
            eval_weight_path = output_dir / f"{args.exp_id}_{iteration}.pth"
            if is_main:
                if not eval_weight_path.exists():
                    with checkpoint_io_lock(args.output_root):
                        atomic_torch_save(
                            module_state_dict_to_cpu(student),
                            eval_weight_path,
                        )
                if ema is not None:
                    ema_eval_path = (
                        output_dir / f"{args.exp_id}_{iteration}_ema.pth"
                    )
                    if not ema_eval_path.exists():
                        with checkpoint_io_lock(args.output_root):
                            atomic_torch_save(ema.state_dict(), ema_eval_path)
                    if args.eval_use_ema:
                        eval_weight_path = ema_eval_path
            if distributed:
                dist.barrier()

            if args.eval_offload_train_state:
                if is_main:
                    optimizer.zero_grad(set_to_none=True)
                    teacher.to("cpu")
                    move_optimizer_state(optimizer, torch.device("cpu"))
                empty_cuda_cache()
            if distributed:
                dist.barrier()

            if is_main:
                eval_environment = {
                    "GT_AUDIO": str(args.eval_gt_audio),
                    "EVAL_TSV": str(args.eval_tsv),
                    "EVAL_NPZ_DIR": str(args.eval_npz_dir),
                    "VAE_WEIGHTS": str(args.eval_vae_weights),
                    "VOCODER_WEIGHTS": str(args.eval_vocoder_dir),
                    "DURATION": str(args.eval_duration),
                    "EVAL_LIMIT": (
                        "" if args.eval_limit is None else str(args.eval_limit)
                    ),
                    "EVAL_SKIP_AV_BENCHMARK": (
                        "1" if args.eval_skip_av_benchmark else "0"
                    ),
                }
                try:
                    with temporary_environment(eval_environment):
                        run_checkpoint_evaluation(
                            eval_entrypoint=Path("drifting/Resonate/test.py"),
                            iteration=iteration,
                            checkpoint_path=eval_weight_path,
                            output_root=args.eval_output_root,
                            exp_id=args.exp_id,
                            gt_cache=args.eval_gt_cache,
                            num_steps=args.eval_num_steps,
                            cfg_strength=args.eval_cfg_strength,
                            use_rope=args.use_rope,
                            eval_metrics_path=eval_metrics_path,
                            logger=logger,
                            stream_output=False,
                        )
                except Exception as error:
                    eval_error = error
                    logger.exception(
                        "Periodic eval failed at iteration %d; fatal=%s",
                        iteration,
                        eval_failure_fatal,
                    )
            if distributed:
                dist.barrier()

            if args.eval_offload_train_state:
                if is_main:
                    teacher.to(device)
                    move_optimizer_state(optimizer, device)
                    freeze_resonate_teacher(teacher)
                    student.train()
                    empty_cuda_cache()
                if distributed:
                    dist.barrier()
            if is_main:
                logger.info(
                    "TRAIN_RESUME_AFTER_EVAL iteration=%d "
                    "student_device=%s teacher_device=%s",
                    iteration,
                    student.device,
                    teacher.device,
                )
                restore_console_logging(console_states)
                progress.unpause()
                if eval_error is not None and eval_failure_fatal:
                    raise eval_error

    progress.close()
    if distributed:
        dist.barrier()
    if is_main:
        with checkpoint_io_lock(args.output_root):
            atomic_torch_save(
                module_state_dict_to_cpu(student),
                output_dir / f"{args.exp_id}_last.pth",
            )
            if ema is not None:
                atomic_torch_save(
                    ema.state_dict(),
                    output_dir / f"{args.exp_id}_ema_last.pth",
                )
            save_training_state(
                training_state_path,
                iteration=last_completed_iteration,
                student=student,
                optimizer=optimizer,
                ema=ema,
                sampler_epoch=sampler_epoch,
            )
        logger.info("Resonate drifting training completed")
    if distributed:
        dist.barrier()
    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
