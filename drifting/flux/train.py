#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import logging
import os
import random
import sys
from datetime import timedelta
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from drifting.eval_helpers import (
    ExponentialMovingAverage,
    empty_cuda_cache,
    find_latest_weight_checkpoint,
    move_optimizer_state,
    run_checkpoint_evaluation,
)
from meanaudio.model.networks import get_mean_audio
from meanaudio.model.teacher_feature_drifting import TeacherFeatureDriftingLoss


def load_torch(path: Path, map_location: str | torch.device):
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


class AudioCapsNpzDataset(Dataset):
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
        data = np.load(self.npz_dir / f"{idx}.npz")
        return {
            "id": self.rows[idx]["id"],
            "caption": self.rows[idx]["caption"],
            "a_mean": torch.from_numpy(data["mean"]),
            "a_std": torch.from_numpy(data["std"]),
            "text_features": torch.from_numpy(data["text_features"]),
            "text_features_c": torch.from_numpy(data["text_features_c"]),
        }


def setup_logger(output_dir: Path, *, enabled: bool = True) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("drifting.flux")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    if not enabled:
        logger.addHandler(logging.NullHandler())
        return logger
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    log_prefix = os.environ.get("LOG_PREFIX", "").strip()
    stream_format = f"{log_prefix} %(asctime)s | %(levelname)s | %(message)s" if log_prefix else "%(asctime)s | %(levelname)s | %(message)s"
    stream_formatter = logging.Formatter(stream_format)

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(stream_formatter)
    file_handler = logging.FileHandler(output_dir / "train.log")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


def disable_console_logging(logger: logging.Logger) -> None:
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            handler.setLevel(logging.CRITICAL + 1)


def suspend_console_logging(logger: logging.Logger) -> list[tuple[logging.Handler, int]]:
    states: list[tuple[logging.Handler, int]] = []
    for handler in logger.handlers:
        if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
            states.append((handler, handler.level))
            handler.setLevel(logging.CRITICAL + 1)
    return states


def restore_console_logging(states: list[tuple[logging.Handler, int]]) -> None:
    for handler, level in states:
        handler.setLevel(level)


def append_metrics(path: Path, row: dict[str, Any]) -> None:
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def load_data_config(path: Path, split: str) -> dict[str, Any]:
    with path.open("r") as f:
        data_cfg = yaml.safe_load(f)
    if split not in data_cfg:
        raise KeyError(f"Split {split!r} not found in {path}")
    return data_cfg


def maybe_load_empty_features(
    weights_dir: Path,
    *,
    text_seq_len: int,
    text_dim: int,
    text_c_dim: int,
    logger: logging.Logger,
):
    text_path = weights_dir / "empty_string_t5.pth"
    text_c_path = weights_dir / "empty_string_clap_c.pth"
    if text_path.exists() and text_c_path.exists():
        empty_text = load_torch(text_path, "cpu")[0]
        empty_text_c = load_torch(text_c_path, "cpu")[0]
        logger.info("Loaded empty string features from %s", weights_dir)
        return empty_text, empty_text_c
    logger.warning("Empty string feature files not found in %s; using zeros.", weights_dir)
    return torch.zeros(text_seq_len, text_dim), torch.zeros(text_c_dim)


def freeze_module(module: torch.nn.Module) -> None:
    module.eval()
    for parameter in module.parameters():
        parameter.requires_grad_(False)


def parse_layers(text: str) -> tuple[str, ...]:
    layers = tuple(item.strip() for item in text.split(",") if item.strip())
    if not layers:
        raise ValueError("At least one feature layer is required.")
    return layers


def parse_radii(text: str) -> tuple[float, ...]:
    radii = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not radii:
        raise ValueError("At least one radius is required.")
    return radii


def setup_distributed() -> tuple[bool, int, int, int]:
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size <= 1:
        return False, 0, 0, 1
    rank = int(os.environ["RANK"])
    local_rank = int(os.environ["LOCAL_RANK"])
    timeout_minutes = int(os.environ.get("DDP_TIMEOUT_MINUTES", "180"))
    dist.init_process_group(backend="nccl", timeout=timedelta(minutes=timeout_minutes))
    torch.cuda.set_device(local_rank)
    return True, rank, local_rank, world_size


def cleanup_distributed(enabled: bool) -> None:
    if enabled and dist.is_initialized():
        dist.destroy_process_group()


def reduce_scalar(value: torch.Tensor, *, distributed: bool, world_size: int) -> torch.Tensor:
    value = value.detach().float()
    if distributed:
        dist.all_reduce(value, op=dist.ReduceOp.SUM)
        value = value / world_size
    return value


class FluxDriftingModule(nn.Module):
    def __init__(
        self,
        *,
        student: nn.Module,
        teacher: nn.Module,
        criterion: TeacherFeatureDriftingLoss,
        feature_layers: tuple[str, ...],
        feature_noise: float,
        lambda_flow: float,
    ) -> None:
        super().__init__()
        self.student = student
        self.teacher = teacher
        self.criterion = criterion
        self.feature_layers = feature_layers
        self.feature_noise = feature_noise
        self.lambda_flow = lambda_flow

    def forward(
        self,
        text_f: torch.Tensor,
        text_f_c: torch.Tensor,
        a_mean: torch.Tensor,
        a_std: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        return one_step_flux_loss(
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
        )


def build_flux_model(
    *,
    weights_path: Path,
    device: torch.device,
    latent_mean: torch.Tensor,
    latent_std: torch.Tensor,
    empty_text: torch.Tensor,
    empty_text_c: torch.Tensor,
    use_rope: bool,
):
    model = get_mean_audio(
        "fluxaudio_s",
        latent_mean=latent_mean,
        latent_std=latent_std,
        empty_string_feat=empty_text,
        empty_string_feat_c=empty_text_c,
        use_rope=use_rope,
        text_c_dim=512,
    ).to(device)
    model.load_weights(load_torch(weights_path, device))
    return model


def one_step_flux_loss(
    *,
    student,
    teacher,
    criterion: TeacherFeatureDriftingLoss,
    text_f: torch.Tensor,
    text_f_c: torch.Tensor,
    a_mean: torch.Tensor,
    a_std: torch.Tensor,
    feature_layers: tuple[str, ...],
    feature_noise: float,
    lambda_flow: float,
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    x_real = a_mean + a_std * torch.randn_like(a_mean)
    x_real = student.normalize(x_real)
    x_noise = torch.randn_like(x_real)

    t_one = torch.ones(x_real.shape[0], device=x_real.device, dtype=x_real.dtype)
    student_conditions = student.preprocess_conditions(text_f, text_f_c)
    pred_flow = student.predict_flow(x_noise, t_one, student_conditions)
    target_flow = x_noise - x_real
    flow_loss = (pred_flow - target_flow).pow(2).mean()
    x_student = x_noise - pred_flow

    sigma = torch.full((x_real.shape[0], 1, 1), feature_noise, device=x_real.device, dtype=x_real.dtype)
    x_real_feat = (1.0 - sigma) * x_real.detach() + sigma * torch.randn_like(x_real)
    x_student_feat = (1.0 - sigma) * x_student + sigma * torch.randn_like(x_student)
    feature_t = torch.full((x_real.shape[0],), feature_noise, device=x_real.device, dtype=x_real.dtype)

    teacher_conditions = teacher.preprocess_conditions(text_f, text_f_c)
    with torch.no_grad():
        positive_features = teacher.extract_features(
            x_real_feat,
            feature_t,
            teacher_conditions,
            layers=feature_layers,
        )
    generated_features = teacher.extract_features(
        x_student_feat,
        feature_t,
        teacher_conditions,
        layers=feature_layers,
    )
    tfd_output = criterion(generated_features, positive_features)
    total_loss = lambda_flow * flow_loss + tfd_output.loss
    return total_loss, {
        "flow_loss": flow_loss.detach(),
        "tfd_loss": tfd_output.loss.detach(),
        "drifting_loss": tfd_output.drifting_loss,
        "anchor_loss": tfd_output.anchor_loss,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train pure FluxAudio one-step student with FluxAudio teacher-feature drifting.")
    parser.add_argument("--exp-id", default="flux_drifting_s_1x4090")
    parser.add_argument("--output-root", type=Path, default=Path("exps/drifting_flux"))
    parser.add_argument("--data-config", type=Path, default=Path("config/data/t5_clap.yaml"))
    parser.add_argument("--train-split", default="AudioCaps_npz")
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument("--teacher-weights", type=Path, default=Path("weights/fluxaudio_s_full.pth"))
    parser.add_argument("--student-init", type=Path, default=Path("weights/fluxaudio_s_full.pth"))
    parser.add_argument("--latent-mean", type=Path, default=Path("sets/latent_mean.pt"))
    parser.add_argument("--latent-std", type=Path, default=Path("sets/latent_std.pt"))
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=1_000_000)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--weight-decay", type=float, default=1e-6)
    parser.add_argument("--seed", type=int, default=14159265)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--use-rope", action="store_true")
    parser.add_argument("--feature-layers", default="joint_3,fused_3,fused_7")
    parser.add_argument("--feature-noise", type=float, default=0.1)
    parser.add_argument("--pool-tokens", type=int, default=64)
    parser.add_argument("--radii", default="0.02,0.05,0.1,0.2")
    parser.add_argument("--lambda-flow", type=float, default=1.0)
    parser.add_argument("--lambda-tfd", type=float, default=0.2)
    parser.add_argument("--lambda-anchor", type=float, default=0.05)
    parser.add_argument("--anchor-margin-alpha", type=float, default=0.5)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=10000, help="Run full evaluation every N iterations; set 0 to disable.")
    parser.add_argument("--eval-output-root", type=Path, default=Path("exps/drifting_flux_eval"))
    parser.add_argument("--eval-gt-cache", type=Path, default=Path("data/audiocaps/test-features"))
    parser.add_argument("--eval-num-steps", type=int, default=1)
    parser.add_argument("--eval-cfg-strength", type=float, default=4.5)
    parser.add_argument("--no-eval-offload-train-state", dest="eval_offload_train_state", action="store_false")
    parser.add_argument("--disable-ema", dest="ema", action="store_false")
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--ema-start", type=int, default=0)
    parser.add_argument("--ema-update-interval", type=int, default=1)
    parser.add_argument("--ema-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--eval-raw", dest="eval_use_ema", action="store_false", help="Evaluate raw student weights instead of EMA weights.")
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.add_argument("--no-auto-resume", dest="auto_resume", action="store_false", help="Start from student init even if exp-id checkpoints exist.")
    parser.set_defaults(eval_offload_train_state=True)
    parser.set_defaults(ema=True, eval_use_ema=True, auto_resume=True)
    args = parser.parse_args()

    distributed, rank, local_rank, world_size = setup_distributed()
    is_main = rank == 0
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    device = torch.device(f"cuda:{local_rank}" if distributed else args.device)
    output_dir = args.output_root / args.exp_id
    logger = setup_logger(output_dir, enabled=is_main)
    metrics_path = output_dir / "metrics.csv"
    eval_metrics_path = output_dir / "eval_metrics.csv"
    logger.info("Writing logs to %s", output_dir / "train.log")
    logger.info("Writing metrics to %s", metrics_path)
    if args.eval_interval > 0:
        logger.info("Writing eval metrics to %s every %d iterations", eval_metrics_path, args.eval_interval)
    if distributed:
        logger.info("Distributed training enabled: world_size=%d", world_size)
    logger.info("Arguments: %s", vars(args))
    if args.ema and args.ema_update_interval < 1:
        raise ValueError("--ema-update-interval must be >= 1")
    if args.ema and args.ema_device == "cuda" and device.type != "cuda":
        raise ValueError("--ema-device cuda requires --device cuda")

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    resume_iteration = 0
    resume_path: Path | None = None
    if args.auto_resume:
        latest = None
        if is_main:
            latest = find_latest_weight_checkpoint(output_dir, args.exp_id)
        if distributed:
            payload: list[tuple[int, str] | None] = [
                None if latest is None else (latest[0], str(latest[1]))
            ]
            dist.broadcast_object_list(payload, src=0)
            latest_payload = payload[0]
            latest = None if latest_payload is None else (latest_payload[0], Path(latest_payload[1]))
        if latest is not None:
            resume_iteration, resume_path = latest
            logger.info("Auto-resume found checkpoint: it=%d path=%s", resume_iteration, resume_path)
            logger.info("Optimizer state is initialized fresh because existing checkpoints are weights-only.")
        else:
            logger.info("Auto-resume found no numeric checkpoint in %s; starting from student init.", output_dir)
    else:
        logger.info("Auto-resume disabled; starting from student init.")

    data_cfg = load_data_config(args.data_config, args.train_split)
    split_cfg = data_cfg[args.train_split]
    dataset = AudioCapsNpzDataset(tsv_path=Path(split_cfg["tsv"]), npz_dir=Path(split_cfg["npz_dir"]))
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    ) if distributed else None
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=(sampler is None),
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda"),
        drop_last=True,
    )
    if len(loader) == 0:
        raise ValueError("DataLoader is empty; reduce batch size or check dataset paths.")
    logger.info("Loaded %d training items from %s", len(dataset), split_cfg["npz_dir"])

    latent_mean = load_torch(args.latent_mean, "cpu")
    latent_std = load_torch(args.latent_std, "cpu")
    empty_text, empty_text_c = maybe_load_empty_features(
        args.weights_dir,
        text_seq_len=77,
        text_dim=1024,
        text_c_dim=512,
        logger=logger,
    )

    logger.info("Loading FluxAudio student init: %s", args.student_init)
    student = build_flux_model(
        weights_path=args.student_init,
        device=device,
        latent_mean=latent_mean,
        latent_std=latent_std,
        empty_text=empty_text,
        empty_text_c=empty_text_c,
        use_rope=args.use_rope,
    )
    if resume_path is not None:
        logger.info("Loading resumed FluxAudio student weights: %s", resume_path)
        student.load_weights(load_torch(resume_path, device))
    logger.info("Loading frozen FluxAudio teacher: %s", args.teacher_weights)
    teacher = build_flux_model(
        weights_path=args.teacher_weights,
        device=device,
        latent_mean=latent_mean,
        latent_std=latent_std,
        empty_text=empty_text,
        empty_text_c=empty_text_c,
        use_rope=args.use_rope,
    )
    freeze_module(teacher)
    student.train()
    ema_device = torch.device(args.ema_device if args.ema_device == "cpu" else device)
    ema = ExponentialMovingAverage(student, decay=args.ema_decay, device=ema_device) if args.ema else None
    if ema is not None:
        if resume_path is not None:
            ema_resume_path = output_dir / f"{args.exp_id}_{resume_iteration}_ema.pth"
            if ema_resume_path.exists():
                logger.info("Loading resumed EMA weights: %s", ema_resume_path)
                ema.load_state_dict(load_torch(ema_resume_path, "cpu"))
            else:
                logger.warning("EMA checkpoint not found at %s; initializing EMA from resumed student weights.", ema_resume_path)
        logger.info(
            "EMA enabled: decay=%.6f start=%d update_interval=%d device=%s eval_use_ema=%s",
            args.ema_decay,
            args.ema_start,
            args.ema_update_interval,
            ema_device,
            args.eval_use_ema,
        )

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
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
    train_module = FluxDriftingModule(
        student=student,
        teacher=teacher,
        criterion=criterion,
        feature_layers=feature_layers,
        feature_noise=args.feature_noise,
        lambda_flow=args.lambda_flow,
    ).to(device)
    if distributed:
        train_module = DistributedDataParallel(
            train_module,
            device_ids=[local_rank],
            output_device=local_rank,
            broadcast_buffers=False,
            find_unused_parameters=False,
        )

    sampler_epoch = 0
    if sampler is not None:
        sampler.set_epoch(sampler_epoch)
    data_iter = iter(loader)
    start_iteration = resume_iteration + 1
    if resume_path is None:
        logger.info(
            "TRAIN_START mode=fresh checkpoint=None start_iteration=%d target_iterations=%d output_dir=%s",
            start_iteration,
            args.iterations,
            output_dir,
        )
    else:
        logger.info(
            "TRAIN_START mode=resume checkpoint=%s resume_iteration=%d start_iteration=%d target_iterations=%d output_dir=%s",
            resume_path,
            resume_iteration,
            start_iteration,
            args.iterations,
            output_dir,
        )
    if start_iteration > args.iterations:
        logger.info(
            "Resume checkpoint iteration %d is already >= target iterations %d; no training steps will run.",
            resume_iteration,
            args.iterations,
        )
    tqdm_position = int(os.environ.get("TQDM_POSITION", "0"))
    tqdm_desc = os.environ.get("TQDM_DESC", "flux-drifting-train")
    eval_console_output = os.environ.get("EVAL_CONSOLE_OUTPUT", "1") == "1"
    eval_failure_fatal = os.environ.get("EVAL_FAILURE_FATAL", "1") == "1"
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
    for iteration in range(start_iteration, args.iterations + 1):
        try:
            batch = next(data_iter)
        except StopIteration:
            sampler_epoch += 1
            if sampler is not None:
                sampler.set_epoch(sampler_epoch)
            data_iter = iter(loader)
            batch = next(data_iter)

        text_f = batch["text_features"].to(device=device, non_blocking=True)
        text_f_c = batch["text_features_c"].to(device=device, non_blocking=True)
        a_mean = batch["a_mean"].to(device=device, non_blocking=True)
        a_std = batch["a_std"].to(device=device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
            total_loss, loss_parts = train_module(text_f, text_f_c, a_mean, a_std)

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), args.clip_grad_norm)
        optimizer.step()
        if ema is not None and iteration >= args.ema_start and iteration % args.ema_update_interval == 0:
            ema.update(student)

        should_log = iteration == 1 or iteration % args.log_interval == 0
        if should_log:
            reduced_total_loss = reduce_scalar(total_loss, distributed=distributed, world_size=world_size)
            reduced_flow_loss = reduce_scalar(loss_parts["flow_loss"], distributed=distributed, world_size=world_size)
            reduced_tfd_loss = reduce_scalar(loss_parts["tfd_loss"], distributed=distributed, world_size=world_size)
            reduced_drifting_loss = reduce_scalar(loss_parts["drifting_loss"], distributed=distributed, world_size=world_size)
            reduced_anchor_loss = reduce_scalar(loss_parts["anchor_loss"], distributed=distributed, world_size=world_size)

        if is_main and should_log:
            row = {
                "iteration": iteration,
                "total_loss": float(reduced_total_loss.item()),
                "flow_loss": float(reduced_flow_loss.item()),
                "tfd_loss": float(reduced_tfd_loss.item()),
                "drifting_loss": float(reduced_drifting_loss.item()),
                "anchor_loss": float(reduced_anchor_loss.item()),
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

        if is_main and iteration % args.save_interval == 0:
            weight_path = output_dir / f"{args.exp_id}_{iteration}.pth"
            torch.save(student.state_dict(), weight_path)
            logger.info("Saved weights to %s", weight_path)
            if ema is not None:
                ema_weight_path = output_dir / f"{args.exp_id}_{iteration}_ema.pth"
                torch.save(ema.state_dict(), ema_weight_path)
                logger.info("Saved EMA weights to %s", ema_weight_path)

        progress.update(1)

        should_eval = args.eval_interval > 0 and iteration % args.eval_interval == 0
        if distributed and should_eval:
            dist.barrier()

        if is_main and should_eval:
            console_states = [] if eval_console_output else suspend_console_logging(logger)
            try:
                eval_weight_path = output_dir / f"{args.exp_id}_{iteration}.pth"
                torch.save(student.state_dict(), eval_weight_path)
                logger.info("Saved eval raw weights to %s", eval_weight_path)
                if ema is not None:
                    ema_eval_weight_path = output_dir / f"{args.exp_id}_{iteration}_ema.pth"
                    torch.save(ema.state_dict(), ema_eval_weight_path)
                    logger.info("Saved eval EMA weights to %s", ema_eval_weight_path)
                    if args.eval_use_ema:
                        eval_weight_path = ema_eval_weight_path

                if args.eval_offload_train_state:
                    logger.info("Offloading train state to CPU before eval.")
                    student.to("cpu")
                    teacher.to("cpu")
                    move_optimizer_state(optimizer, torch.device("cpu"))
                    empty_cuda_cache()
                progress.clear()
                try:
                    try:
                        run_checkpoint_evaluation(
                            eval_entrypoint=Path("drifting/flux/test.py"),
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
                            stream_output=eval_console_output,
                        )
                    except Exception:
                        logger.exception(
                            "Periodic eval failed at iteration %d; fatal=%s. "
                            "See the eval driver log for details.",
                            iteration,
                            eval_failure_fatal,
                        )
                        if eval_failure_fatal:
                            raise
                finally:
                    if args.eval_offload_train_state:
                        logger.info("Restoring train state to %s after eval.", device)
                        student.to(device)
                        teacher.to(device)
                        move_optimizer_state(optimizer, device)
                        freeze_module(teacher)
                        student.train()
                        empty_cuda_cache()
            finally:
                restore_console_logging(console_states)
                progress.refresh()

        if distributed and should_eval:
            dist.barrier()

    progress.close()

    if is_main:
        last_path = output_dir / f"{args.exp_id}_last.pth"
        torch.save(student.state_dict(), last_path)
        logger.info("Saved final weights to %s", last_path)
        if ema is not None:
            ema_last_path = output_dir / f"{args.exp_id}_ema_last.pth"
            torch.save(ema.state_dict(), ema_last_path)
            logger.info("Saved final EMA weights to %s", ema_last_path)
        logger.info("FluxAudio drifting training completed.")
    cleanup_distributed(distributed)


if __name__ == "__main__":
    main()
