#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import logging
import math
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from meanaudio.model.mean_flow import MeanFlow
from meanaudio.model.networks import get_mean_audio
from meanaudio.model.teacher_feature_drifting import TeacherFeatureDriftingLoss
from drifting.eval_helpers import empty_cuda_cache, move_optimizer_state, run_checkpoint_evaluation


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
        first_npz = npz_dir / "0.npz"
        if not first_npz.exists():
            raise FileNotFoundError(f"Missing sample npz: {first_npz}")

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


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("drifting")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    file_handler = logging.FileHandler(output_dir / "train.log")
    file_handler.setFormatter(formatter)

    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)
    return logger


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


def maybe_load_empty_features(weights_dir: Path, text_seq_len: int, text_dim: int, text_c_dim: int, logger: logging.Logger):
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


class ExponentialMovingAverage:
    def __init__(self, model: torch.nn.Module, *, decay: float, device: torch.device) -> None:
        if not 0.0 <= decay < 1.0:
            raise ValueError(f"EMA decay must be in [0, 1), got {decay}")
        self.decay = decay
        self.device = device
        self.num_updates = 0
        self.shadow: dict[str, torch.Tensor] = {
            key: value.detach().to(device=device).clone()
            for key, value in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.num_updates += 1
        model_state = model.state_dict()
        for key, value in model_state.items():
            value = value.detach()
            shadow = self.shadow[key]
            if torch.is_floating_point(shadow):
                shadow.mul_(self.decay).add_(value.to(device=self.device, dtype=shadow.dtype), alpha=1.0 - self.decay)
            else:
                shadow.copy_(value.to(device=self.device))

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            key: value.detach().cpu().clone()
            for key, value in self.shadow.items()
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a one-step MeanAudio student with FluxAudio teacher-feature drifting.")
    parser.add_argument("--exp-id", default="drifting_fluxaudio_s")
    parser.add_argument("--output-root", type=Path, default=Path("exps/drifting"))
    parser.add_argument("--data-config", type=Path, default=Path("config/data/t5_clap.yaml"))
    parser.add_argument("--train-split", default="AudioCaps_npz")
    parser.add_argument("--weights-dir", type=Path, default=Path("weights"))
    parser.add_argument("--teacher-weights", type=Path, default=Path("weights/fluxaudio_s_full.pth"))
    parser.add_argument("--student-init", type=Path, default=Path("weights/fluxaudio_s_full.pth"))
    parser.add_argument("--latent-mean", type=Path, default=Path("sets/latent_mean.pt"))
    parser.add_argument("--latent-std", type=Path, default=Path("sets/latent_std.pt"))
    parser.add_argument("--model", default="meanaudio_s")
    parser.add_argument("--teacher-model", default="fluxaudio_s")
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
    parser.add_argument("--lambda-mf", type=float, default=1.0)
    parser.add_argument("--lambda-tfd", type=float, default=0.2)
    parser.add_argument("--lambda-anchor", type=float, default=0.05)
    parser.add_argument("--anchor-margin-alpha", type=float, default=0.5)
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--save-interval", type=int, default=1000)
    parser.add_argument("--eval-interval", type=int, default=10000, help="Run full evaluation every N iterations; set 0 to disable.")
    parser.add_argument("--eval-output-root", type=Path, default=Path("exps/drifting_eval"))
    parser.add_argument("--eval-gt-cache", type=Path, default=Path("data/audiocaps/test-features"))
    parser.add_argument("--eval-num-steps", type=int, default=1)
    parser.add_argument("--eval-cfg-strength", type=float, default=0.9)
    parser.add_argument("--no-eval-offload-train-state", dest="eval_offload_train_state", action="store_false")
    parser.add_argument("--disable-ema", dest="ema", action="store_false")
    parser.add_argument("--ema-decay", type=float, default=0.9999)
    parser.add_argument("--ema-start", type=int, default=0)
    parser.add_argument("--ema-update-interval", type=int, default=1)
    parser.add_argument("--ema-device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--eval-raw", dest="eval_use_ema", action="store_false", help="Evaluate raw student weights instead of EMA weights.")
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.set_defaults(eval_offload_train_state=True)
    parser.set_defaults(ema=True, eval_use_ema=True)
    args = parser.parse_args()

    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available.")
    device = torch.device(args.device)
    output_dir = args.output_root / args.exp_id
    logger = setup_logger(output_dir)
    metrics_path = output_dir / "metrics.csv"
    eval_metrics_path = output_dir / "eval_metrics.csv"
    logger.info("Writing logs to %s", output_dir / "train.log")
    logger.info("Writing metrics to %s", metrics_path)
    if args.eval_interval > 0:
        logger.info("Writing eval metrics to %s every %d iterations", eval_metrics_path, args.eval_interval)
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

    data_cfg = load_data_config(args.data_config, args.train_split)
    split_cfg = data_cfg[args.train_split]
    dataset = AudioCapsNpzDataset(tsv_path=Path(split_cfg["tsv"]), npz_dir=Path(split_cfg["npz_dir"]))
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
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

    student = get_mean_audio(
        args.model,
        latent_mean=latent_mean,
        latent_std=latent_std,
        empty_string_feat=empty_text,
        empty_string_feat_c=empty_text_c,
        use_rope=args.use_rope,
        text_c_dim=512,
    ).to(device)
    teacher = get_mean_audio(
        args.teacher_model,
        latent_mean=latent_mean,
        latent_std=latent_std,
        empty_string_feat=empty_text,
        empty_string_feat_c=empty_text_c,
        use_rope=args.use_rope,
        text_c_dim=512,
    ).to(device)

    logger.info("Loading student init: %s", args.student_init)
    student.load_weights(load_torch(args.student_init, device))
    logger.info("Loading teacher weights: %s", args.teacher_weights)
    teacher.load_weights(load_torch(args.teacher_weights, device))
    freeze_module(teacher)
    student.train()
    ema_device = torch.device(args.ema_device if args.ema_device == "cpu" else device)
    ema = ExponentialMovingAverage(student, decay=args.ema_decay, device=ema_device) if args.ema else None
    if ema is not None:
        logger.info(
            "EMA enabled: decay=%.6f start=%d update_interval=%d device=%s eval_use_ema=%s",
            args.ema_decay,
            args.ema_start,
            args.ema_update_interval,
            ema_device,
            args.eval_use_ema,
        )

    optimizer = torch.optim.AdamW(student.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    meanflow = MeanFlow()
    tfd_loss = TeacherFeatureDriftingLoss(
        radii=parse_radii(args.radii),
        pool_tokens=args.pool_tokens,
        anchor_margin_alpha=args.anchor_margin_alpha,
        lambda_tfd=args.lambda_tfd,
        lambda_anchor=args.lambda_anchor,
    )
    feature_layers = parse_layers(args.feature_layers)
    use_amp = args.amp and device.type == "cuda"
    autocast_dtype = torch.bfloat16 if use_amp else torch.float32

    data_iter = iter(loader)
    for iteration in tqdm(range(1, args.iterations + 1), desc="drifting-train"):
        try:
            batch = next(data_iter)
        except StopIteration:
            data_iter = iter(loader)
            batch = next(data_iter)

        text_f = batch["text_features"].to(device=device, non_blocking=True)
        text_f_c = batch["text_features_c"].to(device=device, non_blocking=True)
        a_mean = batch["a_mean"].to(device=device, non_blocking=True)
        a_std = batch["a_std"].to(device=device, non_blocking=True)

        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=use_amp):
            x_real = a_mean + a_std * torch.randn_like(a_mean)
            x_real = student.normalize(x_real)

            mf_loss_vec, _, _ = meanflow.loss(
                student,
                x_real,
                text_f.clone(),
                text_f_c.clone(),
                text_f,
                text_f_c,
                student.empty_string_feat,
                student.empty_string_feat_c,
            )
            mf_loss = mf_loss_vec.mean()

            conditions_student = student.preprocess_conditions(text_f, text_f_c)
            x_noise = torch.randn_like(x_real)
            ones = torch.ones(x_real.shape[0], device=device, dtype=x_real.dtype)
            zeros = torch.zeros_like(ones)
            student_flow = student.predict_flow(x_noise, ones, zeros, conditions_student)
            x_student = x_noise - student_flow

            sigma = torch.full((x_real.shape[0], 1, 1), args.feature_noise, device=device, dtype=x_real.dtype)
            x_real_feat = (1.0 - sigma) * x_real.detach() + sigma * torch.randn_like(x_real)
            x_student_feat = (1.0 - sigma) * x_student + sigma * torch.randn_like(x_student)

            teacher_conditions = teacher.preprocess_conditions(text_f, text_f_c)
            feature_t = torch.full((x_real.shape[0],), args.feature_noise, device=device, dtype=x_real.dtype)
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
            tfd_output = tfd_loss(generated_features, positive_features)
            total_loss = args.lambda_mf * mf_loss + tfd_output.loss

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), args.clip_grad_norm)
        optimizer.step()
        if ema is not None and iteration >= args.ema_start and iteration % args.ema_update_interval == 0:
            ema.update(student)

        if iteration == 1 or iteration % args.log_interval == 0:
            row = {
                "iteration": iteration,
                "total_loss": float(total_loss.detach().float().item()),
                "meanflow_loss": float(mf_loss.detach().float().item()),
                "tfd_loss": float(tfd_output.loss.detach().float().item()),
                "drifting_loss": float(tfd_output.drifting_loss.float().item()),
                "anchor_loss": float(tfd_output.anchor_loss.float().item()),
                "grad_norm": float(grad_norm.detach().float().item()),
                "lr": optimizer.param_groups[0]["lr"],
            }
            append_metrics(metrics_path, row)
            logger.info(
                "it=%d total=%.6f mf=%.6f tfd=%.6f drift=%.6f anchor=%.6f grad=%.4f",
                iteration,
                row["total_loss"],
                row["meanflow_loss"],
                row["tfd_loss"],
                row["drifting_loss"],
                row["anchor_loss"],
                row["grad_norm"],
            )

        if iteration % args.save_interval == 0:
            weight_path = output_dir / f"{args.exp_id}_{iteration}.pth"
            torch.save(student.state_dict(), weight_path)
            logger.info("Saved weights to %s", weight_path)
            if ema is not None:
                ema_weight_path = output_dir / f"{args.exp_id}_{iteration}_ema.pth"
                torch.save(ema.state_dict(), ema_weight_path)
                logger.info("Saved EMA weights to %s", ema_weight_path)

        if args.eval_interval > 0 and iteration % args.eval_interval == 0:
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
            try:
                run_checkpoint_evaluation(
                    eval_entrypoint=Path("drifting/test.py"),
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
                )
            finally:
                if args.eval_offload_train_state:
                    logger.info("Restoring train state to %s after eval.", device)
                    student.to(device)
                    teacher.to(device)
                    move_optimizer_state(optimizer, device)
                    freeze_module(teacher)
                    student.train()
                    empty_cuda_cache()

    last_path = output_dir / f"{args.exp_id}_last.pth"
    torch.save(student.state_dict(), last_path)
    logger.info("Saved final weights to %s", last_path)
    if ema is not None:
        ema_last_path = output_dir / f"{args.exp_id}_ema_last.pth"
        torch.save(ema.state_dict(), ema_last_path)
        logger.info("Saved final EMA weights to %s", ema_last_path)
    logger.info("Training completed.")


if __name__ == "__main__":
    main()
