#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import logging
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from drifting.eval_helpers import empty_cuda_cache, move_optimizer_state, run_checkpoint_evaluation
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


def setup_logger(output_dir: Path) -> logging.Logger:
    output_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("drifting.flux")
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
    parser.add_argument("--clip-grad-norm", type=float, default=1.0)
    parser.set_defaults(eval_offload_train_state=True)
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

    data_iter = iter(loader)
    for iteration in tqdm(range(1, args.iterations + 1), desc="flux-drifting-train"):
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
            total_loss, loss_parts = one_step_flux_loss(
                student=student,
                teacher=teacher,
                criterion=criterion,
                text_f=text_f,
                text_f_c=text_f_c,
                a_mean=a_mean,
                a_std=a_std,
                feature_layers=feature_layers,
                feature_noise=args.feature_noise,
                lambda_flow=args.lambda_flow,
            )

        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(student.parameters(), args.clip_grad_norm)
        optimizer.step()

        if iteration == 1 or iteration % args.log_interval == 0:
            row = {
                "iteration": iteration,
                "total_loss": float(total_loss.detach().float().item()),
                "flow_loss": float(loss_parts["flow_loss"].float().item()),
                "tfd_loss": float(loss_parts["tfd_loss"].float().item()),
                "drifting_loss": float(loss_parts["drifting_loss"].float().item()),
                "anchor_loss": float(loss_parts["anchor_loss"].float().item()),
                "grad_norm": float(grad_norm.detach().float().item()),
                "lr": optimizer.param_groups[0]["lr"],
            }
            append_metrics(metrics_path, row)
            logger.info(
                "it=%d total=%.6f flow=%.6f tfd=%.6f drift=%.6f anchor=%.6f grad=%.4f",
                iteration,
                row["total_loss"],
                row["flow_loss"],
                row["tfd_loss"],
                row["drifting_loss"],
                row["anchor_loss"],
                row["grad_norm"],
            )

        if iteration % args.save_interval == 0:
            weight_path = output_dir / f"{args.exp_id}_{iteration}.pth"
            torch.save(student.state_dict(), weight_path)
            logger.info("Saved weights to %s", weight_path)

        if args.eval_interval > 0 and iteration % args.eval_interval == 0:
            eval_weight_path = output_dir / f"{args.exp_id}_{iteration}.pth"
            torch.save(student.state_dict(), eval_weight_path)
            logger.info("Saved eval weights to %s", eval_weight_path)

            if args.eval_offload_train_state:
                logger.info("Offloading train state to CPU before eval.")
                student.to("cpu")
                teacher.to("cpu")
                move_optimizer_state(optimizer, torch.device("cpu"))
                empty_cuda_cache()
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
    logger.info("FluxAudio drifting training completed.")


if __name__ == "__main__":
    main()
