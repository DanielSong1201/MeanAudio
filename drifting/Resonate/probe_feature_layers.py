#!/usr/bin/env python
from __future__ import annotations

import argparse
import csv
import logging
import math
import random
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from drifting.Resonate.config import RESONATE_CONFIG
from drifting.Resonate.data import ResonateNpzDataset
from drifting.Resonate.train import (
    build_training_model,
    load_data_config,
    load_torch,
    parse_layers,
)


DEFAULT_LAYERS = (
    "audio_proj",
    "joint_3",
    "joint_5",
    "joint_7",
    "joint_11",
    "joint_15",
    "fused_0",
    "fused_8",
    "fused_17",
    "fused_26",
    "fused_35",
)


def setup_logging() -> logging.Logger:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    return logging.getLogger("resonate-layer-probe")


def parse_feature_noises(text: str) -> tuple[float, ...]:
    values = tuple(float(item.strip()) for item in text.split(",") if item.strip())
    if not values:
        raise ValueError("At least one feature-noise value is required")
    if any(value < 0 for value in values):
        raise ValueError("Feature-noise values must be non-negative")
    return values


def collate_batch(items: list[dict[str, Any]]) -> dict[str, Any]:
    batch: dict[str, Any] = {}
    for key in ("mean", "std", "text_features", "text_features_c", "teacher_positives"):
        if key in items[0]:
            batch[key] = torch.stack([item[key] for item in items], dim=0)
    batch["id"] = [item["id"] for item in items]
    batch["caption"] = [item["caption"] for item in items]
    return batch


def select_indices(
    dataset_size: int,
    *,
    sample_count: int,
    seed: int,
    strategy: str,
) -> list[int]:
    if sample_count < 1:
        raise ValueError("--sample-count must be >= 1")
    sample_count = min(sample_count, dataset_size)
    if strategy == "first":
        return list(range(sample_count))
    if strategy == "random":
        rng = random.Random(seed)
        return sorted(rng.sample(range(dataset_size), sample_count))
    if strategy == "stride":
        if sample_count == dataset_size:
            return list(range(dataset_size))
        step = dataset_size / sample_count
        return sorted({min(dataset_size - 1, int(i * step)) for i in range(sample_count)})
    raise ValueError(f"Unknown sample strategy: {strategy}")


def pool_and_normalize(feature: torch.Tensor, pool_tokens: int | None) -> torch.Tensor:
    if feature.ndim != 3:
        raise ValueError(f"Expected [samples, tokens, dim], got {tuple(feature.shape)}")
    if pool_tokens is not None and feature.shape[1] > pool_tokens:
        feature = F.adaptive_avg_pool1d(
            feature.transpose(1, 2).float(),
            pool_tokens,
        ).transpose(1, 2).to(dtype=feature.dtype)
    return F.normalize(feature.float(), dim=-1)


def pairwise_prompt_distance(feature: torch.Tensor, pool_tokens: int | None) -> torch.Tensor:
    prepared = pool_and_normalize(feature, pool_tokens)
    if prepared.shape[0] < 2:
        return torch.zeros((), device=feature.device)
    pair_terms = []
    for i in range(prepared.shape[0]):
        for j in range(i + 1, prepared.shape[0]):
            pair_terms.append((prepared[i] - prepared[j]).pow(2).mean())
    return torch.stack(pair_terms).mean()


def feature_centroid(feature: torch.Tensor, pool_tokens: int | None) -> torch.Tensor:
    prepared = pool_and_normalize(feature, pool_tokens)
    return prepared.mean(dim=(0, 1))


def centroid_distance(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return (left.float() - right.float()).pow(2).mean()


def metric_bucket() -> dict[str, list[float]]:
    return defaultdict(list)


def append_metric(metrics: dict[str, dict[str, list[float]]], layer: str, key: str, value: torch.Tensor | float) -> None:
    if torch.is_tensor(value):
        value = float(value.detach().float().cpu().item())
    if math.isfinite(float(value)):
        metrics[layer][key].append(float(value))


def mean_or_nan(values: list[float]) -> float:
    if not values:
        return float("nan")
    return float(np.mean(values))


def summarize(metrics: dict[str, dict[str, list[float]]], layers: tuple[str, ...]) -> list[dict[str, Any]]:
    rows = []
    for layer in layers:
        item = metrics[layer]
        matched = mean_or_nan(item["matched_dist"])
        shuffled = mean_or_nan(item["shuffled_dist"])
        ratio = matched / shuffled if shuffled and math.isfinite(shuffled) else float("nan")
        rows.append(
            {
                "layer": layer,
                "samples": len(item["matched_dist"]),
                "matched_dist": matched,
                "shuffled_dist": shuffled,
                "matched_to_shuffled": ratio,
                "gen_pos_gap": mean_or_nan(item["gen_pos_gap"]),
                "noise_shift_0_to_01": mean_or_nan(item["noise_shift_0_to_01"]),
                "noise_shift_01_to_02": mean_or_nan(item["noise_shift_01_to_02"]),
                "grad_norm": mean_or_nan(item["grad_norm"]),
            }
        )
    return rows


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Probe candidate Resonate teacher hidden layers before full TFD training."
        )
    )
    parser.add_argument("--data-config", type=Path, default=Path("config/data/resonate_flant5_44k.yaml"))
    parser.add_argument("--train-split", default="AudioCaps_npz")
    parser.add_argument(
        "--teacher-positive-dir",
        type=Path,
        default=Path("data/audiocaps_resonate/train-teacher-positives-resonate-grpo-25step-cfg4.5"),
    )
    parser.add_argument("--teacher-positive-count", type=int, default=3)
    parser.add_argument("--teacher-weights", type=Path, default=RESONATE_CONFIG.teacher_weights)
    parser.add_argument("--student-init", type=Path, default=RESONATE_CONFIG.student_init)
    parser.add_argument("--latent-mean", type=Path, default=RESONATE_CONFIG.latent_mean)
    parser.add_argument("--latent-std", type=Path, default=RESONATE_CONFIG.latent_std)
    parser.add_argument("--layers", default=",".join(DEFAULT_LAYERS))
    parser.add_argument("--feature-noises", default="0.0,0.05,0.1,0.2")
    parser.add_argument("--main-feature-noise", type=float, default=0.1)
    parser.add_argument("--pool-tokens", type=int, default=64)
    parser.add_argument("--sample-count", type=int, default=64)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--sample-strategy", choices=("first", "random", "stride"), default="stride")
    parser.add_argument("--seed", type=int, default=14159265)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--dtype", choices=("float32", "bfloat16"), default="bfloat16")
    parser.add_argument("--use-rope", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--samples-per-condition", type=int, default=4)
    parser.add_argument("--grad-sample-count", type=int, default=16)
    parser.add_argument("--output-csv", type=Path, default=Path("exps/drifting_resonate_layer_probe/feature_layer_metrics.csv"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    logger = setup_logging()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    if args.batch_size != 1:
        logger.warning(
            "This probing script reports prompt-local metrics. batch_size=%d is supported, "
            "but batch_size=1 is easiest to interpret.",
            args.batch_size,
        )

    layers = parse_layers(args.layers)
    feature_noises = parse_feature_noises(args.feature_noises)
    if args.main_feature_noise not in feature_noises:
        feature_noises = tuple(sorted((*feature_noises, args.main_feature_noise)))
    if args.samples_per_condition < 1:
        raise ValueError("--samples-per-condition must be >= 1")
    if args.teacher_positive_count + 1 < args.samples_per_condition:
        logger.warning(
            "samples_per_condition=%d but only %d positives are available. "
            "Generation gap will still use %d generated samples.",
            args.samples_per_condition,
            args.teacher_positive_count + 1,
            args.samples_per_condition,
        )

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    data_config = load_data_config(args.data_config, args.train_split)
    split_config = data_config[args.train_split]
    dataset = ResonateNpzDataset(
        tsv_path=Path(split_config["tsv"]),
        npz_dir=Path(split_config["npz_dir"]),
        teacher_positive_dir=args.teacher_positive_dir,
        teacher_positive_count=args.teacher_positive_count,
    )
    indices = select_indices(
        len(dataset),
        sample_count=args.sample_count,
        seed=args.seed,
        strategy=args.sample_strategy,
    )
    subset = Subset(dataset, indices)
    loader = DataLoader(
        subset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_batch,
    )

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.bfloat16
    latent_mean = load_torch(args.latent_mean, "cpu")
    latent_std = load_torch(args.latent_std, "cpu")
    logger.info("Loading frozen Resonate teacher: %s", args.teacher_weights)
    teacher = build_training_model(
        weights_path=args.teacher_weights,
        device=device,
        latent_mean=latent_mean,
        latent_std=latent_std,
        use_rope=args.use_rope,
    ).to(device, dtype).eval()
    logger.info("Loading Resonate student for one-step generated probes: %s", args.student_init)
    student = build_training_model(
        weights_path=args.student_init,
        device=device,
        latent_mean=latent_mean,
        latent_std=latent_std,
        use_rope=args.use_rope,
    ).to(device, dtype).eval()
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    for parameter in student.parameters():
        parameter.requires_grad_(False)

    logger.info("Dataset items=%d, probing samples=%d", len(dataset), len(indices))
    logger.info("Candidate layers=%s", ",".join(layers))
    logger.info("Feature noises=%s", ",".join(str(value) for value in feature_noises))
    logger.info("Output CSV=%s", args.output_csv)

    metrics: dict[str, dict[str, list[float]]] = defaultdict(metric_bucket)
    previous_centroids: dict[tuple[str, float], torch.Tensor] = {}
    processed_conditions = 0

    progress = tqdm(loader, desc="resonate-layer-probe", unit="batch")
    for batch in progress:
        text_f = batch["text_features"].to(device=device, dtype=dtype)
        text_f_c = batch["text_features_c"].to(device=device, dtype=dtype)
        a_mean = batch["mean"].to(device=device, dtype=dtype)
        a_std = batch["std"].to(device=device, dtype=dtype)
        teacher_positives = batch["teacher_positives"].to(device=device, dtype=dtype)
        condition_batch_size = a_mean.shape[0]
        latent_seq_len = a_mean.shape[1]
        teacher.update_seq_lengths(latent_seq_len)
        student.update_seq_lengths(latent_seq_len)

        real_positive = student.normalize(a_mean + a_std * torch.randn_like(a_mean)).unsqueeze(1)
        positive_latents = torch.cat([real_positive, teacher_positives], dim=1)
        positive_count = positive_latents.shape[1]
        positive_flat = positive_latents.reshape(
            condition_batch_size * positive_count,
            *positive_latents.shape[2:],
        )
        positive_text_f = (
            text_f.unsqueeze(1)
            .expand(-1, positive_count, *text_f.shape[1:])
            .reshape(condition_batch_size * positive_count, *text_f.shape[1:])
        )
        positive_text_f_c = (
            text_f_c.unsqueeze(1)
            .expand(-1, positive_count, *text_f_c.shape[1:])
            .reshape(condition_batch_size * positive_count, *text_f_c.shape[1:])
        )
        positive_conditions = teacher.preprocess_conditions(positive_text_f, positive_text_f_c)

        positive_features_by_noise: dict[float, dict[str, torch.Tensor]] = {}
        centroids_by_noise: dict[float, dict[str, torch.Tensor]] = {}
        with torch.no_grad():
            for noise in feature_noises:
                sigma = torch.full(
                    (positive_flat.shape[0], 1, 1),
                    noise,
                    device=device,
                    dtype=dtype,
                )
                positive_noisy = (
                    (1.0 - sigma) * positive_flat
                    + sigma * torch.randn_like(positive_flat)
                )
                feature_t = torch.full(
                    (positive_flat.shape[0],),
                    noise,
                    device=device,
                    dtype=dtype,
                )
                extracted = teacher.extract_features(
                    positive_noisy,
                    feature_t,
                    positive_conditions,
                    layers=layers,
                )
                positive_features_by_noise[noise] = {
                    layer: value.reshape(
                        condition_batch_size,
                        positive_count,
                        *value.shape[1:],
                    )
                    for layer, value in extracted.items()
                }
                centroids_by_noise[noise] = {}
                for layer, value in positive_features_by_noise[noise].items():
                    # This script is intended for prompt-local probing. For batch_size>1
                    # we average per-condition metrics into the same layer bucket.
                    for condition_index in range(condition_batch_size):
                        feature = value[condition_index]
                        if noise == args.main_feature_noise:
                            append_metric(
                                metrics,
                                layer,
                                "matched_dist",
                                pairwise_prompt_distance(feature, args.pool_tokens),
                            )
                        centroid = feature_centroid(feature, args.pool_tokens)
                        centroids_by_noise[noise][f"{layer}:{condition_index}"] = centroid
                        previous = previous_centroids.get((layer, noise))
                        if previous is not None and noise == args.main_feature_noise:
                            append_metric(
                                metrics,
                                layer,
                                "shuffled_dist",
                                centroid_distance(centroid, previous),
                            )
                        previous_centroids[(layer, noise)] = centroid.detach()

        if 0.0 in centroids_by_noise and 0.1 in centroids_by_noise:
            for layer in layers:
                for condition_index in range(condition_batch_size):
                    key = f"{layer}:{condition_index}"
                    append_metric(
                        metrics,
                        layer,
                        "noise_shift_0_to_01",
                        centroid_distance(
                            centroids_by_noise[0.0][key],
                            centroids_by_noise[0.1][key],
                        ),
                    )
        if 0.1 in centroids_by_noise and 0.2 in centroids_by_noise:
            for layer in layers:
                for condition_index in range(condition_batch_size):
                    key = f"{layer}:{condition_index}"
                    append_metric(
                        metrics,
                        layer,
                        "noise_shift_01_to_02",
                        centroid_distance(
                            centroids_by_noise[0.1][key],
                            centroids_by_noise[0.2][key],
                        ),
                    )

        samples_per_condition = args.samples_per_condition
        gen_text_f = (
            text_f.unsqueeze(1)
            .expand(-1, samples_per_condition, *text_f.shape[1:])
            .reshape(condition_batch_size * samples_per_condition, *text_f.shape[1:])
        )
        gen_text_f_c = (
            text_f_c.unsqueeze(1)
            .expand(-1, samples_per_condition, *text_f_c.shape[1:])
            .reshape(condition_batch_size * samples_per_condition, *text_f_c.shape[1:])
        )
        with torch.no_grad():
            x_noise = torch.randn(
                condition_batch_size * samples_per_condition,
                *a_mean.shape[1:],
                device=device,
                dtype=dtype,
            )
            t_one = torch.ones(
                x_noise.shape[0],
                device=device,
                dtype=dtype,
            )
            student_conditions = student.preprocess_conditions(gen_text_f, gen_text_f_c)
            pred_flow = student.predict_flow(x_noise, t_one, student_conditions)
            x_student = x_noise - pred_flow

        gen_sigma = torch.full(
            (x_student.shape[0], 1, 1),
            args.main_feature_noise,
            device=device,
            dtype=dtype,
        )
        x_student_feat = (
            (1.0 - gen_sigma) * x_student
            + gen_sigma * torch.randn_like(x_student)
        )
        x_student_feat = x_student_feat.detach().requires_grad_(True)
        generated_conditions = teacher.preprocess_conditions(gen_text_f, gen_text_f_c)
        generated_t = torch.full(
            (x_student_feat.shape[0],),
            args.main_feature_noise,
            device=device,
            dtype=dtype,
        )
        generated_features = teacher.extract_features(
            x_student_feat,
            generated_t,
            generated_conditions,
            layers=layers,
        )
        for layer, generated in generated_features.items():
            generated_grouped = generated.reshape(
                condition_batch_size,
                samples_per_condition,
                *generated.shape[1:],
            )
            positive_main = positive_features_by_noise[args.main_feature_noise][layer]
            for condition_index in range(condition_batch_size):
                gen_centroid = feature_centroid(
                    generated_grouped[condition_index],
                    args.pool_tokens,
                )
                pos_centroid = feature_centroid(
                    positive_main[condition_index],
                    args.pool_tokens,
                )
                gap = centroid_distance(gen_centroid, pos_centroid)
                append_metric(metrics, layer, "gen_pos_gap", gap)
                if processed_conditions < args.grad_sample_count:
                    grad = torch.autograd.grad(
                        gap,
                        x_student_feat,
                        retain_graph=True,
                        allow_unused=False,
                    )[0]
                    append_metric(
                        metrics,
                        layer,
                        "grad_norm",
                        grad.detach().float().norm() / math.sqrt(grad.numel()),
                    )
        processed_conditions += condition_batch_size
        progress.set_postfix({"samples": processed_conditions})

    rows = summarize(metrics, layers)
    write_csv(args.output_csv, rows)
    logger.info("Wrote layer probe metrics to %s", args.output_csv)
    logger.info("Top layers by matched_to_shuffled:")
    for row in sorted(rows, key=lambda item: item["matched_to_shuffled"])[:8]:
        logger.info(
            "%s ratio=%.6g matched=%.6g shuffled=%.6g gap=%.6g grad=%.6g",
            row["layer"],
            row["matched_to_shuffled"],
            row["matched_dist"],
            row["shuffled_dist"],
            row["gen_pos_gap"],
            row["grad_norm"],
        )


if __name__ == "__main__":
    main()
