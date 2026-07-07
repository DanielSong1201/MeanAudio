#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
from pathlib import Path

from drifting.eval_helpers import (
    append_eval_metrics,
    parse_evaluate_log,
    run_checkpoint_evaluation,
)


DEFAULT_EXP_ID = (
    "resonate_lr1e6_tfd100_anchor1_flow01_hybridpos4_warmup1000_200k"
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate the released Resonate-GRPO initialization as iteration 0 "
            "with the same AV-Benchmark path used by periodic training eval."
        )
    )
    parser.add_argument("--exp-id", default=DEFAULT_EXP_ID)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("weights/Resonate_GRPO.pth"),
    )
    parser.add_argument(
        "--train-output-root",
        type=Path,
        default=Path("exps/drifting_resonate"),
    )
    parser.add_argument(
        "--eval-output-root",
        type=Path,
        default=Path("exps/drifting_resonate_eval"),
    )
    parser.add_argument(
        "--gt-cache",
        type=Path,
        default=Path("data/audiocaps/test-features"),
    )
    parser.add_argument(
        "--gt-audio",
        type=Path,
        default=Path("gt_audio"),
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
        "--vae-weights",
        type=Path,
        default=Path("weights/v1-44.pth"),
    )
    parser.add_argument(
        "--vocoder-weights",
        type=Path,
        default=Path("weights/bigvgan_v2_44khz_128band_512x"),
    )
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--num-steps", type=int, default=1)
    parser.add_argument("--cfg-strength", type=float, default=4.5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--force", action="store_true")
    return parser


def remove_iteration_row(path: Path, iteration: int) -> None:
    if not path.is_file():
        return
    with path.open("r", newline="", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        fieldnames = reader.fieldnames
        rows = [
            row
            for row in reader
            if row.get("iteration") != str(iteration)
        ]
    if not fieldnames:
        path.unlink()
        return
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def has_iteration_row(path: Path, iteration: int) -> bool:
    if not path.is_file():
        return False
    with path.open("r", newline="", encoding="utf-8") as file:
        return any(
            row.get("iteration") == str(iteration)
            for row in csv.DictReader(file)
        )


def main() -> None:
    args = build_parser().parse_args()
    if args.num_steps != 1:
        raise ValueError("The iteration-0 baseline experiment requires --num-steps=1")
    output_dir = args.eval_output_root / args.exp_id / "it_00000000"
    metrics_json = output_dir / "cache" / "output_metrics.json"
    train_output_dir = args.train_output_root / args.exp_id
    train_output_dir.mkdir(parents=True, exist_ok=True)
    eval_metrics_path = train_output_dir / "eval_metrics.csv"
    if metrics_json.is_file() and not args.force:
        if not has_iteration_row(eval_metrics_path, 0):
            parsed_metrics = parse_evaluate_log(output_dir / "evaluate.log")
            if not parsed_metrics:
                parsed_metrics = json.loads(
                    metrics_json.read_text(encoding="utf-8")
                )
            append_eval_metrics(
                eval_metrics_path,
                {
                    "iteration": 0,
                    "checkpoint": str(args.model_path),
                    "output": str(output_dir),
                    "evaluate_log": str(output_dir / "evaluate.log"),
                    "driver_log": str(output_dir / "eval_driver.log"),
                    "metrics_json": json.dumps(parsed_metrics, sort_keys=True),
                },
            )
        print(f"[ok] iteration-0 evaluation already complete: {metrics_json}")
        print(f"[ok] comparison CSV: {eval_metrics_path}")
        print("[ok] Set FORCE=1 in the launcher only to rerun it deliberately.")
        return
    if not args.model_path.is_file():
        raise FileNotFoundError(f"Missing Resonate baseline checkpoint: {args.model_path}")
    if args.force or not metrics_json.is_file():
        metrics_json.unlink(missing_ok=True)
        remove_iteration_row(eval_metrics_path, 0)
    logger = logging.getLogger("drifting.resonate.eval_it0")
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
    )

    environment = {
        "GT_AUDIO": str(args.gt_audio),
        "EVAL_TSV": str(args.eval_tsv),
        "EVAL_NPZ_DIR": str(args.eval_npz_dir),
        "VAE_WEIGHTS": str(args.vae_weights),
        "VOCODER_WEIGHTS": str(args.vocoder_weights),
        "DURATION": str(args.duration),
        "SEED": str(args.seed),
        "EVAL_LIMIT": "",
        "EVAL_SKIP_AV_BENCHMARK": "0",
    }
    previous = {key: os.environ.get(key) for key in environment}
    os.environ.update(environment)
    try:
        metrics = run_checkpoint_evaluation(
            eval_entrypoint=Path("drifting/Resonate/test.py"),
            iteration=0,
            checkpoint_path=args.model_path,
            output_root=args.eval_output_root,
            exp_id=args.exp_id,
            gt_cache=args.gt_cache,
            num_steps=args.num_steps,
            cfg_strength=args.cfg_strength,
            use_rope=True,
            eval_metrics_path=eval_metrics_path,
            logger=logger,
            stream_output=True,
        )
    finally:
        for key, value in previous.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value

    if not metrics_json.is_file():
        raise FileNotFoundError(f"Missing AV-Benchmark result: {metrics_json}")
    logger.info("Iteration-0 Resonate baseline complete: metrics=%s", metrics)
    logger.info("AV-Benchmark JSON: %s", metrics_json)
    logger.info("Comparison CSV: %s", eval_metrics_path)


if __name__ == "__main__":
    main()
