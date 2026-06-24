from __future__ import annotations

import csv
import json
import logging
import re
import subprocess
import sys
from pathlib import Path
from typing import Any, Iterable

import torch


_ANSI_RE = re.compile(r"\x1b\[[0-9;]*m")
_METRIC_RE = re.compile(
    r"([A-Za-z][A-Za-z0-9_ ./()-]{0,64})\s*[:=]\s*(-?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?)"
)


def append_eval_metrics(path: Path, row: dict[str, Any]) -> None:
    fieldnames = [
        "iteration",
        "checkpoint",
        "output",
        "evaluate_log",
        "driver_log",
        "metrics_json",
    ]
    write_header = not path.exists()
    with path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)


def move_optimizer_state(optimizer: torch.optim.Optimizer, device: torch.device) -> None:
    for state in optimizer.state.values():
        for key, value in list(state.items()):
            if torch.is_tensor(value):
                state[key] = value.to(device)


def empty_cuda_cache() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _normalize_metric_name(name: str) -> str:
    name = name.strip().lower()
    name = re.sub(r"[^a-z0-9]+", "_", name)
    return name.strip("_")


def parse_evaluate_log(log_path: Path) -> dict[str, float]:
    if not log_path.exists():
        return {}

    metrics: dict[str, float] = {}
    for raw_line in log_path.read_text(errors="replace").splitlines():
        line = _ANSI_RE.sub("", raw_line).strip()
        if not line:
            continue
        if line.startswith("{") and line.endswith("}"):
            try:
                payload = json.loads(line)
            except json.JSONDecodeError:
                payload = None
            if isinstance(payload, dict):
                for key, value in payload.items():
                    if isinstance(value, int | float):
                        metrics[_normalize_metric_name(str(key))] = float(value)
                continue
        for name, value in _METRIC_RE.findall(line):
            key = _normalize_metric_name(name)
            if key:
                metrics[key] = float(value)
    return metrics


def run_checkpoint_evaluation(
    *,
    eval_entrypoint: Path,
    iteration: int,
    checkpoint_path: Path,
    output_root: Path,
    exp_id: str,
    gt_cache: Path,
    num_steps: int,
    cfg_strength: float,
    use_rope: bool,
    eval_metrics_path: Path,
    logger: logging.Logger,
    extra_args: Iterable[str] = (),
) -> dict[str, float]:
    output_dir = output_root / exp_id / f"it_{iteration:08d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    driver_log = output_dir / "eval_driver.log"
    evaluate_log = output_dir / "evaluate.log"

    cmd = [
        sys.executable,
        str(eval_entrypoint),
        "--mode",
        "eval",
        "--model-path",
        str(checkpoint_path),
        "--output",
        str(output_dir),
        "--gt-cache",
        str(gt_cache),
        "--num-steps",
        str(num_steps),
        "--cfg-strength",
        str(cfg_strength),
        *extra_args,
    ]
    if use_rope:
        cmd.append("--use-rope")

    logger.info("Running eval at it=%d with checkpoint=%s output=%s", iteration, checkpoint_path, output_dir)
    with driver_log.open("w") as f:
        result = subprocess.run(cmd, text=True, stdout=f, stderr=subprocess.STDOUT)
    if result.returncode != 0:
        tail = "\n".join(driver_log.read_text(errors="replace").splitlines()[-80:])
        logger.error("Eval failed at it=%d; driver log=%s\n%s", iteration, driver_log, tail)
        raise RuntimeError(f"Evaluation failed at iteration {iteration}; see {driver_log}")

    metrics = parse_evaluate_log(evaluate_log)
    metrics_json = json.dumps(metrics, sort_keys=True)
    append_eval_metrics(
        eval_metrics_path,
        {
            "iteration": iteration,
            "checkpoint": str(checkpoint_path),
            "output": str(output_dir),
            "evaluate_log": str(evaluate_log),
            "driver_log": str(driver_log),
            "metrics_json": metrics_json,
        },
    )
    logger.info(
        "eval it=%d checkpoint=%s output=%s evaluate_log=%s metrics=%s",
        iteration,
        checkpoint_path,
        output_dir,
        evaluate_log,
        metrics_json,
    )
    return metrics
