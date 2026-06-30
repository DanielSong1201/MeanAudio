from __future__ import annotations

import csv
import json
import logging
import os
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
_CKPT_RE_TEMPLATE = r"^{exp_id}_(\d+)\.pth$"


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

    def load_state_dict(self, state_dict: dict[str, torch.Tensor]) -> None:
        missing = set(self.shadow) - set(state_dict)
        unexpected = set(state_dict) - set(self.shadow)
        if missing or unexpected:
            raise RuntimeError(
                f"EMA state_dict mismatch: missing={sorted(missing)} unexpected={sorted(unexpected)}"
            )
        self.shadow = {
            key: value.detach().to(device=self.device).clone()
            for key, value in state_dict.items()
        }


def find_latest_weight_checkpoint(output_dir: Path, exp_id: str) -> tuple[int, Path] | None:
    if not output_dir.exists():
        return None
    pattern = re.compile(_CKPT_RE_TEMPLATE.format(exp_id=re.escape(exp_id)))
    latest: tuple[int, Path] | None = None
    for path in output_dir.iterdir():
        if not path.is_file():
            continue
        match = pattern.match(path.name)
        if match is None:
            continue
        iteration = int(match.group(1))
        if latest is None or iteration > latest[0]:
            latest = (iteration, path)
    return latest


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


def _first_visible_cuda_device() -> str:
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES", "0")
    first_device = visible_devices.split(",", 1)[0].strip()
    return first_device or "0"


def _run_eval_subprocess(
    cmd: list[str],
    *,
    env: dict[str, str],
    driver_log: Path,
    stream_output: bool,
    stream_prefix: str,
) -> int:
    with driver_log.open("w") as log_file:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            env=env,
            bufsize=0,
        )
        assert process.stdout is not None
        decoder = __import__("codecs").getincrementaldecoder("utf-8")(errors="replace")
        at_line_start = True
        while True:
            chunk = process.stdout.read(1)
            if not chunk:
                break
            text = decoder.decode(chunk)
            if not text:
                continue
            log_file.write(text)
            log_file.flush()
            if stream_output:
                for char in text:
                    if char == "\r":
                        sys.stdout.write(char)
                        at_line_start = False
                        continue
                    if char == "\n":
                        sys.stdout.write(char)
                        at_line_start = True
                        continue
                    if at_line_start:
                        sys.stdout.write(stream_prefix)
                        at_line_start = False
                    sys.stdout.write(char)
                sys.stdout.flush()
        tail = decoder.decode(b"", final=True)
        if tail:
            log_file.write(tail)
            log_file.flush()
            if stream_output:
                if at_line_start:
                    sys.stdout.write(stream_prefix)
                sys.stdout.write(tail)
                sys.stdout.flush()
        return process.wait()


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
    stream_output: bool = False,
    stream_prefix: str | None = None,
    eval_cuda_visible_devices: str | None = None,
) -> dict[str, float]:
    output_dir = output_root / exp_id / f"it_{iteration:08d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    driver_log = output_dir / "eval_driver.log"
    evaluate_log = output_dir / "evaluate.log"

    if extra_args:
        raise ValueError("extra_args is not supported when using eval_drifting_checkpoint.sh")
    env = os.environ.copy()
    for key in ("RANK", "LOCAL_RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT"):
        env.pop(key, None)
    eval_cuda_visible_devices = eval_cuda_visible_devices or _first_visible_cuda_device()
    if stream_prefix is None:
        stream_prefix = f"[GPU{eval_cuda_visible_devices}] "
    eval_tqdm_position = os.environ.get("EVAL_TQDM_POSITION")
    if eval_tqdm_position is None:
        train_tqdm_position = int(os.environ.get("TQDM_POSITION", "0"))
        eval_position_offset = int(os.environ.get("EVAL_TQDM_POSITION_OFFSET", "1"))
        eval_tqdm_position = str(train_tqdm_position + eval_position_offset)
    env.update(
        {
            "CUDA_VISIBLE_DEVICES": eval_cuda_visible_devices,
            "PYTHONUNBUFFERED": "1",
            "EVAL_TQDM_DESC": f"[GPU{eval_cuda_visible_devices}] eval",
            "EVAL_TQDM_POSITION": eval_tqdm_position,
            "EVAL_TQDM_LEAVE": "0",
            "PYTHON": sys.executable,
            "TEST_ENTRYPOINT": str(eval_entrypoint),
            "MODEL_PATH": str(checkpoint_path),
            "OUTPUT_PATH": str(output_dir),
            "GT_CACHE": str(gt_cache),
            "EVAL_TSV": os.environ.get("EVAL_TSV", "sets/test-audiocaps.tsv"),
            "EVAL_NPZ_DIR": os.environ.get("EVAL_NPZ_DIR", "data/audiocaps/test-npz-t5-clap"),
            "VAE_WEIGHTS": os.environ.get("VAE_WEIGHTS", "weights/v1-16.pth"),
            "VOCODER_WEIGHTS": os.environ.get("VOCODER_WEIGHTS", "weights/best_netG.pt"),
            "DURATION": os.environ.get("DURATION", "10"),
            "SEED": os.environ.get("SEED", "42"),
            "NUM_STEPS": str(num_steps),
            "CFG_STRENGTH": str(cfg_strength),
            "USE_ROPE": "1" if use_rope else "0",
            "DDP_TIMEOUT_MINUTES": os.environ.get("DDP_TIMEOUT_MINUTES", "180"),
            "HF_HUB_OFFLINE": os.environ.get("HF_HUB_OFFLINE", "1"),
            "TRANSFORMERS_OFFLINE": os.environ.get("TRANSFORMERS_OFFLINE", "1"),
        }
    )
    cmd = ["bash", "drifting/scripts/eval_drifting_checkpoint.sh"]

    logger.info(
        "Running eval at it=%d with checkpoint=%s output=%s cuda_visible_devices=%s",
        iteration,
        checkpoint_path,
        output_dir,
        eval_cuda_visible_devices,
    )
    returncode = _run_eval_subprocess(
        cmd,
        env=env,
        driver_log=driver_log,
        stream_output=stream_output,
        stream_prefix=stream_prefix,
    )
    if returncode != 0:
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
