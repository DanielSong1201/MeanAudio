#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from tqdm import tqdm

from drifting.Resonate.config import RESONATE_CONFIG


NUMERIC_NPZ = re.compile(r"^(\d+)\.npz$")
SPLIT_KEYS = {
    "train": "AudioCaps_npz",
    "eval": "AudioCaps_val_npz",
    "test": "AudioCaps_test_npz",
}


@dataclass
class ValidationReport:
    errors: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    def error(self, message: str) -> None:
        self.errors.append(message)

    def warning(self, message: str) -> None:
        self.warnings.append(message)


@dataclass(frozen=True)
class SplitState:
    name: str
    tsv_path: Path
    npz_dir: Path
    rows: list[dict[str, str]]
    num_items: int
    available_indices: set[int]


def load_json(path: Path, report: ValidationReport) -> dict[str, Any] | None:
    if not path.is_file():
        report.error(f"missing JSON marker: {path}")
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception as error:
        report.error(f"cannot parse JSON {path}: {error}")
        return None
    if not isinstance(payload, dict):
        report.error(f"JSON root must be an object: {path}")
        return None
    return payload


def load_tsv(path: Path, report: ValidationReport) -> list[dict[str, str]]:
    if not path.is_file():
        report.error(f"missing TSV manifest: {path}")
        return []
    try:
        with path.open("r", newline="", encoding="utf-8") as file:
            rows = list(csv.DictReader(file, delimiter="\t"))
    except Exception as error:
        report.error(f"cannot read TSV {path}: {error}")
        return []
    if not rows:
        report.error(f"TSV has no data rows: {path}")
        return []
    missing_columns = {"id", "caption"} - set(rows[0])
    if missing_columns:
        report.error(f"{path} is missing columns: {sorted(missing_columns)}")
    return rows


def numeric_npz_indices(path: Path, report: ValidationReport) -> set[int]:
    if not path.is_dir():
        report.error(f"missing NPZ directory: {path}")
        return set()
    indices: set[int] = set()
    for child in path.iterdir():
        match = NUMERIC_NPZ.fullmatch(child.name)
        if match and child.is_file():
            indices.add(int(match.group(1)))
    return indices


def summarize_index_set(
    *,
    label: str,
    available: set[int],
    expected_count: int,
    report: ValidationReport,
) -> None:
    expected = set(range(expected_count))
    missing = sorted(expected - available)
    extra = sorted(available - expected)
    if missing:
        report.error(
            f"{label}: {len(missing)} indexed NPZ files are missing; "
            f"first={missing[:20]}"
        )
    if extra:
        report.warning(
            f"{label}: {len(extra)} extra numeric NPZ files are outside "
            f"[0, {expected_count}); first={extra[:20]}"
        )


def selected_indices(
    available: set[int],
    expected_count: int,
    *,
    deep: bool,
    sample_count: int,
) -> list[int]:
    valid = sorted(index for index in available if 0 <= index < expected_count)
    if deep or len(valid) <= sample_count:
        return valid
    positions = {
        round(position * (len(valid) - 1) / (sample_count - 1))
        for position in range(sample_count)
    }
    return [valid[position] for position in sorted(positions)]


def check_finite(
    array: np.ndarray,
    *,
    label: str,
    report: ValidationReport,
) -> None:
    if not np.isfinite(array).all():
        report.error(f"{label} contains NaN or infinity")


def validate_processed_npz(
    *,
    state: SplitState,
    index: int,
    check_values: bool,
    report: ValidationReport,
) -> tuple[int, int] | None:
    path = state.npz_dir / f"{index}.npz"
    try:
        with np.load(path, allow_pickle=False) as data:
            required = ("mean", "std", "text_features", "text_features_c")
            missing = [key for key in required if key not in data]
            if missing:
                report.error(f"{path} is missing arrays: {missing}")
                return None
            mean = data["mean"]
            std = data["std"]
            text_features = data["text_features"]
            text_features_c = data["text_features_c"]
    except Exception as error:
        report.error(f"cannot load processed NPZ {path}: {error}")
        return None

    if mean.ndim != 2 or mean.shape[-1] != RESONATE_CONFIG.latent_dim:
        report.error(
            f"{path}: mean shape {mean.shape} is not "
            f"[latent_tokens, {RESONATE_CONFIG.latent_dim}]"
        )
    if std.shape != mean.shape:
        report.error(f"{path}: std shape {std.shape} != mean shape {mean.shape}")
    if text_features.shape != (
        RESONATE_CONFIG.text_seq_len,
        RESONATE_CONFIG.text_dim,
    ):
        report.error(
            f"{path}: text_features shape {text_features.shape} != "
            f"({RESONATE_CONFIG.text_seq_len}, {RESONATE_CONFIG.text_dim})"
        )
    if text_features_c.shape != (RESONATE_CONFIG.text_c_dim,):
        report.error(
            f"{path}: text_features_c shape {text_features_c.shape} != "
            f"({RESONATE_CONFIG.text_c_dim},)"
        )
    if check_values:
        check_finite(mean, label=f"{path}:mean", report=report)
        check_finite(std, label=f"{path}:std", report=report)
        check_finite(
            text_features,
            label=f"{path}:text_features",
            report=report,
        )
        check_finite(
            text_features_c,
            label=f"{path}:text_features_c",
            report=report,
        )
        if np.any(std < 0):
            report.error(f"{path}: std contains negative values")
    return tuple(mean.shape)


def validate_split(
    *,
    name: str,
    split_config: dict[str, Any],
    deep: bool,
    sample_count: int,
    report: ValidationReport,
) -> SplitState:
    tsv_path = Path(split_config["tsv"])
    npz_dir = Path(split_config["npz_dir"])
    rows = load_tsv(tsv_path, report)
    config = load_json(npz_dir / "config.json", report)
    complete = load_json(npz_dir / "complete.json", report)
    if config is not None and complete is not None and config != complete:
        report.error(
            f"{name}: config.json and complete.json differ under {npz_dir}"
        )

    declared_items = None if config is None else config.get("num_items")
    if not isinstance(declared_items, int) or declared_items < 0:
        report.error(f"{name}: invalid num_items in {npz_dir / 'config.json'}")
        num_items = len(rows)
    else:
        num_items = declared_items
    if rows and len(rows) != num_items:
        report.error(
            f"{name}: TSV rows={len(rows)} but config num_items={num_items}"
        )
    if config is not None:
        expected_fields = {
            "sample_rate": RESONATE_CONFIG.sample_rate,
            "text_seq_len": RESONATE_CONFIG.text_seq_len,
            "text_dim": RESONATE_CONFIG.text_dim,
            "text_c_dim": RESONATE_CONFIG.text_c_dim,
            "latent_dim": RESONATE_CONFIG.latent_dim,
        }
        for key, expected in expected_fields.items():
            if config.get(key) != expected:
                report.error(
                    f"{name}: config {key}={config.get(key)!r}, expected {expected!r}"
                )

    available = numeric_npz_indices(npz_dir, report)
    summarize_index_set(
        label=f"processed-{name}",
        available=available,
        expected_count=num_items,
        report=report,
    )
    state = SplitState(
        name=name,
        tsv_path=tsv_path,
        npz_dir=npz_dir,
        rows=rows,
        num_items=num_items,
        available_indices=available,
    )
    indices = selected_indices(
        available,
        num_items,
        deep=deep,
        sample_count=sample_count,
    )
    for index in tqdm(
        indices,
        desc=f"check-processed-{name}",
        unit="file",
        dynamic_ncols=True,
    ):
        validate_processed_npz(
            state=state,
            index=index,
            check_values=deep,
            report=report,
        )
    print(
        f"[check] processed-{name}: rows={len(rows)} "
        f"expected={num_items} files={len(available)} inspected={len(indices)}"
    )
    return state


def scalar_string(value: np.ndarray) -> str:
    item = np.asarray(value).item()
    if isinstance(item, bytes):
        return item.decode("utf-8")
    return str(item)


def validate_positive_npz(
    *,
    positive_dir: Path,
    train_state: SplitState,
    index: int,
    expected_positive_count: int,
    check_values: bool,
    report: ValidationReport,
) -> None:
    positive_path = positive_dir / f"{index}.npz"
    try:
        with np.load(positive_path, allow_pickle=False) as data:
            if "latents_normalized" not in data:
                report.error(f"{positive_path} has no latents_normalized array")
                return
            positives = data["latents_normalized"]
            item_index = data["item_index"] if "item_index" in data else None
            item_id = data["item_id"] if "item_id" in data else None
    except Exception as error:
        report.error(f"cannot load positive NPZ {positive_path}: {error}")
        return

    if positives.ndim != 3:
        report.error(
            f"{positive_path}: positive shape {positives.shape} must have rank 3"
        )
        return
    if positives.shape[0] < expected_positive_count:
        report.error(
            f"{positive_path}: contains {positives.shape[0]} positives, "
            f"expected at least {expected_positive_count}"
        )
    if positives.shape[-1] != RESONATE_CONFIG.latent_dim:
        report.error(
            f"{positive_path}: latent_dim={positives.shape[-1]}, "
            f"expected {RESONATE_CONFIG.latent_dim}"
        )
    if item_index is None:
        report.error(f"{positive_path}: missing item_index")
    elif int(np.asarray(item_index).item()) != index:
        report.error(
            f"{positive_path}: item_index={np.asarray(item_index).item()} "
            f"does not match filename index={index}"
        )
    expected_id = (
        train_state.rows[index].get("id")
        if index < len(train_state.rows)
        else None
    )
    if item_id is None:
        report.error(f"{positive_path}: missing item_id")
    elif expected_id is not None and scalar_string(item_id) != expected_id:
        report.error(
            f"{positive_path}: item_id={scalar_string(item_id)!r}, "
            f"expected {expected_id!r}"
        )

    real_path = train_state.npz_dir / f"{index}.npz"
    try:
        with np.load(real_path, allow_pickle=False) as real_data:
            real_shape = tuple(real_data["mean"].shape)
    except Exception as error:
        report.error(f"cannot compare real latent {real_path}: {error}")
        real_shape = None
    if real_shape is not None and tuple(positives.shape[1:]) != real_shape:
        report.error(
            f"{positive_path}: positive latent shape {positives.shape[1:]} "
            f"!= real latent shape {real_shape}"
        )
    if check_values:
        check_finite(
            positives,
            label=f"{positive_path}:latents_normalized",
            report=report,
        )


def validate_positive_bank(
    *,
    positive_dir: Path,
    train_state: SplitState,
    expected_positive_count: int,
    deep: bool,
    sample_count: int,
    report: ValidationReport,
) -> None:
    config = load_json(positive_dir / "config.json", report)
    complete = load_json(positive_dir / "complete.json", report)
    if config is not None and complete is not None and config != complete:
        report.error(
            f"positive bank config.json and complete.json differ: {positive_dir}"
        )
    if config is not None:
        if config.get("num_items") != train_state.num_items:
            report.error(
                f"positive bank num_items={config.get('num_items')} but "
                f"train num_items={train_state.num_items}"
            )
        if config.get("normalized_latents") is not True:
            report.error("positive bank is not marked normalized_latents=true")
        available_count = config.get("positives_per_condition")
        if not isinstance(available_count, int) or (
            available_count < expected_positive_count
        ):
            report.error(
                f"positive bank provides {available_count!r} positives, "
                f"expected at least {expected_positive_count}"
            )

    available = numeric_npz_indices(positive_dir, report)
    summarize_index_set(
        label="teacher-positives",
        available=available,
        expected_count=train_state.num_items,
        report=report,
    )
    indices = selected_indices(
        available,
        train_state.num_items,
        deep=deep,
        sample_count=sample_count,
    )
    for index in tqdm(
        indices,
        desc="check-teacher-positives",
        unit="file",
        dynamic_ncols=True,
    ):
        validate_positive_npz(
            positive_dir=positive_dir,
            train_state=train_state,
            index=index,
            expected_positive_count=expected_positive_count,
            check_values=deep,
            report=report,
        )
    print(
        f"[check] teacher-positives: expected={train_state.num_items} "
        f"files={len(available)} inspected={len(indices)}"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate the complete Resonate AudioCaps preprocessing pipeline."
    )
    parser.add_argument(
        "--data-config",
        type=Path,
        default=Path("config/data/resonate_flant5_44k.yaml"),
    )
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
        "--sample-count",
        type=int,
        default=16,
        help="Content samples per split/bank in fast mode.",
    )
    parser.add_argument(
        "--deep",
        action="store_true",
        help="Open and validate every processed and positive NPZ.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.teacher_positive_count < 1:
        raise ValueError("--teacher-positive-count must be positive")
    if args.sample_count < 2:
        raise ValueError("--sample-count must be >= 2")
    if not args.data_config.is_file():
        raise FileNotFoundError(f"Missing data config: {args.data_config}")
    with args.data_config.open("r", encoding="utf-8") as file:
        data_config = yaml.safe_load(file)

    report = ValidationReport()
    states: dict[str, SplitState] = {}
    for name, key in SPLIT_KEYS.items():
        split_config = data_config.get(key)
        if not isinstance(split_config, dict):
            report.error(f"{args.data_config} has no mapping for {key}")
            continue
        try:
            states[name] = validate_split(
                name=name,
                split_config=split_config,
                deep=args.deep,
                sample_count=args.sample_count,
                report=report,
            )
        except Exception as error:
            report.error(f"unexpected error while checking {name}: {error}")

    train_state = states.get("train")
    if train_state is not None:
        validate_positive_bank(
            positive_dir=args.teacher_positive_dir,
            train_state=train_state,
            expected_positive_count=args.teacher_positive_count,
            deep=args.deep,
            sample_count=args.sample_count,
            report=report,
        )
    else:
        report.error("cannot validate teacher positives without a valid train split")

    for warning in report.warnings:
        print(f"[warning] {warning}", file=sys.stderr)
    for error in report.errors:
        print(f"[error] {error}", file=sys.stderr)
    if report.errors:
        print(
            f"[failed] preprocessing validation found {len(report.errors)} error(s) "
            f"and {len(report.warnings)} warning(s)",
            file=sys.stderr,
        )
        raise SystemExit(1)
    print(
        f"[success] Resonate preprocessing is complete "
        f"(deep={args.deep}, warnings={len(report.warnings)})"
    )


if __name__ == "__main__":
    main()
