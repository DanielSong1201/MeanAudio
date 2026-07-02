from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch.utils.data import Dataset


SPLIT_ALIASES = {
    "train": ("train",),
    "eval": ("eval", "val", "validation"),
    "test": ("test",),
}
ID_COLUMNS = ("id", "audio_id", "youtube_id", "audiocap_id")
CAPTION_COLUMNS = ("caption", "prompt", "text")
AUDIO_COLUMNS = ("audio_path", "file_name", "filename", "path")
AUDIO_SUFFIXES = (".wav", ".flac", ".mp3", ".m4a", ".ogg")


@dataclass(frozen=True, slots=True)
class AudioCapsRecord:
    id: str
    caption: str
    audio_path: Path


def _first_value(row: dict[str, Any], keys: Iterable[str]) -> Any | None:
    for key in keys:
        value = row.get(key)
        if value is not None and str(value).strip():
            return value
    return None


def _read_delimited(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8-sig") as file:
        first_line = file.readline()
        file.seek(0)
        delimiter = "\t" if "\t" in first_line else ","
        return list(csv.DictReader(file, delimiter=delimiter))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open("r", encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise TypeError(f"{path}:{line_number} must contain a JSON object")
            rows.append(row)
    return rows


def read_caption_manifest(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"Caption manifest does not exist: {path}")
    if path.suffix.lower() == ".jsonl":
        rows = _read_jsonl(path)
    elif path.suffix.lower() in (".csv", ".tsv"):
        rows = _read_delimited(path)
    else:
        raise ValueError(f"Unsupported caption manifest format: {path}")
    if not rows:
        raise ValueError(f"Caption manifest is empty: {path}")
    return rows


def discover_manifest(
    *,
    dataset_root: Path,
    split: str,
    explicit_path: Path | None = None,
) -> Path:
    if explicit_path is not None:
        if not explicit_path.is_file():
            raise FileNotFoundError(f"Explicit manifest does not exist: {explicit_path}")
        return explicit_path
    if split not in SPLIT_ALIASES:
        raise ValueError(f"Unknown AudioCaps split: {split}")

    candidates: list[Path] = []
    for alias in SPLIT_ALIASES[split]:
        for suffix in (".tsv", ".csv", ".jsonl"):
            candidates.extend(
                [
                    dataset_root / f"{alias}{suffix}",
                    dataset_root / "metadata" / f"{alias}{suffix}",
                    dataset_root / "csv_files" / f"{alias}{suffix}",
                    dataset_root / "csv_files_v1" / f"{alias}{suffix}",
                    dataset_root / alias / f"metadata{suffix}",
                ]
            )

    fallback_alias = "val" if split == "eval" else split
    candidates.extend(
        [
            Path(f"data/audiocaps/{fallback_alias}-memmap.tsv"),
            Path(f"sets/{fallback_alias}-audiocaps.tsv"),
        ]
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    formatted = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        f"Could not discover a caption manifest for split {split!r}. Checked:\n{formatted}\n"
        "Pass --manifest explicitly if your metadata is stored elsewhere."
    )


def discover_audio_dir(dataset_root: Path, split: str) -> Path:
    if split not in SPLIT_ALIASES:
        raise ValueError(f"Unknown AudioCaps split: {split}")
    candidates = []
    for alias in SPLIT_ALIASES[split]:
        candidates.extend(
            [
                dataset_root / alias,
                dataset_root / "audio" / alias,
                dataset_root / "audio_32000Hz" / alias,
                dataset_root / "AUDIOCAPS" / "audio_32000Hz" / alias,
            ]
        )
    for candidate in candidates:
        if candidate.is_dir():
            return candidate
    formatted = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        f"Could not discover the audio directory for split {split!r}. Checked:\n{formatted}"
    )


def _audio_index(audio_dir: Path) -> tuple[dict[str, Path], dict[str, Path]]:
    by_name: dict[str, Path] = {}
    by_stem: dict[str, Path] = {}
    for path in sorted(audio_dir.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in AUDIO_SUFFIXES:
            continue
        by_name.setdefault(path.name, path)
        by_stem.setdefault(path.stem, path)
    if not by_name:
        raise FileNotFoundError(f"No supported audio files found under {audio_dir}")
    return by_name, by_stem


def _resolve_audio(
    *,
    row: dict[str, Any],
    record_id: str,
    audio_dir: Path,
    by_name: dict[str, Path],
    by_stem: dict[str, Path],
) -> Path | None:
    explicit_audio = _first_value(row, AUDIO_COLUMNS)
    if explicit_audio is not None:
        explicit_path = Path(str(explicit_audio))
        for candidate in (
            explicit_path,
            audio_dir / explicit_path,
            audio_dir / explicit_path.name,
        ):
            if candidate.is_file():
                return candidate

    raw_name = Path(record_id).name
    stem = Path(raw_name).stem
    names = [raw_name]
    stems = [stem]
    if not stem.startswith("Y"):
        stems.append(f"Y{stem}")
    for candidate_stem in tuple(stems):
        names.extend(f"{candidate_stem}{suffix}" for suffix in AUDIO_SUFFIXES)
    for name in names:
        if name in by_name:
            return by_name[name]
    for candidate_stem in stems:
        if candidate_stem in by_stem:
            return by_stem[candidate_stem]
    return None


def resolve_audiocaps_records(
    *,
    dataset_root: Path,
    split: str,
    manifest_path: Path | None = None,
    strict_missing_audio: bool = False,
) -> tuple[list[AudioCapsRecord], Path, Path, int]:
    manifest = discover_manifest(
        dataset_root=dataset_root,
        split=split,
        explicit_path=manifest_path,
    )
    audio_dir = discover_audio_dir(dataset_root, split)
    by_name, by_stem = _audio_index(audio_dir)
    rows = read_caption_manifest(manifest)

    records: list[AudioCapsRecord] = []
    missing = 0
    malformed = 0
    for row_number, row in enumerate(rows, start=2):
        record_id = _first_value(row, ID_COLUMNS)
        caption = _first_value(row, CAPTION_COLUMNS)
        if record_id is None or caption is None:
            malformed += 1
            continue
        record_id = str(record_id).strip()
        caption = str(caption).strip()
        audio_path = _resolve_audio(
            row=row,
            record_id=record_id,
            audio_dir=audio_dir,
            by_name=by_name,
            by_stem=by_stem,
        )
        if audio_path is None:
            missing += 1
            if strict_missing_audio:
                raise FileNotFoundError(
                    f"{manifest}:{row_number}: no audio found for id={record_id!r} "
                    f"under {audio_dir}"
                )
            continue
        records.append(
            AudioCapsRecord(
                id=Path(record_id).stem,
                caption=caption,
                audio_path=audio_path,
            )
        )
    if malformed:
        raise ValueError(
            f"{manifest} contains {malformed} rows without a supported id/caption column"
        )
    if not records:
        raise RuntimeError(
            f"No manifest rows could be matched to audio files for split {split!r}"
        )
    return records, manifest, audio_dir, missing


def write_processed_manifest(path: Path, records: list[AudioCapsRecord]) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with temporary.open("w", newline="", encoding="utf-8") as file:
            writer = csv.DictWriter(file, fieldnames=("id", "caption"), delimiter="\t")
            writer.writeheader()
            for record in records:
                writer.writerow({"id": record.id, "caption": record.caption})
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


class ResonateNpzDataset(Dataset):
    """Processed AudioCaps conditions and 44.1 kHz latent distributions."""

    def __init__(
        self,
        *,
        tsv_path: Path,
        npz_dir: Path,
        teacher_positive_dir: Path | None = None,
        teacher_positive_count: int = 3,
    ) -> None:
        with tsv_path.open("r", newline="", encoding="utf-8") as file:
            self.rows = list(csv.DictReader(file, delimiter="\t"))
        if not self.rows:
            raise ValueError(f"No processed rows found in {tsv_path}")
        self.npz_dir = npz_dir
        if not npz_dir.is_dir():
            raise FileNotFoundError(f"Processed NPZ directory does not exist: {npz_dir}")
        self.teacher_positive_dir = teacher_positive_dir
        self.teacher_positive_count = teacher_positive_count
        if teacher_positive_dir is not None:
            if teacher_positive_count < 1:
                raise ValueError("teacher_positive_count must be positive")
            complete_path = teacher_positive_dir / "complete.json"
            if not complete_path.is_file():
                raise FileNotFoundError(
                    f"Teacher-positive bank is incomplete: {complete_path}"
                )

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        path = self.npz_dir / f"{index}.npz"
        if not path.is_file():
            raise FileNotFoundError(f"Missing processed sample: {path}")
        with np.load(path) as data:
            required = ("mean", "std", "text_features", "text_features_c")
            missing = [key for key in required if key not in data]
            if missing:
                raise KeyError(f"{path} is missing arrays: {missing}")
            item = {
                "mean": torch.from_numpy(data["mean"].copy()),
                "std": torch.from_numpy(data["std"].copy()),
                "text_features": torch.from_numpy(data["text_features"].copy()),
                "text_features_c": torch.from_numpy(data["text_features_c"].copy()),
            }
        item["id"] = self.rows[index]["id"]
        item["caption"] = self.rows[index]["caption"]
        if self.teacher_positive_dir is not None:
            positive_path = self.teacher_positive_dir / f"{index}.npz"
            if not positive_path.is_file():
                raise FileNotFoundError(f"Missing teacher positives: {positive_path}")
            with np.load(positive_path) as data:
                if "latents_normalized" not in data:
                    raise KeyError(f"{positive_path} has no latents_normalized array")
                positives = data["latents_normalized"][: self.teacher_positive_count].copy()
            if positives.shape[0] != self.teacher_positive_count:
                raise ValueError(
                    f"{positive_path} contains {positives.shape[0]} positives; "
                    f"expected {self.teacher_positive_count}"
                )
            if tuple(positives.shape[1:]) != tuple(item["mean"].shape):
                raise ValueError(
                    f"{positive_path} latent shape {tuple(positives.shape[1:])} "
                    f"does not match real latent shape {tuple(item['mean'].shape)}"
                )
            item["teacher_positives"] = torch.from_numpy(positives)
        return item
