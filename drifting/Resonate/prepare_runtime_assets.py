#!/usr/bin/env python3
from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import logging
import os
import shutil
import sys
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from huggingface_hub import hf_hub_download
from tqdm.auto import tqdm


LOG = logging.getLogger("resonate-assets")
RESONATE_REPO = "AndreasXi/Resonate"
RESONATE_REVISION = "main"
RESONATE_SOURCE_REVISION = "b08fb6f7887e129623e0efae3e84653783be5c69"
BIGVGAN_REPO = "nvidia/bigvgan_v2_44khz_128band_512x"
CLAP_REPO = "lukewys/laion_clap"
CLAP_FILENAME = "music_speech_audioset_epoch_15_esc_89.98.pt"
SYNCHFORMER_URL = (
    "https://github.com/hkchengrex/MMAudio/releases/download/v0.1/"
    "synchformer_state_dict.pth"
)
STATE_VERSION = 1
RESONATE_MINIMUM_BYTES = {
    "Resonate_GRPO.pth": 900_000_000,
    "Resonate_PT.pth": 1_800_000_000,
    "v1-44.pth": 1_100_000_000,
}


@dataclass(frozen=True)
class Asset:
    name: str
    target: Path
    download: Callable[[Path], None]
    minimum_bytes: int = 1


def configure_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def usable_file(path: Path, minimum_bytes: int = 1) -> bool:
    return path.is_file() and path.stat().st_size >= minimum_bytes


def hf_download(
    *,
    repo_id: str,
    filename: str,
    revision: str = "main",
) -> Callable[[Path], None]:
    def download(target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        LOG.info(
            "Downloading Hugging Face asset: repo=%s file=%s revision=%s",
            repo_id,
            filename,
            revision,
        )
        downloaded = Path(
            hf_hub_download(
                repo_id=repo_id,
                filename=filename,
                revision=revision,
            )
        )
        temporary = target.with_name(f".{target.name}.part-{os.getpid()}")
        try:
            try:
                os.link(downloaded, temporary)
            except OSError:
                shutil.copyfile(downloaded, temporary)
            os.replace(temporary, target)
        finally:
            temporary.unlink(missing_ok=True)

    return download


def url_download(url: str) -> Callable[[Path], None]:
    def download(target: Path) -> None:
        target.parent.mkdir(parents=True, exist_ok=True)
        partial = target.with_suffix(target.suffix + ".part")
        offset = partial.stat().st_size if partial.is_file() else 0
        headers = {"User-Agent": "MeanAudio-Resonate-asset-bootstrap/1.0"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
            LOG.info("Resuming URL download at byte %d: %s", offset, url)
        else:
            LOG.info("Downloading URL asset: %s", url)
        request = urllib.request.Request(url, headers=headers)
        with urllib.request.urlopen(request) as response:
            is_partial = getattr(response, "status", None) == 206
            if offset and not is_partial:
                offset = 0
            content_length = int(response.headers.get("Content-Length", "0"))
            total = offset + content_length if content_length else None
            mode = "ab" if offset and is_partial else "wb"
            with partial.open(mode) as output, tqdm(
                total=total,
                initial=offset,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=f"download-{target.name}",
                dynamic_ncols=True,
            ) as progress:
                while True:
                    chunk = response.read(1024 * 1024)
                    if not chunk:
                        break
                    output.write(chunk)
                    progress.update(len(chunk))
                output.flush()
                os.fsync(output.fileno())
        os.replace(partial, target)

    return download


def released_resonate_asset(path: Path) -> Asset:
    filename = path.name
    supported = {"Resonate_GRPO.pth", "Resonate_PT.pth", "v1-44.pth"}
    if filename not in supported:
        def unavailable_download(target: Path) -> None:
            raise FileNotFoundError(
                f"Custom checkpoint is missing: {target}. Automatic download "
                "only supports released filenames: "
                f"{', '.join(sorted(supported))}"
            )

        return Asset(
            name=f"custom_checkpoint_{filename}",
            target=path,
            download=unavailable_download,
            minimum_bytes=1024,
        )
    return Asset(
        name=filename,
        target=path,
        download=hf_download(
            repo_id=RESONATE_REPO,
            filename=filename,
            revision=RESONATE_REVISION,
        ),
        minimum_bytes=RESONATE_MINIMUM_BYTES[filename],
    )


def build_assets(args: argparse.Namespace) -> list[Asset]:
    assets: list[Asset] = [
        released_resonate_asset(args.teacher_weights),
        released_resonate_asset(args.student_init),
    ]
    raw_root = (
        "https://raw.githubusercontent.com/xiquan-li/Resonate/"
        f"{RESONATE_SOURCE_REVISION}/sets"
    )
    assets.extend(
        [
            Asset(
                name="latent_mean_44k",
                target=args.latent_mean,
                download=url_download(f"{raw_root}/latent_mean_44k.pt"),
                minimum_bytes=1_700,
            ),
            Asset(
                name="latent_std_44k",
                target=args.latent_std,
                download=url_download(f"{raw_root}/latent_std_44k.pt"),
                minimum_bytes=1_700,
            ),
        ]
    )
    if args.with_eval:
        assets.extend(
            [
                released_resonate_asset(args.vae_weights),
                Asset(
                    name="bigvgan_config",
                    target=args.vocoder_dir / "config.json",
                    download=hf_download(
                        repo_id=BIGVGAN_REPO,
                        filename="config.json",
                    ),
                    minimum_bytes=128,
                ),
                Asset(
                    name="bigvgan_generator",
                    target=args.vocoder_dir / "bigvgan_generator.pt",
                    download=hf_download(
                        repo_id=BIGVGAN_REPO,
                        filename="bigvgan_generator.pt",
                    ),
                    minimum_bytes=400_000_000,
                ),
            ]
        )
    if args.with_av_benchmark:
        benchmark_weights = args.av_benchmark_dir / "weights"
        assets.extend(
            [
                Asset(
                    name="av_benchmark_clap",
                    target=benchmark_weights / CLAP_FILENAME,
                    download=hf_download(
                        repo_id=CLAP_REPO,
                        filename=CLAP_FILENAME,
                    ),
                    minimum_bytes=2_000_000_000,
                ),
                Asset(
                    name="av_benchmark_synchformer",
                    target=benchmark_weights / "synchformer_state_dict.pth",
                    download=url_download(SYNCHFORMER_URL),
                    minimum_bytes=500_000_000,
                ),
            ]
        )

    unique: dict[Path, Asset] = {}
    for asset in assets:
        resolved = asset.target.resolve()
        existing = unique.get(resolved)
        if existing is not None and existing.name != asset.name:
            raise ValueError(
                f"Conflicting assets target the same path: {existing.name}, {asset.name}"
            )
        unique[resolved] = asset
    return list(unique.values())


def request_fingerprint(assets: list[Asset]) -> str:
    payload = "\n".join(
        f"{asset.name}\0{asset.target.resolve()}\0{asset.minimum_bytes}"
        for asset in assets
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def state_is_complete(
    state_path: Path,
    assets: list[Asset],
    fingerprint: str,
) -> bool:
    try:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError, OSError):
        return False
    if (
        state.get("version") != STATE_VERSION
        or state.get("fingerprint") != fingerprint
    ):
        return False
    return all(usable_file(asset.target, asset.minimum_bytes) for asset in assets)


def write_state(state_path: Path, assets: list[Asset], fingerprint: str) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": STATE_VERSION,
        "fingerprint": fingerprint,
        "assets": [
            {
                "name": asset.name,
                "path": str(asset.target.resolve()),
                "size": asset.target.stat().st_size,
            }
            for asset in assets
        ],
    }
    temporary = state_path.with_name(f".{state_path.name}.tmp-{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, state_path)


def prepare(args: argparse.Namespace) -> None:
    if args.with_av_benchmark and not (
        args.av_benchmark_dir / "evaluate.py"
    ).is_file():
        raise FileNotFoundError(
            "AV-Benchmark code is missing: "
            f"{args.av_benchmark_dir / 'evaluate.py'}. "
            "Install the benchmark repository first as documented in "
            "drifting/Resonate/INSTALL.md, or disable AV-Benchmark explicitly."
        )
    assets = build_assets(args)
    fingerprint = request_fingerprint(assets)
    state_path = args.state_dir / f"{fingerprint}.json"
    lock_path = args.state_dir / "prepare.lock"
    if state_is_complete(state_path, assets, fingerprint):
        LOG.info("Resonate assets already prepared; completion marker is valid")
        return

    args.state_dir.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock_file:
        try:
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            LOG.info(
                "Another launcher is preparing Resonate assets; waiting for lock: %s",
                lock_path,
            )
            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        if state_is_complete(state_path, assets, fingerprint):
            LOG.info("Resonate assets were prepared by the other launcher; reusing them")
            return

        missing = [
            asset
            for asset in assets
            if not usable_file(asset.target, asset.minimum_bytes)
        ]
        if missing:
            LOG.info(
                "Missing %d model asset(s); downloads run in this single process only",
                len(missing),
            )
            for asset in missing:
                LOG.info("Missing asset: %s -> %s", asset.name, asset.target)
                asset.download(asset.target)
                if not usable_file(asset.target, asset.minimum_bytes):
                    raise RuntimeError(
                        f"Downloaded asset is missing or truncated: {asset.target}"
                    )
                LOG.info(
                    "Asset ready: %s (%.2f MiB)",
                    asset.target,
                    asset.target.stat().st_size / (1024**2),
                )
        else:
            LOG.info("All requested model files exist; recording completion marker")
        write_state(state_path, assets, fingerprint)
        LOG.info("Resonate runtime asset preparation complete")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Prepare released Resonate and evaluation model assets in one "
            "locked, non-distributed process."
        )
    )
    parser.add_argument(
        "--teacher-weights",
        type=Path,
        default=Path("weights/Resonate_GRPO.pth"),
    )
    parser.add_argument(
        "--student-init",
        type=Path,
        default=Path("weights/Resonate_GRPO.pth"),
    )
    parser.add_argument(
        "--latent-mean",
        type=Path,
        default=Path("sets/latent_mean_44k.pt"),
    )
    parser.add_argument(
        "--latent-std",
        type=Path,
        default=Path("sets/latent_std_44k.pt"),
    )
    parser.add_argument(
        "--vae-weights",
        type=Path,
        default=Path("weights/v1-44.pth"),
    )
    parser.add_argument(
        "--vocoder-dir",
        type=Path,
        default=Path("weights/bigvgan_v2_44khz_128band_512x"),
    )
    parser.add_argument(
        "--av-benchmark-dir",
        type=Path,
        default=Path("av-benchmark"),
    )
    parser.add_argument(
        "--state-dir",
        type=Path,
        default=Path("weights/.resonate-runtime-assets"),
    )
    parser.add_argument("--with-eval", action="store_true")
    parser.add_argument("--with-av-benchmark", action="store_true")
    return parser


def main() -> None:
    configure_logging()
    args = build_parser().parse_args()
    if args.with_av_benchmark and not args.with_eval:
        raise ValueError("--with-av-benchmark requires --with-eval")
    LOG.info(
        "Preparing assets in a standalone process (CUDA_VISIBLE_DEVICES=%s)",
        os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"),
    )
    prepare(args)


if __name__ == "__main__":
    try:
        main()
    except Exception:
        LOG.exception("Resonate runtime asset preparation failed")
        sys.exit(1)
