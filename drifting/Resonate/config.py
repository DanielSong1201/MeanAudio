from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True, slots=True)
class ResonateModelConfig:
    """Single source of truth for the released Resonate-GRPO architecture."""

    model_name: str = "fluxaudio_m_44k"
    source_revision: str = "b08fb6f7887e129623e0efae3e84653783be5c69"
    sample_rate: int = 44_100
    spectrogram_frame_rate: int = 512
    latent_downsample_rate: int = 2
    default_duration: float = 10.0

    latent_dim: int = 40
    latent_seq_len: int = 430
    max_latent_seq_len: int = 2048
    text_dim: int = 1024
    text_c_dim: int = 1024
    text_seq_len: int = 77
    hidden_dim: int = 448
    depth: int = 52
    fused_depth: int = 36
    num_heads: int = 7
    mlp_ratio: float = 4.0
    use_rope: bool = True

    # Match the relative positions used by FluxAudio-S:
    # final joint block, middle fused block, final fused block.
    feature_layers: tuple[str, ...] = ("joint_15", "fused_17", "fused_35")

    teacher_weights: Path = Path("weights/Resonate_GRPO.pth")
    student_init: Path = Path("weights/Resonate_GRPO.pth")
    pretrained_weights: Path = Path("weights/Resonate_PT.pth")
    vae_weights: Path = Path("weights/v1-44.pth")
    vocoder_dir: Path = Path("weights/bigvgan_v2_44khz_128band_512x")
    latent_mean: Path = Path("sets/latent_mean_44k.pt")
    latent_std: Path = Path("sets/latent_std_44k.pt")

    def __post_init__(self) -> None:
        if self.depth < 1:
            raise ValueError("depth must be positive")
        if not 0 <= self.fused_depth < self.depth:
            raise ValueError("fused_depth must be in [0, depth)")
        if self.hidden_dim % self.num_heads != 0:
            raise ValueError("hidden_dim must be divisible by num_heads")
        if self.latent_seq_len > self.max_latent_seq_len:
            raise ValueError("latent_seq_len exceeds max_latent_seq_len")

        valid_layers = {"audio_proj"}
        valid_layers.update(f"joint_{i}" for i in range(self.joint_depth))
        valid_layers.update(f"fused_{i}" for i in range(self.fused_depth))
        invalid_layers = set(self.feature_layers) - valid_layers
        if invalid_layers:
            raise ValueError(f"Invalid default feature layers: {sorted(invalid_layers)}")

    @property
    def joint_depth(self) -> int:
        return self.depth - self.fused_depth

    def latent_length_for_duration(self, duration: float) -> int:
        if duration <= 0:
            raise ValueError("duration must be positive")
        length = math.ceil(
            duration
            * self.sample_rate
            / self.spectrogram_frame_rate
            / self.latent_downsample_rate
        )
        if length > self.max_latent_seq_len:
            raise ValueError(
                f"Duration {duration} requires {length} latent tokens, "
                f"exceeding the configured maximum {self.max_latent_seq_len}"
            )
        return length


RESONATE_CONFIG = ResonateModelConfig()
