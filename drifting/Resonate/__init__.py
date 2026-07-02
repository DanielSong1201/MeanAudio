"""Resonate teacher-feature drifting integration.

This package is intentionally isolated from ``drifting.flux`` so the existing
FluxAudio-S training and evaluation paths keep their current behavior.
"""

from .config import RESONATE_CONFIG, ResonateModelConfig
from .model import (
    ResonateFluxAudio,
    build_resonate_model,
    freeze_resonate_teacher,
    load_resonate_checkpoint,
)

__all__ = [
    "RESONATE_CONFIG",
    "ResonateFluxAudio",
    "ResonateModelConfig",
    "build_resonate_model",
    "freeze_resonate_teacher",
    "load_resonate_checkpoint",
]
