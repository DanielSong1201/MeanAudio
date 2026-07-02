from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from drifting.Resonate.config import RESONATE_CONFIG, ResonateModelConfig
from meanaudio.ext.rotary_embeddings import compute_rope_rotations
from meanaudio.model.networks import FluxAudio, PreprocessedConditions


def _load_torch(path: Path, map_location: str | torch.device) -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=True)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _unwrap_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    """Return a plain model state dict without weakening strict loading."""

    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected a checkpoint mapping, got {type(payload).__name__}")

    candidate: Mapping[str, Any] = payload
    for key in ("state_dict", "model", "student"):
        nested = candidate.get(key)
        if isinstance(nested, Mapping):
            candidate = nested
            break

    if not candidate:
        raise ValueError("Checkpoint state dict is empty")
    if not all(isinstance(key, str) for key in candidate):
        raise TypeError("Checkpoint state dict keys must be strings")
    if not all(isinstance(value, torch.Tensor) for value in candidate.values()):
        raise TypeError("Checkpoint state dict values must all be tensors")

    state = dict(candidate)
    if state and all(key.startswith("module.") for key in state):
        state = {key.removeprefix("module."): value for key, value in state.items()}
    # Match the official Resonate loader: these are legacy/non-persistent
    # derived buffers and are not model parameters.
    for key in ("t_embed.freqs", "latent_rot", "text_rot"):
        state.pop(key, None)
    return state


class ResonateFluxAudio(FluxAudio):
    """Released Resonate ``fluxaudio_m_44k`` model with a TFD feature API.

    The parameter hierarchy is kept identical to the official Resonate model.
    Only non-persistent RoPE handling and intermediate-feature exposure are
    implemented here. ``extract_features`` deliberately does not use
    ``torch.no_grad``: callers must disable gradients only for positive
    features, while generated features need a differentiable path back to the
    student output.
    """

    def __init__(
        self,
        *,
        config: ResonateModelConfig = RESONATE_CONFIG,
        latent_mean: torch.Tensor | None = None,
        latent_std: torch.Tensor | None = None,
        empty_string_feat: torch.Tensor | None = None,
        empty_string_feat_c: torch.Tensor | None = None,
        use_rope: bool | None = None,
    ) -> None:
        # FluxAudio.__init__ calls initialize_rotations(), so these plain
        # attributes must exist before nn.Module initialization reaches it.
        object.__setattr__(self, "config", config)
        object.__setattr__(self, "max_latent_seq_len", config.max_latent_seq_len)
        use_rope = config.use_rope if use_rope is None else use_rope
        super().__init__(
            latent_dim=config.latent_dim,
            text_dim=config.text_dim,
            text_c_dim=config.text_c_dim,
            hidden_dim=config.hidden_dim,
            depth=config.depth,
            fused_depth=config.fused_depth,
            num_heads=config.num_heads,
            mlp_ratio=config.mlp_ratio,
            latent_seq_len=config.latent_seq_len,
            text_seq_len=config.text_seq_len,
            latent_mean=latent_mean,
            latent_std=latent_std,
            empty_string_feat=empty_string_feat,
            empty_string_feat_c=empty_string_feat_c,
            use_rope=use_rope,
        )

    def initialize_rotations(self) -> None:
        """Precompute the same variable-length RoPE buffers as Resonate."""

        head_dim = self.hidden_dim // self.num_heads
        latent_rot = compute_rope_rotations(
            self.max_latent_seq_len,
            head_dim,
            10_000,
            freq_scaling=1.0,
            device=self.device,
        )
        text_rot = compute_rope_rotations(
            self._text_seq_len,
            head_dim,
            10_000,
            freq_scaling=1.0,
            device=self.device,
        )
        self.register_buffer("_latent_rot_buffer", latent_rot, persistent=False)
        self.register_buffer("_text_rot_buffer", text_rot, persistent=False)

    def update_seq_lengths(self, latent_seq_len: int) -> None:
        if latent_seq_len < 1:
            raise ValueError("latent_seq_len must be positive")
        if latent_seq_len > self.max_latent_seq_len:
            raise ValueError(
                f"latent_seq_len={latent_seq_len} exceeds maximum "
                f"{self.max_latent_seq_len}"
            )
        self._latent_seq_len = latent_seq_len

    def _rotations(
        self,
        latent_len: int,
        text_len: int,
    ) -> tuple[torch.Tensor | None, torch.Tensor | None]:
        if not self.use_rope:
            return None, None
        if latent_len > self.max_latent_seq_len:
            raise ValueError(
                f"Latent sequence length {latent_len} exceeds RoPE maximum "
                f"{self.max_latent_seq_len}"
            )
        if text_len > self._text_seq_len:
            raise ValueError(
                f"Text sequence length {text_len} exceeds RoPE maximum "
                f"{self._text_seq_len}"
            )
        return (
            self._latent_rot_buffer[:, :latent_len],
            self._text_rot_buffer[:, :text_len],
        )

    def predict_flow(
        self,
        latent: torch.Tensor,
        t: torch.Tensor,
        conditions: PreprocessedConditions,
    ) -> torch.Tensor:
        text_f = conditions.text_f
        text_f_c = conditions.text_f_c
        latent = self.audio_input_proj(latent)
        latent_rot, text_rot = self._rotations(latent.shape[1], text_f.shape[1])

        global_c = self.t_embed(t).unsqueeze(1) + text_f_c.unsqueeze(1)
        extended_c = global_c
        for block in self.joint_blocks:
            latent, text_f = block(
                latent,
                text_f,
                global_c,
                extended_c,
                latent_rot,
                text_rot,
            )
        for block in self.fused_blocks:
            latent = block(latent, extended_c, latent_rot)
        return self.final_layer(latent, extended_c)

    def extract_features(
        self,
        latent: torch.Tensor,
        t: torch.Tensor,
        conditions: PreprocessedConditions,
        *,
        layers: tuple[str, ...] | None = None,
    ) -> dict[str, torch.Tensor]:
        requested = tuple(self.config.feature_layers if layers is None else layers)
        requested_set = set(requested)
        if len(requested_set) != len(requested):
            raise ValueError(f"Duplicate feature layers requested: {requested}")

        valid_layers = {"audio_proj"}
        valid_layers.update(f"joint_{i}" for i in range(len(self.joint_blocks)))
        valid_layers.update(f"fused_{i}" for i in range(len(self.fused_blocks)))
        invalid_layers = requested_set - valid_layers
        if invalid_layers:
            raise ValueError(
                f"Unknown Resonate feature layers: {sorted(invalid_layers)}. "
                f"Valid layers are: {sorted(valid_layers)}"
            )

        features: dict[str, torch.Tensor] = {}
        text_f = conditions.text_f
        text_f_c = conditions.text_f_c
        latent = self.audio_input_proj(latent)
        latent_rot, text_rot = self._rotations(latent.shape[1], text_f.shape[1])
        if "audio_proj" in requested_set:
            features["audio_proj"] = latent

        global_c = self.t_embed(t).unsqueeze(1) + text_f_c.unsqueeze(1)
        extended_c = global_c
        for index, block in enumerate(self.joint_blocks):
            latent, text_f = block(
                latent,
                text_f,
                global_c,
                extended_c,
                latent_rot,
                text_rot,
            )
            name = f"joint_{index}"
            if name in requested_set:
                features[name] = latent

        for index, block in enumerate(self.fused_blocks):
            latent = block(latent, extended_c, latent_rot)
            name = f"fused_{index}"
            if name in requested_set:
                features[name] = latent

        # Preserve the caller's requested ordering for stable loss/log output.
        return {name: features[name] for name in requested}

    def load_weights(self, src_dict: Mapping[str, torch.Tensor] | Any) -> None:
        self.load_state_dict(_unwrap_state_dict(src_dict), strict=True)


def build_resonate_model(
    *,
    config: ResonateModelConfig = RESONATE_CONFIG,
    weights_path: Path | None = None,
    map_location: str | torch.device = "cpu",
    device: str | torch.device | None = None,
    dtype: torch.dtype | None = None,
    latent_mean: torch.Tensor | None = None,
    latent_std: torch.Tensor | None = None,
    empty_string_feat: torch.Tensor | None = None,
    empty_string_feat_c: torch.Tensor | None = None,
    use_rope: bool | None = None,
) -> ResonateFluxAudio:
    model = ResonateFluxAudio(
        config=config,
        latent_mean=latent_mean,
        latent_std=latent_std,
        empty_string_feat=empty_string_feat,
        empty_string_feat_c=empty_string_feat_c,
        use_rope=use_rope,
    )
    if weights_path is not None:
        model.load_weights(_load_torch(weights_path, map_location))
    if device is not None or dtype is not None:
        model = model.to(device=device, dtype=dtype)
    return model


def load_resonate_checkpoint(
    model: ResonateFluxAudio,
    path: Path,
    *,
    map_location: str | torch.device = "cpu",
) -> None:
    model.load_weights(_load_torch(path, map_location))


def freeze_resonate_teacher(model: nn.Module) -> None:
    model.eval()
    model.requires_grad_(False)
