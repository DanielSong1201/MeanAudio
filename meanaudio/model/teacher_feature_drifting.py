from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TeacherFeatureDriftingOutput:
    loss: torch.Tensor
    drifting_loss: torch.Tensor
    anchor_loss: torch.Tensor
    per_layer: dict[str, torch.Tensor]


def _as_radius_tensor(radii: tuple[float, ...], device: torch.device, dtype: torch.dtype) -> torch.Tensor:
    return torch.tensor(radii, device=device, dtype=dtype).clamp_min(1e-8)


def _pool_tokens(feature: torch.Tensor, pool_tokens: int | None) -> torch.Tensor:
    if feature.ndim == 2:
        return feature.unsqueeze(1)
    if feature.ndim != 3:
        raise ValueError(f"Expected feature rank 2 or 3, got shape {tuple(feature.shape)}")
    if pool_tokens is None or feature.shape[1] <= pool_tokens:
        return feature
    feature = feature.transpose(1, 2)
    feature = F.adaptive_avg_pool1d(feature, pool_tokens)
    return feature.transpose(1, 2)


def _prepare_feature(feature: torch.Tensor, pool_tokens: int | None, normalize: bool) -> torch.Tensor:
    feature = _pool_tokens(feature, pool_tokens)
    if normalize:
        feature = F.normalize(feature.float(), dim=-1).to(dtype=feature.dtype)
    return feature.reshape(-1, feature.shape[-1])


def _kernel_weights(query: torch.Tensor, keys: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
    dist_sq = torch.cdist(query.float(), keys.float()).pow(2)
    weights = torch.exp(-dist_sq / radius.float())
    return weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-12)


def _kernel_values(query: torch.Tensor, keys: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
    dist_sq = torch.cdist(query.float(), keys.float()).pow(2)
    return torch.exp(-dist_sq / radius.float())


def _mean_shift(query: torch.Tensor, keys: torch.Tensor, radius: torch.Tensor) -> torch.Tensor:
    weights = _kernel_weights(query, keys, radius)
    return torch.matmul(weights.to(dtype=keys.dtype), keys) - query


class TeacherFeatureDriftingLoss(nn.Module):
    """
    Teacher-feature drifting and anchor-margin coverage loss.

    The generated feature tensors should keep gradients to the student path.
    Positive/anchor tensors are detached internally because they are targets.
    """

    def __init__(
        self,
        *,
        radii: tuple[float, ...] = (0.02, 0.05, 0.1, 0.2),
        pool_tokens: int | None = 64,
        normalize_features: bool = True,
        anchor_margin_alpha: float = 0.5,
        lambda_tfd: float = 1.0,
        lambda_anchor: float = 0.2,
    ) -> None:
        super().__init__()
        if not radii:
            raise ValueError("At least one kernel radius is required.")
        self.radii = tuple(float(radius) for radius in radii)
        self.pool_tokens = pool_tokens
        self.normalize_features = normalize_features
        self.anchor_margin_alpha = anchor_margin_alpha
        self.lambda_tfd = lambda_tfd
        self.lambda_anchor = lambda_anchor

    def forward(
        self,
        generated_features: dict[str, torch.Tensor],
        positive_features: dict[str, torch.Tensor],
        anchor_features: dict[str, torch.Tensor] | None = None,
    ) -> TeacherFeatureDriftingOutput:
        if generated_features.keys() != positive_features.keys():
            raise ValueError(
                f"Feature layer mismatch: generated={sorted(generated_features)}, "
                f"positive={sorted(positive_features)}"
            )
        if anchor_features is not None and generated_features.keys() != anchor_features.keys():
            raise ValueError(
                f"Anchor layer mismatch: generated={sorted(generated_features)}, "
                f"anchor={sorted(anchor_features)}"
            )

        drifting_terms: list[torch.Tensor] = []
        anchor_terms: list[torch.Tensor] = []
        per_layer: dict[str, torch.Tensor] = {}

        for layer_name, generated in generated_features.items():
            positive = positive_features[layer_name]
            anchors_for_layer = None if anchor_features is None else anchor_features[layer_name]

            if generated.ndim in (2, 3):
                generated_groups = (generated,)
                positive_groups = (positive,)
                anchor_groups = (anchors_for_layer,)
            elif generated.ndim == 4:
                if positive.ndim != 4:
                    raise ValueError(
                        f"{layer_name} generated/positive rank mismatch: "
                        f"{generated.ndim} vs {positive.ndim}"
                    )
                if generated.shape[0] != positive.shape[0]:
                    raise ValueError(
                        f"{layer_name} generated/positive condition count mismatch: "
                        f"{generated.shape[0]} vs {positive.shape[0]}"
                    )
                if anchors_for_layer is not None:
                    if anchors_for_layer.ndim != 4 or anchors_for_layer.shape[0] != generated.shape[0]:
                        raise ValueError(
                            f"{layer_name} anchors must have rank 4 and the same condition count as "
                            f"generated features, got {tuple(anchors_for_layer.shape)}"
                        )
                    anchor_groups = tuple(anchors_for_layer.unbind(dim=0))
                else:
                    anchor_groups = (None,) * generated.shape[0]
                generated_groups = tuple(generated.unbind(dim=0))
                positive_groups = tuple(positive.unbind(dim=0))
            else:
                raise ValueError(
                    f"{layer_name} expected feature rank 2, 3, or grouped rank 4, "
                    f"got shape {tuple(generated.shape)}"
                )

            if generated.shape[-1] != positive.shape[-1]:
                raise ValueError(
                    f"{layer_name} generated/positive feature dimension mismatch: "
                    f"{generated.shape[-1]} vs {positive.shape[-1]}"
                )

            condition_drift_terms: list[torch.Tensor] = []
            condition_anchor_terms: list[torch.Tensor] = []
            for generated_group, positive_group, anchor_group in zip(
                generated_groups,
                positive_groups,
                anchor_groups,
                strict=True,
            ):
                query = _prepare_feature(generated_group, self.pool_tokens, self.normalize_features)
                pos_keys = _prepare_feature(
                    positive_group.detach(),
                    self.pool_tokens,
                    self.normalize_features,
                )
                gen_keys = query.detach()
                radii = _as_radius_tensor(self.radii, query.device, query.dtype)

                group_drift_terms = []
                for radius in radii:
                    attraction = _mean_shift(query.detach(), pos_keys, radius)
                    repulsion = _mean_shift(query.detach(), gen_keys, radius)
                    target = (query + attraction - repulsion).detach()
                    group_drift_terms.append(F.mse_loss(query, target))
                condition_drift_terms.append(torch.stack(group_drift_terms).mean())

                if anchor_group is not None:
                    anchors = _prepare_feature(
                        anchor_group.detach(),
                        self.pool_tokens,
                        self.normalize_features,
                    )
                else:
                    anchors = pos_keys
                group_anchor_terms = []
                for radius in radii:
                    support = _kernel_values(anchors.detach(), query, radius).mean(dim=-1)
                    anchor_dist_sq = torch.cdist(anchors.float(), anchors.float()).pow(2)
                    eye = torch.eye(
                        anchor_dist_sq.shape[0],
                        device=anchor_dist_sq.device,
                        dtype=torch.bool,
                    )
                    self_weights = torch.exp(
                        -anchor_dist_sq / (2.0 * radius.float())
                    ).masked_fill(eye, 0.0)
                    denom = (~eye).sum(dim=-1).clamp_min(1)
                    self_support = self_weights.sum(dim=-1) / denom
                    target_support = self.anchor_margin_alpha * self_support.detach()
                    group_anchor_terms.append(F.relu(target_support - support).mean())
                condition_anchor_terms.append(torch.stack(group_anchor_terms).mean())

            layer_drift = torch.stack(condition_drift_terms).mean()
            drifting_terms.append(layer_drift)
            layer_anchor = torch.stack(condition_anchor_terms).mean()
            anchor_terms.append(layer_anchor)
            per_layer[layer_name] = layer_drift.detach()

        drifting_loss = torch.stack(drifting_terms).mean()
        anchor_loss = torch.stack(anchor_terms).mean()
        loss = self.lambda_tfd * drifting_loss + self.lambda_anchor * anchor_loss
        return TeacherFeatureDriftingOutput(
            loss=loss,
            drifting_loss=drifting_loss.detach(),
            anchor_loss=anchor_loss.detach(),
            per_layer=per_layer,
        )
