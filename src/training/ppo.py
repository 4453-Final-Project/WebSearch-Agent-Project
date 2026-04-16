from __future__ import annotations

from dataclasses import asdict, dataclass

import torch


@dataclass(slots=True)
class PPOStats:
    """Scalar outputs from one clipped policy objective computation."""

    loss: torch.Tensor
    ratio_mean: float
    approx_kl: float

    def to_dict(self) -> dict[str, float]:
        return {
            "ratio_mean": self.ratio_mean,
            "approx_kl": self.approx_kl,
        }


def compute_clipped_policy_objective(
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_range: float,
    sample_weights: torch.Tensor | None = None,
) -> PPOStats:
    """Compute the standard clipped policy objective over sequence logprobs."""

    ratios = torch.exp(current_logprobs - old_logprobs)
    unclipped = ratios * advantages
    clipped = torch.clamp(ratios, 1.0 - clip_range, 1.0 + clip_range) * advantages
    objective = torch.minimum(unclipped, clipped)
    if sample_weights is None:
        loss = -torch.mean(objective)
    else:
        weights = sample_weights / sample_weights.sum().clamp(min=1e-8)
        loss = -(objective * weights).sum()
    approx_kl = torch.mean(old_logprobs - current_logprobs).abs().item()
    return PPOStats(
        loss=loss,
        ratio_mean=float(ratios.mean().item()),
        approx_kl=float(approx_kl),
    )
