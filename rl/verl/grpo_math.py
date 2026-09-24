"""Small, framework-independent checks for the standard GRPO objective.

verl performs the production rollout and optimizer update.  These helpers are
used by CPU tests and diagnostics to make the ratio/clipping semantics
explicit without replacing verl's implementation.
"""

from __future__ import annotations

from typing import Any

import torch


def group_relative_advantages(rewards: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    """Compute per-group standardized advantages for a ``[groups, G]`` tensor."""
    if rewards.ndim != 2 or rewards.shape[1] < 2:
        raise ValueError("rewards must have shape [num_groups, num_generations>=2]")
    mean = rewards.mean(dim=1, keepdim=True)
    std = rewards.std(dim=1, unbiased=False, keepdim=True)
    return (rewards - mean) / (std + epsilon)


def clipped_surrogate(
    current_logprobs: torch.Tensor,
    old_logprobs: torch.Tensor,
    advantages: torch.Tensor,
    clip_ratio: float = 0.2,
    mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Return the clipped policy loss and ratio diagnostics.

    ``current_logprobs`` and ``old_logprobs`` are per-token values.  A scalar
    advantage per sampled response is broadcast over the token dimension.
    """
    if current_logprobs.shape != old_logprobs.shape:
        raise ValueError("current_logprobs and old_logprobs must have the same shape")
    if advantages.ndim not in {1, 2}:
        raise ValueError("advantages must be [batch] or [batch, 1]")
    if advantages.ndim == 1:
        advantages = advantages.unsqueeze(1)
    if current_logprobs.ndim != 2 or current_logprobs.shape[0] != advantages.shape[0]:
        raise ValueError("log-prob tensors must be [batch, sequence]")
    ratio = torch.exp(current_logprobs - old_logprobs)
    clipped = ratio.clamp(1.0 - clip_ratio, 1.0 + clip_ratio)
    unclipped_objective = ratio * advantages
    clipped_objective = clipped * advantages
    token_objective = torch.minimum(unclipped_objective, clipped_objective)
    if mask is None:
        mask = torch.ones_like(token_objective)
    else:
        mask = mask.to(token_objective.dtype)
        if mask.shape != token_objective.shape:
            raise ValueError("mask must have the same shape as log-prob tensors")
    denom = mask.sum().clamp_min(1.0)
    loss = -(token_objective * mask).sum() / denom
    active_ratio = ratio[mask.bool()]
    active_delta = (current_logprobs - old_logprobs)[mask.bool()]
    diagnostics = {
        "clip_fraction": float(((active_ratio < 1.0 - clip_ratio) | (active_ratio > 1.0 + clip_ratio)).float().mean().detach().cpu()),
        "approx_policy_kl": float((-active_delta).mean().detach().cpu()) if active_delta.numel() else 0.0,
        "ratio_mean": float(active_ratio.mean().detach().cpu()) if active_ratio.numel() else 1.0,
        "ratio_min": float(active_ratio.min().detach().cpu()) if active_ratio.numel() else 1.0,
        "ratio_max": float(active_ratio.max().detach().cpu()) if active_ratio.numel() else 1.0,
        "old_current_logprob_diff": float(active_delta.mean().detach().cpu()) if active_delta.numel() else 0.0,
    }
    return loss, diagnostics


def diagnostic_record(
    rewards: torch.Tensor,
    advantages: torch.Tensor,
    policy_loss: torch.Tensor,
    ratio_diagnostics: dict[str, float],
    *,
    reference_kl: float | None = None,
    entropy: float | None = None,
    response_length: float | None = None,
) -> dict[str, float]:
    """Produce a stable JSON-friendly metric record for smoke/debug runs."""
    values: dict[str, float] = {
        "reward_mean": float(rewards.mean().detach().cpu()),
        "reward_std": float(rewards.std(unbiased=False).detach().cpu()),
        "advantage_mean": float(advantages.mean().detach().cpu()),
        "advantage_std": float(advantages.std(unbiased=False).detach().cpu()),
        "policy_loss": float(policy_loss.detach().cpu()),
        **ratio_diagnostics,
    }
    if reference_kl is not None:
        values["kl"] = float(reference_kl)
    if entropy is not None:
        values["entropy"] = float(entropy)
    if response_length is not None:
        values["response_length"] = float(response_length)
    return values
