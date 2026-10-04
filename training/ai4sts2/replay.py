import math

import numpy as np
import torch


def episode_returns(rewards, discount):
    rewards = np.asarray(rewards, dtype=np.float32)
    if rewards.ndim != 1 or not rewards.size or not np.isfinite(rewards).all():
        raise ValueError("Use a non-empty finite episode reward sequence.")
    if not math.isfinite(discount) or not 0 < discount <= 1:
        raise ValueError("Use a discount in (0, 1].")
    result = np.empty_like(rewards)
    total = 0.0
    for index in range(len(rewards) - 1, -1, -1):
        total = float(rewards[index]) + discount * total
        result[index] = total
    if not np.isfinite(result).all():
        raise ValueError("Episode returns are not finite.")
    return result


def self_imitation_loss(values, log_probabilities, returns, value_coefficient=0.5):
    if (
        values.ndim != 1
        or values.shape != log_probabilities.shape
        or values.shape != returns.shape
        or not values.numel()
    ):
        raise ValueError("Use matching non-empty vectors of values, log probabilities and returns.")
    if not all(torch.isfinite(value).all() for value in (values, log_probabilities, returns)):
        raise ValueError("Self-imitation inputs must be finite.")
    if not math.isfinite(value_coefficient) or value_coefficient < 0:
        raise ValueError("Use a finite non-negative value coefficient.")
    advantage = (returns.detach() - values).clamp_min(0)
    policy_loss = -(log_probabilities * advantage.detach()).mean()
    value_loss = 0.5 * advantage.square().mean()
    return policy_loss + value_coefficient * value_loss, {
        "policy_loss": policy_loss.detach(),
        "value_loss": value_loss.detach(),
        "positive_advantage_fraction": (advantage > 0).float().mean().detach(),
    }
