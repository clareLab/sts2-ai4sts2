import math

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback


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


class SelfImitationCallback(BaseCallback):
    def __init__(self, buffer, updates, value_coefficient):
        super().__init__()
        if isinstance(updates, bool) or not isinstance(updates, int) or updates < 0:
            raise ValueError("Use a non-negative integer replay update count.")
        if not math.isfinite(value_coefficient) or value_coefficient < 0:
            raise ValueError("Use a finite non-negative value coefficient.")
        self.buffer = buffer
        self.updates = updates
        self.value_coefficient = value_coefficient
        self.diagnostics = []

    def _on_training_start(self):
        self.previous_updates = self.model._n_updates

    def _on_step(self):
        return True

    def _on_rollout_start(self):
        self.update()

    def _on_training_end(self):
        self.update()

    def update(self):
        if self.model._n_updates == self.previous_updates:
            return
        self.previous_updates = self.model._n_updates
        policy = self.model.policy
        policy.set_training_mode(True)
        for _ in range(self.updates):
            batch = self.buffer.sample().to(self.model.device)
            values, log_probabilities, _ = policy.evaluate_actions(
                dict(batch["observation"]), batch["action"], action_masks=batch["action_mask"]
            )
            loss, metrics = self_imitation_loss(
                values.flatten(), log_probabilities, batch["return"], self.value_coefficient
            )
            policy.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), self.model.max_grad_norm, error_if_nonfinite=True
            )
            policy.optimizer.step()
            self.diagnostics.append(
                {
                    "environment_steps": self.model.num_timesteps,
                    "ppo_updates": self.model._n_updates,
                    "loss": float(loss.detach()),
                    "gradient_norm": float(norm),
                    **{key: float(value) for key, value in metrics.items()},
                }
            )
