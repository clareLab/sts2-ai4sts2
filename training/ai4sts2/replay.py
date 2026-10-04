import hashlib
import json
import math
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback


def replay_dataset(manifest, build, discount):
    from tensordict import TensorDict

    from ai4sts2.environment import ACTION_FEATURES, MAX_ACTIONS, STATE_FEATURES, seed_string

    manifest = Path(manifest)
    collection = json.loads(manifest.read_text())
    if not collection.get("complete") or not collection.get("records"):
        raise ValueError("A complete training corpus is required.")
    expected_build = {key: value for key, value in build.items() if key != "trainer"}
    episodes, seen = [], set()
    for record in collection["records"]:
        if (
            not record.get("complete")
            or record.get("truncated") is not False
            or record.get("split") != "train"
            or {key: value for key, value in record["build"].items() if key != "trainer"}
            != expected_build
        ):
            raise ValueError("Replay requires complete, compatible training episodes.")
        case = record["case"]
        identity = (case["character"], case["seed"], case["mode"])
        if identity in seen or case["seed"] != seed_string("train", case["seed_index"]):
            raise ValueError("Duplicate or invalid training seed in replay.")
        seen.add(identity)
        path = manifest.parent / record["trajectory"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != record["sha256"]:
            raise ValueError("Replay episode checksum does not match.")
        with np.load(path, allow_pickle=False) as data:
            if set(data.files) != {"state", "actions", "action_masks", "action_indices", "rewards"}:
                raise ValueError("Unexpected replay episode fields.")
            count = record["episode"]["steps"]
            if (
                count < 1
                or data["state"].shape != (count, STATE_FEATURES)
                or data["actions"].shape != (count, MAX_ACTIONS, ACTION_FEATURES)
                or data["action_masks"].shape != (count, MAX_ACTIONS)
                or data["action_indices"].shape != (count,)
                or data["rewards"].shape != (count,)
                or data["action_indices"].dtype != np.int64
                or data["action_masks"].dtype != np.bool_
                or not all(np.isfinite(data[key]).all() for key in data.files)
            ):
                raise ValueError("Invalid replay episode arrays.")
            indices = data["action_indices"]
            if (
                np.any(indices < 0)
                or np.any(indices >= MAX_ACTIONS)
                or not data["action_masks"][np.arange(count), indices].all()
            ):
                raise ValueError("Replay contains an illegal action.")
            if data["rewards"][:-1].any() or data["rewards"][-1] != (
                1 if record["episode"]["task_success"] else -1
            ):
                raise ValueError("Replay requires complete sparse task rewards.")
            episodes.append(
                TensorDict(
                    {
                        "observation": TensorDict(
                            {
                                key: torch.from_numpy(data[key].copy())
                                for key in ("state", "actions")
                            },
                            batch_size=[count],
                        ),
                        "action": torch.from_numpy(indices.copy()),
                        "action_mask": torch.from_numpy(data["action_masks"].copy()),
                        "return": torch.from_numpy(episode_returns(data["rewards"], discount)),
                    },
                    batch_size=[count],
                )
            )
    return torch.cat(episodes, dim=0)


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
        training = policy.training
        policy.set_training_mode(True)
        try:
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
        finally:
            policy.set_training_mode(training)
