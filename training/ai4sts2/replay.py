import hashlib
import io
import json
import math
from pathlib import Path

import numpy as np
import torch
from stable_baselines3.common.callbacks import BaseCallback


def replay_dataset(manifest, build, discount, capacity=None):
    from tensordict import TensorDict

    from ai4sts2.environment import ACTION_FEATURES, MAX_ACTIONS, STATE_FEATURES, seed_string

    manifest = Path(manifest)
    collection = json.loads(manifest.read_text())
    if not collection.get("complete") or not collection.get("records"):
        raise ValueError("A complete training corpus is required.")
    if (
        capacity is not None
        and sum(row["episode"]["steps"] for row in collection["records"]) > capacity
    ):
        raise ValueError("Replay corpus exceeds its transition capacity.")
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


class TrajectoryCollector(BaseCallback):
    def __init__(self, directory, build, publish=None):
        super().__init__()
        self.directory = Path(directory)
        self.build = build
        self.publish = publish
        self.last_published_step = 0
        self.case = None
        self.pending = []
        self.records = []
        self.episodes = 0
        self.truncated = 0

    def _on_step(self):
        if self.model.n_envs != 1:
            raise ValueError("Trajectory collection requires one environment per learner.")
        info = self.locals["infos"][0]
        if info.get("split") != "train":
            raise ValueError("Only training trajectories can enter replay.")
        if info["steps"] == 1:
            self.case = {
                "character": info["character"],
                "seed": info["seed"],
                "seed_index": info["seed_index"],
                "mode": "online",
            }
            self.pending = []
        if self.case is None:
            return True
        if info["seed"] != self.case["seed"] or info["steps"] != len(self.pending) + 1:
            raise ValueError("Trajectory collection lost its episode boundary.")
        self.pending.append(
            {
                **{key: value[0].copy() for key, value in self.model._last_obs.items()},
                "action_masks": self.locals["action_masks"][0].astype(np.bool_, copy=True),
                "action_indices": np.int64(self.locals["actions"][0]),
                "rewards": np.float32(self.locals["rewards"][0]),
            }
        )
        if self.locals["dones"][0]:
            if info["truncated"]:
                self.truncated += 1
            else:
                self.persist(info)
                self.episodes += 1
            self.case = None
            self.pending = []
        return True

    def _on_rollout_start(self):
        self.flush()

    def _on_training_end(self):
        self.flush()

    def flush(self):
        if self.publish is not None and self.model.num_timesteps - self.last_published_step >= 512:
            self.publish()
            self.last_published_step = self.model.num_timesteps

    def persist(self, info):
        arrays = {key: np.stack([row[key] for row in self.pending]) for key in self.pending[0]}
        if arrays["rewards"][:-1].any() or arrays["rewards"][-1] != (
            1 if info["task_success"] else -1
        ):
            raise ValueError("Collected trajectories require sparse terminal task rewards.")
        stream = io.BytesIO()
        np.savez_compressed(stream, **arrays)
        payload = stream.getvalue()
        digest = hashlib.sha256(payload).hexdigest()
        self.directory.mkdir(parents=True, exist_ok=True)
        target = self.directory / f"{digest}.npz"
        temporary = target.with_suffix(".pending")
        temporary.write_bytes(payload)
        temporary.replace(target)
        record = {
            "case": self.case.copy(),
            "split": "train",
            "complete": True,
            "truncated": False,
            "build": self.build,
            "trajectory": str(target.resolve()),
            "sha256": digest,
            "episode": {
                key: value
                for key, value in info.items()
                if key not in {"terminal_observation", "TimeLimit.truncated"}
            },
        }
        self.records.append(record)

    def snapshot(self):
        return {
            key: getattr(self, key)
            for key in (
                "case",
                "pending",
                "records",
                "episodes",
                "truncated",
                "last_published_step",
            )
        }

    def restore(self, state):
        for key in ("case", "pending", "records", "episodes", "truncated", "last_published_step"):
            setattr(self, key, state[key])


def merge_corpus(manifest, records, directory, capacity=32768):
    from ai4sts2.environment import write_json

    if type(capacity) is not int or capacity < 1:
        raise ValueError("Replay capacity must be a positive number of transitions.")
    manifest = Path(manifest)
    collection = json.loads(manifest.read_text())
    if not collection.get("complete"):
        raise ValueError("Cannot extend an incomplete replay corpus.")
    combined = [
        row | {"trajectory": str((manifest.parent / row["trajectory"]).resolve())}
        for row in collection["records"]
    ]
    indexed = {
        (row["case"]["character"], row["case"]["seed"], row["case"]["mode"]): row["sha256"]
        for row in combined
    }
    for row in records:
        if (
            row.get("split") != "train"
            or not row.get("complete")
            or row.get("truncated") is not False
        ):
            raise ValueError("Only completed training episodes can enter replay.")
        key = (row["case"]["character"], row["case"]["seed"], row["case"]["mode"])
        if key in indexed:
            if indexed[key] != row["sha256"]:
                raise ValueError("The same training episode has conflicting trajectories.")
            continue
        indexed[key] = row["sha256"]
        combined.append(row)
    if any(row["episode"]["steps"] > capacity for row in combined):
        raise ValueError("Replay capacity must hold a complete episode.")
    count = sum(row["episode"]["steps"] for row in combined)
    while count > capacity:
        count -= combined.pop(0)["episode"]["steps"]
    result = {"complete": True, "records": combined, "transitions": count, "capacity": capacity}
    digest = hashlib.sha256(json.dumps(result, sort_keys=True).encode()).hexdigest()
    path = Path(directory) / f"corpus-{digest}.json"
    if not path.exists():
        write_json(path, result)
    return path
