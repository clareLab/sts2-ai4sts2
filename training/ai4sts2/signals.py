import copy
import hashlib
import math

import cloudpickle
import numpy as np
import torch
from stable_baselines3.common.running_mean_std import RunningMeanStd
from syllabus.curricula import LearningProgress
from syllabus.task_space import DiscreteTaskSpace
from tensordict import TensorDict
from torch import nn
from torchrl.objectives import RNDLoss

from ai4sts2.environment import CHARACTERS, STATE_FEATURES, features, visible_observation


def novelty_features(state):
    observation = visible_observation(state)
    observation.pop("turn", None)
    return features(observation, STATE_FEATURES)


class TrainingProgress:
    def __init__(self):
        self.rates = np.zeros(len(CHARACTERS), dtype=np.float32)
        self.totals = np.zeros(len(CHARACTERS), dtype=np.float64)
        self.counts = np.zeros(len(CHARACTERS), dtype=np.int64)

    def update(self, episode):
        index = CHARACTERS.index(episode["character"])
        floor = episode["floor"]
        self.totals[index] += 1.0 if episode["victory"] else floor / (floor + 1)
        self.counts[index] += 1

    def evaluate_agent(self, *args, **kwargs):
        present = self.counts > 0
        self.rates[present] = self.totals[present] / self.counts[present]
        self.totals.fill(0)
        self.counts.fill(0)
        rates = torch.from_numpy(self.rates.copy())
        return None, rates, rates, None


class TrainingSignals:
    def __init__(self, seed, config):
        with torch.random.fork_rng(devices=[]):
            torch.manual_seed(seed)
            predictor = nn.Sequential(nn.Linear(STATE_FEATURES, 64), nn.ReLU(), nn.Linear(64, 32))
            target = nn.Sequential(nn.Linear(STATE_FEATURES, 64), nn.ReLU(), nn.Linear(64, 32))
            self.rnd = RNDLoss(predictor, target, reduction="none", update_fraction=1.0)
        self.optimiser = torch.optim.Adam(predictor.parameters(), lr=0.001)
        self.normaliser = RunningMeanStd(shape=())
        self.curriculum = LearningProgress(
            TrainingProgress(),
            DiscreteTaskSpace(len(CHARACTERS)),
            continuous_progress=True,
            normalize_success=False,
            eval_interval=len(CHARACTERS),
        )
        self.curriculum.eval_and_update()
        self.seen = set()
        self.pending = []
        self.metrics = {"bonus_sum": 0.0, "bonus_steps": 0, "predictor_updates": 0}
        self.configure(config)

    def configure(self, config):
        self.rnd_scale = float(config.get("rnd_scale", 0.0))
        self.curriculum_mix = float(config.get("curriculum_mix", 0.0))
        alpha = float(config.get("curriculum_alpha", 0.1))
        if not math.isfinite(self.rnd_scale) or self.rnd_scale < 0:
            raise ValueError("Invalid exploration reward scale.")
        if not 0 <= self.curriculum_mix <= 0.95 or not 0 < alpha <= 1:
            raise ValueError("Invalid curriculum parameters.")
        self.curriculum.ema_alpha = alpha

    def probabilities(self):
        weights = np.asarray(self.curriculum._sample_distribution(), dtype=np.float64)
        weights = self.curriculum_mix * weights + (1 - self.curriculum_mix) / len(CHARACTERS)
        return weights / weights.sum()

    def character(self, rng):
        return CHARACTERS[int(rng.choice(len(CHARACTERS), p=self.probabilities()))]

    def reset(self, state):
        self.seen = {hashlib.sha256(novelty_features(state).tobytes()).digest()}

    def observe(self, state, episode, done):
        if done and not episode["truncated"]:
            self.curriculum.evaluator.update(episode)
            self.curriculum.update_on_episode(
                float(episode["victory"]),
                episode["steps"],
                CHARACTERS.index(episode["character"]),
                float(episode["victory"]),
            )
        if self.rnd_scale == 0:
            return 0.0
        observation = novelty_features(state)
        key = hashlib.sha256(observation.tobytes()).digest()
        with torch.no_grad():
            error = float(self.rnd(self.batch([observation]))["loss_predictor"].item())
        if not math.isfinite(error):
            raise ValueError("Non-finite exploration reward.")
        self.normaliser.update(np.asarray([error]))
        bonus = (
            self.rnd_scale * min(5.0, error / math.sqrt(float(self.normaliser.var) + 1e-8))
            if not done and key not in self.seen
            else 0.0
        )
        self.seen.add(key)
        self.pending.append(observation)
        if len(self.pending) >= 64:
            loss = self.rnd(self.batch(self.pending))["loss_predictor"].mean()
            self.optimiser.zero_grad(set_to_none=True)
            loss.backward()
            self.optimiser.step()
            self.pending.clear()
            self.metrics["predictor_updates"] += 1
        self.metrics["bonus_sum"] += bonus
        self.metrics["bonus_steps"] += int(bonus > 0)
        return bonus

    @staticmethod
    def batch(observations):
        return TensorDict(
            {"next": {"observation": torch.as_tensor(np.stack(observations))}},
            batch_size=[len(observations)],
        )

    def snapshot(self):
        return copy.deepcopy(
            {
                "rnd": self.rnd.state_dict(),
                "optimiser": self.optimiser.state_dict(),
                "normaliser": self.normaliser,
                "curriculum": cloudpickle.dumps(self.curriculum),
                "seen": self.seen,
                "pending": self.pending,
                "metrics": self.metrics,
            }
        )

    def restore(self, snapshot):
        state = copy.deepcopy(snapshot)
        self.rnd.load_state_dict(state["rnd"])
        self.optimiser.load_state_dict(state["optimiser"])
        self.normaliser = state["normaliser"]
        self.curriculum = cloudpickle.loads(state["curriculum"])
        self.seen = state["seen"]
        self.pending = state["pending"]
        self.metrics = state["metrics"]

    def report(self):
        return self.metrics.copy() | {
            "rnd_scale": self.rnd_scale,
            "curriculum_mix": self.curriculum_mix,
            "character_probabilities": dict(
                zip(CHARACTERS, self.probabilities().tolist(), strict=True)
            ),
        }
