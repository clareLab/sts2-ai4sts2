import copy
import hashlib
import math

import cloudpickle
import numpy as np
import torch
from stable_baselines3.common.running_mean_std import RunningMeanStd
from syllabus.curricula import LearningProgress, SequentialCurriculum
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
        self.totals[index] += float(episode.get("task_success", episode["victory"]))
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
        self.floor_curriculum = None
        self.goal_settings = None
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
        goals = tuple(config.get("floor_goals", ()))
        minimum = config.get("goal_min_episodes", 10)
        success_rate = config.get("goal_success_rate", 0.8)
        if (
            any(type(goal) is not int or goal < 2 for goal in goals)
            or list(goals) != sorted(set(goals))
            or type(minimum) is not int
            or minimum < 1
            or not 0 < success_rate <= 1
        ):
            raise ValueError("Invalid floor curriculum settings.")
        settings = (goals, minimum, success_rate) if goals else None
        if self.floor_curriculum is not None and settings != self.goal_settings:
            raise ValueError("Cannot change an active floor curriculum.")
        if goals and self.floor_curriculum is None:
            tasks = [*goals, 0]
            self.floor_curriculum = SequentialCurriculum(
                tasks,
                [f"episodes>={minimum}&episode_return>={success_rate}"] * len(goals),
                DiscreteTaskSpace(len(tasks), tasks),
                return_buffer_size=100,
            )
            self.goal_settings = settings

    def goal(self):
        if self.floor_curriculum is None:
            return None
        current = self.floor_curriculum.current_curriculum
        return current.task_space.decode(current.sample()[0])

    def probabilities(self):
        weights = np.asarray(self.curriculum._sample_distribution(), dtype=np.float64)
        weights = self.curriculum_mix * weights + (1 - self.curriculum_mix) / len(CHARACTERS)
        return weights / weights.sum()

    def character(self, rng):
        return CHARACTERS[int(rng.choice(len(CHARACTERS), p=self.probabilities()))]

    def reset(self, state):
        self.seen = {hashlib.sha256(novelty_features(state).tobytes()).digest()}

    def observe(self, state, episode, done):
        if done and "goal_floor" in episode and not episode["truncated"]:
            if self.floor_curriculum is None or episode["goal_floor"] != self.goal():
                raise ValueError("Episode goal does not match the floor curriculum.")
            self.floor_curriculum.update_on_episode(
                float(episode["goal_success"]),
                episode["steps"],
                self.floor_curriculum.task_space.encode(episode["goal_floor"]),
                float(episode["goal_success"]),
            )
        elif done and not episode["truncated"]:
            self.curriculum.evaluator.update(episode)
            success = float(episode.get("task_success", episode["victory"]))
            self.curriculum.update_on_episode(
                success,
                episode["steps"],
                CHARACTERS.index(episode["character"]),
                success,
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
                "floor_curriculum": cloudpickle.dumps(self.floor_curriculum),
                "goal_settings": self.goal_settings,
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
        self.floor_curriculum = cloudpickle.loads(
            state.get("floor_curriculum", cloudpickle.dumps(None))
        )
        self.goal_settings = state.get("goal_settings")

    def report(self):
        return self.metrics.copy() | {
            "rnd_scale": self.rnd_scale,
            "curriculum_mix": self.curriculum_mix,
            "character_probabilities": dict(
                zip(CHARACTERS, self.probabilities().tolist(), strict=True)
            ),
            "floor_curriculum": (
                {
                    "goal_floor": self.goal(),
                    "stage_episodes": self.floor_curriculum.n_episodes,
                    "total_episodes": self.floor_curriculum.total_episodes,
                    "recent_success_rate": (
                        float(np.mean(self.floor_curriculum.episode_returns))
                        if self.floor_curriculum.episode_returns
                        else None
                    ),
                }
                if self.floor_curriculum is not None
                else None
            ),
        }
