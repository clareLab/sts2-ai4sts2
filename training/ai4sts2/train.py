import hashlib
import json
import math
import os
import random
import tempfile
import time
from pathlib import Path

import numpy as np
import ray
import torch
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import FloatSchedule
from stable_baselines3.common.vec_env import DummyVecEnv

from ai4sts2.calibration import calibrate, quarantine_execution, selected_execution
from ai4sts2.environment import SCOPES, Sts2Env, evaluate, fingerprint, write_json
from ai4sts2.execution import Execution
from ai4sts2.game import ROOT, WorkerFailure, prepare_game
from ai4sts2.metrics import summarise
from ai4sts2.policy import SharedActionPolicy
from ai4sts2.replay import SelfImitationCallback, replay_dataset
from ai4sts2.resources import TRIAL_MEMORY, budget
from ai4sts2.signals import TrainingSignals


class TimedPPO(MaskablePPO):
    optimisation_seconds = 0.0

    def _excluded_save_params(self):
        return super()._excluded_save_params() + ["learning_diagnostics", "optimisation_seconds"]

    def drain_diagnostics(self):
        rows = getattr(self, "learning_diagnostics", [])
        self.learning_diagnostics = []
        return rows

    def train(self):
        started = time.perf_counter()
        buffer = self.rollout_buffer
        counts = buffer.action_masks.sum(axis=-1)
        row = {
            "environment_steps": self.num_timesteps,
            "samples": int(buffer.rewards.size),
            "positive_rewards": int((buffer.rewards > 0).sum()),
            "negative_rewards": int((buffer.rewards < 0).sum()),
            "choice_fraction": float((counts > 1).mean()),
            "entropy_ceiling": float(np.log(counts.clip(min=1)).mean()),
        }
        for name in ("rewards", "values", "returns", "advantages"):
            values = getattr(buffer, name)
            finite = values[np.isfinite(values)]
            row[name] = {
                "nonfinite": int(values.size - finite.size),
                **{
                    statistic: float(getattr(finite, statistic)()) if finite.size else None
                    for statistic in ("mean", "std", "min", "max")
                },
            }
        heads = {
            name: torch.nn.utils.parameters_to_vector(getattr(self.policy, name).parameters())
            .detach()
            .clone()
            for name in ("action_net", "value_net")
        }
        try:
            super().train()
            row["optimiser"] = {
                key.removeprefix("train/"): float(value) if np.isfinite(value) else None
                for key, value in self.logger.name_to_value.items()
                if key.startswith("train/")
            }
            row["head_update_l2"] = {
                name: float(
                    torch.linalg.vector_norm(
                        torch.nn.utils.parameters_to_vector(
                            getattr(self.policy, name).parameters()
                        ).detach()
                        - before
                    )
                )
                for name, before in heads.items()
            }
            if not hasattr(self, "learning_diagnostics"):
                self.learning_diagnostics = []
            self.learning_diagnostics.append(row)
        finally:
            self.optimisation_seconds += time.perf_counter() - started


def next_sample_count(training_seconds, evaluation_seconds, steps, rollout_size):
    target_seconds = min(120.0, max(15.0, evaluation_seconds * 4))
    estimate = steps * target_seconds / max(training_seconds, 0.001)
    return max(rollout_size, min(2048, math.ceil(estimate / rollout_size) * rollout_size))


def search_space(policy="flat"):
    parameters = {
        "learning_rate": tune.loguniform(1e-5, 3e-3),
        "entropy": tune.loguniform(1e-5, 0.1),
        "gamma": tune.uniform(0.95, 1.0),
        "gae_lambda": tune.uniform(0.85, 1.0),
        "clip_range": tune.uniform(0.1, 0.3),
        "epochs": tune.choice([1, 2, 4]),
        "rnd_scale": tune.choice([0.0, 0.0001, 0.001, 0.01]),
        "curriculum_mix": tune.choice([0.0, 0.25, 0.5, 0.75, 0.95]),
        "curriculum_alpha": tune.loguniform(0.02, 0.3),
    }
    if policy == "shared":
        parameters["temperature"] = tune.loguniform(0.1, 1.0)
    return parameters


def bound_mutations(config):
    for name, domain in search_space(config.get("policy", "flat")).items():
        if hasattr(domain, "lower"):
            config[name] = min(
                math.nextafter(domain.upper, domain.lower), max(domain.lower, config[name])
            )
        elif not domain.is_valid(config[name]):
            config[name] = min(domain.categories, key=lambda value: abs(value - config[name]))
    return config


def mutation_space(policy="flat"):
    return {
        name: domain.categories if hasattr(domain, "categories") else domain
        for name, domain in search_space(policy).items()
    }


def ablation_space(seed, repeats=1, study="signals"):
    if repeats < 1:
        raise ValueError("Use at least one model initialisation.")
    if study not in {"signals", "policies", "encodings"}:
        raise ValueError("Unknown ablation study.")
    if study == "encodings":
        return ablation_space(seed, repeats) | {
            "variant": tune.grid_search(["hash", "tree"]),
            "encoding": tune.sample_from(lambda spec: spec["config"]["variant"]),
            "policy": "shared",
            "rnd_scale": 0.0,
            "curriculum_mix": 0.0,
        }
    if study == "policies":
        return ablation_space(seed, repeats) | {
            "variant": tune.grid_search(["flat", "shared"]),
            "policy": tune.sample_from(lambda spec: spec["config"]["variant"]),
            "rnd_scale": 0.0,
            "curriculum_mix": 0.0,
        }
    return {
        "variant": tune.grid_search(["control", "exploration", "curriculum"]),
        "seed": tune.grid_search(list(range(seed, seed + repeats))),
        "learning_rate": 0.0003,
        "entropy": 0.01,
        "gamma": 0.99,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "epochs": 2,
        "rnd_scale": tune.sample_from(
            lambda spec: 0.001 if spec["config"]["variant"] == "exploration" else 0.0
        ),
        "curriculum_mix": tune.sample_from(
            lambda spec: 0.75 if spec["config"]["variant"] == "curriculum" else 0.0
        ),
        "curriculum_alpha": 0.1,
        "fixed_steps": True,
    }


def fit_or_recover(tuner, experiment_path, trainable):
    try:
        return tuner.fit(), False
    except KeyboardInterrupt:
        return tune.Tuner.restore(str(experiment_path), trainable=trainable).get_results(), True


def pilot_report(results, scope, build, iterations, interrupted=False):
    successful = [result for result in results if not result.error and result.checkpoint]
    return {
        "scope": scope,
        "certifying": False,
        "promoted": False,
        "interrupted": interrupted,
        "complete": bool(results)
        and not interrupted
        and len(successful) == len(results)
        and all(result.metrics.get("training_iteration", 0) >= iterations for result in results),
        "build": build,
        "errors": [str(result.error) for result in results if result.error],
        "trials": [
            {
                "path": result.path,
                "checkpoint": str(result.checkpoint.path),
                "config": result.config,
                "win_rate": result.metrics.get("validation_win_rate"),
                "task_success_rate": result.metrics.get("validation_task_success_rate"),
                "mean_floor": result.metrics.get("validation_mean_floor"),
                "selection_score": result.metrics.get("validation_selection_score"),
                "iterations": result.metrics.get("training_iteration"),
                "training_signals": result.metrics.get("training_signals"),
                "environment_steps": result.metrics.get("environment_steps"),
                "policy_parameters": result.metrics.get("policy_parameters"),
            }
            for result in successful
        ],
    }


class PopulationMember(tune.Trainable):
    def setup(self, config):
        if config.get("initial_checkpoint") and config.get("initial_policy"):
            raise ValueError("Choose checkpoint recovery or policy transfer.")
        torch.set_num_threads(1)
        policy = config.get("policy", "flat")
        if policy not in {"flat", "shared"}:
            raise ValueError("Unknown policy architecture.")
        if policy != "shared" and config.get("temperature", 1.0) != 1.0:
            raise ValueError("Sampling temperature requires the shared policy.")
        encoding = config.get("encoding", "hash")
        if encoding not in {"hash", "tree"} or (encoding == "tree" and policy != "shared"):
            raise ValueError("Structured observations require the shared policy.")
        self.scope = config.get("scope", "run")
        self.ascension = config.get("ascension", 10)
        self.build = fingerprint(self.scope, self.ascension)
        self.replay_buffer = None
        self.replay_identity = None
        self.replay_callback = None
        self.replay_updates = 0
        self.replay_generator = torch.Generator().manual_seed(config["seed"])
        self.recoveries = []
        self.signals = TrainingSignals(config["seed"], config)
        self.execution = selected_execution(self.scope, self.ascension) or Execution(
            **config.get("execution", {})
        )
        self.validation_environment = None
        self.environment = None
        while self.environment is None:
            try:
                self.environment = self.open_environment()
            except WorkerFailure as error:
                self.avoid_failed_execution(error)
        self.evaluation = None
        self.sample_count = config.get("steps_per_iteration", 128)
        self.model = TimedPPO(
            SharedActionPolicy if policy == "shared" else "MultiInputPolicy",
            DummyVecEnv([lambda: self.environment]),
            learning_rate=config["learning_rate"],
            ent_coef=config["entropy"],
            n_steps=64,
            batch_size=32,
            n_epochs=config.get("epochs", 2),
            gamma=config.get("gamma", 0.99),
            gae_lambda=config.get("gae_lambda", 0.95),
            clip_range=config.get("clip_range", 0.2),
            policy_kwargs=(
                {
                    "width": config.get("width", 64),
                    "encoding": encoding,
                    "temperature": config.get("temperature", 1.0),
                }
                if policy == "shared"
                else {"net_arch": {"pi": [64], "vf": [64]}}
            ),
            device="cpu",
            seed=config["seed"],
        )
        try:
            self.configure_replay(config)
            if config.get("initial_checkpoint"):
                self.load_checkpoint(config["initial_checkpoint"])
            elif config.get("initial_policy"):
                self.initialise_policy(config["initial_policy"])
        except Exception:
            self.cleanup()
            raise

    def initialise_policy(self, checkpoint_dir):
        directory = Path(checkpoint_dir)
        build = json.loads((directory / "build.json").read_text())
        if {
            k: v for k, v in build.items() if k not in {"trainer", "mod", "scope", "ascension"}
        } != {
            k: v for k, v in self.build.items() if k not in {"trainer", "mod", "scope", "ascension"}
        }:
            raise ValueError(
                "Policy transfer requires matching game, dependencies and observation schema."
            )
        python_rng, numpy_rng = random.getstate(), np.random.get_state()
        try:
            with torch.random.fork_rng(devices=[]):
                source = TimedPPO.load(directory / "policy.zip", device="cpu")
        finally:
            random.setstate(python_rng)
            np.random.set_state(numpy_rng)
        if (
            type(source.policy) is not type(self.model.policy)
            or source.observation_space != self.model.observation_space
            or source.action_space != self.model.action_space
            or getattr(source.policy, "encoding", "hash") != self.config.get("encoding", "hash")
        ):
            raise ValueError("Policy transfer architecture does not match.")
        self.model.policy.load_state_dict(source.policy.state_dict())

    def step(self, callback=None):
        with tempfile.TemporaryDirectory(prefix="recovery-", dir=self.logdir) as checkpoint:
            self.save_checkpoint(checkpoint)
            recovering = False
            while True:
                try:
                    if recovering:
                        self.environment = self.open_environment()
                        self.load_checkpoint(checkpoint)
                    result = self.train_iteration(callback)
                    result |= {
                        "execution": self.execution.to_dict(),
                        "recovery_events": self.recoveries.copy(),
                    }
                    self.recoveries.clear()
                    return result
                except WorkerFailure as error:
                    self.cleanup()
                    self.avoid_failed_execution(error)
                    recovering = True

    def avoid_failed_execution(self, error):
        self.recoveries.append({"execution": self.execution.to_dict(), "error": str(error)[:2048]})
        self.execution = quarantine_execution(self.execution, error, self.scope, self.ascension)

    def open_environment(self, training=True):
        return Sts2Env(
            self.config["executable"],
            seed=self.config["seed"],
            execution=self.execution,
            scope=self.scope,
            ascension=self.ascension,
            max_steps=256 if self.scope == "first_combat" else 4096,
            signals=self.signals if training else None,
            encoding=self.config.get("encoding", "hash"),
            discount=self.config.get("gamma", 0.99),
            progress_scale=self.config.get("progress_scale", 0.0) if training else 0.0,
        )

    def train_iteration(self, callback=None):
        self.environment.drain_measurements()
        self.model.optimisation_seconds = 0.0
        self.model.drain_diagnostics()
        started = time.monotonic()
        self.replay_callback = SelfImitationCallback(
            self.replay_buffer,
            self.config.get("replay_updates", 0),
            self.config.get("replay_value_coefficient", 0.01),
        )
        self.model.learn(
            total_timesteps=self.sample_count,
            reset_num_timesteps=False,
            callback=[
                self.replay_callback,
                *(callback if isinstance(callback, list) else [callback]),
            ]
            if callback
            else self.replay_callback,
        )
        replay_diagnostics = self.replay_callback.diagnostics
        self.replay_updates += len(replay_diagnostics)
        self.replay_callback = None
        training_seconds = time.monotonic() - started
        training_profile = self.environment.drain_measurements()
        measured_steps = self.sample_count
        training_episodes = self.environment.drain_episodes()
        metrics = {
            "environment_steps": self.model.num_timesteps,
            "policy_parameters": sum(
                parameter.numel() for parameter in self.model.policy.parameters()
            ),
            "training_seconds": training_seconds,
            "optimisation_seconds": self.model.optimisation_seconds,
            "learning_diagnostics": self.model.drain_diagnostics(),
            "auxiliary_updates": self.replay_updates,
            "auxiliary_diagnostics": replay_diagnostics,
            "evaluation_seconds": 0.0,
            "training_profile": training_profile,
            "evaluation_profile": {},
            "sample_count": measured_steps,
            "next_sample_count": self.sample_count,
            "scope": self.scope,
            "certifying": False,
            "validation_performed": False,
            "training_episodes": training_episodes,
            "training_progress": summarise(training_episodes) if training_episodes else None,
            "ongoing_episode_steps": self.environment.steps,
            "training_signals": self.signals.report(),
        }

        self.evaluation = None
        if not self.config.get("validate_each_iteration", True):
            return metrics
        started = time.monotonic()
        if self.validation_environment is None:
            self.validation_environment = self.open_environment(training=False)
        self.validation_environment.drain_measurements()
        result = evaluate(
            self.model,
            self.validation_environment,
            seed=self.config.get("validation_seed", 0),
            max_steps=self.validation_environment.max_steps,
        )
        evaluation_seconds = time.monotonic() - started
        evaluation_profile = self.validation_environment.drain_measurements()
        if not self.config.get("fixed_steps", False):
            self.sample_count = next_sample_count(
                training_seconds, evaluation_seconds, measured_steps, self.model.n_steps
            )
        self.evaluation = result
        write_json(Path(self.logdir) / "evaluation.json", result)
        if not result["eligible"]:
            raise RuntimeError("Validation was truncated; the candidate cannot be ranked.")
        return metrics | {
            "validation_performed": True,
            "validation_win_rate": result["win_rate"],
            "validation_task_success_rate": result["task_success_rate"],
            "validation_mean_floor": result["mean_floor"],
            "validation_median_floor": result["median_floor"],
            "validation_selection_score": result["selection_score"],
            "validation_eligible": result["eligible"],
            "validation_characters": result["characters"],
            "evaluation_id": result["evaluation_id"],
            "validation_episodes": result["episodes"],
            "evaluation_seconds": evaluation_seconds,
            "evaluation_profile": evaluation_profile,
            "next_sample_count": self.sample_count,
        }

    def save_checkpoint(self, checkpoint_dir):
        directory = Path(checkpoint_dir)
        self.model.save(directory / "policy.zip")
        torch.save(
            {
                "python_rng": random.getstate(),
                "numpy_rng": np.random.get_state(),
                "torch_rng": torch.get_rng_state(),
                "environment_rng": self.environment.rng.bit_generator.state,
            },
            directory / "random.pt",
        )
        write_json(directory / "build.json", self.build)
        write_json(directory / "environment.json", self.environment.snapshot())
        torch.save(self.signals.snapshot(), directory / "signals.pt")
        write_json(directory / "schedule.json", {"sample_count": self.sample_count})
        if self.replay_identity is not None:
            torch.save(
                {
                    "generator": self.replay_generator.get_state(),
                    "updates": self.replay_updates
                    + (len(self.replay_callback.diagnostics) if self.replay_callback else 0),
                    "corpus_sha256": self.replay_identity[1],
                },
                directory / "replay.pt",
            )
        else:
            (directory / "replay.pt").unlink(missing_ok=True)
        if self.evaluation is not None:
            write_json(directory / "evaluation.json", self.evaluation)
        else:
            (directory / "evaluation.json").unlink(missing_ok=True)
        return checkpoint_dir

    def load_checkpoint(self, checkpoint_dir):
        while True:
            try:
                if self.environment is None:
                    self.environment = self.open_environment()
                return self.restore_checkpoint(checkpoint_dir)
            except WorkerFailure as error:
                self.cleanup()
                self.avoid_failed_execution(error)

    def restore_checkpoint(self, checkpoint_dir):
        directory = Path(checkpoint_dir)
        if json.loads((directory / "build.json").read_text()) != self.build:
            raise ValueError("Checkpoint game, bridge, dependencies or schema do not match.")
        self.environment.restore(json.loads((directory / "environment.json").read_text()))
        self.signals.restore(torch.load(directory / "signals.pt", weights_only=False))
        self.model = TimedPPO.load(
            directory / "policy.zip",
            env=DummyVecEnv([lambda: self.environment]),
            device="cpu",
            force_reset=False,
        )
        if self.model._last_obs is not None:
            if self.environment.state is None:
                raise ValueError("Checkpoint observation has no matching episode.")
            for key, value in self.environment.encode().items():
                if not np.array_equal(self.model._last_obs[key][0], value):
                    raise ValueError("Checkpoint observation does not match the restored episode.")
        self.sample_count = json.loads((directory / "schedule.json").read_text())["sample_count"]
        evaluation = directory / "evaluation.json"
        self.evaluation = json.loads(evaluation.read_text()) if evaluation.is_file() else None
        state = torch.load(directory / "random.pt", weights_only=False)
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        self.environment.rng.bit_generator.state = state["environment_rng"]
        self.apply_parameters(self.config)
        self.replay_callback = None
        if self.replay_identity is not None:
            replay = torch.load(directory / "replay.pt", weights_only=True)
            if replay["corpus_sha256"] != self.replay_identity[1]:
                raise ValueError("Checkpoint replay corpus does not match.")
            self.replay_generator.set_state(replay["generator"])
            self.replay_updates = replay["updates"]
        elif (directory / "replay.pt").exists():
            raise ValueError("Checkpoint recovery requires its replay corpus configuration.")

    def apply_parameters(self, config):
        expected = config.get("policy", "flat")
        actual = "shared" if isinstance(self.model.policy, SharedActionPolicy) else "flat"
        if expected != actual or (
            actual == "shared" and self.model.policy.width != config.get("width", 64)
        ):
            raise ValueError("Checkpoint policy architecture does not match the configuration.")
        if getattr(self.model.policy, "encoding", "hash") != config.get("encoding", "hash"):
            raise ValueError("Checkpoint observation encoding does not match the configuration.")
        temperature = config.get("temperature", 1.0)
        if actual == "shared":
            self.model.policy.temperature = temperature
            self.model.policy_kwargs["temperature"] = temperature
        elif temperature != 1.0:
            raise ValueError("Sampling temperature requires the shared policy.")
        self.signals.configure(config)
        self.model.learning_rate = config["learning_rate"]
        self.model.lr_schedule = FloatSchedule(config["learning_rate"])
        for group in self.model.policy.optimizer.param_groups:
            group["lr"] = config["learning_rate"]
        self.model.ent_coef = config["entropy"]
        self.model.gamma = config.get("gamma", self.model.gamma)
        self.environment.configure_reward(self.model.gamma, config.get("progress_scale", 0.0))
        self.model.gae_lambda = config.get("gae_lambda", self.model.gae_lambda)
        self.model.rollout_buffer.gamma = self.model.gamma
        self.model.rollout_buffer.gae_lambda = self.model.gae_lambda
        self.model.n_epochs = config.get("epochs", self.model.n_epochs)
        self.model.clip_range = FloatSchedule(config.get("clip_range", 0.2))
        self.configure_replay(config)

    def configure_replay(self, config):
        updates = config.get("replay_updates", 0)
        SelfImitationCallback(None, updates, config.get("replay_value_coefficient", 0.01))
        path = config.get("replay_corpus")
        if updates and not path:
            raise ValueError("Self-imitation updates require a training corpus.")
        if path and (
            config.get("encoding", "hash") != "hash"
            or config.get("rnd_scale", 0)
            or config.get("progress_scale", 0)
            or config.get("floor_goals")
        ):
            raise ValueError("Replay requires matching hash observations and sparse task rewards.")
        identity = (
            (
                str(Path(path).resolve()),
                hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                self.model.gamma,
            )
            if path
            else None
        )
        if identity and config.get("corpus_sha256", identity[1]) != identity[1]:
            raise ValueError("Replay corpus checksum does not match.")
        if self.replay_identity is not None and identity != self.replay_identity:
            raise ValueError("Cannot change an active replay corpus or return discount.")
        self.replay_identity = identity
        if updates and self.replay_buffer is None:
            from torchrl.data import TensorDictReplayBuffer, TensorStorage

            self.replay_buffer = TensorDictReplayBuffer(
                storage=TensorStorage(replay_dataset(path, self.build, self.model.gamma)),
                batch_size=self.model.n_steps,
                generator=self.replay_generator,
            )

    def reset_config(self, new_config):
        self.cleanup()
        self.config = new_config
        self.setup(new_config)
        return True

    def cleanup(self):
        for name in ("environment", "validation_environment"):
            environment = getattr(self, name, None)
            if environment is not None:
                environment.close()
                setattr(self, name, None)


def run(
    minutes=30,
    iterations=4,
    steps=128,
    resume=None,
    checkpoint=None,
    scope="run",
    experiment="pbt",
    seed=0,
    workers=0,
    repeats=1,
    policy="flat",
    width=64,
    encoding="hash",
    initial_policy=None,
    ascension=10,
):
    if not 0 < minutes <= 30:
        raise ValueError("This pilot supports a budget of at most 30 minutes.")
    if iterations < 2 or steps < 64 or steps % 64:
        raise ValueError("Use at least two iterations and a multiple of 64 steps.")
    if scope not in SCOPES:
        raise ValueError("Unknown episode scope.")
    if policy not in {"flat", "shared"} or not isinstance(width, int) or width < 1:
        raise ValueError("Invalid policy architecture or width.")
    if encoding not in {"hash", "tree"} or (encoding == "tree" and policy != "shared"):
        raise ValueError("Structured observations require the shared policy.")
    if sum(bool(value) for value in (resume, checkpoint, initial_policy)) > 1:
        raise ValueError("Choose experiment recovery, checkpoint recovery or policy transfer.")
    if experiment not in {"pbt", "ablation", "policy_ablation", "encoding_ablation"}:
        raise ValueError("Unknown experiment.")
    if experiment != "pbt" and (checkpoint or initial_policy):
        raise ValueError("The ablation requires identical fresh model initialisations.")
    resources = budget().report(workers)
    studies = {
        "ablation": "signals",
        "policy_ablation": "policies",
        "encoding_ablation": "encodings",
    }
    report_name = {"pbt": "pilot", "ablation": "ablation"}.get(experiment, studies.get(experiment))
    report_path = ROOT / f"artifacts/validation/{report_name}.json"
    report = {
        "scope": scope,
        "certifying": False,
        "promoted": False,
        "complete": False,
        "build": fingerprint(scope, ascension),
        "errors": [],
        "trials": [],
        "experiment": experiment,
        "resources": resources,
    }
    write_json(report_path, report)
    executable = prepare_game()
    started = time.monotonic()
    deadline = min(started + minutes * 60, float(os.environ.get("AI4STS2_DEADLINE", "inf")))
    execution = selected_execution(scope, ascension)
    if execution is None:
        calibrate(min(5, minutes / 3), scope, ascension)
        execution = selected_execution(scope, ascension)
    if execution is None:
        raise RuntimeError("Execution calibration did not produce a compatible configuration.")
    remaining_seconds = deadline - time.monotonic() - 10
    if remaining_seconds <= 0:
        raise TimeoutError("The pilot budget was used by calibration.")
    if checkpoint:
        checkpoint = str(Path(checkpoint).resolve())
    ray.init(
        num_cpus=resources["concurrent_trials"],
        num_gpus=0,
        include_dashboard=False,
        object_store_memory=100 * 1024 * 1024,
        _node_ip_address="127.0.0.1",
    )
    try:
        trainable = tune.with_resources(PopulationMember, {"cpu": 1, "memory": TRIAL_MEMORY})
        experiment_path = (
            Path(resume).resolve()
            if resume
            else ROOT / "artifacts/experiments" / time.strftime(f"{experiment}-%Y%m%d-%H%M%S")
        )
        remaining_seconds = deadline - time.monotonic() - 10
        if remaining_seconds <= 0:
            raise TimeoutError("The pilot has no training time remaining.")
        if resume:
            tuner = tune.Tuner.restore(str(Path(resume).resolve()), trainable=trainable)
        else:
            scheduler = None
            parameters = ablation_space(seed, repeats, studies.get(experiment, "signals"))
            if experiment == "pbt":
                scheduler = PopulationBasedTraining(
                    time_attr="training_iteration",
                    metric="validation_selection_score",
                    mode="max",
                    perturbation_interval=2,
                    burn_in_period=2,
                    hyperparam_mutations=mutation_space(policy),
                    custom_explore_fn=bound_mutations,
                    synch=False,
                )
                parameters = search_space(policy) | {
                    "seed": tune.randint(1, 2**30),
                    "policy": policy,
                    "width": width,
                    "encoding": encoding,
                }
                if policy == "shared":
                    parameters["temperature"] = 0.5
            tuner = tune.Tuner(
                trainable,
                tune_config=tune.TuneConfig(
                    scheduler=scheduler,
                    num_samples=2 if experiment == "pbt" else 1,
                    reuse_actors=True,
                    max_concurrent_trials=resources["concurrent_trials"],
                    time_budget_s=remaining_seconds,
                ),
                run_config=tune.RunConfig(
                    name=experiment_path.name,
                    storage_path=str(ROOT / "artifacts/experiments"),
                    stop={"training_iteration": iterations},
                    checkpoint_config=tune.CheckpointConfig(
                        checkpoint_frequency=1, checkpoint_at_end=True, num_to_keep=4
                    ),
                    failure_config=tune.FailureConfig(max_failures=0),
                    verbose=1,
                ),
                param_space=parameters
                | {
                    "executable": str(executable),
                    "steps_per_iteration": steps,
                    "initial_checkpoint": checkpoint,
                    "initial_policy": initial_policy,
                    "execution": execution.to_dict(),
                    "scope": scope,
                    "ascension": ascension,
                },
            )
        results, interrupted = fit_or_recover(tuner, experiment_path, trainable)
        report = pilot_report(
            results, scope, fingerprint(scope, ascension), iterations, interrupted
        )
        report["experiment"] = experiment
        report["resources"] = resources
        write_json(report_path, report)
        if report["errors"] or (not report["trials"] and not interrupted):
            raise RuntimeError(f"Experiment failed. See {report_path}.")
        return report
    finally:
        ray.shutdown()
