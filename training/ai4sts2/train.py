import json
import math
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

from ai4sts2.calibration import calibrate, quarantine_execution, selected_execution
from ai4sts2.environment import Sts2Env, evaluate, fingerprint, write_json
from ai4sts2.execution import Execution
from ai4sts2.game import ROOT, WorkerFailure, prepare_game


class TimedPPO(MaskablePPO):
    optimisation_seconds = 0.0

    def train(self):
        started = time.perf_counter()
        try:
            return super().train()
        finally:
            self.optimisation_seconds += time.perf_counter() - started


def next_sample_count(training_seconds, evaluation_seconds, steps, rollout_size):
    target_seconds = min(120.0, max(15.0, evaluation_seconds * 4))
    estimate = steps * target_seconds / max(training_seconds, 0.001)
    return max(rollout_size, min(2048, math.ceil(estimate / rollout_size) * rollout_size))


def search_space():
    return {
        "learning_rate": tune.loguniform(1e-5, 3e-3),
        "entropy": tune.loguniform(1e-5, 0.1),
        "gamma": tune.uniform(0.95, 1.0),
        "gae_lambda": tune.uniform(0.85, 1.0),
        "clip_range": tune.uniform(0.1, 0.3),
        "epochs": tune.choice([1, 2, 4]),
    }


def bound_mutations(config):
    for name, domain in search_space().items():
        if hasattr(domain, "lower"):
            config[name] = min(
                math.nextafter(domain.upper, domain.lower), max(domain.lower, config[name])
            )
    return config


class PopulationMember(tune.Trainable):
    def setup(self, config):
        torch.set_num_threads(1)
        self.build = fingerprint()
        self.recoveries = []
        self.execution = selected_execution() or Execution(**config.get("execution", {}))
        self.environment = self.open_environment()
        self.evaluation = None
        self.sample_count = config.get("steps_per_iteration", 128)
        self.model = TimedPPO(
            "MultiInputPolicy",
            self.environment,
            learning_rate=config["learning_rate"],
            ent_coef=config["entropy"],
            n_steps=64,
            batch_size=32,
            n_epochs=config.get("epochs", 2),
            gamma=config.get("gamma", 0.99),
            gae_lambda=config.get("gae_lambda", 0.95),
            clip_range=config.get("clip_range", 0.2),
            policy_kwargs={"net_arch": {"pi": [64], "vf": [64]}},
            device="cpu",
            seed=config["seed"],
        )
        if config.get("initial_checkpoint"):
            self.load_checkpoint(config["initial_checkpoint"])

    def step(self):
        self.model._last_obs = None
        with tempfile.TemporaryDirectory(prefix="recovery-", dir=self.logdir) as checkpoint:
            self.save_checkpoint(checkpoint)
            while True:
                try:
                    result = self.train_iteration()
                    result |= {
                        "execution": self.execution.to_dict(),
                        "recovery_events": self.recoveries.copy(),
                    }
                    self.recoveries.clear()
                    return result
                except WorkerFailure as error:
                    self.cleanup()
                    self.avoid_failed_execution(error)
                    self.environment = self.open_environment()
                    self.load_checkpoint(checkpoint)

    def avoid_failed_execution(self, error):
        self.recoveries.append({"execution": self.execution.to_dict(), "error": str(error)[:2048]})
        self.execution = quarantine_execution(self.execution, error)

    def open_environment(self):
        while True:
            try:
                return Sts2Env(
                    self.config["executable"], seed=self.config["seed"], execution=self.execution
                )
            except WorkerFailure as error:
                self.avoid_failed_execution(error)

    def train_iteration(self):
        self.environment.drain_measurements()
        self.model.optimisation_seconds = 0.0
        started = time.monotonic()
        self.model.learn(total_timesteps=self.sample_count, reset_num_timesteps=False)
        training_seconds = time.monotonic() - started
        training_profile = self.environment.drain_measurements()
        started = time.monotonic()
        result = evaluate(self.model, self.environment)
        evaluation_seconds = time.monotonic() - started
        evaluation_profile = self.environment.drain_measurements()
        measured_steps = self.sample_count
        self.sample_count = next_sample_count(
            training_seconds, evaluation_seconds, measured_steps, self.model.n_steps
        )
        self.evaluation = result
        write_json(Path(self.logdir) / "evaluation.json", result)
        return {
            "validation_win_rate": result["win_rate"],
            "environment_steps": self.model.num_timesteps,
            "training_seconds": training_seconds,
            "optimisation_seconds": self.model.optimisation_seconds,
            "evaluation_seconds": evaluation_seconds,
            "training_profile": training_profile,
            "evaluation_profile": evaluation_profile,
            "sample_count": measured_steps,
            "next_sample_count": self.sample_count,
            "scope": "first_combat",
            "certifying": False,
            "validation_episodes": result["episodes"],
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
        write_json(directory / "schedule.json", {"sample_count": self.sample_count})
        if self.evaluation is not None:
            write_json(directory / "evaluation.json", self.evaluation)
        return checkpoint_dir

    def load_checkpoint(self, checkpoint_dir):
        directory = Path(checkpoint_dir)
        if json.loads((directory / "build.json").read_text()) != self.build:
            raise ValueError("Checkpoint game, bridge, dependencies or schema do not match.")
        self.model = TimedPPO.load(directory / "policy.zip", env=self.environment, device="cpu")
        self.sample_count = json.loads((directory / "schedule.json").read_text())["sample_count"]
        evaluation = directory / "evaluation.json"
        self.evaluation = json.loads(evaluation.read_text()) if evaluation.is_file() else None
        state = torch.load(directory / "random.pt", weights_only=False)
        random.setstate(state["python_rng"])
        np.random.set_state(state["numpy_rng"])
        torch.set_rng_state(state["torch_rng"])
        self.environment.rng.bit_generator.state = state["environment_rng"]
        self.apply_parameters(self.config)

    def apply_parameters(self, config):
        self.model.learning_rate = config["learning_rate"]
        self.model.lr_schedule = FloatSchedule(config["learning_rate"])
        for group in self.model.policy.optimizer.param_groups:
            group["lr"] = config["learning_rate"]
        self.model.ent_coef = config["entropy"]
        self.model.gamma = config.get("gamma", self.model.gamma)
        self.model.gae_lambda = config.get("gae_lambda", self.model.gae_lambda)
        self.model.rollout_buffer.gamma = self.model.gamma
        self.model.rollout_buffer.gae_lambda = self.model.gae_lambda
        self.model.n_epochs = config.get("epochs", self.model.n_epochs)
        self.model.clip_range = FloatSchedule(config.get("clip_range", 0.2))

    def reset_config(self, new_config):
        self.cleanup()
        self.config = new_config
        self.setup(new_config)
        return True

    def cleanup(self):
        if hasattr(self, "environment"):
            self.environment.close()


def run(minutes=30, iterations=4, steps=128, resume=None, checkpoint=None):
    if not 0 < minutes <= 30:
        raise ValueError("This pilot supports a budget of at most 30 minutes.")
    if iterations < 2 or steps < 64 or steps % 64:
        raise ValueError("Use at least two iterations and a multiple of 64 steps.")
    executable = prepare_game()
    started = time.monotonic()
    execution = selected_execution()
    if execution is None:
        calibrate(min(5, minutes / 3))
        execution = selected_execution()
    if execution is None:
        raise RuntimeError("Execution calibration did not produce a compatible configuration.")
    remaining_seconds = minutes * 60 - (time.monotonic() - started)
    if remaining_seconds <= 0:
        raise TimeoutError("The pilot budget was used by execution calibration.")
    if resume and checkpoint:
        raise ValueError("Choose either interrupted-experiment recovery or a starting checkpoint.")
    if checkpoint:
        checkpoint = str(Path(checkpoint).resolve())
    ray.init(
        num_cpus=2,
        num_gpus=0,
        include_dashboard=False,
        object_store_memory=100 * 1024 * 1024,
        _node_ip_address="127.0.0.1",
    )
    try:
        trainable = tune.with_resources(PopulationMember, {"cpu": 2})
        if resume:
            tuner = tune.Tuner.restore(str(Path(resume).resolve()), trainable=trainable)
        else:
            scheduler = PopulationBasedTraining(
                time_attr="training_iteration",
                metric="validation_win_rate",
                mode="max",
                perturbation_interval=2,
                burn_in_period=2,
                hyperparam_mutations=search_space() | {"epochs": [1, 2, 4]},
                custom_explore_fn=bound_mutations,
                synch=True,
            )
            tuner = tune.Tuner(
                trainable,
                tune_config=tune.TuneConfig(
                    scheduler=scheduler,
                    num_samples=2,
                    reuse_actors=True,
                    time_budget_s=remaining_seconds,
                ),
                run_config=tune.RunConfig(
                    name=time.strftime("pilot-%Y%m%d-%H%M%S"),
                    storage_path=str(ROOT / "artifacts/experiments"),
                    stop={"training_iteration": iterations},
                    checkpoint_config=tune.CheckpointConfig(
                        checkpoint_frequency=1, checkpoint_at_end=True, num_to_keep=4
                    ),
                    failure_config=tune.FailureConfig(max_failures=0),
                    verbose=1,
                ),
                param_space=search_space()
                | {
                    "executable": str(executable),
                    "seed": tune.randint(1, 2**30),
                    "steps_per_iteration": steps,
                    "initial_checkpoint": checkpoint,
                    "execution": execution.to_dict(),
                },
            )
        results = tuner.fit()
        successful = [result for result in results if not result.error and result.checkpoint]
        report = {
            "scope": "first_combat",
            "certifying": False,
            "promoted": False,
            "build": fingerprint(),
            "errors": [str(result.error) for result in results if result.error],
            "trials": [
                {
                    "path": result.path,
                    "config": result.config,
                    "win_rate": result.metrics.get("validation_win_rate"),
                    "iterations": result.metrics.get("training_iteration"),
                }
                for result in successful
            ],
        }
        write_json(ROOT / "artifacts/validation/pilot.json", report)
        if report["errors"] or not successful:
            raise RuntimeError("Pilot failed. See artifacts/validation/pilot.json.")
        return report
    finally:
        ray.shutdown()
