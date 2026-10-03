import json
import random
import time
from pathlib import Path

import numpy as np
import ray
import torch
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining
from sb3_contrib import MaskablePPO
from stable_baselines3.common.utils import FloatSchedule

from ai4sts2.environment import Sts2Env, evaluate, fingerprint, write_json
from ai4sts2.game import ROOT, prepare_game


class PopulationMember(tune.Trainable):
    def setup(self, config):
        torch.set_num_threads(1)
        self.build = fingerprint()
        self.environment = Sts2Env(config["executable"], seed=config["seed"])
        self.model = MaskablePPO(
            "MultiInputPolicy",
            self.environment,
            learning_rate=config["learning_rate"],
            ent_coef=config["entropy"],
            n_steps=64,
            batch_size=32,
            n_epochs=2,
            policy_kwargs={"net_arch": {"pi": [64], "vf": [64]}},
            device="cpu",
            seed=config["seed"],
        )
        if config.get("initial_checkpoint"):
            self.load_checkpoint(config["initial_checkpoint"])

    def step(self):
        started = time.monotonic()
        self.model.learn(
            total_timesteps=self.config["steps_per_iteration"], reset_num_timesteps=False
        )
        training_seconds = time.monotonic() - started
        result = evaluate(self.model, self.environment)
        self.evaluation = result
        write_json(Path(self.logdir) / "evaluation.json", result)
        return {
            "validation_win_rate": result["win_rate"],
            "environment_steps": self.model.num_timesteps,
            "training_seconds": training_seconds,
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
        if hasattr(self, "evaluation"):
            write_json(directory / "evaluation.json", self.evaluation)
        return checkpoint_dir

    def load_checkpoint(self, checkpoint_dir):
        directory = Path(checkpoint_dir)
        if json.loads((directory / "build.json").read_text()) != self.build:
            raise ValueError("Checkpoint game, bridge, dependencies or schema do not match.")
        self.model = MaskablePPO.load(directory / "policy.zip", env=self.environment, device="cpu")
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
                hyperparam_mutations={
                    "learning_rate": [1e-4, 3e-4, 1e-3],
                    "entropy": [0.0, 0.01, 0.03],
                },
                synch=True,
            )
            tuner = tune.Tuner(
                trainable,
                tune_config=tune.TuneConfig(
                    scheduler=scheduler,
                    num_samples=2,
                    reuse_actors=True,
                    time_budget_s=minutes * 60,
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
                param_space={
                    "executable": str(executable),
                    "seed": tune.randint(1, 2**30),
                    "learning_rate": tune.choice([1e-4, 3e-4, 1e-3]),
                    "entropy": tune.choice([0.0, 0.01, 0.03]),
                    "steps_per_iteration": steps,
                    "initial_checkpoint": checkpoint,
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
