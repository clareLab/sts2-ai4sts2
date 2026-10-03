import argparse
import json
import os
import random
import secrets
import signal
import statistics
import threading
import time
import zipfile
from pathlib import Path

import curve
import evaluate as holdout
import numpy as np
import ray
from ai4sts2.calibration import calibrate, selected_execution
from ai4sts2.environment import evaluation_plan, fingerprint, write_json
from ai4sts2.game import ROOT, prepare_game
from ai4sts2.resources import TRIAL_MEMORY, budget
from ai4sts2.train import PopulationMember, bound_mutations, search_space
from filelock import FileLock
from ray import tune
from ray.tune.schedulers import PopulationBasedTraining


class StudyMember(PopulationMember):
    def setup(self, config):
        super().setup(config)
        self.sample_count = config["steps_per_iteration"]


def runner_identity():
    return {
        name: holdout.digest(Path(__file__).with_name(name))
        for name in ("study.py", "curve.py", "evaluate.py")
    }


def checkpoint_steps(path):
    with zipfile.ZipFile(Path(path) / "policy.zip") as archive:
        return json.loads(archive.read("data"))["num_timesteps"]


def verify_inputs(plan):
    if plan["build"] != fingerprint("run") or plan["runner"] != runner_identity():
        raise ValueError("The saved study has a different build or runner.")
    for trial in plan["sources"] + plan["controls"]:
        if curve.checkpoint_files(trial["checkpoint"]) != trial["files"]:
            raise ValueError("A frozen source checkpoint has changed.")
        if checkpoint_steps(trial["checkpoint"]) != trial["environment_steps"]:
            raise ValueError("Checkpoint weights and step count do not match.")


def training_parameters(config):
    ignored = {
        "seed",
        "initial_checkpoint",
        "steps_per_iteration",
        "fixed_steps",
        "execution",
        "executable",
    }
    return {key: value for key, value in config.items() if key not in ignored}


def prepare_plan(output, initial=None, control=None, steps=None, per_character=None):
    path = output / "plan.json"
    requested = {
        "initial": initial,
        "control": control,
        "steps": steps,
        "per_character": per_character,
    }
    if path.exists():
        plan = json.loads(path.read_text())
        for name, value in requested.items():
            if value is not None:
                value = str(Path(value).resolve()) if name in {"initial", "control"} else value
                if plan["request"][name] != value:
                    raise ValueError("The saved study has a different request.")
        verify_inputs(plan)
        return plan
    if initial is None:
        raise ValueError("A new study requires --initial.")
    build = fingerprint("run")
    sources = curve.sources(json.loads(Path(initial).read_text()), "control", build)
    controls = (
        curve.sources(json.loads(Path(control).read_text()), "control", build) if control else []
    )
    steps = 3072 if steps is None else steps
    per_character = 2 if per_character is None else per_character
    if len(sources) < 2 or len({trial["environment_steps"] for trial in sources}) != 1:
        raise ValueError("Use at least two distinct initialisations at the same step count.")
    if steps < 256 or steps % 256 or per_character < 1:
        raise ValueError("Use positive case counts and a training budget divisible by 256.")
    shared = {
        key: value
        for key, value in sources[0]["config"].items()
        if key not in {"seed", "initial_checkpoint"}
    }
    if any(
        {
            key: value
            for key, value in trial["config"].items()
            if key not in {"seed", "initial_checkpoint"}
        }
        != shared
        for trial in sources
    ):
        raise ValueError("Source initialisations must use matching training parameters.")
    target = sources[0]["environment_steps"] + steps
    if controls and (
        {trial["seed"] for trial in controls} != {trial["seed"] for trial in sources}
        or any(trial["environment_steps"] != target for trial in controls)
    ):
        raise ValueError("Fixed controls must match the source seeds and target steps.")
    if any(
        training_parameters(trial["config"]) != training_parameters(sources[0]["config"])
        for trial in controls
    ):
        raise ValueError("Fixed controls must preserve the source training parameters.")
    seed = secrets.randbits(31)
    rng = np.random.RandomState(seed)
    domains = {key: search_space()[key] for key in ("learning_rate", "entropy")}
    members = {
        str(trial["seed"]): {
            key: float(domain.sample(random_state=rng)) for key, domain in domains.items()
        }
        | {"initial_checkpoint": trial["checkpoint"]}
        for trial in sources
    }
    plan = {
        "build": build,
        "runner": runner_identity(),
        "sources": sources,
        "controls": controls,
        "request": {
            "initial": str(Path(initial).resolve()),
            "control": str(Path(control).resolve()) if control else None,
            "steps": steps,
            "per_character": per_character,
        },
        "target_steps": target,
        "iterations": 4,
        "steps_per_iteration": steps // 4,
        "search_seed": seed,
        "members": members,
        "cases": evaluation_plan("run", secrets.randbits(60), "test", 4096, per_character),
        "certifying": False,
        "promoted": False,
    }
    verify_inputs(plan)
    write_json(path, plan)
    write_json(output / "initial.json", {"complete": True, "build": build, "trials": sources})
    return plan


def training_analysis(experiment):
    return tune.ExperimentAnalysis(experiment) if experiment.exists() else None


def collect_training(experiment, plan):
    analysis = training_analysis(experiment)
    if analysis is None:
        return {"complete": False, "trials": [], "statuses": [], "errors": []}
    statuses = [{"id": trial.trial_id, "status": trial.status} for trial in analysis.trials]
    errors = [trial.trial_id for trial in analysis.trials if trial.status == "ERROR"]
    trials = []
    for index, trial in enumerate(sorted(analysis.trials, key=lambda item: item.trial_id)):
        if trial.status != "TERMINATED":
            continue
        checkpoint = analysis.get_last_checkpoint(trial)
        if checkpoint is None:
            raise ValueError("A finished Ray trial has no checkpoint.")
        directory = Path(checkpoint.path)
        if json.loads((directory / "build.json").read_text()) != plan["build"]:
            raise ValueError("The Ray checkpoint has a different build.")
        steps = checkpoint_steps(directory)
        metrics = trial.last_result
        if (
            steps != plan["target_steps"]
            or metrics.get("environment_steps") != steps
            or metrics.get("training_iteration") != plan["iterations"]
        ):
            raise ValueError("A terminated Ray trial did not reach the frozen target.")
        trials.append(
            {
                "seed": index,
                "trial_id": trial.trial_id,
                "variant": "pbt",
                "checkpoint": str(directory),
                "environment_steps": steps,
                "config": trial.config,
                "metrics": metrics,
                "files": curve.checkpoint_files(directory),
            }
        )
    history = experiment / "pbt_global.txt"
    return {
        "complete": len(trials) == len(plan["sources"]) and not errors,
        "build": plan["build"],
        "trials": trials,
        "statuses": statuses,
        "errors": errors,
        "transfers": len(history.read_text().splitlines()) if history.exists() else 0,
        "certifying": False,
        "promoted": False,
    }


def fit_population(experiment, plan, resources, deadline):
    if deadline - time.monotonic() < 30:
        return
    execution = selected_execution("run")
    if execution is None:
        raise ValueError("A compatible execution calibration is required.")
    executable = prepare_game()
    ray.init(
        num_cpus=resources["concurrent_trials"],
        num_gpus=0,
        include_dashboard=False,
        object_store_memory=100 * 1024 * 1024,
        _node_ip_address="127.0.0.1",
    )
    timer = None
    try:
        trainable = tune.with_resources(StudyMember, {"cpu": 1, "memory": TRIAL_MEMORY})
        analysis = training_analysis(experiment)
        completed = (
            max(
                (trial.last_result.get("training_iteration", 0) for trial in analysis.trials),
                default=0,
            )
            if analysis is not None
            else 0
        )
        random.seed(plan["search_seed"] + completed)
        np.random.seed(plan["search_seed"] + completed)
        members = plan["members"]
        parameters = plan["sources"][0]["config"] | {
            "seed": tune.grid_search([trial["seed"] for trial in plan["sources"]]),
            "initial_checkpoint": tune.sample_from(
                lambda config: members[str(config["seed"])]["initial_checkpoint"]
            ),
            "learning_rate": tune.sample_from(
                lambda config: members[str(config["seed"])]["learning_rate"]
            ),
            "entropy": tune.sample_from(lambda config: members[str(config["seed"])]["entropy"]),
            "executable": str(executable),
            "execution": execution.to_dict(),
            "steps_per_iteration": plan["steps_per_iteration"],
            "fixed_steps": True,
        }
        scheduler = PopulationBasedTraining(
            time_attr="training_iteration",
            metric="validation_selection_score",
            mode="max",
            perturbation_interval=1,
            burn_in_period=completed + 1,
            synch=True,
            hyperparam_mutations={key: search_space()[key] for key in ("learning_rate", "entropy")},
            custom_explore_fn=bound_mutations,
        )
        remaining = deadline - time.monotonic()
        if remaining < 5:
            return
        timer = threading.Timer(remaining, lambda: os.kill(os.getpid(), signal.SIGINT))
        timer.daemon = True
        timer.start()
        try:
            tune.run(
                trainable,
                name=experiment.name,
                storage_path=str(experiment.parent),
                config={} if analysis is not None else parameters,
                scheduler=scheduler,
                reuse_actors=True,
                max_concurrent_trials=resources["concurrent_trials"],
                stop={"training_iteration": plan["iterations"]},
                checkpoint_config=tune.CheckpointConfig(
                    checkpoint_frequency=1, checkpoint_at_end=True, num_to_keep=4
                ),
                max_failures=0,
                raise_on_failed_trial=False,
                resume=analysis is not None,
                verbose=0,
            )
        except KeyboardInterrupt:
            pass
    finally:
        if timer is not None:
            timer.cancel()
        ray.shutdown()


def summary(output, plan, stage, training=None, evaluation=None):
    groups = {}
    if evaluation and evaluation["eligible"]:
        for variant in ("random", "fixed", "pbt"):
            members = [trial for trial in evaluation["trials"] if trial["variant"] == variant]
            episodes = [episode for trial in members for episode in trial["episodes"]]
            groups[variant] = {
                "mean_floor": statistics.mean(episode["floor"] for episode in episodes),
                "wins": sum(episode["victory"] for episode in episodes),
                "episodes": len(episodes),
                "members": {trial["name"]: trial["mean_floor"] for trial in members},
                "act_reach_rates": {
                    str(act + 1): sum(episode["act"] >= act for episode in episodes) / len(episodes)
                    for act in range(3)
                },
            }
    result = {
        "complete": stage == "complete",
        "stage": stage,
        "build": plan["build"],
        "certifying": False,
        "promoted": False,
        "comparison": groups,
        "training": training,
        "evaluation": evaluation,
        "next": (
            "Repeat independently before changing the baseline."
            if stage == "complete"
            else "Inspect the recorded failure."
            if stage == "failed"
            else "Run the same command again to continue the saved study."
        ),
    }
    write_json(output / "summary.json", result)
    return result


def run(output, initial=None, control=None, steps=None, per_character=None, minutes=30, workers=0):
    if not 0 < minutes <= 30:
        raise ValueError("Use a budget between zero and 30 minutes.")
    deadline = (
        min(time.monotonic() + minutes * 60, float(os.environ.get("AI4STS2_DEADLINE", "inf"))) - 20
    )
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(output / "run.lock", timeout=0):
        plan = prepare_plan(output, initial, control, steps, per_character)
        resources = budget().report(workers)
        experiment = output / "population"
        training = collect_training(experiment, plan)
        if training["errors"]:
            return summary(output, plan, "failed", training)
        controls = plan["controls"]
        cached_fixed = output / "fixed" / f"stage-{plan['target_steps']}.json"
        if not controls and cached_fixed.exists():
            fixed = json.loads(cached_fixed.read_text())
            if fixed["complete"]:
                controls = curve.sources(fixed, "control", plan["build"])
                if {trial["seed"] for trial in controls} != {
                    trial["seed"] for trial in plan["sources"]
                } or any(trial["environment_steps"] != plan["target_steps"] for trial in controls):
                    raise ValueError("Saved fixed controls do not match the study.")
                frozen = {trial["seed"]: trial["files"] for trial in fixed["trials"]}
                for trial in controls:
                    if trial["files"] != frozen[trial["seed"]]:
                        raise ValueError("A saved fixed-control checkpoint has changed.")
                    if checkpoint_steps(trial["checkpoint"]) != plan["target_steps"]:
                        raise ValueError("A saved fixed control has the wrong step count.")
        if not controls or not training["complete"]:
            if deadline - time.monotonic() < 30:
                return summary(output, plan, "budget", training)
            if selected_execution("run") is None:
                calibrate(min(5, (deadline - time.monotonic()) / 60), "run")
            if not controls:
                fixed = curve.run(
                    output / "initial.json",
                    output / "fixed",
                    targets=(plan["target_steps"],),
                    minutes=max(0.001, (deadline - time.monotonic()) / 60),
                    workers=workers,
                )
                if not fixed["complete"]:
                    return summary(output, plan, "budget", training)
                controls = fixed["stages"][str(plan["target_steps"])]["trials"]
            if not training["complete"]:
                fit_population(experiment, plan, resources, deadline - 10)
                training = collect_training(experiment, plan)
                write_json(output / "training.json", training)
                if not training["complete"]:
                    return summary(
                        output, plan, "failed" if training["errors"] else "budget", training
                    )
        comparison = {
            "complete": True,
            "build": plan["build"],
            "trials": training["trials"] + [trial | {"variant": "fixed"} for trial in controls],
        }
        write_json(output / "study.json", comparison)
        evaluation_dir = output / "evaluation"
        expected = plan["cases"] | {
            "build": plan["build"],
            "candidates": holdout.candidates(comparison, plan["build"]),
            "evaluator": holdout.digest(holdout.__file__),
        }
        manifest = evaluation_dir / "plan.json"
        if manifest.exists() and json.loads(manifest.read_text()) != expected:
            raise ValueError("The evaluation plan changed after training.")
        write_json(manifest, expected)
        evaluation = holdout.run(
            output / "study.json",
            evaluation_dir,
            minutes=max(0.001, (deadline - time.monotonic()) / 60),
            per_character=plan["request"]["per_character"],
            workers=workers,
            auto_calibrate=True,
        )
        stage = (
            "complete"
            if evaluation["eligible"]
            else "failed"
            if evaluation["complete"] or any(trial["errors"] for trial in evaluation["trials"])
            else "budget"
        )
        return summary(output, plan, stage, training, evaluation)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "artifacts/experiments/study"))
    parser.add_argument("--initial")
    parser.add_argument("--control")
    parser.add_argument("--steps", type=int)
    parser.add_argument("--per-character", type=int)
    parser.add_argument("--minutes", type=float, default=30)
    parser.add_argument("--workers", type=int, default=0)
    result = run(**vars(parser.parse_args()))
    print(
        json.dumps(
            {key: result[key] for key in ("complete", "stage", "comparison", "next")}, indent=2
        )
    )
    if not result["complete"]:
        raise SystemExit(1 if result["stage"] == "failed" else 2)
