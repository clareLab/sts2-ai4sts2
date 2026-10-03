import argparse
import hashlib
import json
import multiprocessing
import os
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from ai4sts2.environment import fingerprint, write_json
from ai4sts2.game import ROOT, prepare_game
from ai4sts2.resources import budget
from ai4sts2.train import PopulationMember
from filelock import FileLock


def checkpoint_files(path):
    names = (
        "policy.zip",
        "build.json",
        "environment.json",
        "random.pt",
        "signals.pt",
        "schedule.json",
    )
    result = {}
    for name in names:
        with (Path(path) / name).open("rb") as stream:
            result[name] = hashlib.file_digest(stream, "sha256").hexdigest()
    return result


def sources(study, variant, build):
    if not study["complete"] or study["build"] != build:
        raise ValueError("A complete study with the current build is required.")
    result = []
    for trial in study["trials"]:
        if trial["variant"] != variant:
            continue
        checkpoint = Path(trial["checkpoint"]).resolve()
        if json.loads((checkpoint / "build.json").read_text()) != build:
            raise ValueError("Checkpoint build does not match.")
        config = trial.get("config")
        if config is None:
            config = json.loads((checkpoint.parent / "params.json").read_text())
        metrics = trial.get("metrics")
        if metrics is None:
            metrics = json.loads((checkpoint.parent / "result.json").read_text().splitlines()[-1])
        if metrics["environment_steps"] != trial["environment_steps"]:
            raise ValueError("Checkpoint timing and step count do not match.")
        result.append(
            {
                "seed": int(trial["seed"]),
                "variant": variant,
                "checkpoint": str(checkpoint),
                "environment_steps": trial["environment_steps"],
                "config": config,
                "metrics": metrics,
                "files": checkpoint_files(checkpoint),
            }
        )
    if not result or len({trial["seed"] for trial in result}) != len(result):
        raise ValueError("Use distinct model initialisations.")
    return result


def forecast_seconds(steps, metrics, parallelism=1):
    if (
        metrics["sample_count"] <= 0
        or metrics["training_seconds"] < 0
        or metrics["evaluation_seconds"] < 0
    ):
        raise ValueError("Invalid training timing.")
    estimate = (
        steps * metrics["training_seconds"] / metrics["sample_count"]
        + metrics["evaluation_seconds"]
    )
    return estimate * parallelism * 1.1 + 30


def saved_stage(directory, target, identity):
    path = directory / f"checkpoint_{target:06}" / "result.json"
    if not path.exists():
        return None
    result = json.loads(path.read_text())
    if result["plan"] != identity or result["environment_steps"] != target:
        raise ValueError("Saved stage does not match the learning curve.")
    if result["files"] != checkpoint_files(path.parent):
        raise ValueError("Saved checkpoint changed after validation.")
    return result


def run_candidate(job):
    source, targets, output, identity, deadline, parallelism = job
    directory = Path(output) / f"seed-{source['seed']}"
    directory.mkdir(parents=True, exist_ok=True)
    current = source
    pending = []
    for target in targets:
        saved = saved_stage(directory, target, identity)
        if saved is not None:
            if pending:
                raise ValueError("A completed stage has an incomplete predecessor.")
            current = saved
        else:
            pending.append(target)
    if not pending:
        return {"seed": source["seed"], "status": "cached"}
    if current is not source:
        parallelism = 1
    estimate = forecast_seconds(
        pending[0] - current["environment_steps"], current["metrics"], parallelism
    )
    if time.monotonic() + estimate >= deadline:
        return {"seed": source["seed"], "status": "budget"}
    config = source["config"] | {"initial_checkpoint": current["checkpoint"], "fixed_steps": True}
    member = object.__new__(PopulationMember)
    member.config = config
    member._logdir = str(directory)
    try:
        member.setup(config)
        if member.model.num_timesteps != current["environment_steps"]:
            raise ValueError("Restored model has the wrong step count.")
        for target in pending:
            steps = target - member.model.num_timesteps
            if (
                time.monotonic() + forecast_seconds(steps, current["metrics"], parallelism)
                >= deadline
            ):
                return {"seed": source["seed"], "status": "budget"}
            member.sample_count = steps
            metrics = member.step()
            if member.model.num_timesteps != target or not metrics["validation_eligible"]:
                raise ValueError("The training stage did not finish cleanly.")
            checkpoint = directory / f"checkpoint_{target:06}"
            checkpoint.mkdir(exist_ok=True)
            member.save_checkpoint(checkpoint)
            current = {
                "plan": identity,
                "seed": source["seed"],
                "variant": source["variant"],
                "environment_steps": target,
                "checkpoint": str(checkpoint.resolve()),
                "config": config,
                "metrics": metrics,
                "files": checkpoint_files(checkpoint),
            }
            write_json(checkpoint / "result.json", current)
            parallelism = 1
            print(
                json.dumps(
                    {
                        "seed": source["seed"],
                        "steps": target,
                        "mean_floor": metrics["validation_mean_floor"],
                        "wins": metrics["validation_win_rate"],
                        "training_seconds": metrics["training_seconds"],
                        "evaluation_seconds": metrics["evaluation_seconds"],
                    }
                ),
                flush=True,
            )
        return {"seed": source["seed"], "status": "complete"}
    finally:
        member.cleanup()


def reports(plan, output, identity):
    result = {}
    for target in plan["targets"]:
        trials = []
        for source in plan["sources"]:
            saved = saved_stage(Path(output) / f"seed-{source['seed']}", target, identity)
            if saved is not None:
                trials.append(saved)
        result[str(target)] = {
            "complete": len(trials) == len(plan["sources"]),
            "certifying": False,
            "promoted": False,
            "build": plan["build"],
            "trials": trials,
        }
        write_json(Path(output) / f"stage-{target}.json", result[str(target)])
    return result


def run(study_path, output, targets=(1536, 3072), variant="control", minutes=20, workers=0):
    if not 0 < minutes <= 30:
        raise ValueError("Use a budget between zero and 30 minutes.")
    started = time.monotonic()
    deadline = min(started + minutes * 60, float(os.environ.get("AI4STS2_DEADLINE", "inf"))) - 15
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(output / "run.lock", timeout=0):
        build = fingerprint("run")
        inputs = sources(json.loads(Path(study_path).read_text()), variant, build)
        targets = list(targets)
        if targets != sorted(set(targets)) or not targets or any(target % 64 for target in targets):
            raise ValueError("Use ascending distinct targets divisible by 64.")
        if targets[0] <= max(source["environment_steps"] for source in inputs):
            raise ValueError("Targets must extend every source checkpoint.")
        plan = {
            "build": build,
            "sources": inputs,
            "targets": targets,
            "runner": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        }
        path = output / "plan.json"
        if path.exists() and json.loads(path.read_text()) != plan:
            raise ValueError("The saved learning curve has a different plan.")
        write_json(path, plan)
        identity = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
        resources = budget().report(workers)
        prepare_game()
        statuses = []
        stages = reports(plan, output, identity)
        pending = [
            source
            for source in inputs
            if saved_stage(output / f"seed-{source['seed']}", targets[-1], identity) is None
        ]
        try:
            if pending:
                with ProcessPoolExecutor(
                    max_workers=resources["concurrent_trials"],
                    mp_context=multiprocessing.get_context("spawn"),
                ) as pool:
                    futures = [
                        pool.submit(
                            run_candidate,
                            (
                                source,
                                targets,
                                str(output),
                                identity,
                                deadline,
                                resources["concurrent_trials"],
                            ),
                        )
                        for source in pending
                    ]
                    for future in as_completed(futures):
                        statuses.append(future.result())
                        stages = reports(plan, output, identity)
        finally:
            stages = reports(plan, output, identity)
            report = {
                "complete": all(stage["complete"] for stage in stages.values()),
                "certifying": False,
                "promoted": False,
                "build": build,
                "resources": resources,
                "seconds": time.monotonic() - started,
                "stages": stages,
                "statuses": statuses,
            }
            write_json(output / "summary.json", report)
        return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True)
    parser.add_argument("--output", default=str(ROOT / "artifacts/experiments/learning-curve"))
    parser.add_argument("--targets", type=int, nargs="+", default=[1536, 3072])
    parser.add_argument("--variant", default="control")
    parser.add_argument("--minutes", type=float, default=20)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    result = run(args.study, args.output, args.targets, args.variant, args.minutes, args.workers)
    print(json.dumps({"complete": result["complete"], "statuses": result["statuses"]}), flush=True)
    if not result["complete"]:
        raise SystemExit(1)
