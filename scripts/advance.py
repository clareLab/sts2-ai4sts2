import argparse
import gc
import hashlib
import json
import multiprocessing
import os
import secrets
import tempfile
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import curve
import evaluate as holdout
from ai4sts2.calibration import ensure_execution
from ai4sts2.environment import evaluation_plan, fingerprint, write_json
from ai4sts2.game import ROOT, prepare_game
from ai4sts2.resources import budget, cgroups
from ai4sts2.train import PopulationMember
from filelock import FileLock


def load(path):
    return json.loads(Path(path).read_text())


def identity(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def verify_trial(trial, build):
    path = Path(trial["checkpoint"])
    if curve.checkpoint_files(path) != trial["files"] or load(path / "build.json") != build:
        raise ValueError("The source checkpoint has changed.")


def prepare(output, initial, steps, per_character, transfer, compare_replay=False):
    if steps < 64 or steps % 64 or per_character < 1:
        raise ValueError("Use positive whole rollouts and positive evaluation case counts.")
    initial = Path(initial).resolve()
    source = load(initial)
    if not source["complete"] or len(source["trials"]) != 1:
        raise ValueError("Use one completed, validation-selected continuation.")
    if source["build"].get("scope") != "act1":
        raise ValueError("Progression selection requires the Act 1 task.")
    trial = source["trials"][0]
    if compare_replay and (
        not transfer
        or not trial["config"].get("replay_corpus")
        or not trial["config"].get("replay_updates", 0)
    ):
        raise ValueError("Replay comparison requires policy transfer and an active replay corpus.")
    verify_trial(trial, source["build"])
    build = fingerprint("act1", source["build"]["ascension"])

    def compatible(value):
        return {k: v for k, v in value.items() if k != "trainer"}

    if compatible(source["build"]) != compatible(build):
        raise ValueError("The game, bridge, dependencies and observation schema must match.")
    if source["build"] != build and not transfer:
        raise ValueError("Trainer changed; use --transfer for weights with a fresh optimiser.")
    request = {
        "initial": str(initial),
        "source_sha256": holdout.digest(initial),
        "steps": steps,
        "per_character": per_character,
        "transfer": transfer,
        "compare_replay": compare_replay,
        "build": build,
        "runner": {
            name: holdout.digest(Path(__file__).with_name(name))
            for name in ("advance.py", "curve.py", "evaluate.py")
        },
    }
    path = output / "plan.json"
    if path.exists():
        plan = load(path)
        if plan["request"] != request:
            raise ValueError("The saved iteration has a different request or code.")
        return plan
    panel = evaluation_plan(
        "act1",
        trial["config"].get("validation_seed", 0),
        "validation",
        4096,
        per_character,
        build["ascension"],
    )
    if source.get("validation_panel") is not None and source["validation_panel"] != panel:
        raise ValueError("Keep the frozen development panel unchanged.")
    plan = {
        "request": request,
        "source": trial,
        "source_build": source["build"],
        "cached_validation": source.get("selected_validation")
        if not transfer and source.get("evaluation_runner") == holdout.digest(holdout.__file__)
        else None,
        "validation_panel": panel,
        "training_seed": secrets.randbits(31) if transfer else trial["seed"],
        "target_steps": steps + (0 if transfer else trial["environment_steps"]),
        "test_used_for_selection": False,
    }
    write_json(path, plan)
    return plan


def persist(member, output, plan, metrics=None):
    steps = member.model.num_timesteps
    target = output / "checkpoints" / f"step-{steps:06}"
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="pending-", dir=target.parent) as temporary:
        directory = Path(temporary) / "checkpoint"
        directory.mkdir()
        member.save_checkpoint(directory)
        trial = {
            "plan": identity(plan),
            "variant": plan.get("variant", "candidate"),
            "seed": member.config["seed"],
            "checkpoint": str(target),
            "environment_steps": steps,
            "config": member.config,
            "files": curve.checkpoint_files(directory),
            "metrics": metrics,
        }
        write_json(directory / "result.json", trial)
        directory.rename(target)
    write_json(output / "latest.json", trial)
    return trial


def train(output, plan, deadline, executable=None):
    build = plan["request"]["build"]
    checkpoints = sorted((output / "checkpoints").glob("step-*/result.json"))
    current = load(checkpoints[-1]) if checkpoints else None
    if current is not None:
        verify_trial(current, build)
        if current["plan"] != identity(plan):
            raise ValueError("The saved training stage belongs to another iteration.")
        if not (output / "reference.json").exists():
            write_json(output / "reference.json", load(checkpoints[0]))
        if current["environment_steps"] == plan["target_steps"]:
            return current
    if time.monotonic() + 60 >= deadline:
        return current
    executable = executable or prepare_game()
    transfer = current is None and plan["request"]["transfer"]
    config = (current or plan["source"])["config"] | {
        "seed": plan["training_seed"],
        "executable": str(executable),
        "fixed_steps": True,
        "validate_each_iteration": False,
        "collect_replay": plan.get(
            "collect_replay",
            plan["source"]["config"].get(
                "collect_replay", bool(plan["source"]["config"].get("replay_corpus"))
            ),
        ),
        "initial_checkpoint": None if transfer else (current or plan["source"])["checkpoint"],
        "initial_policy": plan["source"]["checkpoint"] if transfer else None,
    }
    member = object.__new__(PopulationMember)
    member.config = config
    member._logdir = str(output)
    try:
        member.setup(config)
        if current is None:
            current = persist(member, output, plan)
            write_json(output / "reference.json", current)
        seconds_per_step = 0.2
        while member.model.num_timesteps < plan["target_steps"]:
            steps = min(512, plan["target_steps"] - member.model.num_timesteps)
            if time.monotonic() + seconds_per_step * steps * 1.3 + 15 >= deadline:
                break
            member.sample_count = steps
            try:
                with holdout.evaluation_requests(member.environment.game, deadline, None):
                    metrics = member.step()
            except holdout.EvaluationPaused:
                break
            current = persist(member, output, plan, metrics)
            seconds_per_step = metrics["training_seconds"] / steps
            print(
                json.dumps(
                    {
                        "stage": "training",
                        "variant": plan.get("variant", "candidate"),
                        "steps": member.model.num_timesteps,
                        "target": plan["target_steps"],
                        "seconds": metrics["training_seconds"],
                    }
                ),
                flush=True,
            )
        return current
    finally:
        member.cleanup()


def training_jobs(output, plan):
    if not plan["request"].get("compare_replay"):
        return [(output, plan)]
    return [
        (output / variant, plan | {"variant": variant, "collect_replay": collect})
        for variant, collect in (("fixed", False), ("online", True))
    ]


def train_round(output, plan, deadline, workers):
    jobs = training_jobs(output, plan)
    if len(jobs) == 1:
        return [train(*jobs[0], deadline)]
    executable = prepare_game()
    with ProcessPoolExecutor(
        max_workers=min(workers, len(jobs)), mp_context=multiprocessing.get_context("spawn")
    ) as pool:
        futures = [
            pool.submit(train, directory, variant, deadline, executable)
            for directory, variant in jobs
        ]
        return [future.result() for future in futures]


def training_totals(output, plan):
    totals = []
    for directory, variant in training_jobs(output, plan):
        rows = [load(path) for path in sorted(directory.glob("checkpoints/step-*/result.json"))]
        metrics = [row["metrics"] for row in rows if row["metrics"]]
        episodes = [episode for metric in metrics for episode in metric["training_episodes"]]
        totals.append(
            {
                "variant": variant.get("variant", "candidate"),
                "steps": rows[-1]["environment_steps"] - rows[0]["environment_steps"]
                if rows
                else 0,
                "episodes": len(episodes),
                "act1_clears": sum(episode["task_success"] for episode in episodes),
                "elite_successes": sum(
                    episode.get("act1_elite_wins", 0) > 0 for episode in episodes
                ),
                "seconds": sum(metric["training_seconds"] for metric in metrics),
                "optimisation_seconds": sum(metric["optimisation_seconds"] for metric in metrics),
                "replay_transitions": metrics[-1].get("replay_transitions", 0) if metrics else 0,
            }
        )
    return totals


def cache_reference(output, plan, candidate):
    cached = plan.get("cached_validation")
    if cached is None or (
        not cached["complete"]
        or not cached["eligible"]
        or cached["sha256"] != candidate["sha256"]
        or cached["evaluation_id"] != plan["validation_panel"]["evaluation_id"]
        or len(cached["episodes"]) != len(plan["validation_panel"]["cases"])
    ):
        return 0
    frozen = load(output / "plan.json")
    for index, (case, episode) in enumerate(zip(frozen["cases"], cached["episodes"], strict=True)):
        if episode["character"] != case["character"] or episode["seed"] != case["seed"]:
            raise ValueError("Cached validation case does not match its frozen panel.")
        write_json(
            output / "episodes" / f"{index:04}-{candidate['name']}.json",
            {
                "plan": identity(frozen),
                "case": case,
                "candidate": candidate["name"],
                "episode": episode,
            },
        )
    return len(cached["episodes"])


def validate(output, plan, current, deadline, workers):
    current = current if isinstance(current, list) else [current]
    reference = load(training_jobs(output, plan)[0][0] / "reference.json")
    if not plan["request"]["transfer"]:
        reference = plan["source"]
    trials = [reference | {"variant": "reference"}, *current]
    build = plan["request"]["build"]
    comparison = {"complete": True, "build": build, "trials": trials}
    write_json(output / "comparison.json", comparison)
    models = holdout.candidates(comparison, build)
    evaluation = output / "validation"
    frozen = plan["validation_panel"] | {
        "build": build,
        "candidates": models,
        "evaluator": holdout.digest(holdout.__file__),
    }
    path = evaluation / "plan.json"
    if path.exists() and load(path) != frozen:
        raise ValueError("The saved evaluation belongs to another training checkpoint.")
    write_json(path, frozen)
    reused = cache_reference(evaluation, plan, models[0])
    remaining = deadline - time.monotonic()
    if remaining <= 15:
        return {"complete": False, "eligible": False, "trials": [], "cached_episodes": reused}
    result = holdout.run(
        output / "comparison.json",
        evaluation,
        min(30, remaining / 60),
        plan["request"]["per_character"],
        workers,
        split="validation",
    )
    if result["eligible"]:
        save_selection(output, plan, trials, result)

    return result | {"cached_episodes": reused}


def save_selection(output, plan, trials, result):
    if not result["complete"] or not result["eligible"]:
        raise ValueError("Model selection requires a complete eligible validation panel.")
    best = max(
        range(len(trials)), key=lambda index: holdout.progression_key(result["trials"][index])
    )
    continuation = max(
        range(1, len(trials)), key=lambda index: holdout.progression_key(result["trials"][index])
    )
    for filename, index in (("selection.json", best), ("continuation.json", continuation)):
        write_json(
            output / filename,
            {
                "complete": True,
                "build": plan["request"]["build"],
                "trials": [trials[index] | {"variant": "control"}],
                "validation_panel": plan["validation_panel"],
                "selected_validation": result["trials"][index],
                "evaluation_runner": holdout.digest(holdout.__file__),
                "selected_by_validation": index == best,
                "selected_for_training_only": True,
                "certifying": False,
                "promoted": False,
            },
        )


def run(
    initial,
    output,
    steps=8192,
    minutes=30,
    per_character=2,
    transfer=False,
    workers=0,
    compare_replay=False,
):
    if not 0 < minutes <= 30:
        raise ValueError("Use a budget between zero and 30 minutes.")
    started = time.monotonic()
    deadline = min(started + minutes * 60, float(os.environ.get("AI4STS2_DEADLINE", "inf"))) - 20
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with FileLock(output / "run.lock", timeout=0):
        plan = prepare(output, initial, steps, per_character, transfer, compare_replay)
        resources = budget().report(workers)
        report = {"complete": False, "build": plan["request"]["build"], "resources": resources}
        try:
            phase_started = time.monotonic()
            ensure_execution(
                min(5, max(0, deadline - time.monotonic()) / 60),
                "act1",
                plan["request"]["build"]["ascension"],
            )
            report["calibration_seconds"] = time.monotonic() - phase_started
            phase_started = time.monotonic()
            reserve = max(180, (deadline - started) * 0.3)
            current = train_round(output, plan, deadline - reserve, resources["concurrent_trials"])
            report["training_seconds"] = time.monotonic() - phase_started
            gc.collect()
            finished = all(
                trial is not None and trial["environment_steps"] == plan["target_steps"]
                for trial in current
            )
            report |= {"training_complete": finished, "checkpoints": current}
            if finished:
                result = validate(output, plan, current, deadline, resources["concurrent_trials"])
                report |= {"validation": result, "complete": result["eligible"]}
            return report
        except Exception as error:
            report["error"] = f"{type(error).__name__}: {error}"
            raise
        finally:
            groups = [p for p in cgroups() if (p / "memory.peak").exists()]
            report |= {
                "seconds": time.monotonic() - started,
                "peak_bytes": int((groups[0] / "memory.peak").read_text()) if groups else None,
                "certifying": False,
                "promoted": False,
                "test_episodes": 0,
                "training_totals": training_totals(output, plan),
            }
            write_json(output / "summary.json", report)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", required=True)
    parser.add_argument("--output", default=str(ROOT / "artifacts/experiments/advance"))
    parser.add_argument("--steps", type=int, default=8192)
    parser.add_argument("--minutes", type=float, default=30)
    parser.add_argument("--per-character", type=int, default=2)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--transfer", action="store_true")
    parser.add_argument("--compare-replay", action="store_true")
    result = run(**vars(parser.parse_args()))
    print(json.dumps({"complete": result["complete"], "seconds": result["seconds"]}), flush=True)
    if not result["complete"]:
        raise SystemExit(1)
