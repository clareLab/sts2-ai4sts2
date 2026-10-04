import argparse
import hashlib
import json
import os
import secrets
import statistics
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from ai4sts2.calibration import calibrate, selected_execution
from ai4sts2.environment import Sts2Env, evaluation_plan, fingerprint, write_json
from ai4sts2.game import ROOT, WorkerFailure, prepare_game
from ai4sts2.metrics import compare_evaluations, health_retention, summarise
from ai4sts2.policy import evaluation_action
from ai4sts2.resources import budget
from sb3_contrib import MaskablePPO


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def candidates(study, build, policy_modes=("deterministic",)):
    if (
        not policy_modes
        or len(set(policy_modes)) != len(policy_modes)
        or any(mode not in {"deterministic", "sampled"} for mode in policy_modes)
    ):
        raise ValueError("Use distinct deterministic or sampled policy modes.")
    if not study["complete"] or not study["trials"] or study["build"] != build:
        raise ValueError("A complete study with the current build is required.")
    result = []
    for trial in study["trials"]:
        checkpoint = Path(trial["checkpoint"]).resolve()
        if json.loads((checkpoint / "build.json").read_text()) != build:
            raise ValueError("Checkpoint build does not match the evaluation build.")
        for mode in policy_modes:
            result.append(
                {
                    "name": f"{trial['variant']}-{trial['seed']}"
                    + ("-sampled" if mode == "sampled" else ""),
                    "variant": trial["variant"],
                    "policy_mode": mode,
                    "checkpoint": str(checkpoint),
                    "sha256": digest(checkpoint / "policy.zip"),
                }
            )
    if len({item["name"] for item in result}) != len(result):
        raise ValueError("Candidate names must be unique.")
    return result


class EvaluationPaused(TimeoutError):
    pass


@contextmanager
def evaluation_requests(game, deadline, cancelled):
    original = game.request
    timeout = getattr(game, "timeout", 75)

    def request(*args, **kwargs):
        remaining = deadline - time.monotonic()
        if remaining <= 0 or (cancelled is not None and cancelled()):
            raise EvaluationPaused("Evaluation paused.")
        game.timeout = min(timeout, remaining)
        return original(*args, **kwargs)

    game.request = request
    try:
        yield
    finally:
        game.request = original
        game.timeout = timeout


def episode(
    model,
    environment,
    case,
    deadline,
    resume=None,
    checkpoint=None,
    cancelled=None,
    deterministic=True,
    split="test",
):
    with evaluation_requests(environment.game, deadline, cancelled):
        rng = np.random.default_rng(case["action_seed"])
        if resume is None:
            observation, _ = environment.reset(
                seed=case["seed_index"], options={"character": case["character"], "split": split}
            )
        else:
            if resume.get("deterministic", True) != deterministic:
                raise ValueError("The saved continuation uses a different policy mode.")
            parameters = resume["environment"]["journal"]["parameters"]
            if parameters["seed"] != case["seed"] or parameters["character"] != case["character"]:
                raise ValueError("The saved continuation belongs to a different evaluation case.")
            environment.restore(resume["environment"])
            rng.bit_generator.state = resume["action_rng"]
            observation = environment.encode()
        if environment.journal["parameters"]["seed"] != case["seed"]:
            raise ValueError("The environment seed does not match the evaluation case.")
        first_three = resume.get("first_three_monsters") if resume else None
        if environment.state.get("act1_monster_wins", 0) >= 3 and first_three is None:
            raise ValueError("The saved episode is missing its earlier combat health.")
        terminated = truncated = False
        while not (terminated or truncated):
            if checkpoint is not None:
                checkpoint(
                    {
                        "environment": environment.snapshot(),
                        "action_rng": rng.bit_generator.state,
                        "deterministic": deterministic,
                        "first_three_monsters": first_three,
                    }
                )
            action = evaluation_action(
                model, observation, environment.action_masks(), rng, deterministic
            )
            observation, _, terminated, truncated, info = environment.step(action)
            if info.get("act1_monster_wins", 0) >= 3 and first_three is None:
                player = environment.state["observation"]["player"]
                first_three = {
                    "hp": player["hp"],
                    "max_hp": player["max_hp"],
                    "act1_monster_health": info["act1_monster_health"],
                    "floor": info["floor"],
                    "steps": info["steps"],
                }
                health_retention(first_three)
        if "act1_monster_wins" in info:
            info = info | {"first_three_monsters": first_three}
        environment.drain_episodes()
        return info | {
            "truncated": truncated,
            "trajectory_digest": hashlib.sha256(
                json.dumps(environment.journal, sort_keys=True).encode()
            ).hexdigest(),
        }


def progression_key(report):
    if report.get("scope") != "act1" or not report["complete"] or not report["eligible"]:
        raise ValueError("Progression selection requires a complete Act 1 evaluation.")
    rows = report["episodes"]
    if not rows or any("first_three_monsters" not in row for row in rows):
        raise ValueError("Progression selection requires recorded combat health.")
    return (
        report["task_successes"],
        sum(row["act1_elite_wins"] >= 1 for row in rows),
        statistics.mean(
            health_retention(row["first_three_monsters"])
            if row["first_three_monsters"] is not None
            else -1.0
            for row in rows
        ),
    )


def aggregate(plan, records):
    reports = []
    for candidate in plan["candidates"]:
        rows = [
            records.get(f"{index:04}-{candidate['name']}") for index in range(len(plan["cases"]))
        ]
        episodes = [row["episode"] for row in rows if row and "episode" in row]
        complete = len(episodes) == len(rows)
        summary = summarise(episodes) if episodes else {"eligible": False}
        summary.pop("episodes", None)
        if not complete:
            summary["selection_score"] = -1.0
        reports.append(
            candidate
            | summary
            | {
                "evaluation_id": plan["evaluation_id"],
                "complete": complete,
                "eligible": complete and summary["eligible"],
                "episodes": episodes,
                "errors": [row["error"] for row in rows if row and "error" in row],
            }
        )
    reference = reports[0]
    if reference["eligible"]:
        for report in reports[1:]:
            if report["eligible"]:
                report["reference_comparison"] = {"name": reference["name"]} | compare_evaluations(
                    report, reference
                )
    return {
        "complete": all(report["complete"] for report in reports),
        "eligible": all(report["eligible"] for report in reports),
        "certifying": False,
        "promoted": False,
        "trials": reports,
    }


def run(
    study_path,
    output,
    minutes=25,
    per_character=4,
    workers=0,
    auto_calibrate=False,
    policy_modes=("deterministic",),
    split="test",
):
    if not 0 < minutes <= 30:
        raise ValueError("Use a budget between zero and 30 minutes.")
    started = time.monotonic()
    deadline = min(started + minutes * 60, float(os.environ.get("AI4STS2_DEADLINE", "inf"))) - 10
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    resources = budget().report(workers)
    study = json.loads(Path(study_path).read_text())
    scope = study["build"].get("scope", "run")
    ascension = study["build"].get("ascension", 10)
    build = fingerprint(scope, ascension)
    models = candidates(study, build, policy_modes)
    manifest = output / "plan.json"
    previous = json.loads(manifest.read_text()) if manifest.exists() else None
    if split not in {"validation", "test"}:
        raise ValueError("Use validation or test evaluation seeds.")
    seed = (
        previous["cases"][0]["seed_index"]
        if previous
        else 0
        if split == "validation"
        else secrets.randbits(60)
    )
    plan = evaluation_plan(scope, seed, split, 4096, per_character, ascension) | {
        "build": build,
        "candidates": models,
        "evaluator": digest(__file__),
    }
    if previous is not None and previous != plan:
        raise ValueError("The saved evaluation plan does not match this request.")
    write_json(manifest, plan)
    identity = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    records = {}
    pending = []
    for index, case in enumerate(plan["cases"]):
        for candidate in models:
            key = f"{index:04}-{candidate['name']}"
            path = output / "episodes" / f"{key}.json"
            if path.exists():
                record = json.loads(path.read_text())
                if (
                    record["plan"] != identity
                    or record["case"] != case
                    or record["candidate"] != candidate["name"]
                ):
                    raise ValueError("Cached episode does not match the evaluation plan.")
                records[key] = record
                if "episode" not in record and "error" not in record:
                    if "resume" not in record and not record.get("pending"):
                        raise ValueError("Cached episode has no result or continuation.")
                    pending.append((key, case, candidate, path))
            else:
                pending.append((key, case, candidate, path))
    execution = executable = None
    if pending and time.monotonic() < deadline:
        execution = selected_execution(scope, ascension)
        if execution is None and auto_calibrate:
            if deadline - time.monotonic() < 30:
                deadline = time.monotonic()
            else:
                calibrate(min(5, (deadline - time.monotonic()) / 60), scope, ascension)
                execution = selected_execution(scope, ascension)
        if execution is None and time.monotonic() < deadline:
            raise ValueError("A matching execution calibration is required.")
        if execution is not None:
            executable = prepare_game()
    local = threading.local()
    environments = []
    lock = threading.Lock()
    stopped = threading.Event()
    torch.set_num_threads(1)

    def perform(job):
        key, case, candidate, path = job
        if time.monotonic() >= deadline or stopped.is_set():
            return None
        record = records.get(
            key, {"plan": identity, "case": case, "candidate": candidate["name"]}
        ) | {"pending": True}
        write_json(path, record)

        def checkpoint(snapshot):
            record["resume"] = snapshot
            write_json(path, record)

        try:
            if not hasattr(local, "models"):
                local.models = {}
            if candidate["name"] not in local.models:
                local.models[candidate["name"]] = MaskablePPO.load(
                    Path(candidate["checkpoint"]) / "policy.zip", device="cpu"
                )
            if getattr(local, "environment", None) is None:
                local.environment = Sts2Env(
                    executable,
                    execution=execution,
                    scope=scope,
                    max_steps=4096,
                    ascension=ascension,
                )
                with lock:
                    environments.append(local.environment)
            model = local.models[candidate["name"]]
            local.environment.set_encoding(
                getattr(model.policy, "encoding", "hash") if model is not None else "hash"
            )
            record["episode"] = episode(
                model,
                local.environment,
                case,
                deadline,
                record.get("resume"),
                checkpoint,
                stopped.is_set,
                candidate.get("policy_mode", "deterministic") == "deterministic",
                split,
            )
            record.pop("resume", None)
            record.pop("pending", None)
        except EvaluationPaused:
            pass
        except Exception as error:
            if getattr(local, "environment", None) is not None:
                local.environment.close()
                local.environment = None
            if not isinstance(error, WorkerFailure) or time.monotonic() < deadline:
                record["error"] = f"{type(error).__name__}: {error}"
        write_json(path, record)
        with lock:
            records[key] = record
        return key, record

    def save():
        with lock:
            snapshot = records.copy()
        report = aggregate(plan, snapshot) | {
            "build": build,
            "plan": str(manifest.resolve()),
            "resources": resources,
            "seconds": time.monotonic() - started,
        }
        write_json(output / "summary.json", report)
        return report

    try:
        with ThreadPoolExecutor(max_workers=resources["concurrent_trials"]) as pool:
            try:
                for result in pool.map(perform, pending):
                    if result is None:
                        continue
                    key, record = result
                    save()
                    with lock:
                        completed = sum("episode" in value for value in records.values())
                    print(
                        json.dumps(
                            {
                                "completed": completed,
                                "total": len(plan["cases"]) * len(models),
                                "candidate": record["candidate"],
                                "floor": record.get("episode", {}).get("floor"),
                                "task_success": record.get("episode", {}).get("task_success"),
                                "error": record.get("error"),
                            }
                        ),
                        flush=True,
                    )
            except KeyboardInterrupt:
                stopped.set()
    finally:
        for environment in environments:
            environment.close()
        report = save()
    return report


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--study", required=True)
    parser.add_argument("--output", default=str(ROOT / "artifacts/validation/holdout"))
    parser.add_argument("--minutes", type=float, default=25)
    parser.add_argument("--per-character", type=int, default=4)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--split", choices=("validation", "test"), default="test")
    parser.add_argument(
        "--policy-modes",
        nargs="+",
        choices=("deterministic", "sampled"),
        default=["deterministic"],
    )
    args = parser.parse_args()
    result = run(
        args.study,
        args.output,
        args.minutes,
        args.per_character,
        args.workers,
        policy_modes=args.policy_modes,
        split=args.split,
    )
    print(json.dumps({"complete": result["complete"], "eligible": result["eligible"]}))
    if not result["eligible"]:
        raise SystemExit(1)
