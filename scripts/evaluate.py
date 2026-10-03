import argparse
import hashlib
import json
import os
import secrets
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import torch
from ai4sts2.calibration import selected_execution
from ai4sts2.environment import Sts2Env, evaluation_plan, fingerprint, write_json
from ai4sts2.game import ROOT, prepare_game
from ai4sts2.metrics import compare_evaluations, summarise
from ai4sts2.resources import budget
from sb3_contrib import MaskablePPO


def digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def candidates(study, build):
    if not study["complete"] or not study["trials"] or study["build"] != build:
        raise ValueError("A complete study with the current build is required.")
    result = [{"name": "random", "variant": "random", "checkpoint": None, "sha256": None}]
    for trial in study["trials"]:
        checkpoint = Path(trial["checkpoint"]).resolve()
        if json.loads((checkpoint / "build.json").read_text()) != build:
            raise ValueError("Checkpoint build does not match the evaluation build.")
        result.append(
            {
                "name": f"{trial['variant']}-{trial['seed']}",
                "variant": trial["variant"],
                "checkpoint": str(checkpoint),
                "sha256": digest(checkpoint / "policy.zip"),
            }
        )
    if len({item["name"] for item in result}) != len(result):
        raise ValueError("Candidate names must be unique.")
    return result


def episode(model, environment, case, deadline):
    def check_budget():
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Evaluation budget exhausted.")
        environment.game.timeout = min(75, remaining)

    check_budget()
    observation, _ = environment.reset(
        seed=case["seed_index"], options={"character": case["character"], "split": "test"}
    )
    rng = np.random.default_rng(case["action_seed"])
    terminated = truncated = False
    while not (terminated or truncated):
        check_budget()
        if model is None:
            action = int(rng.choice(np.flatnonzero(environment.action_masks())))
        else:
            action, _ = model.predict(
                observation, deterministic=True, action_masks=environment.action_masks()
            )
        observation, _, terminated, truncated, info = environment.step(action)
    environment.drain_episodes()
    return info | {
        "truncated": truncated,
        "trajectory_digest": hashlib.sha256(
            json.dumps(environment.journal, sort_keys=True).encode()
        ).hexdigest(),
    }


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
    baseline = reports[0]
    if baseline["eligible"]:
        for report in reports[1:]:
            if report["eligible"]:
                report["random_baseline_comparison"] = compare_evaluations(report, baseline)
    return {
        "complete": all(report["complete"] for report in reports),
        "eligible": all(report["eligible"] for report in reports),
        "certifying": False,
        "promoted": False,
        "trials": reports,
    }


def run(study_path, output, minutes=25, per_character=4, workers=0):
    if not 0 < minutes <= 30:
        raise ValueError("Use a budget between zero and 30 minutes.")
    started = time.monotonic()
    deadline = min(started + minutes * 60, float(os.environ.get("AI4STS2_DEADLINE", "inf"))) - 10
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    resources = budget().report(workers)
    build = fingerprint("run")
    models = candidates(json.loads(Path(study_path).read_text()), build)
    manifest = output / "plan.json"
    previous = json.loads(manifest.read_text()) if manifest.exists() else None
    seed = previous["cases"][0]["seed_index"] if previous else secrets.randbits(60)
    plan = evaluation_plan("run", seed, "test", 4096, per_character) | {
        "build": build,
        "candidates": models,
        "evaluator": digest(__file__),
    }
    if previous is not None and previous != plan:
        raise ValueError("The saved evaluation plan does not match this request.")
    write_json(manifest, plan)
    identity = hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    execution = selected_execution("run")
    if execution is None:
        raise ValueError("A matching execution calibration is required.")
    executable = prepare_game()
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
            else:
                pending.append((key, case, candidate, path))
    local = threading.local()
    environments = []
    lock = threading.Lock()
    torch.set_num_threads(1)

    def perform(job):
        key, case, candidate, path = job
        if time.monotonic() >= deadline:
            return None
        record = {"plan": identity, "case": case, "candidate": candidate["name"]}
        write_json(path, record | {"error": "Interrupted before a result was saved."})
        try:
            if not hasattr(local, "models"):
                local.models = {}
            if candidate["name"] not in local.models:
                local.models[candidate["name"]] = (
                    MaskablePPO.load(Path(candidate["checkpoint"]) / "policy.zip", device="cpu")
                    if candidate["checkpoint"]
                    else None
                )
            if getattr(local, "environment", None) is None:
                local.environment = Sts2Env(
                    executable, execution=execution, scope="run", max_steps=4096
                )
                with lock:
                    environments.append(local.environment)
            record["episode"] = episode(
                local.models[candidate["name"]], local.environment, case, deadline
            )
        except Exception as error:
            if getattr(local, "environment", None) is not None:
                local.environment.close()
                local.environment = None
            record["error"] = f"{type(error).__name__}: {error}"
        write_json(path, record)
        return key, record

    def save():
        report = aggregate(plan, records) | {
            "build": build,
            "plan": str(manifest.resolve()),
            "resources": resources,
            "seconds": time.monotonic() - started,
        }
        write_json(output / "summary.json", report)
        return report

    try:
        with ThreadPoolExecutor(max_workers=resources["concurrent_trials"]) as pool:
            for result in pool.map(perform, pending):
                if result is None:
                    continue
                key, record = result
                records[key] = record
                save()
                print(
                    json.dumps(
                        {
                            "completed": len(records),
                            "total": len(plan["cases"]) * len(models),
                            "candidate": record["candidate"],
                            "floor": record.get("episode", {}).get("floor"),
                            "error": record.get("error"),
                        }
                    ),
                    flush=True,
                )
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
    args = parser.parse_args()
    result = run(args.study, args.output, args.minutes, args.per_character, args.workers)
    print(json.dumps({"complete": result["complete"], "eligible": result["eligible"]}))
    if not result["eligible"]:
        raise SystemExit(1)
