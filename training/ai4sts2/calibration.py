import collections
import json
import os
import platform
import random
import time
from pathlib import Path

from filelock import FileLock

from ai4sts2.environment import (
    CHARACTERS,
    fingerprint,
    native_scope,
    probe_action,
    seed_string,
    validate_ascension,
    write_json,
)
from ai4sts2.execution import CANDIDATES, REFERENCE, Execution
from ai4sts2.game import ROOT, OfficialGame, prepare_game
from ai4sts2.resources import capacity


def hardware():
    cpu = platform.processor()
    cpuinfo = Path("/proc/cpuinfo")
    if cpuinfo.is_file():
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in cpuinfo.read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu,
        )
    return {
        "machine": platform.machine(),
        "model": cpu,
        "cpus": os.cpu_count(),
        "cpu_capacity": capacity(available=False)[0],
        "system": platform.system(),
    }


def signature(state):
    if not state.get("audit"):
        raise ValueError("Native audit fingerprint is required.")
    result = {
        key: state[key] for key in ("observation", "actions", "terminated", "victory", "audit")
    }
    for field in ("act1_elite_wins", "act1_monster_wins", "act1_monster_health"):
        if field in state:
            result[field] = state[field]
    return result


def compare(expected, actual, step, diagnostic=None):
    actual = signature(actual)
    for key in expected:
        if expected[key] != actual[key]:
            if diagnostic is not None:
                write_json(
                    diagnostic, {"step": step, "field": key, "expected": expected, "actual": actual}
                )
            raise ValueError(f"Divergence at step {step}: {key}")


def trace(
    game, character, seed, policy, deadline, expected=None, scope="first_combat", ascension=10
):
    validate_ascension(ascension)
    game.timeout = min(75, max(0.01, deadline - time.monotonic()))
    state = game.request(
        "reset",
        {
            "character": character,
            "seed": seed_string("calibration", seed),
            "scope": scope,
            "ascension": ascension,
        },
    )
    if state["observation"].get("ascension") != ascension:
        raise ValueError("Official game ascension does not match calibration.")
    rng = random.Random(seed)
    frames = []
    coverage = collections.Counter()
    limit = 1024 if scope == "run" else 128
    for index in range(limit + 1):
        if time.monotonic() >= deadline:
            raise TimeoutError("Calibration budget exhausted.")
        current = signature(state)
        if expected is not None:
            if index >= len(expected):
                raise ValueError("The candidate did not terminate with the reference.")
            compare(
                expected[index]["state"],
                state,
                index,
                ROOT / "artifacts/validation/divergence.json",
            )
        if state["terminated"] or index == limit:
            frames.append({"state": current, "action": None})
            break
        actions = state["actions"]
        if expected is not None:
            action = expected[index]["action"]
        elif policy == "random":
            action = rng.randrange(len(actions))
        elif policy == "loss":
            action = next(
                (i for i, a in enumerate(actions) if a["kind"] == "end_turn"), probe_action(actions)
            )
        elif scope == "run":
            from ai4sts2.runcheck import probe_decision

            action = probe_decision(state, rng, coverage)
        else:
            action = probe_action(actions)
        chosen = actions[action]
        if chosen["kind"] == "map":
            coverage["room:" + chosen["room"]] += 1
        if chosen["kind"] == "buy":
            coverage["buy:" + chosen["type"]] += 1
        frames.append({"state": current, "action": action})
        game.timeout = min(75, max(0.01, deadline - time.monotonic()))
        state = game.request("step", {"revision": state["revision"], "action": action})
    if expected is not None and len(frames) != len(expected):
        raise ValueError("The candidate terminated before the reference.")
    return frames


def measure(
    executable, execution, cases, deadline, references=None, scope="first_combat", ascension=10
):
    started = time.monotonic()
    traces = []
    remaining = deadline - started
    if remaining < 10:
        raise TimeoutError("Calibration budget exhausted.")
    with OfficialGame(
        executable, timeout=min(75, remaining), execution=execution, audit=True
    ) as game:
        startup = time.monotonic() - started
        game.drain_measurements()
        measured = time.monotonic()
        for index, (character, seed, policy) in enumerate(cases):
            traces.append(
                trace(
                    game,
                    character,
                    seed,
                    policy,
                    deadline,
                    None if references is None else references[index],
                    scope,
                    ascension,
                )
            )
        elapsed = time.monotonic() - measured
        metrics = game.drain_measurements()
    steps = sum(len(frames) - 1 for frames in traces)
    return {
        "execution": execution.to_dict(),
        "valid": True,
        "seconds": elapsed,
        "startup_seconds": startup,
        "steps": steps,
        "steps_per_second": steps / elapsed,
        "metrics": metrics,
        "completed": sum(frames[-1]["state"]["terminated"] for frames in traces),
    }, traces


def select_result(results):
    valid = [r for r in results if r.get("valid") and "runtime_failure" not in r]
    if not valid:
        raise ValueError("No validated execution configuration.")
    return min(valid, key=lambda result: result["seconds"])


def runtime_path(ascension):
    return ROOT / f"artifacts/runtime-a{validate_ascension(ascension)}.json"


def calibrate(minutes=5, scope="first_combat", ascension=10):
    if not 0 < minutes <= 30:
        raise ValueError("Use a budget between zero and 30 minutes.")
    scope = native_scope(scope)
    deadline = time.monotonic() + minutes * 60
    build = fingerprint(scope, ascension)
    executable = prepare_game()
    cases = [(hero, index, "probe") for index, hero in enumerate(CHARACTERS)]
    if scope == "first_combat":
        cases += [("IRONCLAD", 11, "loss"), ("REGENT", 21, "random"), ("SILENT", 22, "random")]
    baseline, references = measure(
        executable, REFERENCE, cases, deadline, scope=scope, ascension=ascension
    )
    results = [baseline]
    write_json(
        ROOT / f"artifacts/validation/reference-traces-a{ascension}.json",
        {"ascension": ascension, "cases": cases, "traces": references},
    )
    print(json.dumps({"calibration": baseline}), flush=True)
    for execution in CANDIDATES:
        if deadline - time.monotonic() < 10:
            break
        try:
            result, _ = measure(
                executable, execution, cases, deadline, references, scope, ascension
            )
        except (RuntimeError, ValueError, TimeoutError) as error:
            result = {"execution": execution.to_dict(), "valid": False, "error": str(error)}
        results.append(result)
        print(json.dumps({"calibration": result}), flush=True)
    selected = select_result(results)
    report = {
        "build": build,
        "hardware": hardware(),
        "scope": scope,
        "selected": selected["execution"],
        "results": results,
        "speedup": baseline["seconds"] / selected["seconds"],
        "coverage": {
            "episodes": len(cases),
            "steps": baseline["steps"],
            "completed": baseline["completed"],
            "full_run": scope == "run" and baseline["completed"] == len(cases),
        },
    }
    write_json(runtime_path(ascension), report)
    return report


def cached_report(scope="first_combat", ascension=10):
    scope = native_scope(scope)
    path = runtime_path(ascension)
    if not path.is_file():
        return None
    report = json.loads(path.read_text())
    if (
        report.get("build") != fingerprint(scope, ascension)
        or report.get("hardware") != hardware()
        or report.get("scope") != scope
    ):
        return None
    return report


def selected_execution(scope="first_combat", ascension=10):
    report = cached_report(scope, ascension)
    if report is None:
        return None
    return Execution(**select_result(report["results"])["execution"])


def quarantine_execution(execution, error, scope="first_combat", ascension=10):
    with FileLock(runtime_path(ascension).with_suffix(".lock"), timeout=30):
        return quarantine_locked(execution, error, scope, ascension)


def quarantine_locked(execution, error, scope, ascension):
    report = cached_report(scope, ascension)
    if report is None:
        raise RuntimeError(
            "No compatible execution calibration is available for recovery."
        ) from error
    matches = [r for r in report["results"] if r["execution"] == execution.to_dict()]
    if not matches:
        raise RuntimeError("The failed execution configuration was not calibrated.") from error
    for result in matches:
        result["runtime_failure"] = str(error)[:2048]
    try:
        selected = select_result(report["results"])["execution"]
    except ValueError:
        selected = None
    report["selected"] = selected
    write_json(runtime_path(ascension), report)
    if selected is None:
        raise RuntimeError(
            "All calibrated execution configurations failed; training stopped."
        ) from error
    return Execution(**selected)
