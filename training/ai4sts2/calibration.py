import collections
import json
import os
import platform
import random
import time
from pathlib import Path

from ai4sts2.environment import CHARACTERS, fingerprint, probe_action, seed_string, write_json
from ai4sts2.execution import CANDIDATES, REFERENCE, Execution
from ai4sts2.game import ROOT, OfficialGame, prepare_game


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
    quotas = []
    cgroup = Path("/proc/self/cgroup")
    if cgroup.is_file():
        relative = next(
            (line[3:] for line in cgroup.read_text().splitlines() if line.startswith("0::")), ""
        )
        root = Path("/sys/fs/cgroup")
        directory = root / relative.lstrip("/")
        for parent in (directory, *directory.parents):
            if parent == root.parent:
                break
            limit = parent / "cpu.max"
            if limit.is_file():
                quota, period = limit.read_text().split()
                if quota != "max":
                    quotas.append(int(quota) / int(period))
    return {
        "machine": platform.machine(),
        "model": cpu,
        "cpus": os.cpu_count(),
        "cpu_quota": min(quotas, default=None),
        "system": platform.system(),
    }


def signature(state):
    if not state.get("audit"):
        raise ValueError("Native audit fingerprint is required.")
    return {key: state[key] for key in ("observation", "actions", "terminated", "victory", "audit")}


def compare(expected, actual, step, diagnostic=None):
    actual = signature(actual)
    for key in expected:
        if expected[key] != actual[key]:
            if diagnostic is not None:
                write_json(
                    diagnostic, {"step": step, "field": key, "expected": expected, "actual": actual}
                )
            raise ValueError(f"Divergence at step {step}: {key}")


def trace(game, character, seed, policy, deadline, expected=None, scope="first_combat"):
    game.timeout = min(75, max(0.01, deadline - time.monotonic()))
    state = game.request(
        "reset",
        {"character": character, "seed": seed_string("calibration", seed), "scope": scope},
    )
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


def measure(executable, execution, cases, deadline, references=None, scope="first_combat"):
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


def calibrate(minutes=5, scope="first_combat"):
    if not 0 < minutes <= 30:
        raise ValueError("Use a budget between zero and 30 minutes.")
    if scope not in {"run", "first_combat"}:
        raise ValueError("Unknown episode scope.")
    deadline = time.monotonic() + minutes * 60
    build = fingerprint(scope)
    executable = prepare_game()
    cases = [(hero, index, "probe") for index, hero in enumerate(CHARACTERS)]
    if scope == "first_combat":
        cases += [("IRONCLAD", 11, "loss"), ("REGENT", 21, "random"), ("SILENT", 22, "random")]
    baseline, references = measure(executable, REFERENCE, cases, deadline, scope=scope)
    results = [baseline]
    write_json(
        ROOT / "artifacts/validation/reference-traces.json", {"cases": cases, "traces": references}
    )
    print(json.dumps({"calibration": baseline}), flush=True)
    for execution in CANDIDATES:
        if deadline - time.monotonic() < 10:
            break
        try:
            result, _ = measure(executable, execution, cases, deadline, references, scope)
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
    write_json(ROOT / "artifacts/runtime.json", report)
    return report


def cached_report(scope="first_combat"):
    path = ROOT / "artifacts/runtime.json"
    if not path.is_file():
        return None
    report = json.loads(path.read_text())
    if (
        report.get("build") != fingerprint(scope)
        or report.get("hardware") != hardware()
        or report.get("scope") != scope
    ):
        return None
    return report


def selected_execution(scope="first_combat"):
    report = cached_report(scope)
    if report is None:
        return None
    return Execution(**select_result(report["results"])["execution"])


def quarantine_execution(execution, error, scope="first_combat"):
    report = cached_report(scope)
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
    write_json(ROOT / "artifacts/runtime.json", report)
    if selected is None:
        raise RuntimeError(
            "All calibrated execution configurations failed; training stopped."
        ) from error
    return Execution(**selected)
