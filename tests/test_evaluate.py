import copy
import importlib.util
import json
import time
from pathlib import Path

import pytest
from ai4sts2.environment import Sts2Env, evaluate, evaluation_plan
from ai4sts2.execution import Execution
from ai4sts2.resources import GIB, Budget
from test_environment import FakeWorker

spec = importlib.util.spec_from_file_location(
    "holdout", Path(__file__).resolve().parents[1] / "scripts/evaluate.py"
)
holdout = importlib.util.module_from_spec(spec)
spec.loader.exec_module(holdout)


@pytest.fixture
def suite(monkeypatch, tmp_path):
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "build.json").write_text('{"game": "test"}')
    (checkpoint / "policy.zip").write_bytes(b"test model")
    study = tmp_path / "study.json"
    study.write_text(
        json.dumps(
            {
                "complete": True,
                "build": {"game": "test"},
                "trials": [{"checkpoint": str(checkpoint), "variant": "control", "seed": 7}],
            }
        )
    )
    environments = []

    def open_environment(*args, **kwargs):
        environment = Sts2Env(worker_factory=FakeWorker, **kwargs)
        environments.append(environment)
        return environment

    monkeypatch.setattr(holdout, "fingerprint", lambda *_: {"game": "test"})
    monkeypatch.setattr(holdout, "budget", lambda: Budget(2, 8 * GIB))
    monkeypatch.setattr(holdout, "prepare_game", lambda: None)
    monkeypatch.setattr(holdout, "selected_execution", lambda *_: Execution())
    monkeypatch.setattr(holdout, "Sts2Env", open_environment)
    monkeypatch.setattr(holdout.MaskablePPO, "load", lambda *_args, **_kwargs: None)
    return study, tmp_path / "results", environments


def test_single_case_evaluation_matches_existing_evaluator():
    first = Sts2Env(worker_factory=FakeWorker, scope="run", max_steps=4096)
    second = Sts2Env(worker_factory=FakeWorker, scope="run", max_steps=4096)
    expected = evaluate(None, first, seed=500, split="test", max_steps=4096)
    actual = [
        holdout.episode(None, second, case, time.monotonic() + 10) for case in expected["cases"]
    ]
    assert actual == expected["episodes"]
    assert first.game.calls == second.game.calls
    assert not second.completed


def test_policy_uses_deterministic_legal_actions():
    class Policy:
        def predict(self, observation, deterministic, action_masks):
            assert deterministic
            return int(action_masks.sum()) - 1, None

    environment = Sts2Env(worker_factory=FakeWorker, scope="run")
    case = evaluation_plan("run", split="test")["cases"][0]
    result = holdout.episode(Policy(), environment, case, time.monotonic() + 10)
    assert result["victory"] and not result["truncated"]
    assert [
        parameters["action"] for method, parameters in environment.game.calls if method == "step"
    ] == [1] * 4


def test_parallel_evaluation_reuses_completed_cases_and_closes_workers(suite):
    study, output, environments = suite
    result = holdout.run(study, output, per_character=1)
    assert result["complete"] and result["eligible"]
    assert not result["certifying"] and not result["promoted"]
    assert 1 <= len(environments) <= 2
    assert all(environment.game.closed for environment in environments)
    assert len(list((output / "episodes").glob("*.json"))) == 10
    model = result["trials"][1]
    assert model["random_baseline_comparison"]["mean_floor_difference"] == 0
    count = len(environments)
    assert holdout.run(study, output, per_character=1)["trials"] == result["trials"]
    assert len(environments) == count
    plan = json.loads((output / "plan.json").read_text())
    assert plan["split"] == "test"
    assert not {case["seed"] for case in plan["cases"]} & {
        case["seed"] for case in evaluation_plan("run")["cases"]
    }


def test_changed_model_or_case_counts_cannot_reuse_cached_results(suite):
    study, output, _ = suite
    holdout.run(study, output, per_character=1)
    with pytest.raises(ValueError, match="plan"):
        holdout.run(study, output, per_character=2)
    (study.parent / "checkpoint/policy.zip").write_bytes(b"changed model")
    with pytest.raises(ValueError, match="plan"):
        holdout.run(study, output, per_character=1)


def test_failed_attempt_is_recorded_and_never_automatically_retried(suite, monkeypatch):
    study, output, environments = suite
    original = holdout.episode
    calls = []

    def fail_once(*args):
        calls.append(args[2])
        if len(calls) == 1:
            raise RuntimeError("Injected worker failure")
        return original(*args)

    monkeypatch.setattr(holdout, "episode", fail_once)
    result = holdout.run(study, output, per_character=1, workers=1)
    assert not result["complete"] and not result["eligible"]
    assert result["trials"][0]["errors"] == ["RuntimeError: Injected worker failure"]
    assert result["trials"][0]["selection_score"] == -1
    assert all(environment.game.closed for environment in environments)
    assert len(calls) == 10
    assert holdout.run(study, output, per_character=1)["trials"] == result["trials"]
    assert len(calls) == 10


def test_expired_budget_starts_no_episodes(suite, monkeypatch):
    study, output, environments = suite
    monkeypatch.setenv("AI4STS2_DEADLINE", str(time.monotonic() - 1))
    result = holdout.run(study, output, per_character=1)
    assert not result["complete"] and not result["eligible"]
    assert environments == []


def test_incompatible_checkpoint_fails_before_starting_game(suite):
    study, output, environments = suite
    (study.parent / "checkpoint/build.json").write_text('{"game": "changed"}')
    with pytest.raises(ValueError, match="Checkpoint build"):
        holdout.run(study, output)
    assert environments == []


def test_corrupt_cache_fails_before_starting_game(suite):
    study, output, environments = suite
    holdout.run(study, output, per_character=1)
    path = next((output / "episodes").glob("*.json"))
    record = json.loads(path.read_text())
    record["case"]["seed"] = "changed"
    path.write_text(json.dumps(record))
    count = len(environments)
    with pytest.raises(ValueError, match="Cached episode"):
        holdout.run(study, output, per_character=1)
    assert len(environments) == count


def test_evaluation_switches_encoding_without_restarting_game(suite, monkeypatch):
    from types import SimpleNamespace

    study, output, environments = suite
    observed = []

    class Policy:
        policy = SimpleNamespace(encoding="tree")

        def predict(self, observation, deterministic, action_masks):
            observed.append(observation["state"].ndim)
            assert observation["state"].ndim == 2
            return int(action_masks.sum()) - 1, None

    monkeypatch.setattr(holdout.MaskablePPO, "load", lambda *_args, **_kwargs: Policy())
    result = holdout.run(study, output, per_character=1, workers=1)
    assert result["complete"] and result["eligible"]
    assert len(observed) == 20
    assert len(environments) == 1
    assert environments[0].game.closed


def test_interrupted_random_episode_resumes_identical_actions_and_result():
    case = evaluation_plan("run", split="test")["cases"][0]
    direct = Sts2Env(worker_factory=FakeWorker, scope="run")
    expected = holdout.episode(None, direct, case, time.monotonic() + 10)
    interrupted = Sts2Env(worker_factory=FakeWorker, scope="run")
    snapshots = []

    def checkpoint(snapshot):
        snapshots.append(copy.deepcopy(snapshot))
        if len(snapshot["environment"]["journal"]["actions"]) == 2:
            raise holdout.EvaluationPaused()

    with pytest.raises(holdout.EvaluationPaused):
        holdout.episode(None, interrupted, case, time.monotonic() + 10, checkpoint=checkpoint)
    resumed = Sts2Env(worker_factory=FakeWorker, scope="run")
    assert holdout.episode(None, resumed, case, time.monotonic() + 10, snapshots[-1]) == expected
    assert resumed.game.calls == direct.game.calls
    assert interrupted.game.calls == direct.game.calls[:3]


def test_paused_cases_are_resumed_but_completed_cases_are_not_repeated(suite, monkeypatch):
    study, output, environments = suite
    original = holdout.episode
    paused = []

    def pause_once(model, environment, case, deadline, resume, checkpoint, cancelled):
        if paused:
            return original(model, environment, case, deadline, resume, checkpoint, cancelled)

        def stop(snapshot):
            checkpoint(snapshot)
            if len(snapshot["environment"]["journal"]["actions"]) == 2:
                paused.append(case)
                raise holdout.EvaluationPaused()

        return original(model, environment, case, deadline, resume, stop, cancelled)

    monkeypatch.setattr(holdout, "episode", pause_once)
    first = holdout.run(study, output, per_character=1, workers=1)
    assert not first["complete"]
    assert sum(len(trial["episodes"]) for trial in first["trials"]) == 9
    assert all(not trial["errors"] for trial in first["trials"])
    saved = json.loads((output / "episodes/0000-random.json").read_text())
    assert len(saved["resume"]["environment"]["journal"]["actions"]) == 2
    files = {path: path.read_bytes() for path in (output / "episodes").glob("*.json")}
    second = holdout.run(study, output, per_character=1, workers=1)
    assert second["eligible"] and len(environments) == 2
    assert all(environment.game.closed for environment in environments)
    assert len(environments[-1].game.calls) == 5
    assert sum(path.read_bytes() != data for path, data in files.items()) == 1
    final = json.loads((output / "episodes/0000-random.json").read_text())
    assert "resume" not in final and "pending" not in final


def test_cached_evaluation_does_not_calibrate_or_prepare_game(suite, monkeypatch):
    study, output, _ = suite
    expected = holdout.run(study, output, per_character=1)
    for name in ("selected_execution", "calibrate", "prepare_game", "Sts2Env"):
        monkeypatch.setattr(holdout, name, lambda *_: pytest.fail("Cached work was repeated"))
    actual = holdout.run(study, output, per_character=1, auto_calibrate=True)
    assert actual["trials"] == expected["trials"]


def test_resume_rejects_changed_case_or_divergent_history_before_new_actions():
    case = evaluation_plan("run", split="test")["cases"][0]
    source = Sts2Env(worker_factory=FakeWorker, scope="run")
    source.reset(seed=case["seed_index"], options={"character": case["character"], "split": "test"})
    source.step(0)
    snapshot = {"environment": source.snapshot(), "action_rng": {}}
    target = Sts2Env(worker_factory=FakeWorker, scope="run")
    with pytest.raises(ValueError, match="different evaluation case"):
        holdout.episode(None, target, case | {"seed": "changed"}, time.monotonic() + 10, snapshot)
    assert target.game.calls == []
    snapshot["environment"]["journal"]["actions"][0]["digest"] = "changed"
    with pytest.raises(ValueError, match="replay diverged"):
        holdout.episode(None, target, case, time.monotonic() + 10, snapshot)
    assert len(target.game.calls) == 2


def test_cancellation_interrupts_replay_and_restores_request_handler():
    case = evaluation_plan("run", split="test")["cases"][0]
    source = Sts2Env(worker_factory=FakeWorker, scope="run")
    source.reset(seed=case["seed_index"], options={"character": case["character"], "split": "test"})
    for _ in range(3):
        source.step(0)
    target = Sts2Env(worker_factory=FakeWorker, scope="run")
    original = target.game.request
    with pytest.raises(holdout.EvaluationPaused):
        holdout.episode(
            None,
            target,
            case,
            time.monotonic() + 10,
            {"environment": source.snapshot()},
            cancelled=lambda: len(target.game.calls) == 2,
        )
    assert len(target.game.calls) == 2
    assert target.game.request == original
