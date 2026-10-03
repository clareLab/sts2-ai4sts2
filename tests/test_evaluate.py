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
