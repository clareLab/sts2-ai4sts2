import copy
import json
import random

import numpy as np
import pytest
import torch
from ai4sts2.environment import Sts2Env, evaluate, evaluation_plan
from ai4sts2.metrics import summarise
from ai4sts2.signals import TrainingSignals
from test_checkpoint import member
from test_environment import FakeWorker


class ActWorker(FakeWorker):
    def request(self, method, parameters):
        result = super().request(method, parameters)
        result["observation"] |= {
            "floor": (0, 8, 17, 18)[self.count],
            "act": int(self.count == 3),
            "room": "Boss" if self.count == 2 else "Map",
        }
        return result


def test_act1_uses_official_run_and_ends_only_at_the_next_act():
    env = Sts2Env(scope="act1", worker_factory=ActWorker, max_steps=3)
    env.reset(seed=2)
    assert env.journal["parameters"]["scope"] == "run"
    for _ in range(2):
        _, reward, terminated, truncated, info = env.step(0)
        assert reward == 0 and not terminated and not truncated
        assert not info["task_success"]
    _, reward, terminated, truncated, info = env.step(0)
    assert reward == 1 and terminated and not truncated
    assert info["task_success"] and not info["victory"]
    assert info["scope"] == "act1" and info["act"] == 1
    calls = len(env.game.calls)
    with pytest.raises(ValueError, match="Illegal"):
        env.step(0)
    assert len(env.game.calls) == calls
    report = summarise(env.drain_episodes())
    assert report["eligible"] and report["selection_score"] == 1
    assert report["win_rate"] == 0 and report["task_success_rate"] == 1
    assert report["boss_reach_rate"] == 1
    assert report["mean_death_floor"] is None
    env.close()


@pytest.mark.parametrize("terminal", [False, True])
def test_boss_death_and_timeout_are_distinct_from_success(terminal):
    class FailedWorker(ActWorker):
        def request(self, method, parameters):
            result = super().request(method, parameters)
            if self.count == 2 and terminal:
                result |= {"terminated": True, "victory": False, "actions": []}
                result["observation"]["player"]["hp"] = 0
            return result

    env = Sts2Env(scope="act1", worker_factory=FailedWorker, max_steps=2)
    env.reset()
    env.step(0)
    _, reward, terminated, truncated, info = env.step(0)
    assert terminated == terminal and truncated != terminal
    assert reward == (-1 if terminal else 0)
    assert not info["task_success"] and not info["victory"]
    report = summarise(env.drain_episodes())
    assert report["eligible"] == terminal
    assert report["selection_score"] == (0 if terminal else -1)
    assert report["boss_reach_rate"] == 1
    env.close()


def test_act1_checkpoint_resume_retains_boundary_and_rejects_scope_change():
    first = Sts2Env(scope="act1", worker_factory=ActWorker)
    second = Sts2Env(scope="act1", worker_factory=ActWorker)
    first.reset(seed=17)
    first.step(0)
    first.step(0)
    saved = first.snapshot()
    second.restore(saved)
    assert second.snapshot() == saved
    assert first.step(0)[1:] == second.step(0)[1:]
    second.restore(first.snapshot())
    with pytest.raises(ValueError, match="Illegal"):
        second.step(0)
    with pytest.raises(ValueError, match="scope"):
        second.restore(saved | {"scope": "run"})
    first.close()
    second.close()


def test_act1_rejects_floor_curriculum_before_opening_worker():
    signals = TrainingSignals(1, {"floor_goals": [8]})
    with pytest.raises(ValueError, match="curricula"):
        Sts2Env(
            scope="act1",
            signals=signals,
            worker_factory=lambda *_: pytest.fail("Worker launched"),
        )


def test_act1_evaluation_is_separate_and_reports_character_success():
    env = Sts2Env(scope="act1", worker_factory=ActWorker)
    report = evaluate(None, env)
    assert report["eligible"] and report["complete"]
    assert report["task_successes"] == 5 and report["wins"] == 0
    assert all(row["task_success_rate"] == 1 for row in report["characters"].values())
    assert evaluation_plan("act1")["evaluation_id"] != evaluation_plan("run")["evaluation_id"]
    env.close()


def test_metrics_reject_mixed_tasks_and_unearned_success():
    row = {
        "scope": "act1",
        "floor": 18,
        "act": 1,
        "character": "REGENT",
        "victory": False,
        "task_success": True,
        "truncated": False,
    }
    with pytest.raises(ValueError, match="scopes"):
        summarise([row, row | {"scope": "run"}])
    with pytest.raises(ValueError, match="Act 2"):
        summarise([row | {"act": 0}])
    with pytest.raises(ValueError, match="truncated"):
        summarise([row | {"truncated": True}])


def test_policy_transfer_keeps_weights_and_fresh_task_state(monkeypatch, tmp_path):
    donor = member(monkeypatch, policy="shared")
    donor.build |= {"ascension": 10}
    donor.model.learn(64)
    donor.save_checkpoint(tmp_path)
    receiver = member(monkeypatch, policy="shared")
    receiver.build |= {"scope": "act1", "trainer": "new", "mod": "telemetry-update", "ascension": 0}
    rng = random.getstate(), np.random.get_state(), torch.get_rng_state().clone()
    environment = copy.deepcopy(receiver.environment.snapshot())
    receiver.initialise_policy(tmp_path)
    with pytest.raises(ValueError, match="do not match"):
        receiver.load_checkpoint(tmp_path)
    assert receiver.model.num_timesteps == 0
    assert receiver.model.policy.optimizer.state_dict()["state"] == {}
    assert receiver.environment.snapshot() == environment
    assert receiver.signals.goal() is None and receiver.evaluation is None
    assert random.getstate() == rng[0]
    np.testing.assert_equal(np.random.get_state(), rng[1])
    assert torch.equal(torch.get_rng_state(), rng[2])
    for key, value in donor.model.policy.state_dict().items():
        assert torch.equal(value, receiver.model.policy.state_dict()[key])
    other = member(monkeypatch)
    with pytest.raises(ValueError, match="architecture"):
        other.initialise_policy(tmp_path)
    other.cleanup()
    build = json.loads((tmp_path / "build.json").read_text())
    for field in ("schema", "game", "dependencies"):
        (tmp_path / "build.json").write_text(json.dumps(build | {field: "changed"}))
        with pytest.raises(ValueError, match="matching"):
            receiver.initialise_policy(tmp_path)
    donor.cleanup()
    receiver.cleanup()


@pytest.mark.parametrize("ascension", [0, 10])
def test_holdout_runner_uses_the_study_task(monkeypatch, tmp_path, ascension):
    from ai4sts2.execution import Execution
    from ai4sts2.resources import GIB, Budget
    from test_evaluate import holdout

    build = {"game": "test", "scope": "act1", "ascension": ascension}
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    (checkpoint / "build.json").write_text(json.dumps(build))
    (checkpoint / "policy.zip").write_bytes(b"model")
    study = tmp_path / "study.json"
    study.write_text(
        json.dumps(
            {
                "complete": True,
                "build": build,
                "trials": [{"checkpoint": str(checkpoint), "variant": "control", "seed": 1}],
            }
        )
    )
    monkeypatch.setattr(holdout, "fingerprint", lambda scope, *_: build | {"scope": scope})
    monkeypatch.setattr(holdout, "budget", lambda: Budget(2, 8 * GIB))
    monkeypatch.setattr(holdout, "prepare_game", lambda: None)
    monkeypatch.setattr(holdout, "selected_execution", lambda *_: Execution())
    monkeypatch.setattr(holdout.MaskablePPO, "load", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(
        holdout, "Sts2Env", lambda *args, **kwargs: Sts2Env(worker_factory=ActWorker, **kwargs)
    )
    report = holdout.run(study, tmp_path / "holdout", per_character=1)
    assert report["complete"] and report["eligible"]
    for trial in report["trials"]:
        assert trial["ascension"] == ascension
        assert trial["scope"] == "act1" and trial["task_successes"] == 5
        assert trial["wins"] == 0


def test_act1_shares_native_calibration_without_sharing_task_identity(monkeypatch, tmp_path):
    import ai4sts2.calibration as calibration
    from ai4sts2.execution import REFERENCE

    scopes = []

    def fingerprint(scope, ascension=10):
        scopes.append(scope)
        return {"scope": scope}

    runtime = tmp_path / "artifacts/runtime-a10.json"
    runtime.parent.mkdir()
    runtime.write_text(
        json.dumps(
            {
                "build": {"scope": "run"},
                "scope": "run",
                "hardware": {},
                "results": [{"execution": REFERENCE.to_dict(), "valid": True, "seconds": 1}],
            }
        )
    )
    monkeypatch.setattr(calibration, "ROOT", tmp_path)
    monkeypatch.setattr(calibration, "fingerprint", fingerprint)
    monkeypatch.setattr(calibration, "hardware", lambda: {})
    assert calibration.selected_execution("act1") == REFERENCE
    assert scopes == ["run"]
