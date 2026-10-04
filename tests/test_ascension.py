import copy
import json

import pytest
from ai4sts2.environment import Sts2Env, evaluate, evaluation_plan
from ai4sts2.metrics import summarise
from test_environment import FakeWorker


@pytest.mark.parametrize("ascension", [0, 5, 10])
def test_official_difficulty_is_passed_reported_and_replayed(ascension):
    env = Sts2Env(worker_factory=FakeWorker, ascension=ascension)
    replay = Sts2Env(worker_factory=FakeWorker, ascension=ascension)
    _, info = env.reset(seed=2)
    assert info["ascension"] == ascension
    assert env.game.calls[0][1]["ascension"] == ascension
    env.step(0)
    saved = env.snapshot()
    replay.restore(saved)
    assert replay.snapshot() == saved
    assert replay.step(0)[1:] == env.step(0)[1:]
    report = evaluate(None, env)
    assert report["ascension"] == ascension
    assert all(row["ascension"] == ascension for row in report["episodes"])
    env.close()
    replay.close()


@pytest.mark.parametrize("ascension", [-1, 11, 0.0, True, None, "0"])
def test_invalid_difficulty_fails_before_worker_start(ascension):
    with pytest.raises(ValueError, match="Ascension"):
        Sts2Env(ascension=ascension, worker_factory=lambda *_: pytest.fail("Worker launched"))
    with pytest.raises(ValueError, match="Ascension"):
        evaluation_plan("act1", ascension=ascension)


def test_native_difficulty_mismatch_is_detected():
    class WrongWorker(FakeWorker):
        def request(self, method, parameters):
            result = super().request(method, parameters)
            result["observation"]["ascension"] = 10
            return result

    env = Sts2Env(ascension=0, worker_factory=WrongWorker)
    with pytest.raises(ValueError, match="Official game ascension"):
        env.reset()
    env.close()


def test_difficulties_cannot_share_checkpoint_or_evaluation_identity():
    easy = Sts2Env(ascension=0, worker_factory=FakeWorker)
    hard = Sts2Env(ascension=10, worker_factory=FakeWorker)
    hard.reset()
    with pytest.raises(ValueError, match="ascension"):
        easy.restore(hard.snapshot())
    forged = copy.deepcopy(hard.snapshot()) | {"ascension": 0}
    with pytest.raises(ValueError, match="journal ascension"):
        easy.restore(forged)
    assert not easy.game.calls
    first = evaluation_plan("act1", ascension=0)
    second = evaluation_plan("act1", ascension=10)
    assert first["cases"] == second["cases"]
    assert first["evaluation_id"] != second["evaluation_id"]
    row = evaluate(None, hard)["episodes"][0]
    with pytest.raises(ValueError, match="ascensions"):
        summarise([row, row | {"ascension": 0}])
    easy.close()
    hard.close()


def test_calibration_cache_is_separate_for_each_difficulty(monkeypatch, tmp_path):
    import ai4sts2.calibration as calibration
    from ai4sts2.execution import REFERENCE

    monkeypatch.setattr(calibration, "ROOT", tmp_path)
    monkeypatch.setattr(calibration, "hardware", lambda: {})
    monkeypatch.setattr(
        calibration, "fingerprint", lambda scope, ascension: {"ascension": ascension}
    )
    path = calibration.runtime_path(0)
    path.parent.mkdir()
    path.write_text(
        json.dumps(
            {
                "scope": "run",
                "build": {"ascension": 0},
                "hardware": {},
                "results": [{"execution": REFERENCE.to_dict(), "valid": True, "seconds": 1}],
            }
        )
    )
    assert calibration.selected_execution("act1", 0) == REFERENCE
    assert calibration.selected_execution("act1", 10) is None
