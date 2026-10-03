import json

import pytest
import torch
from ai4sts2.calibration import quarantine_execution, selected_execution
from ai4sts2.environment import Sts2Env, write_json
from ai4sts2.execution import REFERENCE, Execution
from ai4sts2.game import WorkerFailure
from test_checkpoint import member
from test_environment import FakeWorker


@pytest.mark.parametrize("reason", ["Native crash", ""])
def test_failed_execution_is_persistently_excluded_and_retries_are_bounded(
    monkeypatch, tmp_path, reason
):
    import ai4sts2.calibration as calibration

    monkeypatch.setattr(calibration, "ROOT", tmp_path)
    monkeypatch.setattr(calibration, "fingerprint", lambda: {"game": "test"})
    monkeypatch.setattr(calibration, "hardware", lambda: {"cpu": "test"})
    fast = Execution(fps=0, fixed_fps=60, settle_frames=1, step_frames=1)
    write_json(
        tmp_path / "artifacts/runtime.json",
        {
            "build": {"game": "test"},
            "hardware": {"cpu": "test"},
            "scope": "first_combat",
            "selected": fast.to_dict(),
            "results": [
                {"execution": fast.to_dict(), "valid": True, "seconds": 1},
                {"execution": REFERENCE.to_dict(), "valid": True, "seconds": 2},
            ],
        },
    )
    assert selected_execution() == fast
    assert quarantine_execution(fast, WorkerFailure(reason)) == REFERENCE
    assert selected_execution() == REFERENCE
    with pytest.raises(RuntimeError, match="All calibrated"):
        quarantine_execution(REFERENCE, WorkerFailure(reason))
    with pytest.raises(ValueError, match="No validated"):
        selected_execution()
    report = json.loads((tmp_path / "artifacts/runtime.json").read_text())
    assert report["selected"] is None
    assert all("runtime_failure" in r for r in report["results"])


def test_worker_recovery_restores_the_whole_training_iteration(monkeypatch, tmp_path):
    import ai4sts2.train as training

    baseline = member(monkeypatch)
    baseline._logdir = str(tmp_path / "baseline")
    (tmp_path / "baseline").mkdir()
    baseline.sample_count = 128
    expected = baseline.step()
    expected_weights = {k: v.clone() for k, v in baseline.model.policy.state_dict().items()}
    expected_rng = baseline.environment.rng.bit_generator.state
    baseline.cleanup()

    class FaultWorker(FakeWorker):
        steps = 0
        injected = False

        def request(self, method, parameters):
            if method == "step":
                FaultWorker.steps += 1
                if FaultWorker.steps == 70:
                    FaultWorker.injected = True
                    raise WorkerFailure("Injected native crash after an optimiser update")
            return super().request(method, parameters)

    recovered = member(monkeypatch)
    recovered._logdir = str(tmp_path / "recovered")
    (tmp_path / "recovered").mkdir()
    recovered.sample_count = 128
    recovered.environment.close()
    recovered.environment = Sts2Env(seed=5, worker_factory=FaultWorker)
    recovered.model.set_env(recovered.environment)
    monkeypatch.setattr(
        training,
        "Sts2Env",
        lambda executable, seed, **_: Sts2Env(seed=seed, worker_factory=FaultWorker),
    )
    monkeypatch.setattr(training, "quarantine_execution", lambda *_: REFERENCE)
    result = recovered.step()
    assert FaultWorker.injected and len(result["recovery_events"]) == 1
    assert result["environment_steps"] == expected["environment_steps"] == 128
    assert result["validation_episodes"] == expected["validation_episodes"]
    assert recovered.environment.rng.bit_generator.state == expected_rng
    for key, weight in recovered.model.policy.state_dict().items():
        assert torch.equal(weight, expected_weights[key])
    recovered.cleanup()


def test_logic_errors_do_not_trigger_retries(monkeypatch, tmp_path):
    import ai4sts2.train as training

    candidate = member(monkeypatch)
    candidate._logdir = str(tmp_path)
    retries = []
    monkeypatch.setattr(training, "quarantine_execution", lambda *args: retries.append(args))

    def invalid():
        raise ValueError("Invalid legal action set")

    candidate.train_iteration = invalid
    with pytest.raises(ValueError, match="Invalid legal"):
        candidate.step()
    assert not retries
    candidate.cleanup()
