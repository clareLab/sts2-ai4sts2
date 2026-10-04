import copy
import importlib.util
import json
import time
from pathlib import Path

import pytest
import torch
from ai4sts2.environment import write_json
from test_checkpoint import member
from test_continuation import long_member

spec = importlib.util.spec_from_file_location(
    "curve", Path(__file__).resolve().parents[1] / "scripts/curve.py"
)
curve = importlib.util.module_from_spec(spec)
spec.loader.exec_module(curve)


@pytest.fixture
def source(monkeypatch, tmp_path):
    donor = member(monkeypatch)
    donor.model.learn(total_timesteps=64)
    checkpoint = tmp_path / "initial"
    checkpoint.mkdir()
    donor.save_checkpoint(checkpoint)
    result = {
        "seed": donor.config["seed"],
        "variant": "control",
        "environment_steps": 64,
        "checkpoint": str(checkpoint),
        "config": donor.config,
        "metrics": {
            "environment_steps": 64,
            "sample_count": 64,
            "training_seconds": 1,
            "evaluation_seconds": 1,
        },
    }
    donor.cleanup()
    return result


def job(source, output, targets=(128, 256)):
    return source, targets, str(output), "plan", time.monotonic() + 600, 1


def test_continuation_matches_uninterrupted_training_and_reuses_completed_stages(source, tmp_path):
    source["config"]["initial_policy"] = "earlier-transfer"
    output = tmp_path / "resumed"
    assert curve.run_candidate(job(source, output, (128,)))["status"] == "complete"
    assert curve.run_candidate(job(source, output))["status"] == "complete"
    first = curve.saved_stage(output / "seed-5", 128, "plan")
    last = curve.saved_stage(output / "seed-5", 256, "plan")
    assert first["metrics"]["sample_count"] == 64
    assert last["metrics"]["sample_count"] == 128
    before = last["files"]
    assert curve.run_candidate(job(source, output))["status"] == "cached"
    assert curve.checkpoint_files(last["checkpoint"]) == before
    direct = tmp_path / "direct"
    assert curve.run_candidate(job(source, direct))["status"] == "complete"
    left = curve.PopulationMember.__new__(curve.PopulationMember)
    right = curve.PopulationMember.__new__(curve.PopulationMember)
    try:
        for actor, directory in ((left, output), (right, direct)):
            actor.config = source["config"] | {
                "initial_checkpoint": str(directory / "seed-5/checkpoint_000256"),
                "initial_policy": None,
            }
            actor.setup(actor.config)
        assert left.environment.snapshot() == right.environment.snapshot()
        for name, weight in left.model.policy.state_dict().items():
            assert torch.equal(weight, right.model.policy.state_dict()[name])
    finally:
        left.cleanup()
        right.cleanup()


def test_budget_stops_before_opening_a_worker(source, tmp_path, monkeypatch):
    monkeypatch.setattr(curve.PopulationMember, "setup", lambda *_: pytest.fail("Worker launched"))
    request = (*job(source, tmp_path)[:4], time.monotonic() + 10, 1)
    assert curve.run_candidate(request) == {"seed": 5, "status": "budget"}
    assert list(tmp_path.glob("seed-*/checkpoint*")) == []


def test_modified_checkpoint_is_rejected_on_resume(source, tmp_path):
    curve.run_candidate(job(source, tmp_path, (128,)))
    (tmp_path / "seed-5/checkpoint_000128/random.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed after validation"):
        curve.run_candidate(job(source, tmp_path))


def test_stage_is_complete_only_after_every_initialisation(source, tmp_path):
    plan = {"targets": [128, 256], "sources": [source, source | {"seed": 6}], "build": {}}
    curve.run_candidate(job(source, tmp_path, (128,)))
    result = curve.reports(plan, tmp_path, "plan")
    assert len(result["128"]["trials"]) == 1 and not result["128"]["complete"]
    assert not result["256"]["complete"] and result["256"]["trials"] == []
    curve.run_candidate(job(source | {"seed": 6}, tmp_path, (128,)))
    result = curve.reports(plan, tmp_path, "plan")
    assert result["128"]["complete"] and not result["128"]["promoted"]
    assert not result["128"]["certifying"] and not result["256"]["complete"]


def test_source_checks_build_unique_seeds_and_timing(source):
    build = json.loads((Path(source["checkpoint"]) / "build.json").read_text())
    study = {"complete": True, "build": build, "trials": [source]}
    result = curve.sources(study, "control", build)
    assert result[0]["files"] == curve.checkpoint_files(source["checkpoint"])
    with pytest.raises(ValueError, match="distinct"):
        curve.sources(study | {"trials": [source, source]}, "control", build)
    with pytest.raises(ValueError, match="complete study"):
        curve.sources(study | {"complete": False}, "control", build)
    with pytest.raises(ValueError, match="step count"):
        curve.sources(study | {"trials": [source | {"environment_steps": 128}]}, "control", build)
    write_json(Path(source["checkpoint"]) / "build.json", {"game": "changed"})
    with pytest.raises(ValueError, match="Checkpoint build"):
        curve.sources(study, "control", build)


def test_forecast_accounts_for_contention_and_evaluation():
    metrics = {"sample_count": 64, "training_seconds": 100, "evaluation_seconds": 20}
    assert curve.forecast_seconds(128, metrics, 2) == pytest.approx(514)
    paired = metrics | {"concurrent_trials": 2}
    assert curve.forecast_seconds(128, paired, 2) == pytest.approx(272)
    assert curve.forecast_seconds(128, paired, 1) == pytest.approx(272)
    assert curve.forecast_seconds(128, paired, 4) == pytest.approx(514)
    with pytest.raises(ValueError, match="Invalid training timing"):
        curve.forecast_seconds(128, metrics | {"concurrent_trials": 0})
    with pytest.raises(ValueError, match="Invalid training timing"):
        curve.forecast_seconds(128, metrics | {"sample_count": 0})


def test_rollout_snapshots_preserve_training_and_resume_after_an_update(monkeypatch, tmp_path):
    candidate = long_member(monkeypatch, tmp_path / "donor")
    candidate.config |= {"rnd_scale": 0.001, "curriculum_mix": 0.75}
    candidate.signals.configure(candidate.config)
    initial = tmp_path / "initial"
    initial.mkdir()
    candidate.save_checkpoint(initial)
    candidate.model.learn(total_timesteps=192, reset_num_timesteps=False)
    expected_policy = copy.deepcopy(candidate.model.policy.state_dict())
    expected_optimizer = copy.deepcopy(candidate.model.policy.optimizer.state_dict())
    expected_environment = candidate.environment.snapshot()
    expected_signals = candidate.signals.snapshot()
    candidate.load_checkpoint(initial)
    candidate.evaluation = {"obsolete": True}
    callback = curve.RolloutCheckpoints(candidate, tmp_path / "snapshots", "plan")
    candidate.model.learn(total_timesteps=192, reset_num_timesteps=False, callback=callback)

    def equivalent():
        assert candidate.environment.snapshot() == expected_environment
        assert candidate.signals.metrics == expected_signals["metrics"]
        for key, weight in expected_signals["rnd"].items():
            assert torch.equal(weight, candidate.signals.rnd.state_dict()[key])
        for key, weight in expected_policy.items():
            assert torch.equal(weight, candidate.model.policy.state_dict()[key])
        actual = candidate.model.policy.optimizer.state_dict()
        assert actual["param_groups"] == expected_optimizer["param_groups"]
        for index, values in expected_optimizer["state"].items():
            for key, value in values.items():
                assert torch.equal(value, actual["state"][index][key])

    try:
        equivalent()
        assert len(callback.saved) == 3
        for steps in (64, 128, 192):
            checkpoint = tmp_path / "snapshots" / f"checkpoint_{steps:06}"
            saved = json.loads((checkpoint / "snapshot.json").read_text())
            assert saved["environment_steps"] == steps and not saved["validated"]
            assert saved["files"] == curve.checkpoint_files(checkpoint)
            assert not (checkpoint / "evaluation.json").exists()
            assert not (checkpoint / "result.json").exists()
        candidate.load_checkpoint(tmp_path / "snapshots/checkpoint_000064")
        assert candidate.model.num_timesteps == candidate.environment.steps == 64
        candidate.model.learn(total_timesteps=128, reset_num_timesteps=False)
        equivalent()
    finally:
        candidate.cleanup()


def test_failed_rollout_save_never_publishes_a_partial_checkpoint(monkeypatch, tmp_path):
    candidate = long_member(monkeypatch, tmp_path / "donor")
    callback = curve.RolloutCheckpoints(candidate, tmp_path / "snapshots", "plan")

    def fail(directory):
        (Path(directory) / "policy.zip").write_bytes(b"partial")
        raise OSError("Injected disk failure")

    monkeypatch.setattr(candidate, "save_checkpoint", fail)
    try:
        with pytest.raises(OSError, match="disk failure"):
            candidate.model.learn(total_timesteps=128, callback=callback)
        assert list((tmp_path / "snapshots").iterdir()) == []
    finally:
        candidate.cleanup()
