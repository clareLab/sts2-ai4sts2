import copy
import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest
import torch
from ai4sts2.environment import evaluation_plan, write_json
from test_checkpoint import member
from test_curve import curve
from test_evaluate import holdout

sys.modules.setdefault("curve", curve)
sys.modules.setdefault("evaluate", holdout)
spec = importlib.util.spec_from_file_location(
    "advance", Path(__file__).resolve().parents[1] / "scripts/advance.py"
)
advance = importlib.util.module_from_spec(spec)
spec.loader.exec_module(advance)


def test_deferred_validation_preserves_training_and_never_opens_a_validation_worker(
    monkeypatch, tmp_path
):
    baseline = member(monkeypatch, policy="shared")
    baseline._logdir = str(tmp_path)
    baseline.sample_count = 64
    baseline.config["fixed_steps"] = True
    first = baseline.step()
    expected = copy.deepcopy(baseline.model.policy.state_dict())
    expected_rng = torch.get_rng_state().clone()
    baseline.cleanup()
    candidate = member(monkeypatch, policy="shared")
    candidate._logdir = str(tmp_path)
    candidate.sample_count = 64
    candidate.config |= {"fixed_steps": True, "validate_each_iteration": False}
    candidate.evaluation = {"old": True}
    monkeypatch.setattr(
        candidate,
        "open_environment",
        lambda *_args, **_kwargs: pytest.fail("Validation worker opened"),
    )
    try:
        second = candidate.step()
        assert second["training_episodes"] == first["training_episodes"]
        assert second["evaluation_seconds"] == 0 and not second["validation_performed"]
        assert "validation_selection_score" not in second and "validation_eligible" not in second
        assert candidate.validation_environment is None and candidate.evaluation is None
        for name, weight in expected.items():
            assert torch.equal(candidate.model.policy.state_dict()[name], weight)
        assert torch.equal(torch.get_rng_state(), expected_rng)
        candidate.save_checkpoint(tmp_path)
        assert not (tmp_path / "evaluation.json").exists()
    finally:
        candidate.cleanup()


@pytest.fixture
def frozen(monkeypatch, tmp_path):
    donor = member(monkeypatch, policy="shared")
    donor.build |= {"scope": "act1", "ascension": 0, "trainer": "current"}
    donor.model.learn(total_timesteps=64)
    checkpoint = tmp_path / "donor"
    checkpoint.mkdir()
    donor.save_checkpoint(checkpoint)
    source = {
        "complete": True,
        "build": donor.build,
        "trials": [
            {
                "variant": "control",
                "seed": 5,
                "checkpoint": str(checkpoint),
                "config": donor.config,
                "environment_steps": 64,
                "files": curve.checkpoint_files(checkpoint),
            }
        ],
    }
    path = tmp_path / "source.json"
    write_json(path, source)
    donor.cleanup()
    monkeypatch.setattr(advance, "fingerprint", lambda *_: source["build"])
    output = tmp_path / "advance"
    output.mkdir()
    return path, output, source


def test_plan_freezes_source_and_requires_explicit_policy_transfer(frozen, monkeypatch):
    path, output, source = frozen
    plan = advance.prepare(output, path, 128, 2, False)
    assert plan["target_steps"] == 192
    assert plan["validation_panel"]["split"] == "validation"
    assert advance.prepare(output, path, 128, 2, False) == plan
    with pytest.raises(ValueError, match="different request"):
        advance.prepare(output, path, 256, 2, False)
    monkeypatch.setattr(advance, "fingerprint", lambda *_: source["build"] | {"trainer": "new"})
    with pytest.raises(ValueError, match="Trainer changed"):
        advance.prepare(output, path, 128, 2, False)
    other = output / "transfer"
    other.mkdir()
    assert advance.prepare(other, path, 128, 2, True)["target_steps"] == 128
    monkeypatch.setattr(advance, "fingerprint", lambda *_: source["build"] | {"game": "new"})
    with pytest.raises(ValueError, match="game, bridge"):
        advance.prepare(other, path, 128, 2, True)


def test_native_resume_preserves_optimizer_rng_and_does_not_repeat_completed_training(
    frozen, monkeypatch
):
    path, output, source = frozen
    plan = advance.prepare(output, path, 128, 2, False)
    monkeypatch.setattr(advance, "prepare_game", lambda: "fake")
    import ai4sts2.train as training

    monkeypatch.setattr(training, "fingerprint", lambda *_: source["build"])
    current = advance.train(output, plan, time.monotonic() + 300)
    assert current["environment_steps"] == 192
    assert not current["metrics"]["validation_performed"]
    monkeypatch.setattr(advance, "prepare_game", lambda: pytest.fail("Repeated native startup"))
    assert advance.train(output, plan, time.monotonic() + 300) == current
    (Path(current["checkpoint"]) / "random.pt").write_bytes(b"changed")
    with pytest.raises(ValueError, match="changed"):
        advance.train(output, plan, time.monotonic() + 300)


def test_budget_does_not_launch_training(frozen, monkeypatch):
    path, output, _ = frozen
    plan = advance.prepare(output, path, 128, 2, False)
    monkeypatch.setattr(advance, "prepare_game", lambda: pytest.fail("Worker opened"))
    assert advance.train(output, plan, time.monotonic() + 5) is None


def test_reference_cache_requires_identical_weights_and_validation_panel(tmp_path):
    panel = evaluation_plan("act1", 42, "validation", 4096, 1, 0)
    write_json(tmp_path / "plan.json", panel)
    candidate = {"name": "reference-5", "sha256": "policy"}
    cached = {
        "complete": True,
        "eligible": True,
        "sha256": "policy",
        "evaluation_id": panel["evaluation_id"],
        "episodes": [{"character": c["character"], "seed": c["seed"]} for c in panel["cases"]],
    }
    plan = {"cached_validation": cached, "validation_panel": panel}
    assert advance.cache_reference(tmp_path, plan, candidate) == 5
    assert advance.cache_reference(tmp_path, plan, candidate | {"sha256": "different"}) == 0
    cached["evaluation_id"] = "test-panel"
    assert advance.cache_reference(tmp_path, plan, candidate) == 0
    record = json.loads(next((tmp_path / "episodes").glob("*.json")).read_text())
    assert record["plan"] == advance.identity(panel)


@pytest.mark.parametrize("winner", [0, 1])
def test_training_continues_forward_when_validation_selects_an_earlier_checkpoint(
    monkeypatch, tmp_path, winner
):
    trials = [{"checkpoint": "earlier"}, {"checkpoint": "latest"}]
    reports = [{"rank": int(index == winner)} for index in range(2)]
    plan = {"request": {"build": {}}, "validation_panel": {"split": "validation"}}
    monkeypatch.setattr(holdout, "progression_key", lambda report: report["rank"])
    result = {"complete": True, "eligible": True, "trials": reports}
    advance.save_selection(tmp_path, plan, trials, result)
    selected = json.loads((tmp_path / "selection.json").read_text())
    continuation = json.loads((tmp_path / "continuation.json").read_text())
    assert selected["trials"][0]["checkpoint"] == trials[winner]["checkpoint"]
    assert continuation["trials"][0]["checkpoint"] == "latest"
    assert continuation["selected_validation"] == reports[1]
    assert continuation["selected_by_validation"] == (winner == 1)
    with pytest.raises(ValueError, match="complete eligible"):
        advance.save_selection(tmp_path, plan, trials, result | {"eligible": False})


def test_interrupted_round_resumes_from_the_last_atomic_checkpoint(frozen, monkeypatch):
    path, output, source = frozen
    plan = advance.prepare(output, path, 1024, 2, False)
    monkeypatch.setattr(advance, "prepare_game", lambda: "fake")
    import ai4sts2.train as training

    monkeypatch.setattr(training, "fingerprint", lambda *_: source["build"])
    persist = advance.persist

    def interrupt(*args, **kwargs):
        saved = persist(*args, **kwargs)
        if saved["environment_steps"] == 576:
            raise InterruptedError("Interrupted after an atomic checkpoint")
        return saved

    monkeypatch.setattr(advance, "persist", interrupt)
    with pytest.raises(InterruptedError):
        advance.train(output, plan, time.monotonic() + 300)
    monkeypatch.setattr(advance, "persist", persist)
    resumed = advance.train(output, plan, time.monotonic() + 300)
    direct = output / "direct"
    direct.mkdir()
    uninterrupted = advance.train(direct, plan, time.monotonic() + 300)
    models = [
        training.TimedPPO.load(Path(row["checkpoint"]) / "policy.zip")
        for row in (resumed, uninterrupted)
    ]
    for name, value in models[0].policy.state_dict().items():
        assert torch.equal(value, models[1].policy.state_dict()[name])
    optimisers = [model.policy.optimizer.state_dict() for model in models]
    assert optimisers[0]["param_groups"] == optimisers[1]["param_groups"]
    for index, values in optimisers[0]["state"].items():
        for name, value in values.items():
            assert torch.equal(value, optimisers[1]["state"][index][name])
    assert json.loads((Path(resumed["checkpoint"]) / "environment.json").read_text()) == json.loads(
        (Path(uninterrupted["checkpoint"]) / "environment.json").read_text()
    )
