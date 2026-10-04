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
from test_replay import replay_corpus as replay_corpus

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
    assert advance.cache_reference(tmp_path, plan, candidate | {"policy_mode": "sampled"}) == 0
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
    assert continuation["incumbent"] == selected
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


def test_replay_comparison_has_matched_initialisation_and_frozen_controls(
    frozen, monkeypatch, replay_corpus
):
    import ai4sts2.train as training

    path, output, source = frozen
    collection = json.loads(replay_corpus.read_text())
    for record in collection["records"]:
        record["build"] = source["build"]
    write_json(replay_corpus, collection)
    source["trials"][0]["config"] |= {
        "replay_corpus": str(replay_corpus),
        "replay_updates": 1,
        "collect_replay": True,
    }
    write_json(path, source)
    with pytest.raises(ValueError, match="requires policy transfer"):
        advance.prepare(output, path, 64, 2, False, True)
    plan = advance.prepare(output, path, 64, 2, True, True)
    assert advance.prepare(output, path, 64, 2, True, True) == plan
    with pytest.raises(ValueError, match="different request"):
        advance.prepare(output, path, 64, 2, True, False)
    jobs = advance.training_jobs(output, plan)
    assert [variant["collect_replay"] for _, variant in jobs] == [False, True]
    assert len({variant["training_seed"] for _, variant in jobs}) == 1
    assert all(variant["source"] == plan["source"] for _, variant in jobs)
    monkeypatch.setattr(advance, "prepare_game", lambda: "fake")
    monkeypatch.setattr(training, "fingerprint", lambda *_: source["build"])
    trials = [
        advance.train(directory, variant, time.monotonic() + 300) for directory, variant in jobs
    ]
    assert [trial["variant"] for trial in trials] == ["fixed", "online"]
    assert all(not trial["metrics"]["validation_performed"] for trial in trials)
    models = [training.TimedPPO.load(Path(trial["checkpoint"]) / "policy.zip") for trial in trials]
    for name, value in models[0].policy.state_dict().items():
        assert torch.equal(value, models[1].policy.state_dict()[name])
    snapshots = [
        json.loads((Path(t["checkpoint"]) / "environment.json").read_text()) for t in trials
    ]
    assert snapshots[0] == snapshots[1]
    totals = advance.training_totals(output, plan)
    assert [row["steps"] for row in totals] == [64, 64]
    assert all(row["episodes"] > 0 and row["seconds"] > 0 for row in totals)
    assert trials[0]["config"]["collect_replay"] is False


def test_incomplete_round_never_evaluates_or_selects(frozen, monkeypatch):
    from ai4sts2.resources import Budget

    path, output, source = frozen
    events = []
    monkeypatch.setattr(advance, "budget", lambda: Budget(2, 8 * 1024**3))
    monkeypatch.setattr(advance, "ensure_execution", lambda *_: None)
    monkeypatch.setattr(advance, "train_round", lambda *_: [{"environment_steps": 128}])
    monkeypatch.setattr(advance, "validate", lambda *_: events.append("evaluation"))
    result = advance.run(path, output, steps=128)
    assert not result["complete"] and not result["training_complete"]
    assert events == [] and not (output / "selection.json").exists()
    monkeypatch.setattr(advance, "train_round", lambda *_: [{"environment_steps": 192}])

    def evaluate(*args):
        events.append("evaluation")
        return {"eligible": True, "complete": True}

    monkeypatch.setattr(advance, "validate", evaluate)
    result = advance.run(path, output, steps=128)
    assert result["complete"] and result["training_complete"]
    assert events == ["evaluation"]


@pytest.mark.parametrize("winner", [0, 1, 2])
def test_comparison_keeps_incumbent_and_selects_a_trained_continuation(
    monkeypatch, tmp_path, winner
):
    trials = [{"checkpoint": name} for name in ("incumbent", "fixed", "online")]
    reports = [{"rank": int(index == winner)} for index in range(3)]
    plan = {"request": {"build": {}}, "validation_panel": {"split": "validation"}}
    monkeypatch.setattr(holdout, "progression_key", lambda report: report["rank"])
    advance.save_selection(
        tmp_path, plan, trials, {"complete": True, "eligible": True, "trials": reports}
    )
    selected = json.loads((tmp_path / "selection.json").read_text())
    continuation = json.loads((tmp_path / "continuation.json").read_text())
    assert selected["trials"][0]["checkpoint"] == trials[winner]["checkpoint"]
    assert continuation["trials"][0]["checkpoint"] == ("online" if winner == 2 else "fixed")
    assert continuation["incumbent"] == selected


def test_later_round_keeps_global_incumbent_and_reuses_its_validation(frozen, monkeypatch):
    import ai4sts2.train as training

    path, output, source = frozen
    monkeypatch.setattr(advance, "prepare_game", lambda: "fake")
    monkeypatch.setattr(training, "fingerprint", lambda *_: source["build"])
    monkeypatch.setattr(holdout, "progression_key", lambda report: report["rank"])
    plan = advance.prepare(output, path, 64, 2, False)
    first = advance.train(output, plan, time.monotonic() + 300)
    original = source["trials"][0]
    report = {
        "complete": True,
        "eligible": True,
        "rank": 1,
        "sha256": holdout.digest(Path(original["checkpoint"]) / "policy.zip"),
        "evaluation_id": plan["validation_panel"]["evaluation_id"],
        "episodes": [
            {"character": case["character"], "seed": case["seed"]}
            for case in plan["validation_panel"]["cases"]
        ],
    }
    advance.save_selection(
        output,
        plan,
        [original, first],
        {"complete": True, "eligible": True, "trials": [report, report | {"rank": 0}]},
    )
    next_output = output / "next"
    next_output.mkdir()
    next_plan = advance.prepare(next_output, output / "continuation.json", 64, 2, False)
    assert advance.validation_reserve(next_plan, 1800) == 270
    assert advance.validation_reserve(next_plan, 600) == 180
    assert advance.validation_reserve(next_plan | {"cached_validation": None}, 1800) == 540
    invalid_cache = report | {"sha256": "changed"}
    assert advance.validation_reserve(next_plan | {"cached_validation": invalid_cache}, 1800) == 540
    assert next_plan["source"]["checkpoint"] == first["checkpoint"]
    assert next_plan["incumbent"]["checkpoint"] == original["checkpoint"]
    assert next_plan["cached_validation"] == report
    assert next_plan["training_seed"] == plan["training_seed"]
    assert next_plan["target_steps"] == first["environment_steps"] + 64
    second = advance.train(next_output, next_plan, time.monotonic() + 300)

    def evaluate(study_path, directory, *args, **kwargs):
        comparison = json.loads(Path(study_path).read_text())
        assert [trial["checkpoint"] for trial in comparison["trials"]] == [
            original["checkpoint"],
            second["checkpoint"],
        ]
        assert len(list((directory / "episodes").glob("*.json"))) == 10
        return {"complete": True, "eligible": True, "trials": [report, report | {"rank": 0}]}

    monkeypatch.setattr(holdout, "run", evaluate)
    result = advance.validate(next_output, next_plan, second, time.monotonic() + 300, 2)
    assert result["cached_episodes"] == 10
    continuation = json.loads((next_output / "continuation.json").read_text())
    assert continuation["trials"][0]["checkpoint"] == second["checkpoint"]
    assert continuation["incumbent"]["trials"][0]["checkpoint"] == original["checkpoint"]
    assert "incumbent" not in continuation["incumbent"]


def test_external_incumbent_requires_same_build_panel_and_frozen_source(frozen):
    path, output, source = frozen
    reference = output / "incumbent.json"
    write_json(reference, source)
    plan = advance.prepare(output, path, 64, 2, False, incumbent=reference)
    assert plan["incumbent"] == source["trials"][0]
    with pytest.raises(ValueError, match="without policy transfer"):
        advance.prepare(output, path, 64, 2, True, incumbent=reference)
    changed = copy.deepcopy(source)
    changed["build"]["trainer"] = "old"
    write_json(reference, changed)
    with pytest.raises(ValueError, match="current build"):
        advance.prepare(output, path, 64, 2, False, incumbent=reference)
    changed = source | {"validation_panel": {"split": "test"}}
    write_json(reference, changed)
    with pytest.raises(ValueError, match="different request"):
        advance.prepare(output, path, 64, 2, False, incumbent=reference)
    other = output / "other"
    other.mkdir()
    with pytest.raises(ValueError, match="panel unchanged"):
        advance.prepare(other, path, 64, 2, False, incumbent=reference)


def test_parallel_round_uses_isolated_processes_and_shared_budget(monkeypatch, tmp_path):
    from concurrent.futures import Future

    calls = []

    class Pool:
        def __init__(self, max_workers, mp_context):
            assert max_workers == 2 and mp_context.get_start_method() == "spawn"

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def submit(self, function, directory, plan, deadline, executable):
            assert function is advance.train and deadline == 123 and executable == "prepared"
            calls.append(plan["variant"])
            future = Future()
            future.set_result({"variant": plan["variant"]})
            return future

    monkeypatch.setattr(advance, "ProcessPoolExecutor", Pool)
    monkeypatch.setattr(advance, "prepare_game", lambda: "prepared")
    result = advance.train_round(tmp_path, {"request": {"compare_replay": True}}, 123, 2)
    assert calls == ["fixed", "online"] and len(result) == 2
