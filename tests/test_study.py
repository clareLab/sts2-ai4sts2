import importlib.util
import json
import time
import zipfile
from pathlib import Path
from types import SimpleNamespace

import pytest
from ai4sts2.execution import Execution
from ai4sts2.resources import GIB, Budget
from test_checkpoint import member


@pytest.fixture
def study(monkeypatch, tmp_path):
    scripts = Path(__file__).resolve().parents[1] / "scripts"
    monkeypatch.syspath_prepend(str(scripts))
    spec = importlib.util.spec_from_file_location("study_runner", scripts / "study.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setattr(module, "fingerprint", lambda *_: {"game": "test"})
    monkeypatch.setattr(module, "budget", lambda: Budget(2, 8 * GIB))
    config = {
        "learning_rate": 0.0003,
        "entropy": 0.01,
        "steps_per_iteration": 64,
        "fixed_steps": True,
        "scope": "run",
    }
    for name, steps in (("initial", 64), ("control", 320)):
        trials = []
        for seed in (7, 8):
            checkpoint = tmp_path / name / str(seed)
            checkpoint.mkdir(parents=True)
            with zipfile.ZipFile(checkpoint / "policy.zip", "w") as archive:
                archive.writestr("data", json.dumps({"num_timesteps": steps, "seed": seed}))
            (checkpoint / "build.json").write_text('{"game": "test"}')
            for filename in ("environment.json", "random.pt", "signals.pt", "schedule.json"):
                (checkpoint / filename).write_text("{}")
            trials.append(
                {
                    "seed": seed,
                    "variant": "control",
                    "environment_steps": steps,
                    "checkpoint": str(checkpoint),
                    "config": config | {"seed": seed},
                    "metrics": {"environment_steps": steps},
                }
            )
        (tmp_path / f"{name}.json").write_text(
            json.dumps({"complete": True, "build": {"game": "test"}, "trials": trials})
        )
    return module


def plan(study, tmp_path):
    return study.prepare_plan(
        tmp_path / "output", tmp_path / "initial.json", tmp_path / "control.json", 256, 1
    )


def test_plan_freezes_search_cases_and_rejects_changed_request(study, tmp_path):
    first = plan(study, tmp_path)
    assert study.prepare_plan(tmp_path / "output") == first
    assert len(first["cases"]["cases"]) == 5
    assert first["target_steps"] == 320
    assert not first["certifying"] and not first["promoted"]
    assert first["members"]["7"]["initial_checkpoint"].endswith("initial/7")
    with pytest.raises(ValueError, match="different request"):
        study.prepare_plan(tmp_path / "output", steps=512)


def test_two_round_plan_preserves_whole_rollouts_and_freezes_epoch_search(study, tmp_path):
    frozen = study.prepare_plan(
        tmp_path / "output", tmp_path / "initial.json", steps=256, iterations=2
    )
    assert frozen["iterations"] == 2 and frozen["steps_per_iteration"] == 128
    assert all(member["epochs"] in (2, 4, 8, 16) for member in frozen["members"].values())
    assert study.prepare_plan(tmp_path / "output") == frozen
    with pytest.raises(ValueError, match="different request"):
        study.prepare_plan(tmp_path / "output", iterations=4)


@pytest.mark.parametrize("steps,iterations", [(128, 1), (256, 3), (128, 4)])
def test_plan_rejects_incomplete_rollouts(study, tmp_path, steps, iterations):
    with pytest.raises(ValueError, match="whole rollouts"):
        study.prepare_plan(
            tmp_path / "output", tmp_path / "initial.json", steps=steps, iterations=iterations
        )


def test_optimiser_mutations_leave_reward_and_curriculum_unchanged(study):
    config = {
        "learning_rate": 1.0,
        "entropy": 0.0,
        "epochs": 8,
        "gamma": 1.0,
        "gae_lambda": 0.95,
        "clip_range": 0.2,
        "rnd_scale": 0.0,
        "curriculum_mix": 0.0,
        "progress_scale": 0.0,
    }
    fixed = {key: value for key, value in config.items() if key not in study.optimiser_space()}
    result = study.bound_optimiser(config.copy())
    assert result["epochs"] == 8
    assert all(domain.is_valid(result[key]) for key, domain in study.optimiser_space().items())
    assert {key: result[key] for key in fixed} == fixed


def test_population_uses_frozen_iteration_size_after_loading_source(study, tmp_path, monkeypatch):
    donor = member(monkeypatch)
    checkpoint = tmp_path / "restored"
    checkpoint.mkdir()
    donor.sample_count = 768
    donor.save_checkpoint(checkpoint)
    receiver = object.__new__(study.StudyMember)
    receiver.config = donor.config | {
        "initial_checkpoint": str(checkpoint),
        "steps_per_iteration": 64,
    }
    try:
        receiver.setup(receiver.config)
        assert receiver.sample_count == 64
    finally:
        donor.cleanup()
        receiver.cleanup()


@pytest.mark.parametrize("change", ["checkpoint", "build", "runner"])
def test_resume_rejects_changed_inputs(study, tmp_path, monkeypatch, change):
    plan(study, tmp_path)
    if change == "checkpoint":
        (tmp_path / "initial/7/signals.pt").write_text("changed")
    elif change == "build":
        monkeypatch.setattr(study, "fingerprint", lambda *_: {"game": "changed"})
    else:
        monkeypatch.setattr(study, "runner_identity", lambda: {})
    with pytest.raises(ValueError, match="changed|different"):
        study.prepare_plan(tmp_path / "output")


@pytest.mark.parametrize("change", ["seed", "steps", "config", "weights"])
def test_mismatched_fixed_controls_are_rejected(study, tmp_path, change):
    path = tmp_path / "control.json"
    value = json.loads(path.read_text())
    trial = value["trials"][0]
    if change == "seed":
        trial["seed"] = 9
    elif change == "steps":
        trial["environment_steps"] = trial["metrics"]["environment_steps"] = 384
    elif change == "config":
        trial["config"]["entropy"] = 0.1
    else:
        with zipfile.ZipFile(Path(trial["checkpoint"]) / "policy.zip", "w") as archive:
            archive.writestr("data", json.dumps({"num_timesteps": 384}))
    path.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="Fixed controls|step count"):
        plan(study, tmp_path)


def ray_analysis(study, tmp_path, monkeypatch, status="TERMINATED", steps=320):
    trials = [
        SimpleNamespace(
            trial_id=str(seed),
            status=status,
            config={},
            last_result={"environment_steps": steps, "training_iteration": 4},
        )
        for seed in (7, 8)
    ]
    analysis = SimpleNamespace(
        trials=trials,
        get_last_checkpoint=lambda trial: SimpleNamespace(
            path=str(tmp_path / "control" / trial.trial_id)
        ),
    )
    monkeypatch.setattr(study, "training_analysis", lambda *_: analysis)
    return analysis


def test_ray_results_use_last_checkpoint_and_require_frozen_target(study, tmp_path, monkeypatch):
    frozen = plan(study, tmp_path)
    analysis = ray_analysis(study, tmp_path, monkeypatch)
    result = study.collect_training(tmp_path / "population", frozen)
    assert result["complete"]
    assert [trial["trial_id"] for trial in result["trials"]] == ["7", "8"]
    assert all(trial["environment_steps"] == 320 for trial in result["trials"])
    analysis.trials[0].last_result["environment_steps"] = 256
    with pytest.raises(ValueError, match="frozen target"):
        study.collect_training(tmp_path / "population", frozen)


@pytest.mark.parametrize("status", ["RUNNING", "PAUSED", "PENDING", "ERROR"])
def test_unfinished_or_failed_ray_trials_are_not_complete(study, tmp_path, monkeypatch, status):
    frozen = plan(study, tmp_path)
    ray_analysis(study, tmp_path, monkeypatch, status)
    result = study.collect_training(tmp_path / "population", frozen)
    assert not result["complete"] and result["trials"] == []
    assert bool(result["errors"]) == (status == "ERROR")


def test_resumed_population_synchronises_after_the_last_saved_round(study, tmp_path, monkeypatch):
    frozen = plan(study, tmp_path)
    analysis = ray_analysis(study, tmp_path, monkeypatch, "PAUSED")
    analysis.trials[0].last_result["training_iteration"] = 2
    analysis.trials[1].last_result["training_iteration"] = 3
    calls = []
    monkeypatch.setattr(study.ray, "init", lambda **_: None)
    monkeypatch.setattr(study.ray, "shutdown", lambda: None)
    monkeypatch.setattr(study, "prepare_game", lambda: "/unused")
    monkeypatch.setattr(study, "selected_execution", lambda *_: Execution())
    monkeypatch.setattr(study, "PopulationBasedTraining", lambda **kwargs: kwargs)
    monkeypatch.setattr(study.tune, "run", lambda *args, **kwargs: calls.append(kwargs))
    study.fit_population(
        tmp_path / "population", frozen, {"concurrent_trials": 2}, time.monotonic() + 60
    )
    assert len(calls) == 1 and calls[0]["resume"]
    assert calls[0]["config"] == {}
    assert calls[0]["scheduler"]["burn_in_period"] == 4
    assert set(calls[0]["scheduler"]["hyperparam_mutations"]) == {
        "learning_rate",
        "entropy",
        "epochs",
    }
    assert calls[0]["scheduler"]["hyperparam_mutations"]["epochs"] == [2, 4, 8, 16]
    assert calls[0]["scheduler"]["custom_explore_fn"] is study.bound_optimiser
    assert calls[0]["max_failures"] == 0
    assert calls[0]["stop"] == {"training_iteration": 4}


def test_expired_study_budget_starts_no_training_or_evaluation(study, tmp_path, monkeypatch):
    plan(study, tmp_path)
    monkeypatch.setenv("AI4STS2_DEADLINE", str(time.monotonic() - 1))
    for module, name in (
        (study, "calibrate"),
        (study, "fit_population"),
        (study.curve, "run"),
        (study.holdout, "run"),
    ):
        monkeypatch.setattr(module, name, lambda *_args, **_kwargs: pytest.fail("Worker launched"))
    result = study.run(tmp_path / "output")
    assert result["stage"] == "budget" and not result["complete"]


@pytest.mark.parametrize("eligible", [True, False])
def test_completed_training_is_reused_and_invalid_evaluation_is_failed(
    study, tmp_path, monkeypatch, eligible
):
    plan(study, tmp_path)
    ray_analysis(study, tmp_path, monkeypatch)
    for module, name in ((study, "fit_population"), (study.curve, "run"), (study, "calibrate")):
        monkeypatch.setattr(
            module, name, lambda *_args, **_kwargs: pytest.fail("Training repeated")
        )
    evaluation = {
        "complete": True,
        "eligible": eligible,
        "trials": [
            {
                "name": variant,
                "variant": variant,
                "errors": [],
                "mean_floor": 3,
                "episodes": [{"floor": 3, "act": 0, "victory": False}],
            }
            for variant in ("random", "fixed", "pbt")
        ],
    }
    monkeypatch.setattr(study.holdout, "run", lambda *_args, **_kwargs: evaluation)
    result = study.run(tmp_path / "output")
    assert result["stage"] == ("complete" if eligible else "failed")
    assert result["complete"] == eligible
    assert not result["promoted"]
    assert study.run(tmp_path / "output") == result
