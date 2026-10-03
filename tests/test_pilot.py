from types import SimpleNamespace

import pytest
from ai4sts2.train import ablation_space, fit_or_recover, pilot_report


def test_cleanup_interrupt_recovers_saved_results_without_resuming_training(monkeypatch, tmp_path):
    import ai4sts2.train as training

    saved = [object()]
    calls = []

    def interrupted():
        raise KeyboardInterrupt

    def restore(path, *, trainable):
        calls.append((path, trainable))
        return SimpleNamespace(get_results=lambda: saved)

    monkeypatch.setattr(training.tune.Tuner, "restore", restore)
    results, stopped = fit_or_recover(SimpleNamespace(fit=interrupted), tmp_path, "worker")
    assert results is saved and stopped
    assert calls == [(str(tmp_path), "worker")]


def test_training_errors_are_not_misreported_as_budget_interruptions():
    def failed():
        raise ValueError("Invalid observation")

    with pytest.raises(ValueError, match="Invalid observation"):
        fit_or_recover(SimpleNamespace(fit=failed), None, None)


def test_interrupted_report_keeps_candidate_metrics_without_claiming_completion():
    result = SimpleNamespace(
        error=None,
        checkpoint=SimpleNamespace(path="saved"),
        path="trial",
        config={},
        metrics={
            "training_iteration": 2,
            "validation_win_rate": 0,
            "validation_mean_floor": 7.2,
            "validation_selection_score": 7.2 / 8.2,
        },
    )
    report = pilot_report([result], "run", {"game": "test"}, {}, 2, interrupted=True)
    assert not report["complete"] and report["interrupted"]
    assert not report["certifying"] and not report["promoted"]
    assert report["trials"][0]["mean_floor"] == 7.2
    assert report["trials"][0]["checkpoint"] == "saved"
    assert report["errors"] == []
    assert pilot_report([result], "run", {}, {}, 2)["complete"]
    assert not pilot_report([result], "run", {}, {}, 3)["complete"]
    assert not pilot_report([], "run", {}, {}, 2)["complete"]


def test_ablation_changes_one_factor_and_uses_identical_initialisation():
    from ray.tune.search.variant_generator import generate_variants

    variants = [spec["config"] for _, spec in generate_variants({"config": ablation_space(17)})]
    assert [variant["variant"] for variant in variants] == ["control", "exploration", "curriculum"]
    controls = {"variant", "rnd_scale", "curriculum_mix"}
    common = [
        {key: value for key, value in variant.items() if key not in controls}
        for variant in variants
    ]
    assert common[0] == common[1] == common[2]
    assert common[0]["seed"] == 17 and common[0]["fixed_steps"]
    assert [(v["rnd_scale"], v["curriculum_mix"]) for v in variants] == [
        (0, 0),
        (0.001, 0),
        (0, 0.75),
    ]


def test_repeated_ablation_pairs_all_variants_with_each_model_seed():
    from ray.tune.search.variant_generator import generate_variants

    variants = [spec["config"] for _, spec in generate_variants({"config": ablation_space(7, 3)})]
    assert len(variants) == 9
    assert {(v["seed"], v["variant"]) for v in variants} == {
        (seed, name) for seed in (7, 8, 9) for name in ("control", "exploration", "curriculum")
    }
    with pytest.raises(ValueError):
        ablation_space(7, 0)


def test_policy_ablation_keeps_rewards_and_hyperparameters_paired():
    from ray.tune.search.variant_generator import generate_variants

    variants = [
        spec["config"]
        for _, spec in generate_variants({"config": ablation_space(7, 3, "policies")})
    ]
    assert {(variant["policy"], variant["seed"]) for variant in variants} == {
        (policy, seed) for policy in ("flat", "shared") for seed in (7, 8, 9)
    }
    common = [
        {key: value for key, value in variant.items() if key not in {"variant", "policy", "seed"}}
        for variant in variants
    ]
    assert all(config == common[0] for config in common)
    assert common[0]["rnd_scale"] == common[0]["curriculum_mix"] == 0
    with pytest.raises(ValueError, match="Unknown ablation study"):
        ablation_space(7, study="unknown")
