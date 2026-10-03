from types import SimpleNamespace

import pytest
from ai4sts2.train import fit_or_recover, pilot_report


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
        checkpoint="saved",
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
    assert report["errors"] == []
    assert pilot_report([result], "run", {}, {}, 2)["complete"]
    assert not pilot_report([result], "run", {}, {}, 3)["complete"]
    assert not pilot_report([], "run", {}, {}, 2)["complete"]
