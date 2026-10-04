import copy

import pytest
from ai4sts2.calibration import compare, select_result, signature
from ai4sts2.execution import Execution
from ai4sts2.train import bound_mutations, next_sample_count, search_space
from test_environment import state


def test_fast_but_divergent_candidate_cannot_be_selected():
    reference = {"valid": True, "seconds": 10}
    fast = {"valid": True, "seconds": 3}
    invalid = {"valid": False, "seconds": 0.1}
    assert select_result([reference, fast, invalid]) is fast


def test_hidden_divergence_rejects_identical_public_observations():
    reference = state() | {"audit": "reference"}
    changed = copy.deepcopy(reference)
    changed["audit"] = "divergent"
    with pytest.raises(ValueError, match="audit"):
        compare(signature(reference), changed, 3)
    changed = copy.deepcopy(reference)
    changed["actions"].reverse()
    with pytest.raises(ValueError, match="actions"):
        compare(signature(reference), changed, 3)


def test_missing_audit_cannot_pass_validation():
    with pytest.raises(ValueError, match="required"):
        signature(state())
    with pytest.raises(ValueError, match="No validated"):
        select_result([{"valid": False}])


def test_execution_settings_require_a_settling_frame():
    with pytest.raises(ValueError):
        Execution(settle_frames=0)


def test_expensive_evaluation_increases_sampling_and_preserves_rollout_alignment():
    frequent = next_sample_count(30, 5, 128, 64)
    expensive = next_sample_count(30, 60, 128, 64)
    assert frequent < expensive and expensive % 64 == 0
    assert next_sample_count(0, 60, 128, 64) == 2048


def test_pbt_mutations_remain_within_valid_parameter_domains():
    config = {key: 99 for key in search_space() if key != "epochs"} | {"epochs": 2}
    result = bound_mutations(config)
    assert 0 < result["gamma"] < 1
    assert 0 < result["gae_lambda"] < 1
    for key, domain in search_space().items():
        assert domain.is_valid(result[key])


def test_concurrent_recovery_retains_every_failed_execution(monkeypatch, tmp_path):
    import json
    from concurrent.futures import ThreadPoolExecutor

    import ai4sts2.calibration as calibration
    from ai4sts2.environment import write_json

    path = tmp_path / "artifacts/runtime-a10.json"
    candidates = [Execution(fps=fps) for fps in (60, 120, 240)]
    write_json(
        path,
        {
            "results": [
                {"execution": execution.to_dict(), "valid": True, "seconds": index + 1}
                for index, execution in enumerate(candidates)
            ]
        },
    )
    monkeypatch.setattr(calibration, "ROOT", tmp_path)
    monkeypatch.setattr(calibration, "cached_report", lambda *_: json.loads(path.read_text()))
    with ThreadPoolExecutor(max_workers=2) as pool:
        list(
            pool.map(
                lambda execution: calibration.quarantine_execution(
                    execution, RuntimeError("failed")
                ),
                candidates[:2],
            )
        )
    saved = json.loads(path.read_text())
    assert all("runtime_failure" in result for result in saved["results"][:2])
    assert saved["selected"] == candidates[2].to_dict()
