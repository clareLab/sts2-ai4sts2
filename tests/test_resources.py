import json
from concurrent.futures import ThreadPoolExecutor

import pytest
from ai4sts2.environment import write_json
from ai4sts2.resources import GIB, Budget, budget


def test_memory_and_cpu_both_limit_parallelism():
    assert Budget(2, 8 * GIB).concurrency() == 2
    assert Budget(2, 4 * GIB).concurrency() == 1
    assert Budget(1, 8 * GIB).concurrency() == 1
    assert Budget(2, 8 * GIB).concurrency(1) == 1
    with pytest.raises(ValueError, match="at most 1"):
        Budget(2, 4 * GIB).concurrency(2)
    with pytest.raises(RuntimeError, match="memory budget"):
        Budget(2, 3 * GIB).concurrency()
    with pytest.raises(RuntimeError):
        Budget(float("nan"), 8 * GIB).concurrency()


def test_automatic_budget_preserves_host_headroom(monkeypatch):
    import ai4sts2.resources as resources

    monkeypatch.delenv("AI4STS2_MEMORY_BUDGET", raising=False)
    monkeypatch.setattr(resources, "capacity", lambda **_: (22, 24 * GIB))
    assert budget() == Budget(2, 8 * GIB)
    monkeypatch.setattr(resources, "capacity", lambda **_: (4, 8 * GIB))
    assert budget() == Budget(2, 4 * GIB)
    assert budget().concurrency() == 1


def test_scoped_budget_is_not_reduced_again_and_respects_parent_limits(monkeypatch):
    import ai4sts2.resources as resources

    monkeypatch.setenv("AI4STS2_MEMORY_BUDGET", str(8 * GIB))
    monkeypatch.setenv("AI4STS2_CPU_BUDGET", "2")
    calls = []

    def capacity(available):
        calls.append(available)
        return 1, 6 * GIB

    monkeypatch.setattr(resources, "capacity", capacity)
    assert budget() == Budget(1, 6 * GIB)
    assert calls == [False]


def test_parallel_json_writers_leave_one_complete_result_and_no_temporary_files(tmp_path):
    path = tmp_path / "shared.json"
    values = [{"writer": index, "payload": [index] * 500} for index in range(40)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda value: write_json(path, value), values))
    assert json.loads(path.read_text()) in values
    assert list(tmp_path.iterdir()) == [path]


def test_failed_json_serialisation_preserves_the_previous_result(tmp_path):
    path = tmp_path / "shared.json"
    write_json(path, {"valid": True})
    with pytest.raises(TypeError):
        write_json(path, {"invalid": object()})
    assert json.loads(path.read_text()) == {"valid": True}
    assert list(tmp_path.iterdir()) == [path]
