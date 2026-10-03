import io
import json
import queue
from pathlib import Path
from types import SimpleNamespace

import pytest
from ai4sts2.execution import Execution
from ai4sts2.game import OfficialGame, WorkerFailure


@pytest.fixture
def worker(monkeypatch):
    opened = []
    closed = []

    def open_process(self, **launch):
        number = len(opened)
        opened.append(launch)
        self.sequence = 0
        self.responses = queue.Queue()
        self.journal = io.StringIO()
        self.log_path = Path(f"worker-{number}.log")

        class Input(io.StringIO):
            def write(stream, value):
                request = json.loads(value)
                if request["params"].get("fail"):
                    self.responses.put(None)
                else:
                    self.responses.put(
                        {
                            "id": request["id"],
                            "ok": True,
                            "result": {"worker": number, "parameters": request["params"]},
                        }
                    )
                return super().write(value)

        self.process = SimpleNamespace(stdin=Input(), poll=lambda: None)
        self.request("hello")

    monkeypatch.setattr(OfficialGame, "_open", open_process)
    monkeypatch.setattr(OfficialGame, "close", lambda self: closed.append(len(opened) - 1))
    result = OfficialGame(
        executable="/unused/game", execution=Execution(non_interactive=True), audit=True
    )
    return result, opened, closed


def test_new_episode_replaces_process_and_preserves_configuration_and_measurements(worker):
    game, opened, closed = worker
    parameters = {"character": "DEFECT", "seed": "fixed", "scope": "run"}
    assert game.request("reset", parameters)["worker"] == 0
    assert game.request("step", {"action": 0})["worker"] == 0
    game.timeout = 17
    result = game.request("reset", parameters)
    assert result == {"worker": 1, "parameters": parameters}
    assert game.request("step", {"action": 1})["worker"] == 1
    assert opened[0] == opened[1]
    assert opened[0]["audit"] and opened[0]["execution"].non_interactive
    assert closed == [0] and game.timeout == 17
    measurements = game.drain_measurements()
    assert measurements["hello"]["calls"] == 2
    assert measurements["reset"]["calls"] == 2
    assert measurements["reset"]["restart_ms"] > 0
    assert measurements["step"]["calls"] == 2
    assert game.drain_measurements() == {}


def test_failed_action_does_not_restart_or_retry_the_episode(worker):
    game, opened, closed = worker
    game.request("reset", {"seed": "fixed"})
    with pytest.raises(WorkerFailure):
        game.request("step", {"action": 2, "fail": True})
    assert len(opened) == 1 and closed == [0]
    records = [json.loads(value) for value in game.journal.getvalue().splitlines()]
    assert [record["method"] for record in records] == ["hello", "reset", "step"]
