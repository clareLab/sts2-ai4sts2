import io
import json
import queue
from types import SimpleNamespace

import pytest
from ai4sts2.game import OfficialGame, WorkerFailure


@pytest.mark.parametrize("broken_pipe", [False, True])
def test_failed_native_request_preserves_replay_inputs(tmp_path, broken_pipe):
    class Input(io.StringIO):
        fail = False

        def write(self, value):
            if self.fail:
                raise BrokenPipeError("Injected pipe failure")
            return super().write(value)

    worker = object.__new__(OfficialGame)
    worker.sequence = 0
    worker.timeout = 1
    worker.measurements = {}
    worker.responses = queue.Queue()
    worker.log_path = tmp_path / "game.log"
    path = tmp_path / "requests.jsonl"
    worker.journal = path.open("w", buffering=1)
    worker.process = SimpleNamespace(stdin=Input(), poll=lambda: None)
    worker.close = worker.journal.close
    parameters = {"character": "DEFECT", "seed": "fixed", "scope": "run"}
    worker.responses.put({"id": "1", "ok": True, "result": {"revision": 4}})
    assert worker.request("reset", parameters) == {"revision": 4}
    assert json.loads(path.read_text())["params"] == parameters
    worker.process.stdin.fail = broken_pipe
    worker.responses.put(None)
    with pytest.raises(WorkerFailure):
        worker.request("step", {"revision": 4, "action": 2})
    assert [json.loads(line) for line in path.read_text().splitlines()] == [
        {"id": "1", "method": "reset", "params": parameters},
        {"id": "2", "method": "step", "params": {"revision": 4, "action": 2}},
    ]
