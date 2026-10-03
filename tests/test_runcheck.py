import collections
import copy
import json
import random

import pytest
from ai4sts2.game import OfficialGame
from ai4sts2.runcheck import check, probe_decision
from test_environment import state


def test_probe_uses_potions_before_discarding_them():
    value = state()
    value["actions"] = [{"kind": "discard_potion"}, {"kind": "use_potion"}, {"kind": "end_turn"}]
    assert probe_decision(value, random.Random(0), collections.Counter()) == 1


def test_diagnostic_event_cannot_enter_a_normal_worker():
    with pytest.raises(ValueError, match="diagnostic mode"):
        OfficialGame(diagnostic_event="CRYSTAL_SPHERE")
    with pytest.raises(ValueError, match="diagnostic mode"):
        check(diagnostic_event="CRYSTAL_SPHERE")


def test_interruption_cannot_leave_a_previous_success_as_the_latest_report(monkeypatch, tmp_path):
    import ai4sts2.runcheck as runcheck

    monkeypatch.setattr(runcheck, "ROOT", tmp_path)
    monkeypatch.setattr(runcheck, "fingerprint", lambda scope: {"scope": scope})
    monkeypatch.setattr(runcheck, "prepare_game", lambda: None)
    latest = tmp_path / "artifacts/validation/run.json"
    latest.parent.mkdir(parents=True)
    latest.write_text(json.dumps({"valid": True, "complete": True}))

    def interrupt(*args, **kwargs):
        raise KeyboardInterrupt

    monkeypatch.setattr(runcheck, "OfficialGame", interrupt)
    with pytest.raises(KeyboardInterrupt):
        check(minutes=1, episodes=1)
    result = json.loads(latest.read_text())
    assert not result["valid"] and not result["complete"]


@pytest.mark.parametrize("failure", [False, True])
def test_run_report_keeps_failures_and_does_not_claim_unvisited_coverage(
    monkeypatch, tmp_path, failure
):
    import ai4sts2.runcheck as runcheck

    monkeypatch.setattr(runcheck, "ROOT", tmp_path)
    monkeypatch.setattr(runcheck, "fingerprint", lambda scope: {"scope": scope})
    monkeypatch.setattr(runcheck, "prepare_game", lambda: None)
    calls = []

    class Worker:
        def __init__(self, *args, **kwargs):
            self.timeout = kwargs["timeout"]

        def __enter__(self):
            return self

        def __exit__(self, *_):
            pass

        def request(self, method, parameters):
            calls.append((method, parameters))
            if method == "step" and failure:
                raise RuntimeError("Unsupported native screen")
            value = copy.deepcopy(state(method == "step"))
            value["observation"] |= {"screen": "NCombatRoom", "floor": 2, "act": 0}
            value["actions"] = [] if value["terminated"] else [{"kind": "end_turn"}]
            return value

    monkeypatch.setattr(runcheck, "OfficialGame", Worker)
    report = check(minutes=1, episodes=2)
    assert report["valid"] is not failure
    assert not report["full_run_coverage"] and not report["certifying"]
    assert "shop" in report["missing_coverage"]
    assert len([c for c in calls if c[0] == "reset"]) == 2
    assert all(c[1]["scope"] == "run" for c in calls if c[0] == "reset")
    assert len(report["errors"]) == (2 if failure else 0)
    assert len(list(tmp_path.rglob("failure-*.json"))) == (2 if failure else 0)
