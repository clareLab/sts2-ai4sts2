import copy

import numpy as np
import pytest
from ai4sts2.calibration import signature
from ai4sts2.environment import Sts2Env, encode, evaluate, evaluation_plan, state_digest
from ai4sts2.metrics import summarise
from test_environment import FakeWorker, state


class EliteWorker(FakeWorker):
    def request(self, method, parameters):
        result = super().request(method, parameters)
        result["act1_elite_wins"] = int(self.count >= 3)
        result["observation"] |= {"room": "Elite", "floor": 8, "act": 0}
        return result


def test_elite_task_requires_a_confirmed_win_and_stops_before_another_action():
    env = Sts2Env(scope="act1_elite", worker_factory=EliteWorker, max_steps=3)
    env.reset()
    assert env.journal["parameters"]["scope"] == "run"
    for _ in range(2):
        _, reward, terminated, truncated, info = env.step(0)
        assert reward == 0 and not terminated and not truncated
        assert not info["task_success"] and info["act1_elite_wins"] == 0
    _, reward, terminated, truncated, info = env.step(0)
    assert reward == 1 and terminated and not truncated
    assert info["task_success"] and not info["victory"] and info["act1_elite_wins"] == 1
    calls = len(env.game.calls)
    with pytest.raises(ValueError, match="Illegal"):
        env.step(0)
    assert len(env.game.calls) == calls
    report = summarise(env.drain_episodes())
    assert report["eligible"] and report["selection_score"] == 1
    assert report["win_rate"] == 0 and report["act1_elite_success_rate"] == 1
    env.reset()
    assert env.elite_wins() == 0 and not env.task_finished()
    env.close()


@pytest.mark.parametrize("outcome", ["death", "next_act", "timeout"])
def test_elite_entry_boss_clear_and_timeout_do_not_count_as_elite_victories(outcome):
    class FailedWorker(EliteWorker):
        def request(self, method, parameters):
            result = super().request(method, parameters)
            if self.count == 2:
                if outcome == "death":
                    result |= {"terminated": True, "victory": False, "actions": []}
                    result["observation"]["player"]["hp"] = 0
                elif outcome == "next_act":
                    result["observation"]["act"] = 1
            return result

    env = Sts2Env(scope="act1_elite", worker_factory=FailedWorker, max_steps=2)
    env.reset()
    env.step(0)
    _, reward, terminated, truncated, info = env.step(0)
    assert terminated == (outcome != "timeout")
    assert truncated == (outcome == "timeout")
    assert reward == (0 if truncated else -1)
    assert not info["task_success"] and info["act1_elite_wins"] == 0
    report = summarise(env.drain_episodes())
    assert report["eligible"] == (not truncated)
    assert report["mean_death_floor"] == (8 if outcome == "death" else None)
    env.close()


@pytest.mark.parametrize("count", [None, -1, 1.0, True, 1])
def test_reset_rejects_invalid_or_stale_elite_counters(count):
    class InvalidWorker(EliteWorker):
        def request(self, method, parameters):
            return super().request(method, parameters) | {"act1_elite_wins": count}

    env = Sts2Env(scope="act1_elite", worker_factory=InvalidWorker)
    with pytest.raises(ValueError, match="elite victory"):
        env.reset()
    env.close()


def test_elite_results_are_replayed_but_do_not_change_policy_features():
    first = Sts2Env(scope="act1_elite", worker_factory=EliteWorker)
    second = Sts2Env(scope="act1_elite", worker_factory=EliteWorker)
    first.reset(seed=19)
    first.step(0)
    first.step(0)
    saved = first.snapshot()
    second.restore(saved)
    assert first.step(0)[1:] == second.step(0)[1:]
    second.restore(first.snapshot())
    assert second.elite_wins() == 1
    with pytest.raises(ValueError, match="Illegal"):
        second.step(0)
    original = state() | {"act1_elite_wins": 0, "audit": "test"}
    changed = copy.deepcopy(original) | {"act1_elite_wins": 1}
    assert state_digest(original) != state_digest(changed)
    assert signature(original) != signature(changed)
    for key, value in encode(original).items():
        np.testing.assert_array_equal(value, encode(changed)[key])
    first.close()
    second.close()


def test_elite_evaluation_is_distinct_from_boss_and_run_success():
    env = Sts2Env(scope="act1_elite", worker_factory=EliteWorker)
    report = evaluate(None, env)
    assert report["complete"] and report["eligible"]
    assert report["task_successes"] == report["act1_elite_victories"] == 5
    assert report["wins"] == 0
    assert all(row["task_success_rate"] == 1 for row in report["characters"].values())
    assert len({evaluation_plan(s)["evaluation_id"] for s in ("run", "act1", "act1_elite")}) == 3
    episode = report["episodes"][0]
    with pytest.raises(ValueError, match="requires"):
        summarise([episode | {"act1_elite_wins": 0}])
    with pytest.raises(ValueError, match="Invalid"):
        summarise([episode | {"act1_elite_wins": True}])
    with pytest.raises(ValueError, match="scopes"):
        summarise([episode, episode | {"scope": "run"}])
    env.close()
