import copy

import numpy as np
import pytest
from ai4sts2.calibration import signature
from ai4sts2.environment import Sts2Env, encode, evaluate, evaluation_plan, state_digest
from ai4sts2.metrics import health_fraction, summarise
from test_environment import FakeWorker


class MonsterWorker(FakeWorker):
    def request(self, method, parameters):
        result = super().request(method, parameters)
        result |= {
            "terminated": False,
            "victory": False,
            "actions": [{"kind": "play"}],
            "act1_elite_wins": int(self.count > 0),
            "act1_monster_wins": max(0, self.count - 1),
        }
        result["observation"] |= {"room": "Monster", "floor": 4 + self.count, "act": 0}
        result["observation"]["player"] = {"hp": 25, "max_hp": 100}
        return result


def test_three_normal_combat_wins_end_the_task_and_reward_remaining_health():
    env = Sts2Env(scope="act1_monsters", worker_factory=MonsterWorker)
    env.reset()
    assert env.journal["parameters"]["scope"] == "run"
    for _ in range(3):
        _, reward, terminated, truncated, info = env.step(0)
        assert reward == 0 and not terminated and not truncated
        assert not info["task_success"]
    _, reward, terminated, truncated, info = env.step(0)
    assert reward == 0.25 and terminated and not truncated
    assert info["act1_monster_wins"] == 3 and info["act1_elite_wins"] == 1
    assert info["task_success"] and not info["victory"]
    assert info["task_score"] == 0.25 and info["max_hp"] == 100
    with pytest.raises(ValueError, match="Illegal"):
        env.step(0)
    report = summarise(env.drain_episodes())
    assert report["selection_score"] == report["mean_surviving_health"] == 0.25
    assert report["task_success_rate"] == 1 and report["win_rate"] == 0
    env.reset()
    assert env.combat_wins("monster") == 0
    env.close()


@pytest.mark.parametrize("outcome", ["death", "next_act", "timeout"])
def test_unfinished_three_fights_cannot_receive_health_credit(outcome):
    class FailedWorker(MonsterWorker):
        def request(self, method, parameters):
            result = super().request(method, parameters)
            if self.count == 3:
                if outcome == "death":
                    result |= {"terminated": True, "actions": []}
                    result["observation"]["player"]["hp"] = 0
                elif outcome == "next_act":
                    result["observation"]["act"] = 1
            return result

    env = Sts2Env(scope="act1_monsters", worker_factory=FailedWorker, max_steps=3)
    env.reset()
    for _ in range(3):
        _, reward, terminated, truncated, info = env.step(0)
    assert terminated == (outcome != "timeout")
    assert truncated == (outcome == "timeout")
    assert reward == (0 if truncated else -1)
    assert info["act1_monster_wins"] == 2 and not info["task_success"]
    report = summarise(env.drain_episodes())
    assert report["eligible"] == (not truncated)
    assert report["mean_surviving_health"] == 0
    assert report["mean_death_floor"] == (7 if outcome == "death" else None)
    env.close()


def test_health_quality_is_ranked_and_failed_runs_remain_in_the_score():
    env = Sts2Env(scope="act1_monsters", worker_factory=MonsterWorker)
    report = evaluate(None, env)
    assert report["complete"] and report["eligible"] and report["task_successes"] == 5
    assert report["selection_score"] == 0.25
    row = report["episodes"][0]
    better = row | {"hp": 75, "task_score": 0.75}
    assert summarise([better])["selection_score"] > summarise([row])["selection_score"]
    failure = row | {
        "hp": 0,
        "act1_monster_wins": 2,
        "task_success": False,
        "task_score": -1.0,
    }
    mixed = summarise([better, failure])
    assert mixed["selection_score"] == -0.125
    assert mixed["task_success_rate"] == 0.5 and mixed["mean_surviving_health"] == 0.375
    with pytest.raises(ValueError, match="score"):
        summarise([row | {"hp": 75}])
    with pytest.raises(ValueError, match="requires"):
        summarise([row | {"act1_monster_wins": 2}])
    env.close()


def test_monster_counter_is_replayed_and_excluded_from_policy_features():
    first = Sts2Env(scope="act1_monsters", worker_factory=MonsterWorker)
    second = Sts2Env(scope="act1_monsters", worker_factory=MonsterWorker)
    first.reset(seed=31)
    for _ in range(3):
        first.step(0)
    second.restore(first.snapshot())
    assert first.step(0)[1:] == second.step(0)[1:]
    original = first.state | {"audit": "test"}
    changed = copy.deepcopy(original) | {"act1_monster_wins": 2}
    assert state_digest(original) != state_digest(changed)
    assert signature(original) != signature(changed)
    for key, value in encode(original).items():
        np.testing.assert_array_equal(value, encode(changed)[key])
    assert (
        len({evaluation_plan(s)["evaluation_id"] for s in ("act1_monsters", "act1_elite", "act1")})
        == 3
    )
    first.close()
    second.close()


@pytest.mark.parametrize("hp,maximum", [(float("nan"), 80), (81, 80), (-1, 80), (0, 0), (True, 80)])
def test_invalid_health_is_rejected(hp, maximum):
    with pytest.raises(ValueError, match="health"):
        health_fraction({"hp": hp, "max_hp": maximum})
