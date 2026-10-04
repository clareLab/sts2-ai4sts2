import copy

import numpy as np
import pytest
from ai4sts2.environment import MAX_ACTIONS, Sts2Env, encode, probe_action, seed_string


def state(terminated=False, victory=False):
    return {
        "revision": 1,
        "observation": {
            "ascension": 10,
            "floor": 2,
            "act": 0,
            "player": {"hp": 50, "max_hp": 80},
            "hand": [{"model": "BASH", "cost": 2}],
            "draw": [
                {"model": "STRIKE_IRONCLAD", "cost": 1, "enchantment": "A"},
                {"model": "STRIKE_IRONCLAD", "cost": 1, "enchantment": "B"},
            ],
        },
        "actions": [] if terminated else [{"kind": "play"}, {"kind": "end_turn"}],
        "terminated": terminated,
        "victory": victory,
    }


class FakeWorker:
    def __init__(self, _=None):
        self.calls = []
        self.closed = False
        self.count = 0

    def request(self, method, parameters):
        self.calls.append((method, parameters))
        if method == "reset":
            self.count = 0
            self.ascension = parameters.get("ascension", 10)
        else:
            self.count += 1
        result = state(self.count >= 4, self.count >= 4)
        result["observation"]["ascension"] = self.ascension
        return result

    def close(self):
        self.closed = True

    def drain_measurements(self):
        return {}


def test_hidden_state_and_draw_order_do_not_change_policy_input():
    original = state()
    changed = copy.deepcopy(original)
    changed["seed"] = "secret"
    changed["audit"] = "native-secret"
    changed["timing"] = {"rng_dependent_ms": 55}
    changed["revision"] = 98765
    changed["observation"]["rng"] = {"counter": 991}
    changed["observation"]["player"]["native_state"] = {"secret": 55}
    changed["observation"]["draw"].reverse()
    changed["actions"][0]["internal_id"] = "hidden"
    for key, value in encode(original).items():
        np.testing.assert_array_equal(value, encode(changed)[key])


def test_public_state_changes_are_retained():
    original = state()
    changed = copy.deepcopy(original)
    changed["observation"]["player"]["hp"] = 1
    assert not np.array_equal(encode(original)["state"], encode(changed)["state"])


def test_visible_route_and_purchase_information_reaches_the_policy():
    original = state()
    original["observation"]["map"] = [
        {"row": 1, "column": 0, "room": "Shop", "children": [{"row": 2, "column": 1}]}
    ]
    original["actions"] = [
        {"kind": "buy", "cost": 75, "relic": {"model": "ANCHOR", "description": "Gain Block."}}
    ]
    changed = copy.deepcopy(original)
    changed["observation"]["map"][0]["children"][0]["column"] = 2
    changed["actions"][0]["cost"] = 100
    assert not np.array_equal(encode(original)["state"], encode(changed)["state"])
    assert not np.array_equal(encode(original)["actions"], encode(changed)["actions"])
    changed = copy.deepcopy(original)
    changed["observation"]["map"][0]["hidden_encounter"] = "secret"
    np.testing.assert_array_equal(encode(original)["state"], encode(changed)["state"])


def test_unrevealed_puzzle_contents_never_enter_policy_features():
    original = state()
    original["observation"]["puzzle"] = {
        "cells": [{"row": 1, "column": 2, "hidden": True, "model": "GOLD"}]
    }
    changed = copy.deepcopy(original)
    changed["observation"]["puzzle"]["cells"][0]["model"] = "CURSE"
    changed["observation"]["puzzle"]["cells"][0]["description"] = "Secret item"
    np.testing.assert_array_equal(encode(original)["state"], encode(changed)["state"])
    changed["observation"]["puzzle"]["cells"][0]["hidden"] = False
    assert not np.array_equal(encode(original)["state"], encode(changed)["state"])


def test_full_run_scope_is_explicit_and_invalid_scope_does_not_launch_a_worker():
    environment = Sts2Env(scope="run", worker_factory=FakeWorker)
    environment.reset(seed=0, options={"character": "IRONCLAD"})
    assert environment.game.calls[0][1]["scope"] == "run"
    environment.close()
    with pytest.raises(ValueError, match="scope"):
        Sts2Env(scope="invalid", worker_factory=lambda *_: pytest.fail("Worker launched"))


def test_probe_confirms_selection_and_preserves_visible_selected_state():
    original = state()
    original["actions"] = [{"kind": "select_card", "selected": False}]
    changed = copy.deepcopy(original)
    changed["actions"][0]["selected"] = True
    assert not np.array_equal(encode(original)["actions"], encode(changed)["actions"])
    actions = changed["actions"] + [{"kind": "select_card", "selected": False}]
    assert probe_action(actions) == 1
    actions += [{"kind": "select", "control": "NConfirmButton"}]
    assert probe_action(actions) == 2


def test_capacity_excess_is_not_silently_pruned():
    value = state()
    value["actions"] *= MAX_ACTIONS
    with pytest.raises(ValueError, match="capacity"):
        encode(value)


def test_missing_decision_is_an_error():
    value = state()
    value["actions"] = []
    with pytest.raises(ValueError, match="no legal action"):
        encode(value)
    encode(state(True))


def test_seed_splits_are_disjoint_and_repeatable():
    sets = [
        {seed_string(split, i) for i in range(100)} for split in ("train", "validation", "test")
    ]
    assert len(set.union(*sets)) == 300
    assert seed_string("train", 7) == seed_string("train", 7)


def test_validation_does_not_reseed_training_stream():
    first = Sts2Env(seed=7, worker_factory=FakeWorker)
    second = Sts2Env(seed=7, worker_factory=FakeWorker)
    first.reset(seed=5, options={"character": "IRONCLAD", "split": "validation"})
    first.reset()
    second.reset()
    assert first.game.calls[-1] == second.game.calls[-1]


@pytest.mark.parametrize("action", [-1, 2, 128, 0.5])
def test_illegal_actions_never_reach_game(action):
    env = Sts2Env(worker_factory=FakeWorker)
    observation, _ = env.reset()
    assert env.observation_space.contains(observation)
    assert env.action_masks().sum() == 2
    with pytest.raises(ValueError, match="Illegal"):
        env.step(action)
    assert len(env.game.calls) == 1


def test_budget_truncation_is_not_given_a_death_penalty():
    env = Sts2Env(max_steps=1, worker_factory=FakeWorker)
    env.reset()
    _, reward, terminated, truncated, info = env.step(0)
    assert not terminated and truncated and not info["victory"] and reward == 0


def test_terminal_reward_and_lifecycle():
    env = Sts2Env(worker_factory=FakeWorker)
    env.reset()
    for _ in range(4):
        _, reward, terminated, truncated, info = env.step(0)
    assert terminated and not truncated and reward == 1 and info["victory"]
    with pytest.raises(ValueError):
        env.step(0)
    env.close()
    assert env.game.closed
