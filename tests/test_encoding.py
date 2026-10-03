import copy

import numpy as np
import pytest
import torch
from ai4sts2.encoding import MAX_DEPTH, TreeNetwork, tree
from ai4sts2.environment import Sts2Env, encode
from ai4sts2.policy import SharedActionPolicy
from sb3_contrib import MaskablePPO
from test_environment import FakeWorker, state
from test_policy import probabilities


def test_tree_preserves_numeric_fields_and_nested_relationships():
    first = {"card": {"cost": 1}, "target": {"hp": 2}}
    second = {"target": {"hp": 1}, "card": {"cost": 2}}
    encoded = tree(first, 16)
    torch.manual_seed(17)
    network = TreeNetwork()
    for changed in (second, {"card": {"cost": 1, "hp": 2}}, {"card": {"cost": True}}):
        current = tree(changed, 16)
        assert not np.array_equal(current, encoded)
        assert not torch.allclose(
            network(torch.from_numpy(encoded[None])), network(torch.from_numpy(current[None]))
        )
    np.testing.assert_array_equal(tree(dict(reversed(list(first.items()))), 16), encoded)
    original = network(torch.from_numpy(encoded[None]))
    padded = network(torch.from_numpy(tree(first, 32)[None]))
    torch.testing.assert_close(original, padded)


def test_tree_fails_on_excess_capacity_and_invalid_numbers():
    with pytest.raises(ValueError, match="capacity"):
        tree({"cost": 1}, 1)
    nested = 1
    for _ in range(MAX_DEPTH + 1):
        nested = [nested]
    with pytest.raises(ValueError, match="capacity"):
        tree(nested, 64)
    for value in (np.nan, np.inf, -np.inf):
        with pytest.raises(ValueError, match="Non-finite"):
            tree({"cost": value}, 16)


def test_tree_keeps_hidden_information_out_and_legal_action_order_in():
    original = state()
    changed = copy.deepcopy(original)
    changed["seed"] = "hidden"
    changed["audit"] = "hidden"
    changed["observation"]["draw"].reverse()
    changed["observation"]["rng"] = 456
    changed["actions"][0]["internal_id"] = 998
    expected = encode(original, "tree")
    for name, value in encode(changed, "tree").items():
        np.testing.assert_array_equal(value, expected[name])
    changed["actions"].reverse()
    actual = encode(changed, "tree")
    np.testing.assert_array_equal(actual["actions"][:2], expected["actions"][:2][::-1])
    assert np.all(actual["actions"][2:] == 0)
    changed = copy.deepcopy(original)
    changed["observation"]["puzzle"] = {
        "cells": [{"row": 0, "column": 2, "hidden": True, "model": "GOLD"}]
    }
    hidden = encode(changed, "tree")
    changed["observation"]["puzzle"]["cells"][0]["model"] = "CURSE"
    np.testing.assert_array_equal(hidden["state"], encode(changed, "tree")["state"])


def test_structured_policy_learns_and_preserves_masks_order_and_checkpoints(tmp_path):
    torch.set_num_threads(1)
    environment = Sts2Env(worker_factory=FakeWorker, encoding="tree")
    observation, _ = environment.reset()
    assert environment.observation_space.contains(observation)
    model = MaskablePPO(
        SharedActionPolicy,
        environment,
        n_steps=16,
        batch_size=8,
        n_epochs=1,
        seed=27,
        policy_kwargs={"encoding": "tree"},
    )
    before = model.policy.features_extractor.encoder.node[0].weight.detach().clone()
    model.learn(32)
    assert not torch.equal(before, model.policy.features_extractor.encoder.node[0].weight)
    observations = {name: value[None] for name, value in observation.items()}
    masks = environment.action_masks()[None]
    expected, values = probabilities(model.policy, observations, masks)
    assert torch.all(expected[~masks] == 0)
    order = np.random.default_rng(8).permutation(128)
    reordered = observations | {"actions": observations["actions"][:, order]}
    actual, actual_values = probabilities(model.policy, reordered, masks[:, order])
    torch.testing.assert_close(actual, expected[:, order])
    torch.testing.assert_close(actual_values, values)
    model.save(tmp_path / "model.zip")
    restored = MaskablePPO.load(tmp_path / "model.zip", device="cpu")
    assert restored.policy.encoding == "tree"
    torch.testing.assert_close(
        probabilities(restored.policy, observations, masks)[0], expected, rtol=0, atol=0
    )
    model.policy.save(tmp_path / "policy.pt")
    restored_policy = SharedActionPolicy.load(tmp_path / "policy.pt", device="cpu")
    torch.testing.assert_close(
        probabilities(restored_policy, observations, masks)[0], expected, rtol=0, atol=0
    )
    environment.close()


def test_structured_checkpoint_replays_with_identical_encoding():
    original = Sts2Env(worker_factory=FakeWorker, encoding="tree")
    original.reset()
    original.step(1)
    snapshot = original.snapshot()
    restored = Sts2Env(worker_factory=FakeWorker, encoding="tree")
    restored.restore(snapshot)
    for key, value in original.encode().items():
        np.testing.assert_array_equal(value, restored.encode()[key])
    incompatible = Sts2Env(worker_factory=FakeWorker)
    with pytest.raises(ValueError, match="encoding"):
        incompatible.restore(snapshot)
    for environment in (original, restored, incompatible):
        environment.close()


def test_unknown_encoding_does_not_start_game():
    with pytest.raises(ValueError, match="encoding"):
        Sts2Env(encoding="unknown", worker_factory=lambda *_: pytest.fail("Worker launched"))
