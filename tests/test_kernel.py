import random

import numpy as np
import pytest

from ai4sts2.kernel import Kernel, engine

pytestmark = pytest.mark.game


@pytest.fixture(scope="module")
def kernel():
    with Kernel(engine()) as instance:
        yield instance


def play(kernel, character, seed, steps):
    rng = random.Random(seed)
    trace = []
    decision = kernel.call("begin", character=character, seed=seed, ascension=10)
    while decision["kind"] != "end" and len(trace) < steps:
        index = rng.randrange(len(decision["options"]))
        trace.append(
            (
                decision["kind"],
                tuple(decision["options"]),
                index,
                decision["state"]["hp"],
                fingerprint(kernel, decision),
            )
        )
        decision = kernel.call("step", index=index)
    return trace, decision


def fingerprint(kernel, decision):
    observation = decision["observation"]
    tables = []
    for name, values in observation.items():
        columns = kernel.schema[name]
        named = values.copy()
        for index, column in enumerate(columns):
            if column in {"model", "zone", "kind", "key", "type", "character", "room", "decision"}:
                named[:, index] = [hash(kernel.vocab[int(v)]) % 1000003 for v in values[:, index]]
        tables.append((name, named.shape, named.tobytes()))
    return tuple(tables)


def test_observation_matches_schema(kernel):
    decision = kernel.call(
        "begin", character=kernel.hello["characters"][0], seed="SCHEMA", ascension=10
    )
    observation = decision["observation"]
    assert set(observation) == set(kernel.schema)
    assert observation["global"].shape == (1, len(kernel.schema["global"]))
    assert observation["option"].shape[0] == len(decision["options"])
    assert np.isfinite(observation["card"]).all()


def test_characters(kernel):
    assert len(kernel.hello["characters"]) >= 5


def test_replay_is_deterministic(kernel):
    character = kernel.hello["characters"][0]
    first, _ = play(kernel, character, "DETERMINISM", 400)
    second, _ = play(kernel, character, "DETERMINISM", 400)
    assert first == second
    assert len(first) > 20


def test_every_character_reaches_a_decision(kernel):
    for character in kernel.hello["characters"]:
        trace, decision = play(kernel, character, f"SMOKE-{character}", 60)
        assert trace or decision["kind"] == "end"


def test_map_segments_match_official(monkeypatch):
    monkeypatch.setenv("AI4STS2_VERIFY_MAPS", "1")
    rng = random.Random(0)
    with Kernel(engine()) as instance:
        characters = instance.hello["characters"]
        for index in range(40):
            seed = f"MAPS-{index}"
            instance.call(
                "begin", character=characters[index % len(characters)], seed=seed, ascension=10
            )
        decision = instance.call(
            "begin", character=characters[0], seed="MAPS-ACTS", ascension=10, invincible=True
        )
        while decision["kind"] != "end":
            decision = instance.call("step", index=rng.randrange(len(decision["options"])))
        assert decision["state"]["act"] >= 2
