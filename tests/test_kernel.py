import json
import random
import re

import numpy as np
import pytest

from ai4sts2.features import CATEGORICAL
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
    tables = []
    for name, values in decision["observation"].items():
        named = values.astype(object)
        for index, column in enumerate(kernel.schema[name]):
            if column in CATEGORICAL or re.fullmatch(r"var\d+", column):
                named[:, index] = [kernel.vocab[int(v)] for v in values[:, index]]
        tables.append((name, named.shape, tuple(map(tuple, named))))
    state = json.dumps(decision["state"])
    return decision["kind"], tuple(decision["options"]), state, tuple(tables)


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


def test_processes_agree():
    rng = random.Random(1)
    with Kernel(engine()) as first, Kernel(engine()) as second:
        for character in first.hello["characters"]:
            arguments = {"character": character, "seed": f"PAIR-{character}", "ascension": 10}
            a = first.call("begin", invincible=True, **arguments)
            b = second.call("begin", invincible=True, **arguments)
            while True:
                assert fingerprint(first, a) == fingerprint(second, b)
                if a["kind"] == "end":
                    break
                index = rng.randrange(len(a["options"]))
                a, b = first.call("step", index=index), second.call("step", index=index)


def test_determinize_keeps_the_observation_and_resamples_the_future(kernel):
    arguments = {"character": kernel.hello["characters"][0], "seed": "RESAMPLE", "ascension": 10}
    rng = random.Random(5)
    prefix = []
    decision = kernel.call("begin", **arguments)
    while decision["kind"] != "combat" or "end_turn" not in decision["options"]:
        prefix.append(rng.randrange(len(decision["options"])))
        decision = kernel.call("step", index=prefix[-1])
    kernel.call("begin", **arguments)
    replayed = kernel.call("replay", actions=prefix)
    assert fingerprint(kernel, replayed) == fingerprint(kernel, decision)
    futures = []
    for seed in (1, 2, 3, 4, 1):
        kernel.call("begin", **arguments)
        kernel.call("replay", actions=prefix)
        resampled = kernel.call("determinize", seed=seed)
        assert fingerprint(kernel, resampled) == fingerprint(kernel, decision)
        after = kernel.call("step", index=resampled["options"].index("end_turn"))
        futures.append(fingerprint(kernel, after))
    assert futures[0] == futures[-1]
    assert len(set(futures)) > 1


def branch(kernel, decision, seed, steps=40):
    rng = random.Random(seed)
    trace = []
    while decision["kind"] not in ("end", "horizon") and len(trace) < steps:
        decision = kernel.call("step", index=rng.randrange(len(decision["options"])))
        trace.append(fingerprint(kernel, decision))
    return trace


def test_snapshots_restore_in_place_and_match_replay(kernel):
    arguments = {"character": kernel.hello["characters"][1], "seed": "SNAPSHOT", "ascension": 10}
    rng = random.Random(7)
    prefix = []
    kernel.call("horizon", enabled=True)
    try:
        decision = kernel.call("begin", **arguments)
        while decision["kind"] != "combat" or len(decision["options"]) < 3:
            prefix.append(rng.randrange(len(decision["options"])))
            decision = kernel.call("step", index=prefix[-1])
        root = fingerprint(kernel, decision)
        snapshot = kernel.call("snapshot")["id"]
        traces = {}
        for _ in range(3):
            for seed in range(4):
                restored = kernel.call("restore", id=snapshot)
                assert fingerprint(kernel, restored) == root
                trace = branch(kernel, restored, seed)
                assert traces.setdefault(seed, trace) == trace
        assert any(trace and trace[-1][0] in ("end", "horizon") for trace in traces.values())
        for seed, trace in traces.items():
            kernel.call("begin", **arguments)
            replayed = kernel.call("replay", actions=prefix)
            assert fingerprint(kernel, replayed) == root
            assert branch(kernel, replayed, seed) == trace
    finally:
        kernel.call("horizon", enabled=False)
