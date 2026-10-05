import random

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
        trace.append((decision["kind"], len(decision["options"]), index, decision["state"]["hp"]))
        decision = kernel.call("step", index=index)
    return trace, decision


def test_characters(kernel):
    assert len(kernel.hello["characters"]) >= 5


def test_replay_is_deterministic(kernel):
    character = kernel.hello["characters"][0]
    first = play(kernel, character, "DETERMINISM", 400)
    second = play(kernel, character, "DETERMINISM", 400)
    assert first == second
    assert len(first[0]) > 20


def test_every_character_reaches_a_decision(kernel):
    for character in kernel.hello["characters"]:
        trace, decision = play(kernel, character, f"SMOKE-{character}", 60)
        assert trace or decision["kind"] == "end"
