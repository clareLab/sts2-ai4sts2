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


def test_replay_is_deterministic(kernel):
    character = kernel.hello["characters"][0]
    first, _ = play(kernel, character, "DETERMINISM", 400)
    second, _ = play(kernel, character, "DETERMINISM", 400)
    assert first == second
    assert len(first) > 20


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
            other = second.call("begin", character=character, seed="OTHER", ascension=10)
            for _ in range(20):
                if other["kind"] != "end":
                    other = second.call("step", index=0)
            a = first.call("begin", invincible=True, **arguments)
            b = second.call("begin", invincible=True, **arguments)
            for _ in range(1000):
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
    while decision["kind"] != "end" and decision["state"]["combat"] and len(trace) < steps:
        decision = kernel.call("step", index=rng.randrange(len(decision["options"])))
        trace.append(fingerprint(kernel, decision))
    return trace


def wander(kernel, decision, seed, steps=30):
    rng = random.Random(seed)
    trace = []
    while decision["kind"] != "end" and len(trace) < steps:
        decision = kernel.call("step", index=rng.randrange(len(decision["options"])))
        trace.append(fingerprint(kernel, decision))
    return trace


def test_restore_after_the_run_ends(kernel):
    arguments = {"character": kernel.hello["characters"][1], "seed": "LETHAL", "ascension": 10}
    rng = random.Random(4)
    prefix = []
    decision = kernel.call("begin", **arguments)
    while not (decision["snapshot"] and decision["kind"] == "combat"):
        prefix.append(rng.randrange(len(decision["options"])))
        decision = kernel.call("step", index=prefix[-1])
    root = kernel.call("snapshot")["id"]
    for _ in range(5000):
        if decision["kind"] == "end":
            break
        decision = kernel.call("step", index=rng.randrange(len(decision["options"])))
    assert decision["kind"] == "end" and not decision["victory"]
    restored = kernel.call("restore", id=root)
    kernel.call("release", id=root)
    after = wander(kernel, restored, 6)
    kernel.call("begin", **arguments)
    assert wander(kernel, kernel.call("replay", actions=prefix), 6) == after


@pytest.mark.parametrize("which", range(5))
def test_snapshots_restore_every_kind_of_decision(kernel, which):
    character = kernel.hello["characters"][which]
    arguments = {"character": character, "seed": f"BRANCHES-{which}", "ascension": 10}
    rng = random.Random(11 + which)
    prefix, roots, refused = [], {}, set()
    decision = kernel.call("begin", invincible=True, **arguments)
    while decision["kind"] != "end" and len(prefix) < 900:
        if not decision["snapshot"]:
            refused.add(decision["kind"])
        elif len(decision["options"]) > 1 and decision["kind"] not in roots:
            snapshot = kernel.call("snapshot")["id"]
            roots[decision["kind"]] = (list(prefix), fingerprint(kernel, decision), snapshot)
        prefix.append(rng.randrange(len(decision["options"])))
        decision = kernel.call("step", index=prefix[-1])
    assert {"combat", "map", "rewards", "event"} <= set(roots)
    assert "select" in refused
    traces = {}
    for kind, (path, root, snapshot) in reversed(roots.items()):
        for _ in range(2):
            restored = kernel.call("restore", id=snapshot)
            assert fingerprint(kernel, restored) == root, kind
            trace = wander(kernel, restored, len(path))
            assert traces.setdefault(kind, trace) == trace, kind
    for kind, (path, root, _) in roots.items():
        kernel.call("begin", invincible=True, **arguments)
        replayed = kernel.call("replay", actions=path)
        assert fingerprint(kernel, replayed) == root, kind
        assert wander(kernel, replayed, len(path)) == traces[kind], kind


def test_snapshots_restore_in_place_and_match_replay(kernel):
    arguments = {"character": kernel.hello["characters"][1], "seed": "SNAPSHOT", "ascension": 10}
    rng = random.Random(7)
    prefix = []
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
    assert any(trace and json.loads(trace[-1][2])["combat"] is False for trace in traces.values())
    for seed, trace in traces.items():
        kernel.call("begin", **arguments)
        replayed = kernel.call("replay", actions=prefix)
        assert fingerprint(kernel, replayed) == root
        assert branch(kernel, replayed, seed) == trace


def test_resampling_covers_every_stream_drawn(monkeypatch):
    monkeypatch.setenv("AI4STS2_AUDIT", "1")
    rng = random.Random(13)
    audits, kinds = [], set()
    with Kernel(engine()) as instance:
        for character in instance.hello["characters"]:
            arguments = {"character": character, "seed": f"AUDIT-{character}", "ascension": 10}
            decision = instance.call("begin", invincible=True, **arguments)
            quota = {True: 2, False: 4}
            while decision["kind"] != "end" and any(quota.values()):
                combat = decision["kind"] == "combat"
                later = combat or decision["state"]["floor"] >= 3
                if decision["snapshot"] and later and quota[combat] and rng.random() < 0.5:
                    audits.append(instance.call("audit", seed=rng.getrandbits(62)))
                    kinds.add(decision["kind"])
                    quota[combat] -= 1
                decision = instance.call("step", index=rng.randrange(len(decision["options"])))
    assert audits and sum(audit["drawn"] > 0 for audit in audits) > len(audits) // 2
    assert {"combat", "map"} <= kinds and len(kinds) >= 4
    assert [audit["leaks"] for audit in audits if audit["leaks"]] == []


def replay(events, pulls):
    pointer = 0
    for pull in pulls:
        if pull["fits"] is None:
            pointer += 1
            continue
        while events[pointer] in pull["fits"]:
            pointer += 1
        assert events[pointer] not in pull["fits"]
        assert pull["replaced"] or events[pointer] == pull["seen"]
        pointer += pull["consumed"]
    return pointer


def cycles(sequence, start, size):
    return [sequence[i : i + size] for i in range(start, len(sequence), size)]


def test_determinize_conditions_hidden_rooms_on_what_was_seen(kernel):
    arguments = {"character": kernel.hello["characters"][2], "seed": "HIDDEN", "ascension": 10}
    rng = random.Random(5)
    decision = kernel.call("begin", invincible=True, **arguments)
    while True:
        quiet = decision["snapshot"] and decision["kind"] != "combat"
        if quiet and decision["state"]["floor"] >= 10:
            rooms = kernel.call("rooms")
            act = rooms["acts"][rooms["current"]]
            pulls = [p for p in act["pulls"] if p["fits"] is not None]
            if len(pulls) >= 2 and replay(act["events"], act["pulls"]) > len(act["pulls"]):
                break
        decision = kernel.call("step", index=rng.randrange(len(decision["options"])))
    root = kernel.call("snapshot")["id"]
    observed = fingerprint(kernel, decision)
    before = kernel.call("rooms")
    act = before["current"]
    real = before["acts"][act]
    assert real["normal_visited"] >= 3
    weak = 3
    regular = len(set(real["normal"][weak:]))
    elites = len(set(real["elite"]))
    assert replay(real["events"], real["pulls"]) == real["events_visited"]
    futures, pasts = set(), set()
    for seed in range(12):
        kernel.call("restore", id=root)
        assert fingerprint(kernel, kernel.call("determinize", seed=seed)) == observed
        acts = kernel.call("rooms")["acts"]
        now = acts[act]
        for key in ("normal", "elite"):
            visited = real[f"{key}_visited"]
            assert now[key][:visited] == real[key][:visited]
            assert all(
                a != b
                for a, b in zip(
                    now[key][max(visited, 1) - 1 :], now[key][max(visited, 1) :], strict=False
                )
            )
        assert all(len(set(c)) == len(c) for c in cycles(now["normal"][:weak], 0, weak))
        assert all(len(set(c)) == len(c) for c in cycles(now["normal"], weak, regular))
        assert all(len(set(c)) == len(c) for c in cycles(now["elite"], 0, elites))
        assert sorted(now["events"]) == sorted(real["events"])
        assert replay(now["events"], real["pulls"]) == now["events_visited"]
        pasts.add(tuple(now["events"][: now["events_visited"]]))
        assert (now["boss"], now["second"], now["ancient"]) == (
            real["boss"],
            real["second"],
            real["ancient"],
        )
        assert acts[:act] == before["acts"][:act]
        futures.add(tuple((a["boss"], a["second"], a["ancient"]) for a in acts[act + 1 :]))
    assert len(futures) > 1 and len(pasts) > 1
    kernel.call("restore", id=root)
    assert kernel.call("rooms") == before


def test_determinize_resamples_the_next_room(kernel):
    columns = {table: {c: i for i, c in enumerate(cols)} for table, cols in kernel.schema.items()}
    arguments = {"character": kernel.hello["characters"][0], "seed": "ROOMS", "ascension": 10}
    rng = random.Random(3)
    decision = kernel.call("begin", invincible=True, **arguments)
    checked = 0
    while decision["kind"] != "end" and checked < 2:
        observation = decision["observation"]
        nodes = [int(row[columns["option"]["node"]]) for row in observation["option"]]
        kinds = [
            kernel.vocab[int(observation["node"][n][columns["node"]["type"]])] if n >= 0 else None
            for n in nodes
        ]
        if decision["kind"] == "map" and decision["snapshot"] and "MapPointType.Monster" in kinds:
            root = kernel.call("snapshot")["id"]
            rooms = set()
            for seed in range(8):
                kernel.call("restore", id=root)
                kernel.call("determinize", seed=seed)
                entered = kernel.call("step", index=kinds.index("MapPointType.Monster"))
                assert entered["kind"] == "combat"
                creatures = entered["observation"]["creature"]
                rooms.add(
                    tuple(
                        sorted(
                            kernel.vocab[int(row[columns["creature"]["model"]])]
                            for row in creatures
                            if kernel.vocab[int(row[columns["creature"]["side"]])]
                            == "CombatSide.Enemy"
                        )
                    )
                )
            assert len(rooms) > 1
            decision = kernel.call("restore", id=root)
            checked += 1
        decision = kernel.call("step", index=rng.randrange(len(decision["options"])))
    assert checked == 2
