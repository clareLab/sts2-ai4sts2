import collections
import json
import random
import time
from datetime import UTC, datetime

from ai4sts2.environment import CHARACTERS, encode, fingerprint, seed_string, write_json
from ai4sts2.execution import REFERENCE
from ai4sts2.game import ROOT, OfficialGame, prepare_game


def probe_decision(state, rng, coverage):
    observation = state["observation"]
    player = observation["player"]
    actions = state["actions"]
    incoming = sum(
        (intent.get("damage") or 0) * (intent.get("repeats") or 1)
        for creature in observation.get("creatures", [])
        if creature.get("side") == "Enemy" and creature["hp"] > 0
        for intent in creature.get("intents") or []
    )

    def score(action):
        kind = action["kind"]
        if kind == "map":
            room = action["room"]
            return 100 - 10 * coverage["room:" + room] - (30 if room == "Elite" else 0)
        if kind == "use_potion":
            return 90
        if kind == "play":
            card = action["card"]
            variables = {v["model"]: v["amount"] for v in card.get("variables", [])}
            damage = variables.get("Damage", 0)
            target = action.get("target")
            lethal = target and damage >= target["hp"] + target.get("block", 0)
            block = min(variables.get("Block", 0), max(0, incoming - player.get("block", 0)))
            return 50 + 100 * bool(lethal) + damage + 1.2 * block - 2 * card.get("cost", 0)
        if kind == "buy":
            return 60 - 10 * coverage["buy:" + action["type"]]
        if kind in {"reward", "take_relic"}:
            return 50
        if kind == "rest":
            healing = action.get("model", "").lower() == "heal"
            return 70 if healing and player["hp"] < player["max_hp"] * 0.7 else 50
        if kind in {"choose_card", "select_card"}:
            return 40
        if kind in {"finish_selection", "skip_selection"}:
            return 30
        if kind == "reward_alternative":
            return 20
        if kind == "event":
            return 40
        if kind == "discard_potion":
            return -20
        if kind in {"end_turn", "proceed", "leave_shop"}:
            return 0
        return 10

    ranks = [(score(action), rng.random(), index) for index, action in enumerate(actions)]
    return max(ranks)[2]


def check(minutes=5, episodes=5, steps=2048, seed=0, diagnostic=False, diagnostic_event=None):
    if not 0 < minutes <= 30 or episodes < 1 or steps < 1:
        raise ValueError("Use a positive episode/step count and a budget of at most 30 minutes.")
    if diagnostic_event and not diagnostic:
        raise ValueError("A diagnostic event requires diagnostic mode.")
    deadline = time.monotonic() + minutes * 60
    directory = ROOT / "artifacts/validation" / datetime.now(UTC).strftime("run-%Y%m%d-%H%M%S")
    directory.mkdir(parents=True)
    report = {
        "build": fingerprint("diagnostic_run" if diagnostic else "run"),
        "scope": "run",
        "diagnostic": diagnostic,
        "diagnostic_event": diagnostic_event,
        "certifying": False,
        "policy": "coverage_probe",
        "execution": REFERENCE.to_dict(),
        "requested_episodes": episodes,
        "episodes": [],
        "errors": [],
        "valid": False,
        "complete": False,
        "full_run_coverage": False,
    }
    name = "diagnostic-run.json" if diagnostic else "run.json"

    def save():
        write_json(directory / "report.json", report)
        write_json(ROOT / "artifacts/validation" / name, report | {"path": str(directory)})

    save()
    coverage = collections.Counter()
    executable = prepare_game()
    with (directory / "steps.jsonl").open("w") as stream:
        for episode in range(episodes):
            if time.monotonic() >= deadline:
                break
            hero = CHARACTERS[episode % len(CHARACTERS)]
            parameters = {
                "character": hero,
                "seed": seed_string("validation", seed + episode),
                "scope": "run",
            }
            state = None
            result = {"character": hero, "seed": parameters["seed"], "terminated": False}
            rng = random.Random(seed + episode)
            previous = None
            repeated = 0
            try:
                with OfficialGame(
                    executable,
                    execution=REFERENCE,
                    timeout=min(75, deadline - time.monotonic()),
                    diagnostic=diagnostic,
                    diagnostic_event=diagnostic_event,
                ) as game:
                    state = game.request("reset", parameters)
                    for step in range(steps + 1):
                        encode(state)
                        observation = state["observation"]
                        coverage["screen:" + observation["screen"]] += 1
                        coverage["act:" + str(observation["act"])] += 1
                        result |= {
                            "steps": step,
                            "floor": observation["floor"],
                            "act": observation["act"],
                            "hp": observation["player"]["hp"],
                            "terminated": state["terminated"],
                            "victory": state["victory"],
                        }
                        index = None
                        if not state["terminated"] and step < steps and time.monotonic() < deadline:
                            index = probe_decision(state, rng, coverage)
                        stream.write(
                            json.dumps({"episode": episode, "state": state, "action": index}) + "\n"
                        )
                        stream.flush()
                        if index is None:
                            break
                        action = state["actions"][index]
                        coverage["action:" + action["kind"]] += 1
                        if action["kind"] == "map":
                            coverage["room:" + action["room"]] += 1
                        if action["kind"] == "buy":
                            coverage["buy:" + action["type"]] += 1
                        visible = json.dumps([observation, state["actions"]], sort_keys=True)
                        repeated = repeated + 1 if visible == previous else 0
                        previous = visible
                        if repeated >= 8:
                            raise RuntimeError(
                                "The probe made no observable progress for eight steps."
                            )
                        game.timeout = min(75, max(0.01, deadline - time.monotonic()))
                        state = game.request(
                            "step", {"revision": state["revision"], "action": index}
                        )
            except (RuntimeError, ValueError, TimeoutError) as error:
                result["error"] = str(error)
                report["errors"].append({"episode": episode, "error": str(error)})
                write_json(
                    directory / f"failure-{episode}.json",
                    {"parameters": parameters, "state": state, "error": str(error)},
                )
            report["episodes"].append(result)
            print(json.dumps({"run_check": result}), flush=True)
            report["coverage"] = dict(sorted(coverage.items()))
            save()
    report["valid"] = (
        not report["errors"]
        and len(report["episodes"]) == episodes
        and all(result["terminated"] for result in report["episodes"])
    )
    required = {
        "map": "action:map",
        "rewards": "action:reward",
        "shop": "action:buy",
        "rest_site": "action:rest",
        "treasure": "action:take_relic",
        "second_act": "act:1",
        "third_act": "act:2",
    }
    report["missing_coverage"] = [name for name, key in required.items() if not coverage[key]]
    report["full_run_coverage"] = (
        report["valid"]
        and not report["missing_coverage"]
        and any(result.get("victory") for result in report["episodes"])
    )
    report["complete"] = len(report["episodes"]) == episodes
    save()
    return report
