import collections
import json
import time

from ai4sts2.environment import CHARACTERS, fingerprint, probe_action, seed_string, write_json
from ai4sts2.execution import REFERENCE
from ai4sts2.game import ROOT, OfficialGame, prepare_game


def equivalent(expected, actual):
    return all(
        expected[key] == actual[key] for key in ("observation", "audit", "terminated", "victory")
    )


def match_action(action, actions):
    if action["kind"] == "choose_card":
        matches = [
            index
            for index, item in enumerate(actions)
            if item["kind"] == "select_card"
            and not item["selected"]
            and item["card"] == action["card"]
        ]
    elif action["kind"] == "finish_selection":
        matches = [
            index for index, item in enumerate(actions) if item.get("control") == "NConfirmButton"
        ]
    elif action["kind"] == "skip_selection":
        matches = [
            index for index, item in enumerate(actions) if item.get("control") == "NBackButton"
        ]
        if not matches:
            matches = [
                index
                for index, item in enumerate(actions)
                if item.get("control") == "NConfirmButton"
            ]
    else:
        matches = [index for index, item in enumerate(actions) if item == action]
    if not matches:
        raise ValueError(f"No native action corresponds to {action}.")
    return matches[0]


def check_options(expected, actual):
    if expected["observation"].get("selection") is None:
        if expected["actions"] != actual["actions"]:
            raise ValueError("Native actions differ outside card selection.")
        return
    expected_cards = [a["card"] for a in expected["actions"] if a["kind"] == "choose_card"]
    actual_cards = [
        a["card"] for a in actual["actions"] if a["kind"] == "select_card" and not a["selected"]
    ]

    def canonical(values):
        return collections.Counter(json.dumps(value, sort_keys=True) for value in values)

    if canonical(expected_cards) != canonical(actual_cards):
        raise ValueError("Card selection does not expose all native candidates.")


def validate(minutes=5, cases=None):
    if not 0 < minutes <= 30:
        raise ValueError("Use a budget between zero and 30 minutes.")
    deadline = time.monotonic() + minutes * 60
    executable = prepare_game()
    cases = cases or [
        (hero, index, option) for index, hero in enumerate(CHARACTERS) for option in range(3)
    ]
    traces = []
    with OfficialGame(executable, execution=REFERENCE, audit=True) as semantic:
        for hero, seed, option in cases:
            parameters = {
                "character": hero,
                "seed": seed_string("validation", seed),
                "scope": "first_combat",
            }
            state = semantic.request("reset", parameters)
            trace = []
            for _ in range(128):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Selection validation budget exhausted.")
                if state["terminated"]:
                    trace.append({"state": state, "action": None})
                    break
                event = [
                    i
                    for i, a in enumerate(state["actions"])
                    if a.get("control") == "NEventOptionButton"
                ]
                finish = [
                    i
                    for i, a in enumerate(state["actions"])
                    if a["kind"] in {"finish_selection", "skip_selection"}
                ]
                index = (
                    event[min(option, len(event) - 1)]
                    if event
                    else finish[0]
                    if finish and option > 0
                    else probe_action(state["actions"])
                )
                action = state["actions"][index]
                trace.append({"state": state, "action": action})
                state = semantic.request("step", {"revision": state["revision"], "action": index})
            if not state["terminated"]:
                raise ValueError(f"Selection probe did not finish: {hero}, option {option}.")
            traces.append((parameters, trace))
            print(
                json.dumps({"selection_reference": [hero, option], "steps": len(trace) - 1}),
                flush=True,
            )
    results = []
    with OfficialGame(executable, execution=REFERENCE, audit=True, raw_selection=True) as native:
        for (hero, _, option), (parameters, trace) in zip(cases, traces, strict=True):
            actual = native.request("reset", parameters)
            selections = []
            for step, frame in enumerate(trace):
                if time.monotonic() >= deadline:
                    raise TimeoutError("Selection validation budget exhausted.")
                expected = frame["state"]
                for _ in range(4):
                    if equivalent(expected, actual):
                        break
                    confirmations = [
                        i
                        for i, a in enumerate(actual["actions"])
                        if a.get("control") == "NConfirmButton"
                    ]
                    if actual["observation"].get("selection") is None or not confirmations:
                        break
                    actual = native.request(
                        "step", {"revision": actual["revision"], "action": confirmations[0]}
                    )
                if not equivalent(expected, actual):
                    write_json(
                        ROOT / "artifacts/validation/selection-divergence.json",
                        {
                            "case": [hero, option],
                            "step": step,
                            "expected": expected,
                            "actual": actual,
                        },
                    )
                    raise ValueError(
                        f"Native selection differs: {hero}, option {option}, step {step}."
                    )
                try:
                    check_options(expected, actual)
                except ValueError:
                    write_json(
                        ROOT / "artifacts/validation/selection-divergence.json",
                        {
                            "case": [hero, option],
                            "step": step,
                            "expected": expected,
                            "actual": actual,
                        },
                    )
                    raise
                action = frame["action"]
                if action is None:
                    break
                if expected["observation"].get("selection"):
                    selections.append(expected["observation"]["selection"]["prompt"])
                index = match_action(action, actual["actions"])
                actual = native.request("step", {"revision": actual["revision"], "action": index})
            result = {
                "character": hero,
                "option": option,
                "steps": len(trace) - 1,
                "selections": selections,
                "valid": True,
            }
            results.append(result)
            print(json.dumps({"selection_check": result}), flush=True)
    report = {"build": fingerprint(), "cases": results, "valid": True}
    write_json(ROOT / "artifacts/validation/selections.json", report)
    return report
