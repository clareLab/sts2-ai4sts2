import copy

import numpy as np
import pytest
from ai4sts2.environment import encode
from ai4sts2.selection import check_options, match_action
from test_environment import state


def selection():
    value = state()
    value["observation"]["selection"] = {
        "minimum": 1,
        "maximum": 2,
        "prompt": "Select cards to discard.",
        "selected": [],
    }
    value["actions"] = [
        {"kind": "choose_card", "card": {"model": "DEFEND"}, "upgrade": {"amount": 8}},
        {"kind": "choose_card", "card": {"model": "DEFEND"}, "upgrade": {"amount": 8}},
    ]
    return value


def test_selection_context_and_upgrade_preview_reach_the_policy():
    original = selection()
    for key, value in (
        ("minimum", 0),
        ("maximum", 1),
        ("skippable", True),
        ("selected", [{"model": "BASH"}]),
        ("prompt", "Select cards to upgrade."),
    ):
        changed = copy.deepcopy(original)
        changed["observation"]["selection"][key] = value
        assert not np.array_equal(encode(original)["state"], encode(changed)["state"])
    changed = copy.deepcopy(original)
    changed["actions"][0]["upgrade"]["amount"] = 5
    assert not np.array_equal(encode(original)["actions"], encode(changed)["actions"])


def test_native_comparison_preserves_distinct_copies_of_a_card():
    expected = selection()
    native = copy.deepcopy(expected)
    native["actions"] = [
        {"kind": "select_card", "card": action["card"], "selected": False}
        for action in expected["actions"]
    ]
    check_options(expected, native)
    native["actions"].pop()
    with pytest.raises(ValueError, match="all native candidates"):
        check_options(expected, native)


def test_reference_driver_does_not_deselect_an_existing_choice():
    action = selection()["actions"][0]
    native = [
        {"kind": "select_card", "card": action["card"], "selected": True},
        {"kind": "select_card", "card": action["card"], "selected": False},
    ]
    assert match_action(action, native) == 1
    with pytest.raises(ValueError, match="No native action"):
        match_action(action, native[:1])


def test_empty_selection_uses_the_native_exit_or_confirmation():
    action = {"kind": "skip_selection"}
    assert match_action(action, [{"control": "NBackButton"}]) == 0
    assert match_action(action, [{"control": "NConfirmButton"}]) == 0
    with pytest.raises(ValueError, match="No native action"):
        match_action(action, [{"control": "NTickbox"}])


def test_known_preview_and_random_outcomes_remain_distinct():
    original = selection()
    original["actions"][0]["preview"] = {
        "kind": "transform",
        "random": True,
        "options": ["BASH", "MAUL"],
    }
    changed = copy.deepcopy(original)
    changed["actions"][0]["preview"] = {
        "kind": "transform",
        "random": False,
        "card": {"model": "MAUL"},
    }
    assert not np.array_equal(encode(original)["actions"], encode(changed)["actions"])
