import copy
import time

import numpy as np
import pytest
from ai4sts2.environment import CHARACTERS, Sts2Env, evaluate, evaluation_plan
from test_environment import FakeWorker, state


class ChoiceWorker(FakeWorker):
    def request(self, method, parameters):
        super().request(method, parameters)
        result = state(self.count >= 500, False)
        if not result["terminated"]:
            result["actions"] = [
                {"kind": "select"},
                {"kind": "skip"},
                {"kind": "confirm", "control": "NConfirmButton"},
            ]
        return result


def test_diagnostic_random_actions_are_uniform_includes_skip_and_does_not_change_training_rng():
    environment = Sts2Env(worker_factory=ChoiceWorker, seed=12)
    rng = copy.deepcopy(environment.rng.bit_generator.state)
    np.random.seed(73)
    expected = np.random.random(4)
    np.random.seed(73)
    result = evaluate(None, environment, max_steps=600)
    calls = [
        parameters["action"] for method, parameters in environment.game.calls if method == "step"
    ]
    expected_actions = []
    for case in result["cases"]:
        generator = np.random.default_rng(case["action_seed"])
        expected_actions.extend(int(generator.choice([0, 1, 2])) for _ in range(500))
    assert calls == expected_actions
    assert all(700 < count < 1000 for count in np.bincount(calls, minlength=3))
    assert environment.rng.bit_generator.state == rng
    np.testing.assert_array_equal(np.random.random(4), expected)
    assert result["policy"] == "uniform_random" and result["eligible"]
    assert result["win_rate"] == 0
    environment.close()


def test_diagnostic_random_actions_replay_the_same_cases_and_trajectories():
    first = Sts2Env(worker_factory=FakeWorker, seed=0)
    second = Sts2Env(worker_factory=FakeWorker, seed=999)
    result = evaluate(None, first, per_character=2)
    assert result == evaluate(None, second, per_character=2)
    assert [case["character"] for case in result["cases"]] == list(CHARACTERS) * 2
    assert first.game.calls == second.game.calls
    assert all(e["trajectory_digest"] for e in result["episodes"])
    first.close()
    second.close()


def test_case_identity_covers_limits_seed_split_and_scope():
    original = evaluation_plan("run")
    for plan in (
        evaluation_plan("first_combat"),
        evaluation_plan("run", seed=1),
        evaluation_plan("run", split="test"),
        evaluation_plan("run", per_character=2),
        evaluation_plan("run", max_steps=100),
    ):
        assert original["evaluation_id"] != plan["evaluation_id"]
    assert original == evaluation_plan("run")


def test_partial_evaluation_is_saved_but_never_eligible():
    environment = Sts2Env(worker_factory=FakeWorker)
    reports = []
    evaluate(None, environment, on_episode=reports.append)
    assert len(reports) == 5
    assert all(not r["complete"] and not r["eligible"] for r in reports[:-1])
    assert all(r["selection_score"] == -1 for r in reports[:-1])
    assert reports[-1]["complete"] and reports[-1]["eligible"]
    environment.close()


def test_expired_evaluation_budget_cannot_submit_an_action():
    environment = Sts2Env(worker_factory=FakeWorker, max_steps=17)
    with pytest.raises(TimeoutError, match="budget"):
        evaluate(None, environment, deadline=time.monotonic() - 1)
    assert environment.game.calls == []
    assert environment.max_steps == 17
    environment.close()
