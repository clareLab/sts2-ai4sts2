import copy

import numpy as np
import pytest
import torch
from ai4sts2.environment import Sts2Env, evaluate
from ai4sts2.metrics import summarise
from ai4sts2.signals import TrainingSignals
from test_checkpoint import member
from test_rewards import RouteWorker


def curriculum(goals=(2, 4), minimum=2):
    return TrainingSignals(
        5, {"floor_goals": goals, "goal_min_episodes": minimum, "goal_success_rate": 0.75}
    )


def environment(signals, floors=(0, 1, 2, 3, 4), *, victory=False, terminal=False):
    return Sts2Env(
        scope="run",
        signals=signals,
        worker_factory=lambda *_: RouteWorker(floors, victory, terminal),
        max_steps=len(floors) - 1,
    )


def test_reaching_a_goal_ends_training_without_claiming_game_victory():
    signals = curriculum()
    env = environment(signals)
    env.reset()
    _, reward, terminated, truncated, _ = env.step(0)
    assert reward == 0 and not terminated and not truncated
    _, reward, terminated, truncated, info = env.step(0)
    assert reward == 1 and terminated and not truncated
    assert info["goal_success"] and not info["victory"]
    assert info["goal_floor"] == 2 and not env.state["terminated"]
    report = summarise(env.drain_episodes())
    assert report["curriculum_successes"] == 1 and report["wins"] == 0
    assert report["mean_death_floor"] is None
    assert not report["eligible"] and report["selection_score"] == -1
    with pytest.raises(ValueError, match="Illegal"):
        env.step(0)
    assert env.game.count == 2
    env.close()


def test_curriculum_advances_only_after_sufficient_success_and_finishes_on_full_runs():
    signals = curriculum()
    env = environment(signals)
    for expected in (2, 2, 4, 4):
        env.reset()
        assert env.goal_floor == expected
        for _ in range(expected):
            result = env.step(0)
        assert result[2] and result[4]["goal_success"]
    assert signals.goal() == 0
    env.reset()
    for _ in range(4):
        result = env.step(0)
    assert result[3] and not result[2] and not result[4]["goal_success"]
    assert signals.floor_curriculum.total_episodes == 4
    assert signals.curriculum.completed_episodes == 0
    env.close()


@pytest.mark.parametrize("victory", [False, True])
def test_natural_game_result_takes_precedence_over_goal_crossing(victory):
    env = environment(curriculum(), (0, 2), victory=victory, terminal=True)
    env.reset()
    _, reward, terminated, truncated, info = env.step(0)
    assert terminated and not truncated
    assert reward == (1 if victory else -1)
    assert info["goal_success"] == info["victory"] == victory
    env.close()


def test_failure_and_timeout_do_not_advance_curriculum():
    signals = curriculum()
    env = environment(signals, (0, 1), terminal=True)
    for _ in range(3):
        env.reset()
        env.step(0)
    assert signals.goal() == 2
    assert signals.floor_curriculum.total_episodes == 3
    env.close()
    env = environment(signals, (0, 1))
    env.reset()
    assert env.step(0)[3]
    assert signals.floor_curriculum.total_episodes == 3
    env.close()


def test_evaluation_rejects_training_goals_before_requesting_a_game_reset():
    env = environment(curriculum())
    with pytest.raises(ValueError, match="evaluation"):
        evaluate(None, env)
    assert not env.game.calls
    with pytest.raises(ValueError, match="full-run"):
        Sts2Env(signals=curriculum(), worker_factory=lambda *_: pytest.fail("Worker launched"))
    env.close()


def test_goals_do_not_change_features_actions_or_seed_streams():
    first, second = environment(None), environment(curriculum())
    for _ in range(2):
        observation, _ = first.reset()
        actual, _ = second.reset()
        for key in observation:
            np.testing.assert_array_equal(observation[key], actual[key])
        assert first.journal == second.journal
        np.testing.assert_array_equal(first.action_masks(), second.action_masks())
        for _ in range(2):
            first.step(0)
            second.step(0)
        assert first.journal == second.journal
    first.close()
    second.close()


def test_environment_and_curriculum_resume_mid_goal_and_after_success():
    first, second = environment(curriculum()), environment(curriculum())
    first.reset()
    first.step(0)
    second.restore(first.snapshot())
    second.signals.restore(first.signals.snapshot())
    assert first.step(0)[1:] == second.step(0)[1:]
    second.restore(first.snapshot())
    second.signals.restore(first.signals.snapshot())
    with pytest.raises(ValueError, match="Illegal"):
        second.step(0)
    for _ in range(3):
        first.reset()
        second.reset()
        for _ in range(first.goal_floor):
            assert first.step(0)[1:] == second.step(0)[1:]
    assert first.signals.report() == second.signals.report()
    assert first.snapshot() == second.snapshot()
    first.close()
    second.close()


@pytest.mark.parametrize(
    "settings",
    [
        {"floor_goals": [2, 2]},
        {"floor_goals": [4, 2]},
        {"floor_goals": [True]},
        {"floor_goals": [1]},
        {"floor_goals": [2.5]},
        {"goal_min_episodes": 0},
        {"goal_success_rate": float("nan")},
        {"goal_success_rate": 1.1},
    ],
)
def test_invalid_curriculum_settings_are_rejected(settings):
    with pytest.raises(ValueError, match="curriculum"):
        TrainingSignals(5, settings)


def test_active_curriculum_cannot_silently_change_on_resume():
    signals = curriculum()
    with pytest.raises(ValueError, match="active"):
        signals.configure({"floor_goals": [2, 8]})


def test_legacy_checkpoint_keeps_ongoing_run_until_next_training_reset():
    original = environment(None)
    original.reset()
    original.step(0)
    snapshot = original.snapshot()
    snapshot.pop("goal_floor")
    receiver = environment(curriculum())
    receiver.restore(snapshot)
    for _ in range(3):
        result = receiver.step(0)
    assert result[3] and "goal_floor" not in result[4]
    assert receiver.signals.floor_curriculum.total_episodes == 0
    receiver.reset()
    assert receiver.goal_floor == 2
    original.close()
    receiver.close()


def test_ppo_goal_training_resumes_identical_weights_and_curriculum(monkeypatch, tmp_path):
    import ai4sts2.train as training

    donor = member(monkeypatch, policy="shared")
    donor.cleanup()
    monkeypatch.setattr(
        training,
        "Sts2Env",
        lambda *args, signals=None, **kwargs: environment(
            signals, tuple(index // 16 for index in range(81)), terminal=True
        ),
    )
    donor.config |= {"floor_goals": [2, 4], "goal_min_episodes": 2}
    donor.setup(donor.config)
    donor.model.learn(total_timesteps=64)
    donor.save_checkpoint(tmp_path)
    donor.model.learn(total_timesteps=192, reset_num_timesteps=False)
    weights = copy.deepcopy(donor.model.policy.state_dict())
    snapshot = donor.environment.snapshot()
    report = donor.signals.report()
    donor.load_checkpoint(tmp_path)
    donor.model.learn(total_timesteps=192, reset_num_timesteps=False)
    assert donor.environment.snapshot() == snapshot
    assert donor.signals.report() == report
    for key, weight in weights.items():
        torch.testing.assert_close(donor.model.policy.state_dict()[key], weight, rtol=0, atol=0)
    validation = donor.open_environment(training=False)
    validation.reset()
    assert validation.goal_floor is None and validation.signals is None
    validation.close()
    donor.cleanup()
