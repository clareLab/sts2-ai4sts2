import copy

import numpy as np
import torch
from ai4sts2.environment import CHARACTERS, Sts2Env
from ai4sts2.signals import TrainingSignals, novelty_features
from test_environment import FakeWorker, state


def episode(character="IRONCLAD", floor=3, victory=False, truncated=False):
    return {
        "character": character,
        "floor": floor,
        "victory": victory,
        "truncated": truncated,
        "steps": 10,
    }


def test_signals_do_not_change_global_random_streams():
    torch.manual_seed(11)
    expected = torch.rand(4)
    torch.manual_seed(11)
    np.random.seed(13)
    numpy_state = np.random.get_state()
    TrainingSignals(20, {})
    torch.testing.assert_close(torch.rand(4), expected, rtol=0, atol=0)
    actual = np.random.get_state()
    np.testing.assert_array_equal(actual[1], numpy_state[1])
    assert actual[2:] == numpy_state[2:]


def test_novelty_ignores_turn_counters_hidden_data_and_unordered_piles():
    first = state()
    first["observation"]["turn"] = 1
    second = copy.deepcopy(first)
    second["observation"]["turn"] = 999
    second["observation"]["draw"].reverse()
    second["observation"]["rng"] = {"counter": 100}
    second["audit"] = "secret"
    np.testing.assert_array_equal(novelty_features(first), novelty_features(second))
    second["observation"]["player"]["hp"] -= 1
    assert not np.array_equal(novelty_features(first), novelty_features(second))


def test_exploration_cannot_reward_repeated_states_or_terminal_results():
    signals = TrainingSignals(5, {"rnd_scale": 0.001})
    original = state()
    signals.reset(original)
    assert signals.observe(original, episode(), False) == 0
    changed = copy.deepcopy(original)
    changed["observation"]["player"]["hp"] -= 1
    bonus = signals.observe(changed, episode(), False)
    assert 0 < bonus <= 0.005
    assert signals.observe(changed, episode(), False) == 0
    changed["observation"]["player"]["hp"] = 0
    assert signals.observe(changed, episode(), True) == 0


def test_predictor_learns_and_target_stays_frozen():
    signals = TrainingSignals(5, {"rnd_scale": 0.001})
    observation = state()
    target = copy.deepcopy(signals.rnd.target_network.state_dict())
    batch = signals.batch([novelty_features(observation)])
    before = float(signals.rnd(batch)["loss_predictor"].detach().item())
    for _ in range(256):
        signals.observe(observation, episode(), False)
    after = float(signals.rnd(batch)["loss_predictor"].detach().item())
    assert after < before
    assert signals.metrics["predictor_updates"] == 4
    for key, value in signals.rnd.target_network.state_dict().items():
        torch.testing.assert_close(value, target[key], rtol=0, atol=0)
    assert all(parameter.grad is None for parameter in signals.rnd.target_network.parameters())


def test_task_success_adapts_curriculum_without_floor_credit_or_starving_a_character():
    signals = TrainingSignals(5, {"curriculum_mix": 0.75})
    np.testing.assert_allclose(signals.probabilities(), np.full(5, 0.2))
    for _ in range(10):
        signals.observe(state(), episode(floor=100), True)
    assert signals.curriculum.task_rates[0] == 0
    for _ in range(10):
        signals.observe(state(), episode(floor=8) | {"task_success": True}, True)
    weights = signals.probabilities()
    assert weights[0] > weights[1]
    assert np.all(weights >= 0.05 - 1e-12)
    assert signals.curriculum.task_rates[0] == 1
    counts = signals.curriculum.completed_episodes
    signals.observe(state(), episode(floor=100, truncated=True), True)
    assert signals.curriculum.completed_episodes == counts


def test_curriculum_does_not_change_the_game_seed_stream():
    signals = TrainingSignals(5, {"curriculum_mix": 0.75})
    first = Sts2Env(seed=14, worker_factory=FakeWorker)
    second = Sts2Env(seed=14, worker_factory=FakeWorker, signals=signals)
    for _ in range(20):
        first.reset()
        second.reset()
        assert first.character == second.character
        assert first.journal["parameters"]["seed"] == second.journal["parameters"]["seed"]
    first.close()
    second.close()


def test_auxiliary_checkpoint_restores_learning_sampling_and_episode_memory():
    config = {"rnd_scale": 0.001, "curriculum_mix": 0.75}
    first = TrainingSignals(5, config)
    observation = state()
    for step in range(70):
        observation["observation"]["player"]["hp"] = step
        first.observe(observation, episode(), step % 10 == 0)
    saved = first.snapshot()
    second = TrainingSignals(99, config)
    second.restore(saved)
    for step in range(70, 145):
        observation["observation"]["player"]["hp"] = step
        assert first.observe(observation, episode(), False) == second.observe(
            observation, episode(), False
        )
    for key, value in first.rnd.state_dict().items():
        torch.testing.assert_close(value, second.rnd.state_dict()[key], rtol=0, atol=0)
    assert first.report() == second.report()
    assert first.seen == second.seen


def test_control_and_validation_keep_the_original_reward():
    signals = TrainingSignals(5, {})
    environment = Sts2Env(worker_factory=FakeWorker, signals=signals)
    environment.reset()
    for _ in range(3):
        _, reward, _, _, info = environment.step(0)
        assert reward == info["intrinsic_reward"] == 0
    _, reward, terminated, _, info = environment.step(0)
    assert reward == 1 and terminated
    assert info["episode"]["r"] == 1
    assert set(signals.report()["character_probabilities"]) == set(CHARACTERS)
    environment.close()
