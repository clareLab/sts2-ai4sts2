import numpy as np
import pytest
from ai4sts2.environment import Sts2Env
from test_environment import FakeWorker, state


class RouteWorker(FakeWorker):
    def __init__(self, floors, victory, terminal=True):
        super().__init__()
        self.floors = floors
        self.victory = victory
        self.terminal = terminal

    def request(self, method, parameters):
        super().request(method, parameters)
        ended = self.terminal and self.count == len(self.floors) - 1
        result = state(ended, ended and self.victory)
        result["observation"]["floor"] = self.floors[self.count]
        return result


def route(floors, victory, discount, scale, terminal=True):
    return Sts2Env(
        worker_factory=lambda *_: RouteWorker(floors, victory, terminal),
        scope="run",
        max_steps=len(floors) - 1,
        discount=discount,
        progress_scale=scale,
    )


@pytest.mark.parametrize("discount", [0.99, 1.0])
@pytest.mark.parametrize("victory", [False, True])
@pytest.mark.parametrize("floors", [[0, 1, 3, 8], [0, 1, 1, 1, 3, 8], [4, 5, 4, 5, 9]])
def test_complete_returns_preserve_outcomes_and_remove_final_potential(discount, victory, floors):
    returns = []
    histories = []
    for scale in (0, 1):
        environment = route(floors, victory, discount, scale)
        environment.reset(seed=42, options={"character": "IRONCLAD"})
        total = 0
        for step in range(len(floors) - 1):
            _, reward, terminated, truncated, info = environment.step(0)
            total += discount**step * reward
        assert terminated and not truncated
        assert info["episode"]["r"] == (1 if victory else -1)
        returns.append(total)
        histories.append(environment.journal)
        environment.close()
    assert histories[0] == histories[1]
    assert returns[1] == pytest.approx(returns[0] - floors[0] / (1 + floors[0]))
    if discount == 1:
        assert returns[0] == (1 if victory else -1)


@pytest.mark.parametrize("discount", [0.99, 1.0])
def test_truncation_retains_potential_for_value_bootstrap(discount):
    floors = [2, 3, 5]
    initial = floors[0] / (1 + floors[0])
    final = floors[-1] / (1 + floors[-1])
    values = []
    for scale in (0, 1):
        environment = route(floors, False, discount, scale, terminal=False)
        environment.reset()
        rewards = []
        for _ in range(2):
            observation, reward, terminated, truncated, _ = environment.step(0)
            rewards.append(reward)
        assert truncated and not terminated
        assert environment.observation_space.contains(observation)
        assert environment.action_masks().any()
        values.append(sum(discount**i * r for i, r in enumerate(rewards)))
        environment.close()
    assert values[1] + discount**2 * (0.37 - final) == pytest.approx(
        values[0] + discount**2 * 0.37 - initial
    )


def test_undiscounted_idle_actions_have_no_progress_reward():
    environment = route([2, 2, 2, 3], False, 1, 1)
    environment.reset()
    for _ in range(2):
        _, reward, terminated, truncated, info = environment.step(0)
        assert reward == info["progress_reward"] == 0
        assert not terminated and not truncated
    environment.close()


def test_reward_configuration_and_future_feedback_survive_replay():
    source = route([2, 3, 4, 6], True, 1, 1)
    restored = route([2, 3, 4, 6], True, 0.99, 0)
    source.reset(seed=9)
    source.step(0)
    restored.restore(source.snapshot())
    assert restored.discount == 1 and restored.progress_scale == 1
    for _ in range(2):
        first, second = source.step(0), restored.step(0)
        assert first[1:] == second[1:]
        for key in first[0]:
            np.testing.assert_array_equal(first[0][key], second[0][key])
    source.close()
    restored.close()


@pytest.mark.parametrize(
    "parameters",
    [{"discount": 0}, {"discount": 1.1}, {"progress_scale": -1}, {"progress_scale": float("nan")}],
)
def test_invalid_reward_settings_do_not_start_a_worker(parameters):
    with pytest.raises(ValueError, match="reward"):
        Sts2Env(worker_factory=lambda *_: pytest.fail("Worker launched"), **parameters)
