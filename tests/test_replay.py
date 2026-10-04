import numpy as np
import pytest
import torch
from ai4sts2.replay import episode_returns, self_imitation_loss


@pytest.mark.parametrize("terminal", [-1.0, 1.0])
def test_complete_episode_return_reaches_every_preceding_action(terminal):
    rewards = np.zeros(257, dtype=np.float32)
    rewards[-1] = terminal
    np.testing.assert_array_equal(episode_returns(rewards, 1.0), np.full(257, terminal))
    np.testing.assert_allclose(
        episode_returns(rewards, 0.99), terminal * 0.99 ** np.arange(256, -1, -1), rtol=1e-6
    )
    assert rewards[:-1].sum() == 0


def test_positive_advantage_weights_policy_without_leaking_into_value_gradient():
    values = torch.tensor([-0.5, 0.7], requires_grad=True)
    log_probabilities = torch.tensor([-0.4, -0.2], requires_grad=True)
    returns = torch.tensor([1.0, -1.0], requires_grad=True)
    loss, metrics = self_imitation_loss(values, log_probabilities, returns)
    assert loss.item() == pytest.approx(0.3 + 0.28125)
    assert metrics["positive_advantage_fraction"].item() == 0.5
    loss.backward()
    torch.testing.assert_close(log_probabilities.grad, torch.tensor([-0.75, 0.0]))
    torch.testing.assert_close(values.grad, torch.tensor([-0.375, 0.0]))
    assert returns.grad is None


def test_worse_than_expected_experience_produces_no_update():
    values = torch.tensor([0.0, 1.0], requires_grad=True)
    log_probabilities = torch.tensor([-0.4, -0.2], requires_grad=True)
    loss, metrics = self_imitation_loss(values, log_probabilities, torch.tensor([-1.0, 1.0]))
    loss.backward()
    assert loss.item() == metrics["positive_advantage_fraction"].item() == 0
    assert torch.equal(values.grad, torch.zeros(2))
    assert torch.equal(log_probabilities.grad, torch.zeros(2))


@pytest.mark.parametrize("rewards", [[], [[1]], [float("nan")], [float("inf")]])
def test_invalid_episode_rewards_are_rejected(rewards):
    with pytest.raises(ValueError):
        episode_returns(rewards, 1.0)


@pytest.mark.parametrize("discount", [0, -1, 1.1, float("nan")])
def test_invalid_discount_is_rejected(discount):
    with pytest.raises(ValueError):
        episode_returns([0, 1], discount)


@pytest.mark.parametrize("values", [[], [[1.0]], [float("nan")], [float("inf")]])
def test_invalid_self_imitation_inputs_are_rejected(values):
    with pytest.raises(ValueError):
        self_imitation_loss(torch.tensor(values), torch.zeros(1), torch.ones(1))


def test_torchrl_replay_increases_successful_legal_action_probability():
    from ai4sts2.environment import Sts2Env
    from ai4sts2.policy import SharedActionPolicy
    from sb3_contrib import MaskablePPO
    from tensordict import TensorDict
    from test_environment import FakeWorker
    from torchrl.data import TensorDictReplayBuffer, TensorStorage

    torch.set_num_threads(1)
    environment = Sts2Env(worker_factory=FakeWorker)
    try:
        model = MaskablePPO(
            SharedActionPolicy,
            environment,
            n_steps=8,
            batch_size=8,
            seed=9,
            policy_kwargs={"temperature": 0.5},
        )
        observation, _ = environment.reset(seed=11)
        observations = {
            key: torch.tensor(value)[None].repeat(8, 1, *([1] * (value.ndim - 1)))
            for key, value in observation.items()
        }
        masks = torch.tensor(environment.action_masks())[None].repeat(8, 1)
        data = TensorDict(
            {
                "observation": TensorDict(observations, batch_size=[8]),
                "action": torch.zeros(8, dtype=torch.long),
                "action_mask": masks,
                "return": torch.ones(8),
            },
            batch_size=[8],
        )
        buffer = TensorDictReplayBuffer(
            storage=TensorStorage(data), batch_size=8, generator=torch.Generator().manual_seed(7)
        )
        with torch.no_grad():
            before = model.policy.get_distribution(
                observations, action_masks=masks
            ).distribution.probs.clone()
        for _ in range(8):
            batch = buffer.sample()
            values, probabilities, _ = model.policy.evaluate_actions(
                dict(batch["observation"]), batch["action"], action_masks=batch["action_mask"]
            )
            loss, _ = self_imitation_loss(values.flatten(), probabilities, batch["return"])
            model.policy.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                model.policy.parameters(), model.max_grad_norm, error_if_nonfinite=True
            )
            model.policy.optimizer.step()
        with torch.no_grad():
            after = model.policy.get_distribution(
                observations, action_masks=masks
            ).distribution.probs
        assert torch.all(after[:, 0] > before[:, 0])
        assert torch.all(after[~masks] == 0)
        assert len(buffer) == 8 and model.num_timesteps == 0
    finally:
        environment.close()
