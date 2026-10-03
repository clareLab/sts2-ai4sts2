import numpy as np
import pytest
import torch
from ai4sts2.environment import Sts2Env
from ai4sts2.policy import SharedActionPolicy
from sb3_contrib import MaskablePPO
from test_environment import FakeWorker


def sample():
    rng = np.random.default_rng(19)
    observations = {
        "state": rng.uniform(-1, 1, (3, 512)).astype(np.float32),
        "actions": np.zeros((3, 128, 64), dtype=np.float32),
    }
    for row, count in enumerate((2, 5, 17)):
        observations["actions"][row, :count] = rng.uniform(-1, 1, (count, 64))
    masks = np.any(observations["actions"] != 0, axis=-1)
    return observations, masks


def probabilities(policy, observations, masks):
    with torch.no_grad():
        tensors = policy.obs_to_tensor(observations)[0]
        values = policy.predict_values(tensors)
        distribution = policy.get_distribution(tensors, action_masks=masks)
        return distribution.distribution.probs.clone(), values


@pytest.mark.parametrize("width", [16, 64])
def test_action_order_preserves_probabilities_and_value_after_learning(width, tmp_path):
    torch.set_num_threads(1)
    environment = Sts2Env(worker_factory=FakeWorker)
    model = MaskablePPO(
        SharedActionPolicy,
        environment,
        n_steps=32,
        batch_size=16,
        n_epochs=2,
        seed=21,
        policy_kwargs={"width": width},
    )
    before = {name: value.clone() for name, value in model.policy.state_dict().items()}
    model.learn(64)
    for prefix in ("action_net", "mlp_extractor.action_encoder", "mlp_extractor.state_encoder"):
        assert any(
            not torch.equal(value, before[name])
            for name, value in model.policy.state_dict().items()
            if name.startswith(prefix)
        )
    observations, masks = sample()
    original, values = probabilities(model.policy, observations, masks)
    assert torch.all(original[~masks] == 0)
    order = np.random.default_rng(7).permutation(128)
    changed = observations | {"actions": observations["actions"][:, order]}
    reordered, reordered_values = probabilities(model.policy, changed, masks[:, order])
    torch.testing.assert_close(reordered, original[:, order])
    torch.testing.assert_close(values, reordered_values)
    model.save(tmp_path / "model.zip")
    restored = MaskablePPO.load(tmp_path / "model.zip", device="cpu")
    actual, actual_values = probabilities(restored.policy, observations, masks)
    torch.testing.assert_close(actual, original, rtol=0, atol=0)
    torch.testing.assert_close(actual_values, values, rtol=0, atol=0)
    model.policy.save(tmp_path / "policy.pt")
    policy = SharedActionPolicy.load(tmp_path / "policy.pt", device="cpu")
    assert policy.width == width
    torch.testing.assert_close(probabilities(policy, observations, masks)[0], original)
    environment.close()


def test_scoring_uses_both_state_and_action_content():
    torch.manual_seed(31)
    environment = Sts2Env(worker_factory=FakeWorker)
    policy = SharedActionPolicy(
        environment.observation_space, environment.action_space, lambda _: 0.001
    )
    observations, masks = sample()
    original, _ = probabilities(policy, observations, masks)
    for field in ("state", "actions"):
        changed = observations | {field: -observations[field]}
        current, _ = probabilities(policy, changed, masks)
        assert not torch.allclose(current, original)
    empty = {name: np.zeros_like(value) for name, value in observations.items()}
    tensors = policy.obs_to_tensor(empty)[0]
    assert torch.isfinite(policy.predict_values(tensors)).all()
    environment.close()
