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


@pytest.mark.parametrize("updates", [0, 1, 4])
def test_replay_runs_after_each_ppo_update_and_never_before_it(updates):
    from ai4sts2.environment import Sts2Env
    from ai4sts2.policy import SharedActionPolicy
    from ai4sts2.replay import SelfImitationCallback
    from sb3_contrib import MaskablePPO
    from tensordict import TensorDict
    from test_environment import FakeWorker
    from torchrl.data import TensorDictReplayBuffer, TensorStorage

    torch.set_num_threads(1)
    environment = Sts2Env(worker_factory=FakeWorker)
    try:
        model = MaskablePPO(
            SharedActionPolicy, environment, n_steps=8, batch_size=8, n_epochs=1, seed=7
        )
        observation, _ = environment.reset(seed=11)
        data = TensorDict(
            {
                "observation": TensorDict(
                    {key: torch.tensor(value)[None] for key, value in observation.items()},
                    batch_size=[1],
                ),
                "action": torch.zeros(1, dtype=torch.long),
                "action_mask": torch.tensor(environment.action_masks())[None],
                "return": torch.ones(1),
            },
            batch_size=[1],
        )
        generator = torch.Generator().manual_seed(7)
        buffer = TensorDictReplayBuffer(
            storage=TensorStorage(data), batch_size=8, generator=generator
        )
        callback = SelfImitationCallback(buffer, updates, 0.01)
        rng_before = generator.get_state().clone()
        model.learn(total_timesteps=24, callback=callback)
        assert model.num_timesteps == 24 and model._n_updates == 3
        assert len(callback.diagnostics) == updates * 3
        assert [row["ppo_updates"] for row in callback.diagnostics] == [
            count for count in (1, 2, 3) for _ in range(updates)
        ]
        assert [row["environment_steps"] for row in callback.diagnostics] == [
            count for count in (8, 16, 24) for _ in range(updates)
        ]
        if not updates:
            assert torch.equal(generator.get_state(), rng_before)
        model.learn(total_timesteps=8, reset_num_timesteps=False, callback=callback)
        assert len(callback.diagnostics) == updates * 4
        if updates:
            assert callback.diagnostics[-1]["ppo_updates"] == 4
    finally:
        environment.close()


@pytest.mark.parametrize("updates", [-1, 0.5, True])
def test_invalid_replay_update_counts_are_rejected(updates):
    from ai4sts2.replay import SelfImitationCallback

    with pytest.raises(ValueError):
        SelfImitationCallback(None, updates, 0.01)


def test_interrupted_rollout_does_not_replay_without_a_ppo_update():
    from ai4sts2.environment import Sts2Env
    from ai4sts2.replay import SelfImitationCallback
    from sb3_contrib import MaskablePPO
    from test_environment import FakeWorker

    class Stop(SelfImitationCallback):
        def _on_step(self):
            return False

    environment = Sts2Env(worker_factory=FakeWorker)
    try:
        model = MaskablePPO("MultiInputPolicy", environment, n_steps=8, batch_size=8)
        callback = Stop(None, 1, 0.01)
        model.learn(total_timesteps=8, callback=callback)
        assert model.num_timesteps == 1 and model._n_updates == 0
        assert callback.diagnostics == []
    finally:
        environment.close()


@pytest.fixture
def replay_corpus(tmp_path):
    import hashlib
    import json

    from ai4sts2.environment import encode, seed_string
    from test_environment import state

    observation = encode(state())
    path = tmp_path / "episode.npz"
    np.savez_compressed(
        path,
        **{key: np.stack([value, value]) for key, value in observation.items()},
        action_masks=np.tile(np.arange(128) < 2, (2, 1)),
        action_indices=np.array([0, 1], dtype=np.int64),
        rewards=np.array([0, 1], dtype=np.float32),
    )
    record = {
        "complete": True,
        "truncated": False,
        "split": "train",
        "build": {"game": "test", "schema": 1},
        "case": {
            "character": "IRONCLAD",
            "seed": seed_string("train", 12),
            "seed_index": 12,
            "mode": "sampled",
        },
        "trajectory": path.name,
        "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "episode": {"steps": 2, "task_success": True},
    }
    manifest = tmp_path / "corpus.json"
    manifest.write_text(json.dumps({"complete": True, "records": [record]}))
    return manifest


def test_corpus_allows_new_trainer_with_unchanged_game_and_schema(replay_corpus):
    from ai4sts2.replay import replay_dataset

    data = replay_dataset(replay_corpus, {"game": "test", "schema": 1, "trainer": "new"}, 0.5)
    torch.testing.assert_close(data["return"], torch.tensor([0.5, 1.0]))
    assert data["action_mask"][torch.arange(2), data["action"]].all()


@pytest.mark.parametrize("split", ["validation", "test"])
def test_replay_rejects_evaluation_data(replay_corpus, split):
    import json

    from ai4sts2.replay import replay_dataset

    saved = json.loads(replay_corpus.read_text())
    saved["records"][0]["split"] = split
    replay_corpus.write_text(json.dumps(saved))
    with pytest.raises(ValueError, match="training episodes"):
        replay_dataset(replay_corpus, {"game": "test", "schema": 1}, 1.0)


@pytest.mark.parametrize("change", ["build", "checksum", "duplicate", "incomplete", "seed"])
def test_corpus_provenance_errors_are_rejected(replay_corpus, change):
    import json

    from ai4sts2.replay import replay_dataset

    saved = json.loads(replay_corpus.read_text())
    row = saved["records"][0]
    if change == "build":
        row["build"]["game"] = "different"
    elif change == "checksum":
        row["sha256"] = "wrong"
    elif change == "duplicate":
        saved["records"].append(row)
    elif change == "incomplete":
        row["truncated"] = True
    else:
        row["case"]["seed"] = "wrong"
    replay_corpus.write_text(json.dumps(saved))
    with pytest.raises(ValueError):
        replay_dataset(replay_corpus, {"game": "test", "schema": 1}, 1.0)


@pytest.mark.parametrize("change", ["illegal", "shape", "reward", "nonfinite"])
def test_corpus_rejects_bad_arrays_even_with_valid_checksums(replay_corpus, change):
    import hashlib
    import json

    from ai4sts2.replay import replay_dataset

    saved = json.loads(replay_corpus.read_text())
    row = saved["records"][0]
    path = replay_corpus.parent / row["trajectory"]
    with np.load(path) as archive:
        data = {key: archive[key].copy() for key in archive.files}
    if change == "illegal":
        data["action_indices"][0] = 3
    elif change == "shape":
        data["state"] = data["state"][:1]
    elif change == "reward":
        data["rewards"][0] = 1
    else:
        data["actions"][0, 0, 0] = np.nan
    np.savez_compressed(path, **data)
    row["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    replay_corpus.write_text(json.dumps(saved))
    with pytest.raises(ValueError):
        replay_dataset(replay_corpus, {"game": "test", "schema": 1}, 1.0)


def test_integrated_replay_recovers_exactly_and_validation_never_updates_it(
    monkeypatch, tmp_path, replay_corpus
):
    from ai4sts2.environment import evaluate
    from test_checkpoint import member

    candidate = member(monkeypatch, policy="shared")
    candidate._logdir = str(tmp_path)
    candidate.config |= {
        "replay_corpus": str(replay_corpus),
        "replay_updates": 4,
        "replay_value_coefficient": 0.01,
        "fixed_steps": True,
    }
    candidate.apply_parameters(candidate.config)
    candidate.sample_count = 64
    try:
        first = candidate.step()
        assert first["auxiliary_updates"] == 4
        saved = tmp_path / "saved"
        saved.mkdir()
        candidate.save_checkpoint(saved)
        second = candidate.step()
        weights = {key: value.clone() for key, value in candidate.model.policy.state_dict().items()}
        generator = candidate.replay_generator.get_state().clone()
        snapshot = candidate.environment.snapshot()
        evaluate(candidate.model, candidate.validation_environment)
        assert torch.equal(candidate.replay_generator.get_state(), generator)
        assert candidate.replay_updates == 8
        candidate.cleanup()
        candidate.load_checkpoint(saved)
        repeated = candidate.step()
        assert second["auxiliary_diagnostics"] == repeated["auxiliary_diagnostics"]
        assert repeated["auxiliary_updates"] == 8
        assert candidate.environment.snapshot() == snapshot
        assert torch.equal(candidate.replay_generator.get_state(), generator)
        for key, value in weights.items():
            assert torch.equal(candidate.model.policy.state_dict()[key], value)
    finally:
        candidate.cleanup()


def test_disabling_replay_preserves_sampler_and_allows_reenabling(
    monkeypatch, tmp_path, replay_corpus
):
    from test_checkpoint import member

    candidate = member(monkeypatch, policy="shared")
    candidate._logdir = str(tmp_path)
    candidate.config |= {
        "replay_corpus": str(replay_corpus),
        "replay_updates": 0,
        "fixed_steps": True,
    }
    candidate.apply_parameters(candidate.config)
    candidate.sample_count = 64
    try:
        before = candidate.replay_generator.get_state().clone()
        assert candidate.replay_buffer is None
        assert candidate.step()["auxiliary_updates"] == 0
        assert torch.equal(before, candidate.replay_generator.get_state())
        candidate.config["replay_updates"] = 1
        candidate.apply_parameters(candidate.config)
        assert candidate.step()["auxiliary_updates"] == 1
        candidate.config["replay_updates"] = 0
        candidate.apply_parameters(candidate.config)
        before = candidate.replay_generator.get_state().clone()
        assert candidate.step()["auxiliary_updates"] == 1
        assert torch.equal(before, candidate.replay_generator.get_state())
    finally:
        candidate.cleanup()
