import json

import pytest
import torch
from ai4sts2.environment import Sts2Env
from ai4sts2.train import PopulationMember
from test_environment import FakeWorker


def member(monkeypatch, learning_rate=0.0003, entropy=0.01, policy="flat", encoding="hash"):
    import ai4sts2.train as training

    monkeypatch.setattr(training, "fingerprint", lambda *_: {"game": "test", "schema": 1})
    monkeypatch.setattr(training, "selected_execution", lambda *_: None)
    monkeypatch.setattr(
        training,
        "Sts2Env",
        lambda executable, seed, **kwargs: Sts2Env(seed=seed, worker_factory=FakeWorker, **kwargs),
    )
    instance = object.__new__(PopulationMember)
    instance.config = {
        "executable": None,
        "learning_rate": learning_rate,
        "entropy": entropy,
        "seed": 5,
        "policy": policy,
        "encoding": encoding,
    }
    instance.setup(instance.config)
    return instance


def test_pbt_restores_weights_and_optimizer_then_applies_mutations(monkeypatch, tmp_path):
    donor = member(monkeypatch)
    donor.model.learn(total_timesteps=64)
    donor.evaluation = {"scope": "first_combat", "episodes": [{"victory": True}]}
    donor.save_checkpoint(str(tmp_path))
    assert json.loads((tmp_path / "evaluation.json").read_text()) == donor.evaluation
    receiver = member(monkeypatch, learning_rate=0.001, entropy=0.03)
    receiver.config |= {
        "gamma": 0.97,
        "gae_lambda": 0.9,
        "clip_range": 0.15,
        "epochs": 4,
        "progress_scale": 0.5,
    }
    receiver.load_checkpoint(str(tmp_path))
    for key, weight in donor.model.policy.state_dict().items():
        assert torch.equal(weight, receiver.model.policy.state_dict()[key])
    assert receiver.model.policy.optimizer.state_dict()["state"]
    assert receiver.model.ent_coef == 0.03
    assert receiver.model.lr_schedule(0.5) == 0.001
    assert all(group["lr"] == 0.001 for group in receiver.model.policy.optimizer.param_groups)
    assert receiver.model.num_timesteps == donor.model.num_timesteps
    assert receiver.model.gamma == receiver.model.rollout_buffer.gamma == 0.97
    assert receiver.environment.discount == 0.97
    assert receiver.environment.progress_scale == 0.5
    validation = receiver.open_environment(training=False)
    assert validation.progress_scale == 0
    validation.close()
    assert receiver.model.gae_lambda == receiver.model.rollout_buffer.gae_lambda == 0.9
    assert receiver.model.clip_range(0.5) == 0.15
    assert receiver.model.n_epochs == 4
    assert receiver.sample_count == donor.sample_count
    assert receiver.evaluation == donor.evaluation
    assert receiver.environment.rng.bit_generator.state == donor.environment.rng.bit_generator.state
    donor.cleanup()
    receiver.cleanup()


def test_checkpoint_rejects_incompatible_game(monkeypatch, tmp_path):
    instance = member(monkeypatch)
    instance.save_checkpoint(str(tmp_path))
    (tmp_path / "build.json").write_text(json.dumps({"game": "different", "schema": 1}))
    with pytest.raises(ValueError, match="do not match"):
        instance.load_checkpoint(str(tmp_path))
    instance.cleanup()


def test_checkpoint_rejects_cross_architecture_population_transfer(monkeypatch, tmp_path):
    donor = member(monkeypatch, policy="shared")
    receiver = member(monkeypatch)
    try:
        donor.model.learn(total_timesteps=64)
        donor.save_checkpoint(tmp_path)
        with pytest.raises(ValueError, match="policy architecture"):
            receiver.load_checkpoint(tmp_path)
    finally:
        donor.cleanup()
        receiver.cleanup()


def test_reused_actor_starts_an_independent_candidate(monkeypatch):
    instance = member(monkeypatch)
    instance.model.learn(total_timesteps=64)
    old_worker = instance.environment.game
    assert instance.reset_config(instance.config | {"seed": 6})
    assert old_worker.closed
    assert instance.model.num_timesteps == 0
    assert instance.environment.game is not old_worker
    instance.cleanup()


def test_structured_member_resumes_optimizer_and_unfinished_episode(monkeypatch, tmp_path):
    donor = member(monkeypatch, policy="shared", encoding="tree")
    receiver = member(monkeypatch, policy="shared", encoding="tree")
    incompatible = member(monkeypatch, policy="shared")
    try:
        donor.model.learn(total_timesteps=64)
        donor.save_checkpoint(tmp_path)
        receiver.load_checkpoint(tmp_path)
        donor.model.learn(total_timesteps=64, reset_num_timesteps=False)
        receiver.load_checkpoint(tmp_path)
        receiver.model.learn(total_timesteps=64, reset_num_timesteps=False)
        for key, weight in donor.model.policy.state_dict().items():
            torch.testing.assert_close(
                weight, receiver.model.policy.state_dict()[key], rtol=0, atol=0
            )
        with pytest.raises(ValueError, match="encoding"):
            incompatible.load_checkpoint(tmp_path)
    finally:
        for instance in (donor, receiver, incompatible):
            instance.cleanup()


def test_temperature_mutation_survives_model_and_policy_serialisation(monkeypatch, tmp_path):
    from ai4sts2.policy import SharedActionPolicy
    from sb3_contrib import MaskablePPO
    from test_policy import probabilities, sample

    donor = member(monkeypatch, policy="shared")
    receiver = member(monkeypatch, policy="shared")
    try:
        donor.config |= {"temperature": 0.5}
        donor.apply_parameters(donor.config)
        donor.model.learn(64)
        donor.save_checkpoint(tmp_path)
        receiver.config |= {"temperature": 0.2}
        receiver.load_checkpoint(tmp_path)
        assert receiver.model.policy.temperature == 0.2
        for name, value in donor.model.policy.state_dict().items():
            torch.testing.assert_close(
                receiver.model.policy.state_dict()[name], value, rtol=0, atol=0
            )
        observations, masks = sample()
        expected, _ = probabilities(receiver.model.policy, observations, masks)
        receiver.model.save(tmp_path / "mutated.zip")
        receiver.model.policy.save(tmp_path / "policy.pt")
        restored = MaskablePPO.load(tmp_path / "mutated.zip", device="cpu")
        standalone = SharedActionPolicy.load(tmp_path / "policy.pt", device="cpu")
        for policy in (restored.policy, standalone):
            assert policy.temperature == 0.2
            torch.testing.assert_close(
                probabilities(policy, observations, masks)[0], expected, rtol=0, atol=0
            )
    finally:
        donor.cleanup()
        receiver.cleanup()
