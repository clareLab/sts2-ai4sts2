import json

import pytest
import torch
from ai4sts2.environment import Sts2Env
from ai4sts2.train import PopulationMember
from test_environment import FakeWorker


def member(monkeypatch, learning_rate=0.0003, entropy=0.01):
    import ai4sts2.train as training

    monkeypatch.setattr(training, "fingerprint", lambda: {"game": "test", "schema": 1})
    monkeypatch.setattr(
        training,
        "Sts2Env",
        lambda executable, seed: Sts2Env(seed=seed, worker_factory=FakeWorker),
    )
    instance = object.__new__(PopulationMember)
    instance.config = {
        "executable": None,
        "learning_rate": learning_rate,
        "entropy": entropy,
        "seed": 5,
    }
    instance.setup(instance.config)
    return instance


def test_pbt_restores_weights_and_optimizer_then_applies_mutations(monkeypatch, tmp_path):
    donor = member(monkeypatch)
    donor.model.learn(total_timesteps=64)
    donor.save_checkpoint(str(tmp_path))
    receiver = member(monkeypatch, learning_rate=0.001, entropy=0.03)
    receiver.load_checkpoint(str(tmp_path))
    for key, weight in donor.model.policy.state_dict().items():
        assert torch.equal(weight, receiver.model.policy.state_dict()[key])
    assert receiver.model.policy.optimizer.state_dict()["state"]
    assert receiver.model.ent_coef == 0.03
    assert receiver.model.lr_schedule(0.5) == 0.001
    assert all(group["lr"] == 0.001 for group in receiver.model.policy.optimizer.param_groups)
    assert receiver.model.num_timesteps == donor.model.num_timesteps
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


def test_reused_actor_starts_an_independent_candidate(monkeypatch):
    instance = member(monkeypatch)
    instance.model.learn(total_timesteps=64)
    old_worker = instance.environment.game
    assert instance.reset_config(instance.config | {"seed": 6})
    assert old_worker.closed
    assert instance.model.num_timesteps == 0
    assert instance.environment.game is not old_worker
    instance.cleanup()
