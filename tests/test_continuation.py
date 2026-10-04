import copy

import numpy as np
import pytest
import torch
from ai4sts2.environment import Sts2Env, evaluate
from test_checkpoint import member
from test_environment import FakeWorker, state


class LongWorker(FakeWorker):
    def request(self, method, parameters):
        super().request(method, parameters)
        result = state(self.count >= 150, self.count >= 150)
        result["observation"]["player"]["hp"] = 200 - self.count
        result["revision"] = len(self.calls)
        return result


def long_member(monkeypatch, directory):
    import ai4sts2.train as training

    instance = member(monkeypatch)
    instance.cleanup()
    monkeypatch.setattr(
        training,
        "Sts2Env",
        lambda executable, seed, **kwargs: Sts2Env(seed=seed, worker_factory=LongWorker, **kwargs),
    )
    instance.setup(instance.config)
    instance._logdir = str(directory)
    directory.mkdir(exist_ok=True)
    return instance


def test_validation_and_iteration_boundaries_preserve_an_unfinished_run(monkeypatch, tmp_path):
    candidate = long_member(monkeypatch, tmp_path)
    candidate.sample_count = 64
    first = candidate.step()
    training_worker = candidate.environment.game
    episode_seed = candidate.environment.journal["parameters"]["seed"]
    assert candidate.environment.steps == 64
    assert first["training_episodes"] == []
    assert len(first["validation_episodes"]) == 5
    assert candidate.validation_environment.game is not training_worker
    candidate.sample_count = 64
    second = candidate.step()
    assert second["environment_steps"] == 128
    assert candidate.environment.steps == 128
    assert candidate.environment.journal["parameters"]["seed"] == episode_seed
    assert sum(method == "reset" for method, _ in training_worker.calls) == 1
    candidate.sample_count = 64
    third = candidate.step()
    assert third["training_episodes"][0]["steps"] == 150
    assert candidate.environment.steps == 42
    assert third["environment_steps"] == 192
    candidate.cleanup()


@pytest.mark.parametrize("progress_scale", [0.0, 1.0])
def test_checkpoint_continues_identical_actions_weights_and_seed_stream(
    monkeypatch, tmp_path, progress_scale
):
    donor = long_member(monkeypatch, tmp_path / "donor")
    donor.config |= {"gamma": 1.0, "progress_scale": progress_scale}
    donor.apply_parameters(donor.config)
    donor.model.learn(total_timesteps=64)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    donor.save_checkpoint(checkpoint)
    donor.model.learn(total_timesteps=128, reset_num_timesteps=False)
    expected_weights = {
        key: value.clone() for key, value in donor.model.policy.state_dict().items()
    }
    expected_environment = donor.environment.snapshot()
    expected_actions = [call for call in donor.environment.game.calls if call[0] == "step"]
    donor.cleanup()
    receiver = long_member(monkeypatch, tmp_path / "receiver")
    receiver.config |= {"gamma": 1.0, "progress_scale": progress_scale}
    receiver.load_checkpoint(checkpoint)
    assert receiver.environment.steps == 64
    assert receiver.model.num_timesteps == 64
    receiver.model.learn(total_timesteps=128, reset_num_timesteps=False)
    assert receiver.environment.snapshot() == expected_environment
    assert receiver.model.num_timesteps == 192
    assert [
        call for call in receiver.environment.game.calls if call[0] == "step"
    ] == expected_actions
    for key, weight in receiver.model.policy.state_dict().items():
        assert torch.equal(weight, expected_weights[key])
    receiver.cleanup()


def test_auxiliary_training_resumes_exactly_and_validation_never_updates_it(monkeypatch, tmp_path):
    config = {"rnd_scale": 0.001, "curriculum_mix": 0.75}
    donor = long_member(monkeypatch, tmp_path / "donor")
    donor.config |= config
    donor.signals.configure(donor.config)
    donor.model.learn(total_timesteps=64)
    checkpoint = tmp_path / "checkpoint"
    checkpoint.mkdir()
    donor.save_checkpoint(checkpoint)
    donor.model.learn(total_timesteps=128, reset_num_timesteps=False)
    expected_weights = copy.deepcopy(donor.model.policy.state_dict())
    expected_signals = donor.signals.snapshot()
    expected_environment = donor.environment.snapshot()
    donor.cleanup()
    receiver = long_member(monkeypatch, tmp_path / "receiver")
    receiver.config |= config
    receiver.load_checkpoint(checkpoint)
    receiver.model.learn(total_timesteps=128, reset_num_timesteps=False)
    assert receiver.environment.snapshot() == expected_environment
    assert receiver.signals.metrics == expected_signals["metrics"]
    for key, weight in expected_weights.items():
        assert torch.equal(weight, receiver.model.policy.state_dict()[key])
    for key, weight in expected_signals["rnd"].items():
        assert torch.equal(weight, receiver.signals.rnd.state_dict()[key])
    validation = receiver.open_environment(training=False)
    assert validation.signals is None
    evaluate(receiver.model, validation)
    assert receiver.signals.metrics == expected_signals["metrics"]
    validation.close()
    receiver.cleanup()


def test_checkpoint_divergence_stops_before_any_new_choice():
    environment = Sts2Env(worker_factory=LongWorker)
    environment.reset()
    environment.step(0)
    snapshot = environment.snapshot()
    snapshot["journal"]["actions"][0]["digest"] = "changed"
    with pytest.raises(ValueError, match="diverged at step 1"):
        environment.restore(snapshot)
    assert environment.game.count == 1
    environment.close()


def test_snapshot_is_independent_and_never_exposes_seed_to_features():
    environment = Sts2Env(worker_factory=LongWorker)
    environment.reset()
    observation = environment.encode()
    snapshot = environment.snapshot()
    environment.step(0)
    assert snapshot["journal"]["actions"] == []
    environment.restore(snapshot)
    for key, value in observation.items():
        np.testing.assert_array_equal(environment.encode()[key], value)
    environment.close()


def test_training_validation_uses_the_predeclared_seed_panel(monkeypatch, tmp_path):
    from ai4sts2.environment import evaluation_plan

    candidate = long_member(monkeypatch, tmp_path)
    try:
        candidate.config |= {"validation_seed": 812349}
        candidate.sample_count = 64
        result = candidate.step()
        expected = evaluation_plan(
            candidate.scope, 812349, max_steps=candidate.validation_environment.max_steps
        )
        assert result["evaluation_id"] == expected["evaluation_id"]
    finally:
        candidate.cleanup()


def test_evaluation_cannot_clear_training_observation(monkeypatch, tmp_path):
    candidate = long_member(monkeypatch, tmp_path)
    candidate.model.learn(total_timesteps=64)
    observation = copy.deepcopy(candidate.model._last_obs)
    starts = candidate.model._last_episode_starts.copy()
    environment = Sts2Env(worker_factory=FakeWorker)
    evaluate(candidate.model, environment)
    for key in observation:
        np.testing.assert_array_equal(candidate.model._last_obs[key], observation[key])
    np.testing.assert_array_equal(candidate.model._last_episode_starts, starts)
    environment.close()
    candidate.cleanup()


def test_no_additional_actions_after_a_truncation():
    environment = Sts2Env(worker_factory=LongWorker, max_steps=1)
    environment.reset()
    environment.step(0)
    with pytest.raises(ValueError, match="Illegal"):
        environment.step(0)
    assert environment.game.count == 1
    environment.close()


@pytest.mark.parametrize("validation_failure", [False, True])
def test_recovery_keeps_an_existing_run_and_does_not_change_training(
    monkeypatch, tmp_path, validation_failure
):
    import ai4sts2.train as training
    from ai4sts2.execution import REFERENCE
    from ai4sts2.game import WorkerFailure
    from stable_baselines3.common.vec_env import DummyVecEnv

    baseline = long_member(monkeypatch, tmp_path / "baseline")
    baseline.model.learn(total_timesteps=64)
    checkpoint = tmp_path / "saved"
    checkpoint.mkdir()
    baseline.save_checkpoint(checkpoint)
    baseline.sample_count = 64
    expected = baseline.step()
    weights = {key: value.clone() for key, value in baseline.model.policy.state_dict().items()}
    snapshot = baseline.environment.snapshot()
    baseline.cleanup()

    class FaultWorker(LongWorker):
        injected = False

        def request(self, method, parameters):
            if method == "step" and self.count == 70 and not FaultWorker.injected:
                FaultWorker.injected = True
                raise WorkerFailure("Injected worker failure")
            return super().request(method, parameters)

    recovered = long_member(monkeypatch, tmp_path / "recovered")
    recovered.load_checkpoint(checkpoint)
    recovered.sample_count = 64
    faulty = Sts2Env(scope="run", max_steps=4096, worker_factory=FaultWorker)
    if validation_failure:
        recovered.validation_environment = faulty
    else:
        faulty.restore(recovered.environment.snapshot())
        recovered.environment.close()
        recovered.environment = faulty
        recovered.model.set_env(DummyVecEnv([lambda: faulty]), force_reset=False)
    monkeypatch.setattr(training, "quarantine_execution", lambda *_: REFERENCE)
    actual = recovered.step()
    assert FaultWorker.injected and len(actual["recovery_events"]) == 1
    assert actual["environment_steps"] == expected["environment_steps"] == 128
    assert actual["validation_episodes"] == expected["validation_episodes"]
    assert recovered.environment.snapshot() == snapshot
    for key, weight in recovered.model.policy.state_dict().items():
        assert torch.equal(weight, weights[key])
    recovered.cleanup()
