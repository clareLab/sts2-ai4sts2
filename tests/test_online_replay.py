import copy
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from ai4sts2.environment import Sts2Env, encode, evaluate, seed_string
from ai4sts2.replay import TrajectoryCollector, merge_corpus, replay_dataset
from test_checkpoint import member
from test_environment import FakeWorker, state
from test_replay import replay_corpus as replay_corpus


@pytest.fixture
def collector_member(monkeypatch, tmp_path, replay_corpus):
    candidate = member(monkeypatch, policy="shared")
    candidate._logdir = str(tmp_path)
    candidate.config |= {
        "replay_corpus": str(replay_corpus),
        "replay_updates": 4,
        "replay_value_coefficient": 0.01,
        "collect_replay": True,
        "fixed_steps": True,
        "validate_each_iteration": False,
    }
    candidate.apply_parameters(candidate.config)
    candidate.sample_count = 512
    yield candidate
    candidate.cleanup()


def test_complete_training_episodes_feed_the_next_block_and_restore_exactly(
    collector_member, tmp_path
):
    candidate = collector_member
    first = candidate.step()
    assert first["replay_transitions"] == 514 and first["collected_episodes"] == 128
    saved = tmp_path / "checkpoint"
    saved.mkdir()
    candidate.save_checkpoint(saved)
    source = candidate.config["replay_corpus"]
    original_manifest = Path(source).read_bytes()
    second = candidate.step()
    assert second["replay_transitions"] == 1026 and second["collected_episodes"] == 256
    assert candidate.config["replay_corpus"] != source
    assert Path(source).read_bytes() == original_manifest
    weights = copy.deepcopy(candidate.model.policy.state_dict())
    generator = candidate.replay_generator.get_state().clone()
    corpus = candidate.replay_identity
    validation = candidate.open_environment(training=False)
    try:
        evaluate(candidate.model, validation)
    finally:
        validation.close()
    assert candidate.replay_identity == corpus
    assert candidate.trajectory_collector.episodes == 256
    candidate.load_checkpoint(saved)
    assert candidate.config["replay_corpus"] == source
    assert candidate.trajectory_collector.episodes == 128
    repeated = candidate.step()
    assert repeated["auxiliary_diagnostics"] == second["auxiliary_diagnostics"]
    assert candidate.replay_identity == corpus
    assert torch.equal(candidate.replay_generator.get_state(), generator)
    for name, value in weights.items():
        assert torch.equal(candidate.model.policy.state_dict()[name], value)
    records = json.loads(Path(corpus[0]).read_text())["records"]
    assert all(row["split"] == "train" for row in records)
    assert all(
        row["case"]["seed"] == seed_string("train", row["case"]["seed_index"]) for row in records
    )


def test_passive_collection_preserves_weights_actions_and_rng(monkeypatch, tmp_path, replay_corpus):
    results = []
    for collect in (False, True):
        candidate = member(monkeypatch, policy="shared")
        candidate._logdir = str(tmp_path)
        candidate.config |= {
            "replay_corpus": str(replay_corpus),
            "replay_updates": 0,
            "collect_replay": collect,
            "validate_each_iteration": False,
        }
        candidate.apply_parameters(candidate.config)
        candidate.sample_count = 512
        metrics = candidate.step()
        results.append(
            (
                copy.deepcopy(candidate.model.policy.state_dict()),
                torch.get_rng_state().clone(),
                metrics["training_episodes"],
            )
        )
        candidate.cleanup()
    assert results[0][2] == results[1][2]
    assert torch.equal(results[0][1], results[1][1])
    for name, value in results[0][0].items():
        assert torch.equal(value, results[1][0][name])


def test_collection_keeps_pre_action_observations_and_pending_episode_across_checkpoints(
    collector_member, monkeypatch, tmp_path
):
    import ai4sts2.train as training
    from stable_baselines3.common.vec_env import DummyVecEnv

    class SevenSteps(FakeWorker):
        def request(self, method, parameters):
            result = super().request(method, parameters)
            result["terminated"] = result["victory"] = self.count >= 7
            result["actions"] = [] if result["terminated"] else state()["actions"]
            result["observation"]["floor"] = 2 + self.count
            return result

    candidate = collector_member
    candidate.environment.close()
    candidate.environment = Sts2Env(seed=5, worker_factory=SevenSteps, scope="run", max_steps=4096)
    candidate.model.set_env(DummyVecEnv([lambda: candidate.environment]))
    monkeypatch.setattr(
        training,
        "Sts2Env",
        lambda executable, seed, **kwargs: Sts2Env(seed=seed, worker_factory=SevenSteps, **kwargs),
    )
    candidate.step()
    assert len(candidate.trajectory_collector.pending) == 1
    saved = tmp_path / "partial"
    saved.mkdir()
    candidate.save_checkpoint(saved)
    candidate.step()
    expected = candidate.replay_identity
    weights = copy.deepcopy(candidate.model.policy.state_dict())
    candidate.load_checkpoint(saved)
    assert len(candidate.trajectory_collector.pending) == 1
    candidate.step()
    assert candidate.replay_identity == expected
    for name, value in weights.items():
        assert torch.equal(value, candidate.model.policy.state_dict()[name])
    records = json.loads(Path(expected[0]).read_text())["records"]
    record = next(row for row in records if row["case"]["mode"] == "online")
    with np.load(record["trajectory"]) as data:
        for index in range(7):
            observation = state()
            observation["observation"]["floor"] = index + 2
            np.testing.assert_array_equal(data["state"][index], encode(observation)["state"])


def test_truncated_episodes_are_excluded(collector_member):
    candidate = collector_member
    candidate.environment.max_steps = 2
    result = candidate.step()
    assert result["replay_transitions"] == 2
    assert result["collected_episodes"] == 0
    assert candidate.trajectory_collector.truncated == 256
    assert not candidate.trajectory_collector.pending


@pytest.mark.parametrize("split", ["validation", "test"])
def test_collector_rejects_evaluation_before_writing(tmp_path, split):
    collector = TrajectoryCollector(tmp_path, {})
    collector.model = SimpleNamespace(n_envs=1)
    collector.locals = {"infos": [{"split": split}]}
    with pytest.raises(ValueError, match="Only training"):
        collector._on_step()
    assert not list(tmp_path.iterdir())


def test_corpus_retention_is_bounded_idempotent_and_keeps_complete_episodes(
    replay_corpus, tmp_path
):
    original = json.loads(replay_corpus.read_text())["records"][0]
    rows = []
    for index in (13, 14):
        rows.append(
            original
            | {
                "trajectory": str((replay_corpus.parent / original["trajectory"]).resolve()),
                "case": original["case"]
                | {"seed_index": index, "seed": seed_string("train", index)},
            }
        )
    merged = merge_corpus(replay_corpus, rows, tmp_path, capacity=4)
    data = json.loads(merged.read_text())
    assert data["transitions"] == 4
    assert [row["case"]["seed_index"] for row in data["records"]] == [13, 14]
    assert len(replay_dataset(merged, original["build"], 1)) == 4
    assert merge_corpus(merged, rows, tmp_path, capacity=4) == merged
    with pytest.raises(ValueError, match="conflicting"):
        merge_corpus(merged, [rows[-1] | {"sha256": "changed"}], tmp_path, capacity=4)
    with pytest.raises(ValueError, match="complete episode"):
        merge_corpus(replay_corpus, rows, tmp_path, capacity=1)


def test_replay_updates_do_not_depend_on_training_block_boundaries(collector_member, tmp_path):
    candidate = collector_member
    saved = tmp_path / "initial"
    saved.mkdir()
    candidate.save_checkpoint(saved)
    candidate.sample_count = 576
    candidate.step()
    weights = copy.deepcopy(candidate.model.policy.state_dict())
    sampler = candidate.replay_generator.get_state().clone()
    corpus = candidate.replay_identity
    candidate.load_checkpoint(saved)
    candidate.sample_count = 64
    for _ in range(9):
        candidate.step()
    assert candidate.replay_identity == corpus
    assert torch.equal(candidate.replay_generator.get_state(), sampler)
    for name, value in weights.items():
        assert torch.equal(value, candidate.model.policy.state_dict()[name])


def test_capacity_is_checked_before_loading_episode_arrays(replay_corpus):
    with pytest.raises(ValueError, match="capacity"):
        replay_dataset(replay_corpus, {"game": "test", "schema": 1}, 1, capacity=1)


def test_native_failure_rewinds_collection_with_the_training_checkpoint(
    collector_member, monkeypatch, tmp_path
):
    import ai4sts2.train as training
    from ai4sts2.execution import REFERENCE
    from ai4sts2.game import WorkerFailure

    candidate = collector_member
    initial = tmp_path / "initial"
    initial.mkdir()
    candidate.save_checkpoint(initial)
    candidate.sample_count = 1024
    candidate.step()
    expected = copy.deepcopy(candidate.model.policy.state_dict())
    sampler = candidate.replay_generator.get_state().clone()
    corpus = candidate.replay_identity
    candidate.load_checkpoint(initial)
    candidate.sample_count = 1024
    original = candidate.environment.game.request
    calls = 0

    def fault(method, parameters):
        nonlocal calls
        if method == "step":
            calls += 1
            if calls == 550:
                raise WorkerFailure("Injected failure after corpus refresh")
        return original(method, parameters)

    monkeypatch.setattr(candidate.environment.game, "request", fault)
    monkeypatch.setattr(training, "quarantine_execution", lambda *_: REFERENCE)
    result = candidate.step()
    assert len(result["recovery_events"]) == 1
    assert result["collected_episodes"] == 256
    assert candidate.replay_identity == corpus
    assert torch.equal(candidate.replay_generator.get_state(), sampler)
    for name, value in expected.items():
        assert torch.equal(value, candidate.model.policy.state_dict()[name])
