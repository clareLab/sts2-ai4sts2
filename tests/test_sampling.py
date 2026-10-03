import copy
import json
import time

import numpy as np
import pytest
import torch
from ai4sts2.environment import Sts2Env, evaluate, evaluation_plan
from ai4sts2.policy import SharedActionPolicy, evaluation_action
from sb3_contrib import MaskablePPO
from test_environment import FakeWorker
from test_evaluate import holdout


@pytest.fixture
def model():
    torch.set_num_threads(1)
    environment = Sts2Env(worker_factory=FakeWorker)
    result = MaskablePPO(SharedActionPolicy, environment, n_steps=8, batch_size=8, seed=17)
    yield result
    environment.close()


def test_sampled_policy_uses_local_rng_and_respects_masks(model):
    environment = Sts2Env(worker_factory=FakeWorker)
    observation, _ = environment.reset(seed=3)
    rng = np.random.default_rng(87)
    state = copy.deepcopy(rng.bit_generator.state)
    torch_state = torch.get_rng_state().clone()
    actions = [
        evaluation_action(model, observation, environment.action_masks(), rng, False)
        for _ in range(80)
    ]
    assert set(actions) == {0, 1}
    torch.testing.assert_close(torch.get_rng_state(), torch_state, rtol=0, atol=0)
    torch.manual_seed(991)
    np.random.seed(192)
    restored = np.random.default_rng()
    restored.bit_generator.state = state
    assert actions == [
        evaluation_action(model, observation, environment.action_masks(), restored, False)
        for _ in range(80)
    ]
    mask = environment.action_masks()
    mask[0] = False
    assert evaluation_action(model, observation, mask, rng, False) == 1
    environment.close()


def test_sampled_episode_resume_matches_uninterrupted_evaluation(model):
    case = evaluation_plan("run", split="test")["cases"][0]
    direct = Sts2Env(worker_factory=FakeWorker, scope="run")
    expected = holdout.episode(model, direct, case, time.monotonic() + 10, deterministic=False)
    interrupted = Sts2Env(worker_factory=FakeWorker, scope="run")
    snapshots = []

    def checkpoint(snapshot):
        snapshots.append(copy.deepcopy(snapshot))
        if len(snapshot["environment"]["journal"]["actions"]) == 2:
            raise holdout.EvaluationPaused()

    with pytest.raises(holdout.EvaluationPaused):
        holdout.episode(
            model,
            interrupted,
            case,
            time.monotonic() + 10,
            checkpoint=checkpoint,
            deterministic=False,
        )
    resumed = Sts2Env(worker_factory=FakeWorker, scope="run")
    with pytest.raises(ValueError, match="different policy mode"):
        holdout.episode(model, resumed, case, time.monotonic() + 10, snapshots[-1])
    assert resumed.game.calls == []
    actual = holdout.episode(
        model, resumed, case, time.monotonic() + 10, snapshots[-1], deterministic=False
    )
    assert actual == expected
    assert resumed.game.calls == direct.game.calls


def test_both_evaluators_sample_identical_trajectories(model):
    first = Sts2Env(worker_factory=FakeWorker, scope="run", max_steps=4096)
    second = Sts2Env(worker_factory=FakeWorker, scope="run", max_steps=4096)
    expected = evaluate(model, first, seed=719, split="test", max_steps=4096, deterministic=False)
    actual = [
        holdout.episode(model, second, case, time.monotonic() + 10, deterministic=False)
        for case in expected["cases"]
    ]
    assert expected["policy"] == "learned_sampled"
    assert actual == expected["episodes"]
    assert first.game.calls == second.game.calls


def test_candidate_modes_share_weights_but_have_distinct_cache_identities(tmp_path):
    (tmp_path / "build.json").write_text('{"game": "test"}')
    (tmp_path / "policy.zip").write_bytes(b"test")
    study = {
        "complete": True,
        "build": {"game": "test"},
        "trials": [{"checkpoint": str(tmp_path), "variant": "control", "seed": 7}],
    }
    models = holdout.candidates(study, study["build"], ("deterministic", "sampled"))
    assert len(models) == 3
    assert len({model["name"] for model in models}) == 3
    assert models[1]["sha256"] == models[2]["sha256"]
    assert models[1]["checkpoint"] == models[2]["checkpoint"]
    assert json.dumps(models) != json.dumps(holdout.candidates(study, study["build"]))
    for modes in ((), ("sampled", "sampled"), ("invalid",)):
        with pytest.raises(ValueError, match="policy modes"):
            holdout.candidates(study, study["build"], modes)
