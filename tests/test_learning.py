import json
import zipfile

import numpy as np
import pytest
import torch
from ai4sts2.environment import Sts2Env
from ai4sts2.policy import SharedActionPolicy
from ai4sts2.train import TimedPPO
from sb3_contrib import MaskablePPO
from test_environment import FakeWorker


@pytest.mark.parametrize("policy", ["MultiInputPolicy", SharedActionPolicy])
def test_diagnostics_preserve_learning_and_capture_every_rollout(policy, tmp_path):
    torch.set_num_threads(1)
    results = []
    for algorithm in (MaskablePPO, TimedPPO):
        environment = Sts2Env(worker_factory=FakeWorker)
        model = algorithm(policy, environment, n_steps=16, batch_size=8, n_epochs=2, seed=21)
        model.learn(48)
        results.append(
            (
                {key: value.clone() for key, value in model.policy.state_dict().items()},
                torch.get_rng_state().clone(),
                np.random.get_state(),
            )
        )
        environment.close()
    for key in results[0][0]:
        torch.testing.assert_close(results[0][0][key], results[1][0][key], rtol=0, atol=0)
    torch.testing.assert_close(results[0][1], results[1][1], rtol=0, atol=0)
    np.testing.assert_equal(results[0][2], results[1][2])
    model.save(tmp_path / "policy.zip")
    with zipfile.ZipFile(tmp_path / "policy.zip") as archive:
        data = json.loads(archive.read("data"))
        assert "learning_diagnostics" not in data
        assert "optimisation_seconds" not in data
    rows = model.drain_diagnostics()
    assert [row["environment_steps"] for row in rows] == [16, 32, 48]
    assert sum(row["samples"] for row in rows) == 48
    assert sum(row["positive_rewards"] for row in rows) == 12
    assert sum(row["negative_rewards"] for row in rows) == 0
    for row in rows:
        assert row["choice_fraction"] == 1
        assert row["entropy_ceiling"] == pytest.approx(np.log(2))
        assert row["head_update_l2"]["action_net"] > 0
        assert row["head_update_l2"]["value_net"] > 0
        assert row["advantages"]["std"] > 0
        assert row["optimiser"]["value_loss"] >= 0
        for name in ("rewards", "values", "returns", "advantages"):
            assert row[name]["nonfinite"] == 0
    json.dumps(rows, allow_nan=False)
    assert model.drain_diagnostics() == []
    restored = TimedPPO.load(tmp_path / "policy.zip", device="cpu")
    assert restored.drain_diagnostics() == []
