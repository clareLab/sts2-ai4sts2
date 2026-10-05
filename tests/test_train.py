import random

import numpy as np
import pytest
import torch

from ai4sts2.features import Batch, Layout, Translator
from ai4sts2.kernel import Kernel, engine
from ai4sts2.model import Net
from ai4sts2.search import Evaluator, search
from ai4sts2.train import ACTS, COMBATS, Outcome, vtrace, wilson
from tests.test_kernel import branch, fingerprint


def distributions(rng, count, bins):
    values = rng.random((count, bins)).astype(np.float32)
    return values / values.sum(1, keepdims=True)


def test_vtrace_reduces_to_monte_carlo_on_policy():
    rng = np.random.default_rng(0)
    values = distributions(rng, 20, 9)
    after = distributions(rng, 1, 9)[0]
    targets, _ = vtrace(values, after, np.ones(20), 1.0, np.arange(9.0))
    assert np.allclose(targets, after, atol=1e-6)


def test_vtrace_matches_td_lambda_on_policy():
    rng = np.random.default_rng(1)
    values = distributions(rng, 12, 7)
    after = distributions(rng, 1, 7)[0]
    lam = 0.9
    targets, _ = vtrace(values, after, np.ones(12), lam, np.arange(7.0))
    following, expected = after, np.empty_like(values)
    for t in range(11, -1, -1):
        proposed = values[t + 1] if t < 11 else after
        following = (1 - lam) * proposed + lam * following
        expected[t] = following
    assert np.allclose(targets, expected, atol=1e-6)


def test_vtrace_targets_stay_distributions():
    rng = np.random.default_rng(2)
    values = distributions(rng, 30, 11)
    after = distributions(rng, 1, 11)[0]
    targets, advantages = vtrace(values, after, rng.random(30), 0.95, rng.random(11))
    assert np.allclose(targets.sum(1), 1, atol=1e-5)
    assert (targets >= -1e-7).all()
    assert np.isfinite(advantages).all()


@pytest.mark.parametrize("name", ["wkch", "wfh"])
def test_utility_is_monotone_in_outcome_order(name):
    outcome = Outcome(name)
    rng = np.random.default_rng(3)
    mass = rng.random(outcome.bins)
    utility = outcome.utility(mass / mass.sum())
    assert (np.diff(utility[:-1]) >= -1e-9).all()
    assert utility[-1] >= utility[:-1].max() - 1e-9
    assert not outcome.utility(None).any()


def test_outcome_targets_split_within_a_cell():
    outcome = Outcome("wkch")
    state = {"act": 1, "won": 4, "floor": 22, "foe": 0.25}
    target = outcome.target(state, False)
    cell = COMBATS + 4
    assert target.sum() == pytest.approx(1)
    assert target[2 * cell] == pytest.approx(0.25)
    assert target[2 * cell + 1] == pytest.approx(0.75)
    assert outcome.target(state, True)[-1] == 1
    assert outcome.bins == 2 * ACTS * COMBATS + 1


def test_wilson_interval_brackets_the_estimate():
    low, high = wilson(30, 100)
    assert low < 0.3 < high
    assert wilson(0, 0) == (0.0, 1.0)


@pytest.mark.game
def test_batched_forward_matches_single_observations():
    torch.manual_seed(0)
    rng = random.Random(4)
    observations = []
    with Kernel(engine()) as kernel:
        layout = Layout.from_schema(kernel.schema)
        translator = Translator()
        for run, character in enumerate(kernel.hello["characters"]):
            arguments = {"character": character, "seed": f"COLLATE-{run}", "ascension": 10}
            decision = kernel.call("begin", **arguments)
            while decision["kind"] != "end" and len(observations) < 30 * (run + 1):
                translator.sync(kernel.vocab)
                observations.append(translator.canonical(decision["observation"], layout))
                decision = kernel.call("step", index=rng.randrange(len(decision["options"])))
    net = Net(layout, 9).eval()
    with torch.no_grad():
        logits, values = net(Batch(observations, layout))
        for row, observation in enumerate(observations):
            single, value = net(Batch([observation], layout))
            width = single.shape[1]
            assert torch.allclose(logits[row, :width], single[0], atol=1e-4)
            assert torch.isinf(logits[row, width:]).all()
            assert torch.allclose(values[row], value[0], atol=1e-4)


@pytest.mark.game
def test_search_leaves_the_real_game_untouched():
    arguments = {"seed": "SEARCH", "ascension": 10}
    rng = random.Random(9)
    with Kernel(engine()) as kernel:
        arguments["character"] = kernel.hello["characters"][2]
        layout = Layout.from_schema(kernel.schema)
        outcome = Outcome("wkch")
        net = Net(layout, outcome.bins).eval()
        evaluator = Evaluator(
            kernel, net, layout, Translator(), outcome, np.linspace(0, 1, outcome.bins)
        )
        prefix = []
        decision = kernel.call("begin", **arguments)
        while decision["kind"] != "combat" or len(decision["options"]) < 3:
            prefix.append(rng.randrange(len(decision["options"])))
            decision = kernel.call("step", index=prefix[-1])
        root = fingerprint(kernel, decision)
        action, improved = search(evaluator, decision, 8, 4, np.random.default_rng(0))
        assert improved.sum() == pytest.approx(1)
        restored = kernel.call("replay", actions=[])
        assert fingerprint(kernel, restored) == root
        searched = branch(kernel, kernel.call("step", index=action), 3)
        kernel.call("begin", **arguments)
        kernel.call("replay", actions=prefix)
        assert branch(kernel, kernel.call("step", index=action), 3) == searched
