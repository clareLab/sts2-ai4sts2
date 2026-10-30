import json
import os
import random
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from ai4sts2.evaluate import compare, summarize, wilson
from ai4sts2.features import Batch, Layout, Translator
from ai4sts2.kernel import Kernel, engine, sweep
from ai4sts2.model import Net
from ai4sts2.search import Evaluator, search
from ai4sts2.train import ACTS, COMBATS, IDLE, Outcome, Progress, chunks, vtrace
from tests.test_kernel import fingerprint, wander


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
    targets, advantages = vtrace(values, after, np.ones(12), lam, np.arange(7.0))
    following, expected = after, np.empty_like(values)
    for t in range(11, -1, -1):
        proposed = values[t + 1] if t < 11 else after
        following = (1 - lam) * proposed + lam * following
        expected[t] = following
    assert np.allclose(targets, expected, atol=1e-6)
    nexts = [*expected[1:], after]
    reference = [np.arange(7.0) @ (n - v) for n, v in zip(nexts, values, strict=True)]
    assert np.allclose(advantages, reference, atol=1e-5)


def test_vtrace_targets_stay_distributions():
    rng = np.random.default_rng(2)
    values = distributions(rng, 30, 11)
    after = distributions(rng, 1, 11)[0]
    targets, advantages = vtrace(values, after, rng.random(30), 0.95, rng.random(11))
    assert np.allclose(targets.sum(1), 1, atol=1e-5)
    assert (targets >= -1e-7).all()
    assert np.isfinite(advantages).all()


def test_utility_is_monotone_in_outcome_order():
    outcome = Outcome()
    rng = np.random.default_rng(3)
    mass = rng.random(outcome.bins)
    utility = outcome.utility(mass / mass.sum())
    assert (np.diff(utility[:-1]) >= -1e-9).all()
    assert utility[-1] >= utility[:-1].max() - 1e-9
    assert not outcome.utility(None).any()


def test_outcome_targets_split_within_a_cell():
    outcome = Outcome()
    state = {"act": 1, "won": 4, "floor": 22, "foe": 0.25}
    target = outcome.target(state, False)
    cell = COMBATS + 4
    assert target.sum() == pytest.approx(1)
    assert target[2 * cell] == pytest.approx(0.25)
    assert target[2 * cell + 1] == pytest.approx(0.75)
    assert outcome.target(state, True)[-1] == 1
    assert outcome.bins == 2 * ACTS * COMBATS + 1


def test_chunks_bound_attention_cells_and_cover_every_observation():
    rng = np.random.default_rng(0)
    lengths = np.concatenate([rng.integers(10, 120, 4000), [400, 900, 3000]]).tolist()
    parts = list(chunks(lengths, 512, 1 << 23))
    assert sorted(np.concatenate(parts).tolist()) == list(range(len(lengths)))
    for part in parts:
        longest = max(lengths[i] for i in part)
        assert len(part) <= 512
        assert len(part) == 1 or len(part) * longest**2 <= 1 << 23
    assert max(len(p) for p in parts) == 512


def test_sweep_keeps_live_and_fresh_kernel_directories(tmp_path):
    for name, pid in (("live", os.getpid()), ("dead", 2**22 + 1), ("fresh", None)):
        directory = tmp_path / f"kernel-{name}"
        directory.mkdir()
        if pid is not None:
            (directory / "pid").write_text(str(pid))
    sweep(tmp_path)
    assert sorted(p.name for p in tmp_path.iterdir()) == ["kernel-fresh", "kernel-live"]


def test_progress_tolerates_damaging_loops_and_flags_idle_ones():
    progress = Progress()
    assert not any(progress.stalled({"floor": 1, "foe": 1 - i / 1e5}) for i in range(5 * IDLE))
    assert not progress.stalled({"floor": 2, "foe": None})
    flags = [progress.stalled({"floor": 2, "foe": 0.5}) for _ in range(IDLE)]
    assert flags[-1] and not any(flags[:-1])


def test_summaries_and_paired_comparisons_count_runs_exactly(tmp_path):
    base = {"character": "A", "won": 0, "foe": 0.5, "elite": False, "farming": 0, "available": 1}
    base |= {"capped": False, "stalled": False, "start": 0.5}
    runs = [
        base | {"seed": "1", "act": 2, "floor": 49, "victory": True, "bosses": 4},
        base | {"seed": "2", "act": 0, "floor": 10, "victory": False, "bosses": 0},
        base | {"seed": "3", "act": 1, "floor": 25, "victory": False, "bosses": 1},
        base | {"seed": "4", "act": 2, "floor": 48, "victory": False, "bosses": 3},
        base | {"seed": "5", "act": 0, "floor": 2, "victory": False, "bosses": 0, "stalled": True},
    ]
    summary = summarize(runs)
    entry = summary["all"]
    assert summary["A"] == entry
    assert [entry[f"act{k}"][:2] for k in (1, 2, 3)] == [[3, 5], [2, 3], [1, 2]]
    assert entry["win"][:2] == [1, 5] and entry["bosses"] == [3, 2, 2, 1]
    assert (entry["stalled"], entry["capped"]) == (1, 0)
    first, second = tmp_path / "first.jsonl", tmp_path / "second.jsonl"
    first.write_text("".join(json.dumps(r) + "\n" for r in runs))
    better = [r | {"act": 1, "floor": 20} if r["seed"] == "2" else r for r in runs[:4]]
    second.write_text("".join(json.dumps(r) + "\n" for r in better))
    result = compare(first, second)
    assert result["pairs"] == 4 and result["unpaired"] == [1, 0]
    assert (result["act1"]["second_better"], result["act1"]["first_better"]) == (1, 0)


def test_wilson_interval_brackets_the_estimate():
    low, high = wilson(30, 100)
    assert low < 0.3 < high
    assert wilson(0, 0) == (0.0, 1.0)


@pytest.mark.parametrize("training", [False, True])
def test_encoder_matches_the_official_transformer(training):
    torch.manual_seed(0)
    net = Net(SimpleNamespace(width=lambda table: 1), 9).train(training)
    lengths = torch.tensor([5, 1, 9, 3])
    width = int(lengths.max())
    mask = torch.arange(width)[None, :] < lengths[:, None]
    tokens = mask.reshape(-1).nonzero().squeeze(1)
    batch = SimpleNamespace(size=len(lengths), width=width, mask=mask, tokens=tokens)
    flat = torch.randn(len(lengths) * width, net.width) * mask.reshape(-1, 1)
    with torch.set_grad_enabled(training):
        hidden = flat.view(len(lengths), width, net.width)
        official = net.encoder(hidden, src_key_padding_mask=~mask).reshape(-1, net.width)
        packed = net.encode(flat, batch)
    assert torch.allclose(packed[tokens], official[tokens], atol=1e-5)
    assert not packed[~mask.reshape(-1)].any()


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
@pytest.mark.parametrize("combat", [True, False])
def test_search_leaves_the_real_game_untouched(combat):
    arguments = {"seed": "SEARCH", "ascension": 10, "invincible": True}
    rng = random.Random(9)
    with Kernel(engine()) as kernel:
        arguments["character"] = kernel.hello["characters"][2]
        layout = Layout.from_schema(kernel.schema)
        outcome = Outcome()
        net = Net(layout, outcome.bins).eval()
        evaluator = Evaluator(
            kernel, net, layout, Translator(), outcome, np.linspace(0, 1, outcome.bins)
        )
        prefix = []
        decision = kernel.call("begin", **arguments)
        while (
            not decision["snapshot"]
            or (decision["kind"] == "combat") != combat
            or len(decision["options"]) < 3
            or decision["state"]["floor"] < 4
        ):
            prefix.append(rng.randrange(len(decision["options"])))
            decision = kernel.call("step", index=prefix[-1])
        root = fingerprint(kernel, decision)
        action, improved = search(evaluator, decision, 8, 4, np.random.default_rng(0))
        assert improved.sum() == pytest.approx(1)
        restored = kernel.call("replay", actions=[])
        assert fingerprint(kernel, restored) == root
        searched = wander(kernel, kernel.call("step", index=action), 3)
        kernel.call("begin", **arguments)
        kernel.call("replay", actions=prefix)
        assert wander(kernel, kernel.call("step", index=action), 3) == searched
