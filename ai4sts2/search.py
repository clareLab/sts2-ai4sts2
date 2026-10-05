import math

import numpy as np
import torch

from ai4sts2.features import Batch

VISIT = 50
SCALE = 0.1


class Evaluator:
    def __init__(self, kernel, net, layout, translator, outcome, utility):
        self.kernel, self.net, self.layout = kernel, net, layout
        self.translator, self.outcome, self.utility = translator, outcome, utility

    def forward(self, decision):
        self.translator.sync(self.kernel.vocab)
        observation = self.translator.canonical(decision["observation"], self.layout)
        with torch.inference_mode():
            logits, values = self.net(Batch([observation], self.layout))
        return logits[0].numpy(), float(self.utility @ torch.softmax(values[0], -1).numpy())

    def leaf(self, decision):
        if decision["kind"] == "end":
            return float(self.utility @ self.outcome.target(decision["state"], decision["victory"]))
        return self.forward(decision)[1]

    def simulate(self, root, action, seed, turn, rng):
        self.kernel.call("restore", id=root)
        self.kernel.call("determinize", seed=seed)
        decision = self.kernel.call("step", index=action)
        while decision["kind"] not in ("end", "horizon"):
            if decision["kind"] == "combat" and decision["state"]["turn"] > turn:
                break
            logits, _ = self.forward(decision)
            weights = np.exp(logits - logits.max())
            action = int(rng.choice(len(weights), p=weights / weights.sum()))
            decision = self.kernel.call("step", index=action)
        return self.leaf(decision)


def completed(values, visits, prior, value):
    visited = visits > 0
    if visited.any():
        weights = prior[visited] / prior[visited].sum()
        mixed = (value + visits.sum() * (weights @ values[visited])) / (visits.sum() + 1)
    else:
        mixed = value
    result = np.where(visited, values, mixed)
    low, high = result.min(), result.max()
    return (result - low) / max(high - low, 1e-8)


def search(evaluator, decision, simulations, width, rng, noise=True):
    logits, value = evaluator.forward(decision)
    count = len(logits)
    if count == 1:
        return 0, np.ones(1)
    prior = np.exp(logits - logits.max())
    prior /= prior.sum()
    gumbel = rng.gumbel(size=count) if noise else np.zeros(count)
    candidates = list(np.argsort(-(gumbel + logits))[: min(width, count)])
    totals, visits = np.zeros(count), np.zeros(count)
    root = evaluator.kernel.call("snapshot")["id"]
    evaluator.kernel.call("horizon", enabled=True)
    turn = decision["state"]["turn"]
    rounds = max(1, math.ceil(math.log2(len(candidates))))
    try:
        for _ in range(rounds):
            share = max(1, simulations // (rounds * len(candidates)))
            for _ in range(share):
                seed = int(rng.integers(1 << 62))
                for action in candidates:
                    totals[action] += evaluator.simulate(root, int(action), seed, turn, rng)
                    visits[action] += 1
            if len(candidates) == 1:
                break
            values = np.divide(totals, visits, out=np.zeros(count), where=visits > 0)
            sigma = (VISIT + visits.max()) * SCALE * completed(values, visits, prior, value)
            score = gumbel + logits + sigma
            candidates = sorted(candidates, key=lambda a: -score[a])[: max(1, len(candidates) // 2)]
        evaluator.kernel.call("restore", id=root)
    finally:
        evaluator.kernel.call("horizon", enabled=False)
        evaluator.kernel.call("release", id=root)
    values = np.divide(totals, visits, out=np.zeros(count), where=visits > 0)
    sigma = (VISIT + visits.max()) * SCALE * completed(values, visits, prior, value)
    improved = np.exp(logits + sigma - (logits + sigma).max())
    return int(candidates[0]), improved / improved.sum()
