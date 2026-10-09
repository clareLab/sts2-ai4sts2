import math

import numpy as np
import torch

from ai4sts2.features import Batch

VISIT = 50
SCALE = 0.1
FLOOR = 0.05
DEPTH = 200
LINKS = ("card", "creature", "relic", "potion", "node", "cell")


class Evaluator:
    def __init__(self, kernel, net, layout, translator, outcome, utility):
        self.kernel, self.net, self.layout = kernel, net, layout
        self.translator, self.outcome, self.utility = translator, outcome, utility

    def observe(self, decision):
        self.translator.sync(self.kernel.vocab)
        return self.translator.canonical(decision["observation"], self.layout)

    def forward(self, decision, observation=None):
        observation = observation or self.observe(decision)
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
        for _ in range(DEPTH):
            if decision["kind"] == "end":
                break
            if decision["state"]["combat"]:
                if decision["kind"] == "combat" and decision["state"]["turn"] > turn:
                    break
            elif decision["snapshot"]:
                break
            logits, _ = self.forward(decision)
            weights = np.exp(logits - logits.max())
            action = int(rng.choice(len(weights), p=weights / weights.sum()))
            decision = self.kernel.call("step", index=action)
        return self.leaf(decision)


def groups(observation, layout):
    options = observation["option"]
    columns = layout.columns["option"]
    keys = {}
    links = [columns[link] for link in LINKS]
    for index, row in enumerate(options):
        plain = row.copy()
        plain[links] = 0
        key = [plain.tobytes()]
        for link, column in zip(LINKS, links, strict=True):
            target = int(row[column])
            key.append(observation[link][target].tobytes() if target >= 0 else b"-")
        keys.setdefault(b"|".join(key), []).append(index)
    return list(keys.values())


def schedule(considered, simulations):
    if considered <= 1:
        return list(range(simulations))
    rounds = math.ceil(math.log2(considered))
    sequence, visits = [], [0] * considered
    while len(sequence) < simulations:
        extra = max(1, int(simulations / (rounds * considered)))
        for _ in range(extra):
            sequence.extend(visits[:considered])
            for i in range(considered):
                visits[i] += 1
        considered = max(2, considered // 2)
    return sequence[:simulations]


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
    observation = evaluator.observe(decision)
    raw, value = evaluator.forward(decision, observation)
    members = groups(observation, evaluator.layout)
    count = len(members)
    if len(raw) == 1:
        return 0, np.ones(1)
    logits = np.array([np.logaddexp.reduce(raw[m]) for m in members])
    prior = np.exp(logits - logits.max())
    prior = (1 - FLOOR) * prior / prior.sum() + FLOOR / count
    logits = np.log(prior)
    gumbel = rng.gumbel(size=count) if noise else np.zeros(count)
    considered = min(width, count)
    totals, visits = np.zeros(count), np.zeros(count)
    seeds = {}
    root = evaluator.kernel.call("snapshot")["id"]
    turn = decision["state"]["turn"] if decision["state"]["combat"] else 0

    def sigma():
        values = np.divide(totals, visits, out=np.zeros(count), where=visits > 0)
        return (VISIT + visits.max()) * SCALE * completed(values, visits, prior, value)

    try:
        top = set(np.argsort(-(gumbel + logits))[:considered].tolist())
        for visit in schedule(considered, simulations if count > 1 else 0):
            score = gumbel + logits + sigma()
            eligible = [a for a in top if visits[a] == visit]
            if not eligible:
                continue
            action = max(eligible, key=lambda a: score[a])
            if visit not in seeds:
                seeds[visit] = int(rng.integers(1 << 62))
            real = members[action][0]
            totals[action] += evaluator.simulate(root, real, seeds[visit], turn, rng)
            visits[action] += 1
        evaluator.kernel.call("restore", id=root)
    finally:
        if not evaluator.kernel.failed:
            evaluator.kernel.call("release", id=root)
    score = gumbel + logits + sigma()
    most = visits.max()
    choice = max((a for a in top if visits[a] == most), key=lambda a: score[a])
    improved = np.exp(logits + sigma() - (logits + sigma()).max())
    improved /= improved.sum()
    policy = np.zeros(len(raw))
    for group, mass in zip(members, improved, strict=True):
        policy[group] = mass / len(group)
    return members[choice][0], policy
