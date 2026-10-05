import argparse
import collections
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from ai4sts2.features import Batch, Layout, Translator, Vocabulary, length
from ai4sts2.kernel import ROOT, KernelError
from ai4sts2.model import Net
from ai4sts2.pool import Pool


class Normalizer:
    def __init__(self):
        self.count, self.mean, self.square = 1e-4, 0.0, 1.0

    def update(self, values):
        values = np.asarray(values, dtype=np.float64)
        total = self.count + values.size
        delta = values.mean() - self.mean
        self.mean += delta * values.size / total
        self.square = (
            self.square * self.count
            + values.var() * values.size
            + delta**2 * self.count * values.size / total
        ) / total
        self.count = total

    @property
    def std(self):
        return math.sqrt(self.square) + 1e-6

    def state_dict(self):
        return {"count": self.count, "mean": self.mean, "square": self.square}

    def load_state_dict(self, state):
        self.count, self.mean, self.square = state["count"], state["mean"], state["square"]


class Trainer:
    def __init__(self, args):
        self.args = args
        self.directory = Path(args.output)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.vocabulary = Vocabulary(self.directory / "vocab.json")
        self.rng = random.Random(args.seed)
        torch.manual_seed(args.seed)
        self.pool = Pool(args.workers)
        self.schema = self.pool.kernels[0].schema
        self.characters = self.pool.kernels[0].hello["characters"]
        self.layout = Layout.from_schema(self.schema)
        self.translators = [Translator(self.vocabulary) for _ in range(args.workers)]
        self.net = Net(self.layout, width=args.width, depth=args.depth)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr)
        self.returns = Normalizer()
        self.updates = 0
        self.steps = 0
        self.episodes = collections.deque(maxlen=200)
        self.completed = 0
        self.victories = 0
        checkpoint = self.directory / "model.pt"
        if checkpoint.exists():
            state = torch.load(checkpoint, weights_only=True)
            self.net.load_state_dict(state["net"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.returns.load_state_dict(state["returns"])
            self.updates, self.steps = state["updates"], state["steps"]
            self.completed, self.victories = state["completed"], state["victories"]
        self.decisions = [None] * args.workers
        self.floors = [0] * args.workers
        self.trajectories = [[] for _ in range(args.workers)]
        self.meta = [None] * args.workers

    def begin(self, worker):
        character = self.characters[self.rng.randrange(len(self.characters))]
        seed = "".join(self.rng.choice("0123456789ABCDEFGHJKLMNPQRSTUVWXYZ") for _ in range(10))
        self.meta[worker] = {"character": character, "seed": seed}
        self.pool.send(
            worker, "begin", character=character, seed=seed, ascension=self.args.ascension
        )

    def absorb(self, worker, decision):
        kernel = self.pool.kernels[worker]
        self.translators[worker].sync(kernel.vocab)
        trajectory = self.trajectories[worker]
        if trajectory and not trajectory[-1]["closed"]:
            trajectory[-1]["reward"] += decision["state"]["floor"] - self.floors[worker]
        self.floors[worker] = decision["state"]["floor"]
        if decision["kind"] == "end":
            if trajectory and not trajectory[-1]["closed"]:
                trajectory[-1]["done"] = True
                trajectory[-1]["closed"] = True
            self.completed += 1
            self.victories += bool(decision["victory"])
            self.episodes.append(
                {
                    **self.meta[worker],
                    "floor": decision["state"]["floor"],
                    "victory": decision["victory"],
                }
            )
            self.decisions[worker] = None
            self.begin(worker)
            return
        self.decisions[worker] = decision

    def act(self, workers):
        batch = Batch(
            [self.decisions[w]["observation"] for w in workers],
            [self.translators[w] for w in workers],
            self.layout,
            self.schema,
            self.vocabulary,
        )
        with torch.no_grad():
            logits, values = self.net(batch)
            distribution = torch.distributions.Categorical(logits=logits)
            actions = distribution.sample()
            logps = distribution.log_prob(actions)
        for row, worker in enumerate(workers):
            trajectory = self.trajectories[worker]
            if trajectory and not trajectory[-1]["closed"]:
                trajectory[-1]["closed"] = True
                trajectory[-1]["next_value"] = (
                    float(values[row]) * self.returns.std + self.returns.mean
                )
            action = int(actions[row])
            trajectory.append(
                {
                    "observation": self.decisions[worker]["observation"],
                    "translator": self.translators[worker],
                    "action": action,
                    "logp": float(logps[row]),
                    "value": float(values[row]) * self.returns.std + self.returns.mean,
                    "reward": 0.0,
                    "done": False,
                    "closed": False,
                }
            )
            self.decisions[worker] = None
            self.pool.send(worker, "step", index=action)
            self.steps += 1

    def collect(self, size):
        started = time.perf_counter()
        target = self.steps + size
        while self.steps < target:
            idle = [
                w
                for w, d in enumerate(self.decisions)
                if d is not None and w not in self.pool.pending
            ]
            if idle:
                self.act(idle)
            for worker, decision in self.pool.receive():
                self.absorb(worker, decision)
        while self.pool.pending:
            for worker, decision in self.pool.receive():
                self.absorb(worker, decision)
        waiting = [w for w in range(len(self.decisions)) if self.decisions[w] is not None]
        if waiting:
            batch = Batch(
                [self.decisions[w]["observation"] for w in waiting],
                [self.translators[w] for w in waiting],
                self.layout,
                self.schema,
                self.vocabulary,
            )
            with torch.no_grad():
                _, values = self.net(batch)
            for row, worker in enumerate(waiting):
                trajectory = self.trajectories[worker]
                if trajectory and not trajectory[-1]["closed"]:
                    trajectory[-1]["next_value"] = (
                        float(values[row]) * self.returns.std + self.returns.mean
                    )
                    trajectory[-1]["closed"] = True
        return time.perf_counter() - started

    def advantages(self):
        samples = []
        for worker, trajectory in enumerate(self.trajectories):
            advantage = 0.0
            for index in range(len(trajectory) - 1, -1, -1):
                step = trajectory[index]
                if step["done"]:
                    following, advantage = 0.0, 0.0
                else:
                    following = step["next_value"]
                    if index == len(trajectory) - 1:
                        advantage = 0.0
                delta = step["reward"] + following - step["value"]
                advantage = delta + self.args.lam * advantage
                step["advantage"] = advantage
                step["return"] = advantage + step["value"]
            samples.extend(trajectory)
            self.trajectories[worker] = []
        return samples

    def optimize(self, samples):
        returns = np.array([s["return"] for s in samples])
        self.returns.update(returns)
        advantages = np.array([s["advantage"] for s in samples], dtype=np.float32)
        advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
        targets = (returns - self.returns.mean) / self.returns.std
        lengths = np.array([length(s["observation"], self.schema) for s in samples])
        stats = collections.defaultdict(list)
        for _ in range(self.args.epochs):
            order = np.random.permutation(len(samples))
            group = self.args.minibatch * 8
            chunks = []
            for begin in range(0, len(order), group):
                block = order[begin : begin + group]
                block = block[np.argsort(lengths[block], kind="stable")]
                chunks += [
                    block[i : i + self.args.minibatch]
                    for i in range(0, len(block), self.args.minibatch)
                ]
            np.random.shuffle(chunks)
            for chunk in chunks:
                batch = Batch(
                    [samples[i]["observation"] for i in chunk],
                    [samples[i]["translator"] for i in chunk],
                    self.layout,
                    self.schema,
                    self.vocabulary,
                )
                logits, values = self.net(batch)
                distribution = torch.distributions.Categorical(logits=logits)
                actions = torch.tensor([samples[i]["action"] for i in chunk])
                old = torch.tensor([samples[i]["logp"] for i in chunk])
                advantage = torch.from_numpy(advantages[chunk])
                ratio = torch.exp(distribution.log_prob(actions) - old)
                clipped = torch.clamp(ratio, 1 - self.args.clip, 1 + self.args.clip)
                policy_loss = -torch.min(ratio * advantage, clipped * advantage).mean()
                value_loss = 0.5 * (values - torch.from_numpy(targets[chunk]).float()).pow(2).mean()
                entropy = distribution.entropy().mean()
                loss = policy_loss + value_loss - self.args.entropy * entropy
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()
                stats["policy"].append(policy_loss.item())
                stats["value"].append(value_loss.item())
                stats["entropy"].append(entropy.item())
                stats["clip"].append(float(((ratio - 1).abs() > self.args.clip).float().mean()))
        self.updates += 1
        return {key: float(np.mean(value)) for key, value in stats.items()}

    def save(self):
        self.vocabulary.save()
        state = {
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "returns": self.returns.state_dict(),
            "updates": self.updates,
            "steps": self.steps,
            "completed": self.completed,
            "victories": self.victories,
            "args": vars(self.args),
        }
        temporary = self.directory / "model.pt.tmp"
        torch.save(state, temporary)
        temporary.replace(self.directory / "model.pt")

    def run(self):
        for worker in range(len(self.decisions)):
            self.begin(worker)
        deadline = time.monotonic() + self.args.minutes * 60
        try:
            while time.monotonic() < deadline:
                collect = self.collect(self.args.rollout)
                samples = self.advantages()
                started = time.perf_counter()
                stats = self.optimize(samples)
                optimize = time.perf_counter() - started
                recent = list(self.episodes)
                report = {
                    "update": self.updates,
                    "steps": self.steps,
                    "samples": len(samples),
                    "collect_sps": len(samples) / collect,
                    "optimize_s": optimize,
                    "episodes": self.completed,
                    "victories": self.victories,
                    "floor": float(np.mean([e["floor"] for e in recent])) if recent else None,
                    "vocab": len(self.vocabulary),
                    **stats,
                }
                print(json.dumps(report), flush=True)
                with (self.directory / "log.jsonl").open("a") as stream:
                    stream.write(json.dumps(report) + "\n")
                if self.updates % self.args.save_every == 0:
                    self.save()
        finally:
            self.save()
            self.pool.close()


def main():
    cpus = len(os.sched_getaffinity(0))
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "artifacts/runs/ppo"))
    parser.add_argument("--workers", type=int, default=max(1, cpus // 2))
    parser.add_argument("--threads", type=int, default=0)
    parser.add_argument("--minutes", type=float, default=30)
    parser.add_argument("--ascension", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--rollout", type=int, default=4096)
    parser.add_argument("--minibatch", type=int, default=512)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--save-every", type=int, default=10)
    args = parser.parse_args()
    torch.set_num_threads(args.threads or max(1, cpus - args.workers))
    try:
        Trainer(args).run()
    except KernelError as error:
        raise SystemExit(str(error)[:4000]) from error


if __name__ == "__main__":
    main()
