import argparse
import collections
import json
import math
import multiprocessing as mp
import multiprocessing.connection
import os
import queue
import random
import signal
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import torch

from ai4sts2.features import Batch, Layout, Translator, length
from ai4sts2.kernel import ROOT, Kernel, die_with_parent, engine
from ai4sts2.model import Net
from ai4sts2.pool import Pool

ACTS = 3
COMBATS = 18
FLOORS = 64
TARGET = 10
SEED_ALPHABET = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"


class Outcome:
    def __init__(self, name):
        self.name = name
        self.cells = ACTS * COMBATS if name == "wkch" else FLOORS
        self.bins = 2 * self.cells + 1

    def target(self, state, victory):
        result = np.zeros(self.bins, np.float32)
        if victory:
            result[-1] = 1
            return result
        if self.name == "wkch":
            cell = min(state["act"], ACTS - 1) * COMBATS + min(state["won"], COMBATS - 1)
        else:
            cell = min(state["floor"], FLOORS - 1)
        foe = state["foe"]
        h = 0.0 if self.name == "floor" or foe is None else 1.0 - foe
        result[2 * cell], result[2 * cell + 1] = 1 - h, h
        return result

    def utility(self, mass):
        if self.name == "floor":
            cells = np.arange(self.cells, dtype=np.float32)
            return np.append(np.stack([cells, cells + 1], 1).reshape(-1), self.cells)
        if mass is None:
            return np.zeros(self.bins, np.float32)
        cells = mass[:-1:2] + mass[1:-1:2]
        below = np.cumsum(cells) - cells
        utility = np.stack([below, below + cells], 1).reshape(-1)
        return np.append(utility, 1 - mass[-1] / 2).astype(np.float32)


class Statistics:
    def __init__(self, size, state=()):
        self.history = collections.defaultdict(lambda: collections.deque(maxlen=size))
        for group, records in state:
            self.history[tuple(group)].extend((np.asarray(t), v) for t, v in records)

    def add(self, group, target, victory):
        self.history[group].append((target, victory))

    def mass(self, group):
        records = self.history.get(group)
        return np.mean([target for target, _ in records], axis=0) if records else None

    def wins(self, group):
        records = self.history.get(group, ())
        return sum(victory for _, victory in records), len(records)

    def state(self):
        return [
            [list(group), [[t.tolist(), v] for t, v in records]]
            for group, records in self.history.items()
        ]


def wilson(successes, trials, z=1.959964):
    if trials == 0:
        return 0.0, 1.0
    p = successes / trials
    centre = p + z * z / (2 * trials)
    margin = z * math.sqrt(p * (1 - p) / trials + z * z / (4 * trials * trials))
    scale = 1 + z * z / trials
    return (centre - margin) / scale, (centre + margin) / scale


def vtrace(values, after, rho, lam, utility):
    targets = np.empty_like(values)
    advantages = np.empty(len(values), np.float32)
    following = proposed = after
    for t in range(len(values) - 1, -1, -1):
        value = values[t]
        advantages[t] = utility @ (following - value)
        following = value + rho[t] * (proposed - value + lam * (following - proposed))
        targets[t], proposed = following, value
    return targets, advantages


def make_net(layout, outcome, args):
    return Net(layout, outcome.bins, width=args.width, depth=args.depth)


class Outbox:
    def __init__(self, connection, size):
        self.queue = queue.Queue(maxsize=size)
        threading.Thread(target=self.pump, args=(connection,), daemon=True).start()

    def pump(self, connection):
        while True:
            connection.send(self.queue.get())

    def send(self, message):
        self.queue.put(message)


def actor(index, args, executable, weights_path, schedule, connection):
    die_with_parent()
    channel = Outbox(connection, max(1, 2 * args.rollout // args.segment // args.actors))
    torch.set_num_threads(1)
    rng = random.Random(f"{args.seed}:{index}:{time.time_ns()}")
    outcome = Outcome(args.outcome)
    pool = Pool(args.kernels, executable)
    schema, characters = pool.kernels[0].schema, pool.kernels[0].hello["characters"]
    layout = Layout.from_schema(schema)
    net = make_net(layout, outcome, args)
    seen, stamp = -1, None
    translators = [Translator() for _ in range(len(pool))]
    episodes = [None] * len(pool)
    segments = [None] * len(pool)
    decisions = [None] * len(pool)
    weights, frontier, utilities = schedule
    archive = collections.deque(maxlen=args.archive)

    def begin(worker):
        if archive and rng.random() < args.restarts:
            entry = rng.choices(archive, weights=[e["spread"] + 1e-6 for e in archive])[0]
            group, seed = entry["group"], entry["seed"]
            episodes[worker] = {"restart": entry, "phase": "replay", "actions": []}
        else:
            mix = list(weights[:])
            c = rng.choices(range(len(characters)), weights=mix)[0]
            ascension = frontier[c] if frontier[c] < TARGET and rng.random() < 0.5 else TARGET
            seed = "".join(rng.choice(SEED_ALPHABET) for _ in range(10))
            group = (characters[c], ascension)
            episodes[worker] = {"restart": None, "phase": None, "actions": [], "start": None}
        segments[worker] = {"group": group, "seed": seed, "version": seen, "steps": []}
        pool.send(worker, "begin", character=group[0], seed=seed, ascension=group[1])

    def send(worker, op, **arguments):
        try:
            pool.send(worker, op, **arguments)
        except Exception as error:
            fail(worker, error)

    def start(worker):
        try:
            begin(worker)
        except Exception as error:
            fail(worker, error)

    def fail(worker, error):
        decisions[worker] = None
        while True:
            channel.send(("failure", repr(error)[-2000:]))
            try:
                pool.restart(worker)
                translators[worker] = Translator()
                begin(worker)
                return
            except Exception as again:
                error = again
                time.sleep(5)

    def reload():
        nonlocal seen, stamp
        try:
            current = os.stat(weights_path).st_mtime_ns
            if current == stamp:
                return
            state = torch.load(weights_path, weights_only=True)
        except (FileNotFoundError, EOFError, RuntimeError):
            return
        net.load_state_dict(state["net"])
        seen, stamp = state["version"], current

    def flush(worker, **ending):
        segment = segments[worker]
        if segment["steps"]:
            channel.send(("segment", {**segment, **ending}))

    def remember(worker):
        start, segment = episodes[worker]["start"], segments[worker]
        archive.append({"group": segment["group"], "seed": segment["seed"]} | start)

    def absorb(worker, decision):
        if isinstance(decision, Exception):
            fail(worker, decision)
            return
        translators[worker].sync(pool.kernels[worker].vocab)
        episode, segment = episodes[worker], segments[worker]
        if episode["phase"] == "replay":
            episode["phase"] = "resample"
            send(worker, "replay", actions=episode["restart"]["prefix"])
            return
        if episode["phase"] == "resample":
            episode["phase"] = None
            observation = translators[worker].canonical(decision["observation"], layout)
            expected = episode["restart"]["observation"]
            if any(not np.array_equal(observation[k], expected[k]) for k in expected):
                channel.send(("divergence", segment["group"], segment["seed"]))
                start(worker)
                return
            send(worker, "determinize", seed=rng.getrandbits(63))
            return
        state = decision["state"]
        if decision["kind"] == "end":
            victory = bool(decision["victory"])
            target = outcome.target(state, victory)
            flush(worker, target=target)
            if episode["restart"] is None:
                if episode["start"] is not None and episode["start"]["room"] == "Monster":
                    remember(worker)
                channel.send(("episode", segment["group"], segment["seed"], target, victory, state))
            else:
                channel.send(("restart", episode["restart"]["room"], victory, state))
            start(worker)
            return
        observation = translators[worker].canonical(decision["observation"], layout)
        if episode["restart"] is not None and not state["combat"]:
            flush(worker, bootstrap=observation)
            channel.send(("restart", episode["restart"]["room"], True, state))
            start(worker)
            return
        if episode["restart"] is None:
            if state["combat"] and episode["start"] is None:
                prefix = list(episode["actions"])
                episode["start"] = {"prefix": prefix, "room": state["room"], "spread": None}
                episode["start"]["observation"] = observation
            elif not state["combat"]:
                episode["start"] = None
        if len(segment["steps"]) >= args.segment:
            flush(worker, bootstrap=observation)
            segments[worker] = {**segment, "version": seen, "steps": []}
        decisions[worker] = observation

    def spread(group, logits):
        c = characters.index(group[0])
        offset = (c * (TARGET + 1) + group[1]) * outcome.bins
        utility = np.frombuffer(utilities, np.float32)[offset : offset + outcome.bins]
        probabilities = torch.softmax(logits, -1).numpy()
        mean = probabilities @ utility
        return float(np.sqrt(max(probabilities @ utility**2 - mean**2, 0.0)))

    reload()
    for worker in range(len(pool)):
        start(worker)
    activity, reported = np.zeros(3), time.monotonic()
    while True:
        if time.monotonic() - reported > 5:
            channel.send(("actor", activity))
            activity, reported = np.zeros(3), time.monotonic()
        reload()
        idle = [w for w, d in enumerate(decisions) if d is not None and w not in pool.pending]
        clock = time.perf_counter()
        if idle:
            batch = Batch([decisions[w] for w in idle], layout)
            with torch.no_grad():
                logits, values = net(batch)
                distribution = torch.distributions.Categorical(logits=logits)
                actions = distribution.sample()
                logps = distribution.log_prob(actions)
            activity += (len(idle), time.perf_counter() - clock, 0)
            for row, worker in enumerate(idle):
                episode, segment = episodes[worker], segments[worker]
                opening = episode.get("start")
                if opening is not None and opening["spread"] is None:
                    opening["spread"] = spread(segment["group"], values[row])
                    if opening["room"] in ("Elite", "Boss"):
                        remember(worker)
                action = int(actions[row])
                segment["steps"].append((decisions[worker], action, float(logps[row])))
                episode["actions"].append(action)
                decisions[worker] = None
                send(worker, "step", index=action)
        results = pool.receive(timeout=0 if idle else 5)
        working = time.perf_counter()
        for worker, decision in results:
            absorb(worker, decision)
        activity[2] += time.perf_counter() - working + (working - clock if idle else 0)


class Learner:
    def __init__(self, args):
        self.args = args
        self.directory = Path(args.output)
        self.directory.mkdir(parents=True, exist_ok=True)
        self.device = torch.device("xpu" if torch.xpu.is_available() else "cpu")
        torch.manual_seed(args.seed)
        self.outcome = Outcome(args.outcome)
        self.executable = engine()
        with Kernel(self.executable) as kernel:
            self.schema, self.hello = kernel.schema, kernel.hello
        self.characters = self.hello["characters"]
        self.layout = Layout.from_schema(self.schema)
        self.context = mp.get_context("spawn")
        self.weights_path = Path("/dev/shm") / f"ai4sts2-{os.getpid()}.pt"
        self.weights = self.context.Array("d", [1.0] * len(self.characters), lock=False)
        self.frontier = self.context.Array(
            "i", [0 if args.frontier else TARGET] * len(self.characters), lock=False
        )
        self.utilities = self.context.Array(
            "f", len(self.characters) * (TARGET + 1) * self.outcome.bins, lock=False
        )
        self.restarts = collections.defaultdict(lambda: [0, 0])
        self.divergences = 0
        self.activity = np.zeros(3)
        self.net = make_net(self.layout, self.outcome, args).to(self.device)
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=args.lr)
        self.updates = self.samples = self.episodes = self.failures = 0
        statistics = ()
        path = self.directory / "model.pt"
        if path.exists():
            state = torch.load(path, map_location="cpu", weights_only=True)
            self.net.load_state_dict(state["net"])
            self.optimizer.load_state_dict(state["optimizer"])
            self.updates, self.samples = state["updates"], state["samples"]
            self.episodes, statistics = state["episodes"], state["statistics"]
            if args.frontier:
                self.frontier[:] = state["frontier"]
        self.statistics = Statistics(args.reference, statistics)
        self.publish()
        self.commit = subprocess.run(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, capture_output=True, text=True
        ).stdout.strip()
        self.readers = {}
        self.actors = [self.spawn(i) for i in range(args.actors)]
        self.pending, self.size, self.respawns = [], 0, 0

    def spawn(self, index):
        shared = (self.weights, self.frontier, self.utilities)
        reader, writer = self.context.Pipe(duplex=False)
        process = self.context.Process(
            target=actor,
            args=(index, self.args, self.executable, self.weights_path, shared, writer),
            daemon=True,
        )
        process.start()
        writer.close()
        self.readers[index] = reader
        return process

    def supervise(self):
        for index, process in enumerate(self.actors):
            if not process.is_alive() or index not in self.readers:
                process.kill()
                process.join()
                self.respawns += 1
                print(json.dumps({"respawn": index, "exitcode": process.exitcode}), flush=True)
                self.actors[index] = self.spawn(index)

    def publish(self):
        state = {k: v.detach().cpu() for k, v in self.net.state_dict().items()}
        temporary = self.weights_path.with_suffix(".tmp")
        torch.save({"net": state, "version": self.updates}, temporary)
        temporary.replace(self.weights_path)

    def receive(self, block):
        ready = multiprocessing.connection.wait(list(self.readers.values()), 5 if block else 0)
        for reader in ready:
            try:
                self.handle(*reader.recv())
            except (EOFError, OSError):
                index = next(i for i, r in self.readers.items() if r is reader)
                self.readers.pop(index).close()
        if not ready or len(self.readers) < len(self.actors):
            self.supervise()
        return bool(ready)

    def handle(self, kind, *payload):
        if kind == "segment":
            self.pending.append(payload[0])
            self.size += len(payload[0]["steps"])
        elif kind == "episode":
            self.record(*payload)
        elif kind == "restart":
            room, survived, _ = payload
            self.restarts[room][0] += survived
            self.restarts[room][1] += 1
        elif kind == "divergence":
            self.divergences += 1
        elif kind == "actor":
            self.activity += payload[0]
        elif kind == "failure":
            self.failures += 1
            print(json.dumps({"failure": payload[0]}), flush=True)

    def record(self, group, seed, target, victory, state):
        self.episodes += 1
        self.statistics.add(group, target, victory)
        record = {"character": group[0], "ascension": group[1], "seed": seed, "victory": victory}
        record |= {k: state[k] for k in ("act", "floor", "won", "foe", "hp", "deck", "relics")}
        with (self.directory / "episodes.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")

    def schedule(self):
        losses = []
        for c, character in enumerate(self.characters):
            wins, trials = self.statistics.wins((character, TARGET))
            losses.append(1 - wins / trials if trials else 1.0)
            while self.frontier[c] < TARGET:
                wins, trials = self.statistics.wins((character, self.frontier[c]))
                if trials < self.args.reference or wilson(wins, trials)[0] < 0.5:
                    break
                self.frontier[c] += 1
        total = sum(losses)
        share = [loss / total if total else 1 / len(losses) for loss in losses]
        self.weights[:] = [0.5 / len(losses) + 0.5 * s for s in share]
        table = np.zeros((len(self.characters), TARGET + 1, self.outcome.bins), np.float32)
        for c, character in enumerate(self.characters):
            for ascension in range(TARGET + 1):
                table[c, ascension] = self.outcome.utility(
                    self.statistics.mass((character, ascension))
                )
        np.frombuffer(self.utilities, np.float32)[:] = table.reshape(-1)

    def run(self):
        deadline = time.monotonic() + self.args.minutes * 60
        saved = cycle = time.monotonic()
        snapshot = self.samples // self.args.snapshot_samples * self.args.snapshot_samples
        try:
            while time.monotonic() < deadline and self.samples < self.args.samples:
                started = time.monotonic()
                while self.size < self.args.rollout and time.monotonic() < deadline:
                    self.receive(block=True)
                if self.size < self.args.rollout:
                    break
                self.supervise()
                waited = time.monotonic() - started
                block, self.pending, self.size = self.pending, [], 0
                lag = float(np.mean([self.updates - s["version"] for s in block]))
                stats = self.update(block)
                self.schedule()
                now = time.monotonic()
                report = {
                    "update": self.updates,
                    "samples": self.samples,
                    "cycle": round(now - cycle, 2),
                    "wait": round(waited, 2),
                    "actors": self.throughput(now - cycle),
                    "lag": round(lag, 2),
                    "episodes": self.episodes,
                    "failures": self.failures,
                    "respawns": self.respawns,
                    "restarts": {room: v for room, v in sorted(self.restarts.items())},
                    "divergences": self.divergences,
                    **stats,
                    "characters": self.summary(),
                }
                cycle = now
                print(json.dumps(report), flush=True)
                with (self.directory / "log.jsonl").open("a") as stream:
                    stream.write(json.dumps(report) + "\n")
                if time.monotonic() - saved > self.args.save_minutes * 60:
                    self.save()
                    saved = time.monotonic()
                every = self.args.snapshot_samples
                if self.samples >= snapshot + every:
                    snapshot = self.samples // every * every
                    self.save(self.directory / "snapshots" / f"{snapshot}.pt")
        finally:
            self.save()
            for process in self.actors:
                process.kill()
            self.weights_path.unlink(missing_ok=True)

    def throughput(self, elapsed):
        steps, inference, busy = self.activity
        self.activity = np.zeros(3)
        return {
            "sps": round(steps / elapsed, 1),
            "inference": round(inference / max(busy, 1e-9), 3),
            "load": round(busy / elapsed / self.args.actors, 3),
        }

    def batches(self, observations):
        order = np.argsort([length(o) for o in observations], kind="stable")
        for begin in range(0, len(order), self.args.minibatch):
            chunk = order[begin : begin + self.args.minibatch]
            yield chunk, Batch([observations[i] for i in chunk], self.layout).to(self.device)

    def update(self, segments):
        clock = [time.perf_counter()]
        steps = [step for segment in segments for step in segment["steps"]]
        count = len(steps)
        tails = [s["bootstrap"] for s in segments if "bootstrap" in s]
        actions = np.array([action for _, action, _ in steps] + [0] * len(tails))
        behaviour = np.array([logp for _, _, logp in steps] + [0] * len(tails), np.float32)
        prepared = list(self.batches([observation for observation, _, _ in steps] + tails))
        clock.append(time.perf_counter())
        bins = self.outcome.bins
        current = np.zeros(count + len(tails), np.float32)
        distributions = np.zeros((count + len(tails), bins), np.float32)
        with torch.no_grad():
            for chunk, batch in prepared:
                logits, values = self.net(batch)
                distributions[chunk] = torch.softmax(values.float(), -1).cpu().numpy()
                taken = torch.from_numpy(actions[chunk]).to(self.device)
                logp = torch.log_softmax(logits.float(), -1).gather(1, taken[:, None])[:, 0]
                current[chunk] = logp.cpu().numpy()
        clock.append(time.perf_counter())
        rho = np.minimum(1.0, np.exp(current - behaviour))
        targets = np.zeros((count + len(tails), bins), np.float32)
        advantages = np.zeros(count + len(tails), np.float32)
        groups = {segment["group"] for segment in segments}
        utilities = {g: self.outcome.utility(self.statistics.mass(g)) for g in groups}
        stop, tail = 0, count
        for segment in segments:
            start, stop = stop, stop + len(segment["steps"])
            if "bootstrap" in segment:
                after = distributions[tail]
                tail += 1
            else:
                after = segment["target"]
            targets[start:stop], advantages[start:stop] = vtrace(
                distributions[start:stop],
                after,
                rho[start:stop],
                self.args.lam,
                utilities[segment["group"]],
            )
        scale = np.sqrt(np.mean(advantages[:count] ** 2))
        advantages[:count] = advantages[:count] / scale if scale > 1e-6 else 0
        clock.append(time.perf_counter())
        stats = collections.defaultdict(list)
        passes = 0
        while passes < self.args.passes and (passes == 0 or self.size < self.args.rollout):
            passes += 1
            for k in np.random.permutation(len(prepared)):
                while self.size < self.args.rollout and self.receive(block=False):
                    pass
                chunk, batch = prepared[k]
                if not (chunk < count).any():
                    continue
                inside = torch.from_numpy(chunk < count).to(self.device)
                weight = inside.float() / self.args.minibatch
                logits, values = self.net(batch)
                distribution = torch.distributions.Categorical(logits=logits)
                taken = torch.from_numpy(actions[chunk]).to(self.device)
                old = torch.from_numpy(behaviour[chunk]).to(self.device)
                ratio = torch.exp(distribution.log_prob(taken) - old)
                advantage = torch.from_numpy(advantages[chunk]).to(self.device)
                clipped = torch.clamp(ratio, 1 - self.args.clip, 1 + self.args.clip)
                surrogate = torch.min(ratio * advantage, clipped * advantage)
                policy_loss = -(surrogate * weight).sum()
                target = torch.from_numpy(targets[chunk]).to(self.device)
                value_loss = -((target * torch.log_softmax(values, -1)).sum(-1) * weight).sum()
                entropy = (distribution.entropy() * weight).sum()
                loss = policy_loss + value_loss - self.args.entropy * entropy
                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.optimizer.step()
                clip = ((ratio - 1).abs() > self.args.clip).float()
                stats["policy"].append(policy_loss.item())
                stats["value"].append(value_loss.item())
                stats["entropy"].append(entropy.item())
                stats["clip"].append((clip * weight).sum().item())
        self.updates += 1
        self.samples += count
        self.publish()
        clock.append(time.perf_counter())
        result = {key: round(float(np.mean(value)), 5) for key, value in stats.items()}
        result["passes"] = passes
        result["rho"] = round(float(rho[:count].mean()), 4)
        result["time"] = [round(b - a, 2) for a, b in zip(clock, clock[1:], strict=False)]
        return result

    def summary(self):
        result = {}
        for c, character in enumerate(self.characters):
            records = self.statistics.history.get((character, TARGET), ())
            if not records:
                continue
            targets = np.array([target for target, _ in list(records)[-200:]])
            result[character] = {"win": round(float(targets[:, -1].mean()), 4)}
            result[character]["frontier"] = self.frontier[c]
            if self.outcome.name == "wkch":
                losses = targets[:, -1] < 1
                if losses.any():
                    cells = targets[losses, :-1:2] + targets[losses, 1:-1:2]
                    cells = cells.reshape(-1, ACTS, COMBATS)
                    acts, won = cells.sum(2) @ np.arange(ACTS), cells.sum(1) @ np.arange(COMBATS)
                    result[character] |= {"act": round(float(acts.mean()), 3)}
                    result[character] |= {"won": round(float(won.mean()), 2)}
        return result

    def save(self, path=None):
        state = {
            "net": self.net.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "updates": self.updates,
            "samples": self.samples,
            "episodes": self.episodes,
            "statistics": self.statistics.state(),
            "frontier": list(self.frontier),
            "game": self.hello.get("game"),
            "commit": self.commit,
            "args": vars(self.args),
        }
        path = path or self.directory / "model.pt"
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        torch.save(state, temporary)
        temporary.replace(path)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=str(ROOT / "artifacts/runs/train"))
    parser.add_argument("--outcome", choices=("wkch", "wfh", "floor"), default="wkch")
    parser.add_argument("--frontier", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--actors", type=int, default=4)
    parser.add_argument("--kernels", type=int, default=4)
    parser.add_argument("--minutes", type=float, default=60)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--rollout", type=int, default=4096)
    parser.add_argument("--segment", type=int, default=64)
    parser.add_argument("--minibatch", type=int, default=512)
    parser.add_argument("--passes", type=int, default=4)
    parser.add_argument("--clip", type=float, default=0.2)
    parser.add_argument("--entropy", type=float, default=0.01)
    parser.add_argument("--lam", type=float, default=0.95)
    parser.add_argument("--reference", type=int, default=1024)
    parser.add_argument("--restarts", type=float, default=0.0)
    parser.add_argument("--archive", type=int, default=256)
    parser.add_argument("--save-minutes", type=float, default=10)
    parser.add_argument("--samples", type=int, default=10**12)
    parser.add_argument("--snapshot-samples", type=int, default=10**6)
    args = parser.parse_args()
    torch.set_num_threads(2)
    signal.signal(signal.SIGTERM, lambda *_: sys.exit(0))
    Learner(args).run()


if __name__ == "__main__":
    main()
