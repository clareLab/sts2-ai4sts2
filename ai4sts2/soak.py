import argparse
import collections
import json
import random
import time

from ai4sts2.kernel import Kernel, KernelError, engine

PASSIVE = {"end_turn", "discard_potion", "skip_rewards", "skip_selection", "leave"}


class Worker:
    def __init__(self, executable, index, characters, ascension, invincible, seed):
        self.executable = executable
        self.index = index
        self.characters = characters
        self.ascension = ascension
        self.invincible = invincible
        self.rng = random.Random(seed * 100003 + index)
        self.kernel = None
        self.run = 0
        self.steps = 0
        self.seed = None

    def start(self):
        if self.kernel is None:
            self.kernel = Kernel(self.executable, connect=False)
            self.kernel.connect()
        character = self.characters[(self.index + self.run) % len(self.characters)]
        self.seed = f"S{self.index:03d}R{self.run:05d}"
        self.character = character
        self.steps = 0
        self.kernel.send(
            "begin",
            character=character,
            seed=self.seed,
            ascension=self.ascension,
            invincible=self.invincible,
        )

    def choose(self, decision):
        options = decision["options"]
        weights = [0.2 if o["kind"] in PASSIVE else 1.0 for o in options]
        return self.rng.choices(range(len(options)), weights)[0]


def soak(runs, workers, ascension=10, invincible=False, seed=0, limit=20000):
    import selectors

    executable = engine()
    hello = None
    pool = []
    selector = selectors.DefaultSelector()
    stats = collections.Counter()
    failures = []
    outcomes = []
    started = time.perf_counter()
    try:
        for index in range(workers):
            worker = Worker(executable, index, [], ascension, invincible, seed)
            worker.kernel = Kernel(executable, connect=False)
            pool.append(worker)
        for worker in pool:
            worker.kernel.connect()
            hello = hello or worker.kernel.hello
            worker.characters = hello["characters"]
        launched = 0
        for worker in pool:
            if launched < runs:
                worker.start()
                launched += 1
                selector.register(worker.kernel, selectors.EVENT_READ, worker)
        while selector.get_map():
            for key, _ in selector.select():
                worker = key.data
                try:
                    decision = worker.kernel.receive()
                except KernelError as error:
                    selector.unregister(key.fileobj)
                    failures.append(
                        {
                            "character": worker.character,
                            "seed": worker.seed,
                            "steps": worker.steps,
                            "error": str(error)[:4000],
                        }
                    )
                    worker.kernel.close()
                    worker.kernel = None
                    worker.run += 1
                    if launched < runs:
                        worker.start()
                        launched += 1
                        selector.register(worker.kernel, selectors.EVENT_READ, worker)
                    continue
                stats[decision["kind"]] += 1
                if decision["kind"] == "end" or worker.steps >= limit:
                    outcomes.append(
                        {
                            "character": worker.character,
                            "seed": worker.seed,
                            "victory": decision.get("victory"),
                            "steps": worker.steps,
                            **decision["state"],
                        }
                    )
                    worker.run += 1
                    if launched < runs:
                        worker.start()
                        launched += 1
                    else:
                        selector.unregister(key.fileobj)
                    continue
                worker.steps += 1
                stats["steps"] += 1
                worker.kernel.send("step", index=worker.choose(decision))
    finally:
        for worker in pool:
            if worker.kernel is not None:
                worker.kernel.close()
    elapsed = time.perf_counter() - started
    return {
        "game": hello and hello["game"],
        "runs": len(outcomes),
        "failures": failures,
        "victories": sum(bool(o["victory"]) for o in outcomes),
        "mean_floor": sum(o["floor"] for o in outcomes) / max(1, len(outcomes)),
        "steps": stats["steps"],
        "seconds": elapsed,
        "steps_per_second": stats["steps"] / elapsed,
        "decisions": {k: v for k, v in stats.items() if k != "steps"},
        "outcomes": outcomes,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--runs", type=int, default=50)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--ascension", type=int, default=10)
    parser.add_argument("--invincible", action="store_true")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    result = soak(args.runs, args.workers, args.ascension, args.invincible, args.seed)
    outcomes = result.pop("outcomes")
    print(json.dumps(result, indent=1))
    print(json.dumps(collections.Counter(o["floor"] // 17 for o in outcomes)))


if __name__ == "__main__":
    main()
