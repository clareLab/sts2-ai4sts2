import argparse
import collections
import json
import multiprocessing as mp
import random
from pathlib import Path

import numpy as np
import torch
from scipy import stats

from ai4sts2.features import Batch, Layout, Translator
from ai4sts2.kernel import Kernel, KernelError, die_with_parent, engine
from ai4sts2.search import Evaluator, search
from ai4sts2.train import ACTS, COMBATS, TARGET, Outcome, Statistics, make_net, wilson

COMBAT_NODES = {"MapPointType.Monster", "MapPointType.Elite", "MapPointType.Boss"}


def load(checkpoint, layout):
    if checkpoint is None:
        return None, None, None
    state = torch.load(checkpoint, weights_only=True)
    args = argparse.Namespace(**state["args"])
    outcome = Outcome(args.outcome)
    net = make_net(layout, outcome, args)
    net.load_state_dict(state["net"])
    return net.eval(), outcome, Statistics(args.reference, state["statistics"])


def play(index, args, executable, jobs, results):
    die_with_parent()
    torch.set_num_threads(1)
    kernel = Kernel(executable)
    layout = Layout.from_schema(kernel.schema)
    net, outcome, statistics = load(args.checkpoint, layout)
    generator = np.random.default_rng(index)
    for character, seed in jobs:
        rng = random.Random(f"{args.policy}:{character}:{seed}")
        translator = Translator()
        record = {"character": character, "seed": seed, "steps": 0, "elite": False}
        farming, available, start = 0, 0, None
        try:
            decision = kernel.call(
                "begin",
                character=character,
                seed=seed,
                ascension=args.ascension,
                invincible=args.invincible,
            )
            while decision["kind"] != "end" and record["steps"] < args.cap:
                translator.sync(kernel.vocab)
                state = decision["state"]
                if decision["kind"] == "rewards" and state["room"] == "Elite":
                    record["elite"] = True
                count = len(decision["options"])
                if net is None:
                    action = rng.randrange(count)
                elif args.policy == "search" and decision["kind"] == "combat" and count > 1:
                    utility = outcome.utility(statistics.mass((character, args.ascension)))
                    evaluator = Evaluator(kernel, net, layout, translator, outcome, utility)
                    action, _ = search(
                        evaluator, decision, args.simulations, args.width, generator, False
                    )
                else:
                    observation = translator.canonical(decision["observation"], layout)
                    with torch.no_grad():
                        logits, values = net(Batch([observation], layout))
                    if start is None and outcome.name == "wkch":
                        mass = torch.softmax(values[0], -1).numpy()
                        start = float(mass[2 * COMBATS : -1].sum() + mass[-1])
                    if args.policy != "sample":
                        action = int(logits[0].argmax())
                    else:
                        weights = torch.softmax(logits[0], -1).tolist()
                        action = rng.choices(range(count), weights=weights)[0]
                if decision["kind"] == "map":
                    kinds = map_kinds(kernel, decision, layout)
                    if COMBAT_NODES & set(kinds):
                        available += 1
                        farming += kinds[action] not in COMBAT_NODES
                decision = kernel.call("step", index=action)
                record["steps"] += 1
        except KernelError as error:
            results.put(("failure", character, seed, str(error)[-2000:]))
            kernel.close()
            kernel = Kernel(executable)
            continue
        state = decision["state"]
        record |= {k: state[k] for k in ("act", "floor", "won", "foe", "hp", "deck", "relics")}
        record["victory"] = bool(decision.get("victory")) and decision["kind"] == "end"
        record["capped"] = decision["kind"] != "end"
        record["farming"], record["available"], record["start"] = farming, available, start
        results.put(("run", record))
    kernel.close()
    results.put(("done", index))


def map_kinds(kernel, decision, layout):
    observation = decision["observation"]
    node = observation["option"][:, layout.columns["option"]["node"]].astype(int)
    types = observation["node"][:, layout.columns["node"]["type"]].astype(int)
    return [kernel.vocab[types[n]] if n >= 0 else None for n in node]


def evaluate(args):
    executable = engine()
    with Kernel(executable) as kernel:
        characters = kernel.hello["characters"]
    jobs = [(c, f"{args.panel}-{c}-{i}") for i in range(args.runs) for c in characters]
    context = mp.get_context("spawn")
    results = context.Queue()
    workers = [
        context.Process(
            target=play, args=(i, args, executable, jobs[i :: args.workers], results), daemon=True
        )
        for i in range(args.workers)
    ]
    for process in workers:
        process.start()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    done, failures = 0, 0
    with output.open("w") as stream:
        while done < len(workers):
            kind, *payload = results.get()
            if kind == "run":
                stream.write(json.dumps(payload[0]) + "\n")
            elif kind == "failure":
                failures += 1
                print(json.dumps({"failure": payload}), flush=True)
            else:
                done += 1
    report(summarize(read(output)) | {"failures": failures})


def report(result):
    for key, value in result.items():
        print(json.dumps({key: value}), flush=True)


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def interval(successes, trials):
    low, high = wilson(successes, trials)
    return [successes, trials, round(low, 4), round(high, 4)]


def summarize(records):
    result = {}
    groups = collections.defaultdict(list)
    for record in records:
        groups[record["character"]].append(record)
    for character, runs in sorted(groups.items()):
        entry = {"runs": len(runs), "win": interval(sum(r["victory"] for r in runs), len(runs))}
        for k in range(ACTS):
            reached = [r for r in runs if r["act"] >= k or r["victory"]]
            cleared = sum(r["act"] > k or r["victory"] for r in reached)
            entry[f"act{k + 1}"] = interval(cleared, len(reached))
        entry["elite"] = interval(sum(r["elite"] for r in runs), len(runs))
        first = [r for r in runs if r["act"] == 0 and not r["victory"]]
        entry["won_act1"] = round(float(np.mean([r["won"] for r in first])), 3) if first else None
        available = sum(r["available"] for r in runs)
        farming = sum(r["farming"] for r in runs)
        entry["farming"] = round(farming / available, 4) if available else None
        entry["floor"] = round(float(np.mean([r["floor"] for r in runs])), 2)
        entry["capped"] = sum(r["capped"] for r in runs)
        predicted = [
            (r["start"], r["act"] >= 1 or r["victory"]) for r in runs if r["start"] is not None
        ]
        if predicted:
            entry["ece_act1"] = round(calibration(predicted), 4)
        result[character] = entry
    return result


def calibration(pairs, bins=10):
    probabilities = np.array([p for p, _ in pairs])
    outcomes = np.array([o for _, o in pairs], dtype=np.float64)
    index = np.minimum((probabilities * bins).astype(int), bins - 1)
    error = 0.0
    for b in range(bins):
        members = index == b
        if members.any():
            error += members.mean() * abs(probabilities[members].mean() - outcomes[members].mean())
    return error


def order(record):
    if record["victory"]:
        return (1,)
    foe = record["foe"]
    return (0, record["act"], record["won"], 0.0 if foe is None else 1.0 - foe)


def compare(first, second):
    a = {(r["character"], r["seed"]): r for r in read(first)}
    b = {(r["character"], r["seed"]): r for r in read(second)}
    keys = sorted(a.keys() & b.keys())
    result = {"pairs": len(keys)}
    for name, metric in (
        ("act1", lambda r: r["act"] >= 1 or r["victory"]),
        ("win", lambda r: r["victory"]),
        ("elite", lambda r: r["elite"]),
    ):
        better = sum(metric(b[k]) and not metric(a[k]) for k in keys)
        worse = sum(metric(a[k]) and not metric(b[k]) for k in keys)
        p = stats.binomtest(better, better + worse).pvalue if better + worse else 1.0
        result[name] = {"second_better": better, "first_better": worse, "mcnemar_p": p}
    pooled = sorted({order(r) for k in keys for r in (a[k], b[k])})
    ranks = {value: rank for rank, value in enumerate(pooled)}
    differences = [ranks[order(b[k])] - ranks[order(a[k])] for k in keys]
    if any(differences):
        result["order"] = {
            "median_rank_shift": float(np.median(differences)),
            "wilcoxon_p": float(stats.wilcoxon(differences, zero_method="zsplit").pvalue),
        }
    return result


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--checkpoint")
    run.add_argument("--policy", choices=("argmax", "sample", "random", "search"), default="argmax")
    run.add_argument("--simulations", type=int, default=32)
    run.add_argument("--width", type=int, default=8)
    run.add_argument("--panel", default="EVAL")
    run.add_argument("--runs", type=int, default=200)
    run.add_argument("--ascension", type=int, default=TARGET)
    run.add_argument("--workers", type=int, default=8)
    run.add_argument("--cap", type=int, default=20000)
    run.add_argument("--invincible", action="store_true")
    run.add_argument("--output", required=True)
    pair = commands.add_parser("compare")
    pair.add_argument("first")
    pair.add_argument("second")
    summary = commands.add_parser("summary")
    summary.add_argument("results")
    args = parser.parse_args()
    if args.command == "run":
        if args.policy != "random" and not args.checkpoint:
            parser.error("--checkpoint is required unless --policy random")
        if args.policy == "random":
            args.checkpoint = None
        evaluate(args)
    elif args.command == "compare":
        report(compare(args.first, args.second))
    else:
        report(summarize(read(args.results)))


if __name__ == "__main__":
    main()
