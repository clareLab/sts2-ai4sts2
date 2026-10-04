import argparse
import json
import time


def main():
    parser = argparse.ArgumentParser(prog="ai4sts2")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("check")
    calibration = commands.add_parser("calibrate")
    calibration.add_argument("--minutes", type=float, default=5)
    calibration.add_argument("--scope", choices=("run", "first_combat"), default="first_combat")
    baseline = commands.add_parser("baseline")
    baseline.add_argument("--minutes", type=float, default=5)
    baseline.add_argument("--per-character", type=int, default=1)
    baseline.add_argument("--seed", type=int, default=0)
    baseline.add_argument("--scope", choices=("run", "act1", "first_combat"), default="run")
    baseline.add_argument("--steps", type=int, default=4096)
    baseline.add_argument("--refresh", action="store_true")
    selection = commands.add_parser("check-selection")
    selection.add_argument("--minutes", type=float, default=5)
    runs = commands.add_parser("check-run")
    runs.add_argument("--minutes", type=float, default=5)
    runs.add_argument("--episodes", type=int, default=5)
    runs.add_argument("--steps", type=int, default=2048)
    runs.add_argument("--seed", type=int, default=0)
    runs.add_argument("--diagnostic", action="store_true")
    runs.add_argument("--diagnostic-event")
    probe = commands.add_parser("probe")
    probe.add_argument("--character", default="IRONCLAD")
    probe.add_argument("--steps", type=int, default=80)
    pilot = commands.add_parser("pilot")
    pilot.add_argument("--minutes", type=float, default=30)
    pilot.add_argument("--iterations", type=int, default=4)
    pilot.add_argument("--steps", type=int, default=128)
    pilot.add_argument("--scope", choices=("run", "act1", "first_combat"), default="run")
    pilot.add_argument("--resume", help="Recover an interrupted Ray experiment.")
    pilot.add_argument("--checkpoint", help="Start another bounded pilot from a saved model.")
    pilot.add_argument("--initial-policy", help="Transfer weights into a fresh training task.")
    pilot.add_argument("--policy", choices=("flat", "shared"), default="flat")
    pilot.add_argument("--width", type=int, default=64)
    pilot.add_argument("--encoding", choices=("hash", "tree"), default="hash")
    pilot.add_argument(
        "--workers", type=int, default=0, help="Concurrent trials; zero selects automatically."
    )
    ablation = commands.add_parser("ablate")
    ablation.add_argument("--minutes", type=float, default=20)
    ablation.add_argument("--iterations", type=int, default=2)
    ablation.add_argument("--steps", type=int, default=256)
    ablation.add_argument("--seed", type=int, default=0)
    ablation.add_argument("--repeats", type=int, default=1)
    ablation.add_argument(
        "--study", choices=("signals", "policies", "encodings"), default="signals"
    )
    ablation.add_argument(
        "--workers", type=int, default=0, help="Concurrent trials; zero selects automatically."
    )
    ablation.add_argument("--scope", choices=("run", "act1", "first_combat"), default="run")
    for command in (calibration, baseline, pilot, ablation):
        command.add_argument("--ascension", type=int, choices=range(11), default=10)
    args = parser.parse_args()
    if args.command == "baseline":
        from ai4sts2.baseline import run

        result = run(
            args.minutes,
            args.per_character,
            args.seed,
            args.scope,
            args.steps,
            args.refresh,
            args.ascension,
        )
        print(json.dumps(result, indent=2))
        if not result["eligible"]:
            raise SystemExit(1)
    elif args.command == "check-run":
        from ai4sts2.runcheck import check

        result = check(
            args.minutes,
            args.episodes,
            args.steps,
            args.seed,
            args.diagnostic,
            args.diagnostic_event,
        )
        print(json.dumps(result, indent=2))
        if not result["valid"]:
            raise SystemExit(1)
    elif args.command == "check-selection":
        from ai4sts2.selection import validate

        print(json.dumps(validate(args.minutes), indent=2))
    elif args.command == "calibrate":
        from ai4sts2.calibration import calibrate

        print(json.dumps(calibrate(args.minutes, args.scope, args.ascension), indent=2))
    elif args.command in {"pilot", "ablate"}:
        from ai4sts2.train import run

        print(
            json.dumps(
                run(
                    args.minutes,
                    args.iterations,
                    args.steps,
                    getattr(args, "resume", None),
                    getattr(args, "checkpoint", None),
                    args.scope,
                    (
                        "pbt"
                        if args.command == "pilot"
                        else "policy_ablation"
                        if args.study == "policies"
                        else "encoding_ablation"
                        if args.study == "encodings"
                        else "ablation"
                    ),
                    getattr(args, "seed", 0),
                    args.workers,
                    getattr(args, "repeats", 1),
                    getattr(args, "policy", "flat"),
                    getattr(args, "width", 64),
                    getattr(args, "encoding", "hash"),
                    getattr(args, "initial_policy", None),
                    args.ascension,
                ),
                indent=2,
            )
        )
    elif args.command == "probe":
        from ai4sts2.game import probe as run_probe

        run_probe(args.character, args.steps)
    else:
        from ai4sts2.environment import Sts2Env, evaluate, fingerprint, write_json
        from ai4sts2.game import ROOT

        started = time.monotonic()
        build = fingerprint()
        environment = Sts2Env()
        try:
            result = evaluate(None, environment)
            result |= {"seconds": time.monotonic() - started, "build": build}
            write_json(ROOT / "artifacts/validation/environment.json", result)
            print(json.dumps(result, indent=2))
            if any(episode["truncated"] for episode in result["episodes"]):
                raise RuntimeError("The environment probe did not finish every episode.")
        finally:
            environment.close()


if __name__ == "__main__":
    main()
