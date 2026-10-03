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
    pilot.add_argument("--scope", choices=("run", "first_combat"), default="run")
    pilot.add_argument("--resume", help="Recover an interrupted Ray experiment.")
    pilot.add_argument("--checkpoint", help="Start another bounded pilot from a saved model.")
    args = parser.parse_args()
    if args.command == "check-run":
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

        print(json.dumps(calibrate(args.minutes, args.scope), indent=2))
    elif args.command == "pilot":
        from ai4sts2.train import run

        print(
            json.dumps(
                run(
                    args.minutes,
                    args.iterations,
                    args.steps,
                    args.resume,
                    args.checkpoint,
                    args.scope,
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
