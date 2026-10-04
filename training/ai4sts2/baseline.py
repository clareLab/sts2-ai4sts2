import json
import time

from ai4sts2.calibration import selected_execution
from ai4sts2.environment import SCOPES, Sts2Env, evaluate, evaluation_plan, fingerprint, write_json
from ai4sts2.execution import REFERENCE
from ai4sts2.game import ROOT, prepare_game


def run(
    minutes=5, per_character=1, seed=0, scope="run", max_steps=4096, refresh=False, ascension=10
):
    if not 0 < minutes <= 30:
        raise ValueError("Use a budget between zero and 30 minutes.")
    if scope not in SCOPES:
        raise ValueError("Unknown episode scope.")
    plan = evaluation_plan(
        scope, seed=seed, max_steps=max_steps, per_character=per_character, ascension=ascension
    )
    build = fingerprint(scope, ascension)
    path = ROOT / f"artifacts/validation/random-baseline-{scope}-a{ascension}.json"
    if not refresh and path.is_file():
        cached = json.loads(path.read_text())
        if (
            cached.get("build") == build
            and cached.get("evaluation_id") == plan["evaluation_id"]
            and cached.get("complete")
            and cached.get("eligible")
        ):
            return cached
    started = time.monotonic()
    deadline = started + minutes * 60
    execution = selected_execution(scope, ascension) or REFERENCE
    report = plan | {
        "build": build,
        "policy": "uniform_random",
        "execution": execution.to_dict(),
        "certifying": False,
        "complete": False,
        "eligible": False,
        "episodes": [],
    }
    write_json(path, report)
    environment = None

    def record(result):
        report.update(result | {"seconds": time.monotonic() - started})
        write_json(path, report)
        print(
            json.dumps(
                {
                    "baseline_episodes": len(result["episodes"]),
                    "win_rate": result["win_rate"],
                    "task_success_rate": result["task_success_rate"],
                    "mean_floor": result["mean_floor"],
                    "complete": result["complete"],
                }
            ),
            flush=True,
        )

    try:
        environment = Sts2Env(
            prepare_game(),
            scope=scope,
            execution=execution,
            max_steps=max_steps,
            ascension=ascension,
        )
        evaluate(
            None,
            environment,
            seed=seed,
            per_character=per_character,
            max_steps=max_steps,
            deadline=deadline,
            on_episode=record,
        )
        return report
    except Exception as error:
        report.update(complete=False, eligible=False, error=str(error))
        write_json(path, report)
        raise
    finally:
        if environment is not None:
            environment.close()
