import copy
import hashlib
import json
import math
import tempfile
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ai4sts2.encoding import ACTION_NODES, NODE_SIZE, STATE_NODES, tree
from ai4sts2.execution import REFERENCE
from ai4sts2.game import OfficialGame
from ai4sts2.metrics import summarise

CHARACTERS = ("IRONCLAD", "SILENT", "REGENT", "NECROBINDER", "DEFECT")
MAX_ACTIONS = 128
STATE_FEATURES = 512
ACTION_FEATURES = 64
SCHEMA = 4
VISIBLE_FIELDS = frozenset(
    "character ascension floor act screen player gold deck relics potions energy stars orbs turn "
    "hand draw discard exhaust creatures model type cost upgrades enchantment side hp max_hp "
    "block powers amount intents damage repeats passive evoke kind card target row column room "
    "control label selected keywords variables selection prompt minimum maximum "
    "skippable upgrade preview random options description relic potion usage "
    "position map children puzzle tool cells hidden cards".split()
)


def public_fields(value):
    if isinstance(value, dict):
        if value.get("hidden") is True:
            return {key: value[key] for key in ("row", "column", "hidden") if key in value}
        return {key: public_fields(item) for key, item in value.items() if key in VISIBLE_FIELDS}
    if isinstance(value, list):
        return [public_fields(item) for item in value]
    return value


def features(value, size):
    output = np.zeros(size, dtype=np.float32)

    def add(key, amount):
        digest = hashlib.blake2b(key.encode(), digest_size=8).digest()
        index = int.from_bytes(digest, "little") % size
        output[index] += amount

    def visit(item, path):
        if isinstance(item, dict):
            for key in sorted(item):
                visit(item[key], f"{path}/{key}")
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{path}/{index}")
        elif isinstance(item, bool):
            add(f"{path}={item}", 1.0)
        elif isinstance(item, (int, float)):
            if not math.isfinite(item):
                raise ValueError("Non-finite observation.")
            add(path, math.copysign(math.log1p(abs(item)), item) / 5)
        elif isinstance(item, str):
            add(f"{path}={item}", 1.0)
        elif item is not None:
            raise TypeError(type(item))

    visit(value, "")
    return np.tanh(output)


def visible_observation(state):
    observation = public_fields(state["observation"])
    for pile in ("draw", "discard", "exhaust", "deck"):
        if pile in observation:
            observation[pile] = sorted(
                observation[pile], key=lambda card: json.dumps(card, sort_keys=True)
            )
    return observation


def encode(state, encoding="hash"):
    if encoding not in {"hash", "tree"}:
        raise ValueError("Unknown observation encoding.")
    actions = state["actions"]
    if len(actions) > MAX_ACTIONS:
        raise ValueError(f"Action capacity exceeded: {len(actions)} > {MAX_ACTIONS}")
    if not actions and not state["terminated"]:
        raise ValueError("Non-terminal state has no legal action.")
    observation = visible_observation(state)
    if encoding == "tree":
        matrix = np.zeros((MAX_ACTIONS, ACTION_NODES, NODE_SIZE), dtype=np.float32)
        for index, action in enumerate(actions):
            matrix[index] = tree(public_fields(action), ACTION_NODES)
        return {"state": tree(observation, STATE_NODES), "actions": matrix}
    matrix = np.zeros((MAX_ACTIONS, ACTION_FEATURES), dtype=np.float32)
    for index, action in enumerate(actions):
        matrix[index] = features(public_fields(action), ACTION_FEATURES)
    return {"state": features(observation, STATE_FEATURES), "actions": matrix}


def seed_string(split, seed):
    if split not in {"train", "validation", "test", "calibration"}:
        raise ValueError("Unknown seed split.")
    return hashlib.sha256(f"ai4sts2:{split}:{seed}".encode()).hexdigest()[:16].upper()


def state_digest(state):
    visible = {key: state[key] for key in ("observation", "actions", "terminated", "victory")}
    return hashlib.sha256(json.dumps(visible, sort_keys=True).encode()).hexdigest()


class Sts2Env(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        executable=None,
        seed=0,
        max_steps=1024,
        worker_factory=OfficialGame,
        execution=REFERENCE,
        scope="first_combat",
        signals=None,
        encoding="hash",
    ):
        super().__init__()
        if scope not in {"run", "first_combat"}:
            raise ValueError("Unknown episode scope.")
        self.set_encoding(encoding)
        self.game = (
            worker_factory(executable, execution=execution)
            if worker_factory is OfficialGame
            else worker_factory(executable)
        )
        self.scope = scope
        self.encoding_seconds = 0.0
        self.rng = np.random.default_rng(seed)
        self.task_rng = np.random.default_rng(np.random.SeedSequence([seed, 1]))
        self.signals = signals
        self.max_steps = max_steps
        self.state = None
        self.steps = 0
        self.character = None
        self.journal = None
        self.completed = []
        self.action_space = spaces.Discrete(MAX_ACTIONS)

    def set_encoding(self, encoding):
        if encoding not in {"hash", "tree"}:
            raise ValueError("Unknown observation encoding.")
        if getattr(self, "encoding", None) == encoding:
            return
        self.encoding = encoding
        state_shape, action_shape = (
            ((STATE_NODES, NODE_SIZE), (MAX_ACTIONS, ACTION_NODES, NODE_SIZE))
            if encoding == "tree"
            else ((STATE_FEATURES,), (MAX_ACTIONS, ACTION_FEATURES))
        )
        limit = np.inf if encoding == "tree" else 1
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(-limit, limit, state_shape, dtype=np.float32),
                "actions": spaces.Box(-limit, limit, action_shape, dtype=np.float32),
            }
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.character = options.get("character") or (
            self.signals.character(self.task_rng)
            if self.signals is not None
            else CHARACTERS[
                int(
                    self.task_rng.choice(
                        len(CHARACTERS), p=np.full(len(CHARACTERS), 1 / len(CHARACTERS))
                    )
                )
            ]
        )
        if self.character not in CHARACTERS:
            raise ValueError("Unknown character.")
        split = options.get("split", "train")
        episode_seed = int(self.rng.integers(2**63)) if seed is None else seed
        parameters = {
            "character": self.character,
            "seed": seed_string(split, episode_seed),
            "scope": self.scope,
        }
        self.state = self.game.request("reset", parameters)
        self.journal = {
            "parameters": parameters,
            "initial": state_digest(self.state),
            "actions": [],
        }
        self.steps = 0
        if self.signals is not None:
            self.signals.reset(self.state)
        return self.encode(), {"scope": self.scope, "character": self.character}

    def encode(self):
        started = time.perf_counter()
        observation = encode(self.state, self.encoding)
        self.encoding_seconds += time.perf_counter() - started
        return observation

    def action_masks(self):
        return np.arange(MAX_ACTIONS) < len(self.state["actions"])

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError("Illegal action index.")
        action = int(action)
        if (
            self.state is None
            or self.state["terminated"]
            or self.steps >= self.max_steps
            or not 0 <= action < len(self.state["actions"])
        ):
            raise ValueError("Illegal action.")
        self.state = self.game.request(
            "step", {"revision": self.state["revision"], "action": action}
        )
        self.steps += 1
        self.journal["actions"].append({"action": action, "digest": state_digest(self.state)})
        terminated = self.state["terminated"]
        truncated = not terminated and self.steps >= self.max_steps
        victory = bool(terminated and self.state["victory"])
        reward = float(1 if victory else -1) if terminated else 0.0
        info = {
            "scope": self.scope,
            "character": self.character,
            "victory": victory,
            "hp": self.state["observation"]["player"]["hp"],
            "steps": self.steps,
            "truncated": truncated,
            "screen": self.state["observation"].get("screen"),
            "floor": self.state["observation"].get("floor"),
            "act": self.state["observation"].get("act"),
            "seed": self.journal["parameters"]["seed"],
        }
        if terminated or truncated:
            self.completed.append(info.copy())
            info["episode"] = {"r": reward, "l": self.steps}
        if self.signals is not None:
            bonus = self.signals.observe(self.state, info, terminated or truncated)
            info["intrinsic_reward"] = bonus
            reward += bonus
        return self.encode(), reward, terminated, truncated, info

    def snapshot(self):
        return copy.deepcopy(
            {
                "scope": self.scope,
                "encoding": self.encoding,
                "max_steps": self.max_steps,
                "journal": self.journal,
                "completed": self.completed,
                "rng": self.rng.bit_generator.state,
                "task_rng": self.task_rng.bit_generator.state,
            }
        )

    def restore(self, snapshot):
        if snapshot["scope"] != self.scope or snapshot["max_steps"] != self.max_steps:
            raise ValueError("Checkpoint episode scope or step limit does not match.")
        if snapshot.get("encoding", "hash") != self.encoding:
            raise ValueError("Checkpoint observation encoding does not match.")
        journal = snapshot["journal"]
        if journal is None:
            self.state = None
            self.character = None
            self.steps = 0
        else:
            state = self.game.request("reset", journal["parameters"])
            if state_digest(state) != journal["initial"]:
                raise ValueError("Checkpoint replay diverged at reset.")
            for index, entry in enumerate(journal["actions"]):
                state = self.game.request(
                    "step", {"revision": state["revision"], "action": entry["action"]}
                )
                if state_digest(state) != entry["digest"]:
                    raise ValueError(f"Checkpoint replay diverged at step {index + 1}.")
            self.state = state
            self.character = journal["parameters"]["character"]
            self.steps = len(journal["actions"])
        self.journal = copy.deepcopy(journal)
        self.completed = copy.deepcopy(snapshot["completed"])
        self.rng.bit_generator.state = snapshot["rng"]
        self.task_rng.bit_generator.state = snapshot["task_rng"]

    def drain_episodes(self):
        episodes, self.completed = self.completed, []
        return episodes

    def drain_measurements(self):
        result = self.game.drain_measurements()
        result["encoding_seconds"] = self.encoding_seconds
        self.encoding_seconds = 0.0
        return result

    def close(self):
        self.game.close()


def probe_action(actions):
    for index, action in enumerate(actions):
        if action.get("control") == "NConfirmButton":
            return index
    for index, action in enumerate(actions):
        if not action.get("selected", False):
            return index
    return 0


def evaluation_plan(scope, seed=0, split="validation", max_steps=256, per_character=1):
    if per_character < 1 or max_steps < 1:
        raise ValueError("Use positive episode and step counts.")
    cases = []
    for repeat in range(per_character):
        for index, character in enumerate(CHARACTERS):
            episode_seed = seed + repeat * len(CHARACTERS) + index
            action_seed = hashlib.sha256(
                f"ai4sts2:random-actions:{split}:{episode_seed}:{character}".encode()
            ).hexdigest()[:16]
            cases.append(
                {
                    "character": character,
                    "seed_index": episode_seed,
                    "seed": seed_string(split, episode_seed),
                    "action_seed": int(action_seed, 16),
                }
            )
    plan = {"scope": scope, "split": split, "max_steps": max_steps, "cases": cases}
    return plan | {
        "evaluation_id": hashlib.sha256(json.dumps(plan, sort_keys=True).encode()).hexdigest()
    }


def evaluate(
    model,
    environment,
    seed=0,
    split="validation",
    max_steps=256,
    per_character=1,
    deadline=None,
    on_episode=None,
):
    plan = evaluation_plan(environment.scope, seed, split, max_steps, per_character)
    results = []
    previous_limit = environment.max_steps
    previous_timeout = getattr(environment.game, "timeout", None)
    environment.max_steps = max_steps

    def check_budget():
        if deadline is not None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError("Evaluation budget exhausted.")
            if previous_timeout is not None:
                environment.game.timeout = min(previous_timeout, remaining)

    def report():
        summary = summarise(results)
        summary.pop("episodes")
        complete = len(results) == len(plan["cases"])
        if not complete:
            summary.update(eligible=False, selection_score=-1.0)
        return (
            plan
            | summary
            | {
                "policy": "uniform_random" if model is None else "learned_deterministic",
                "certifying": False,
                "complete": complete,
                "episodes": results.copy(),
            }
        )

    try:
        for case in plan["cases"]:
            check_budget()
            observation, _ = environment.reset(
                seed=case["seed_index"], options={"character": case["character"], "split": split}
            )
            rng = np.random.default_rng(case["action_seed"])
            terminated = truncated = False
            while not (terminated or truncated):
                check_budget()
                if model is None:
                    action = int(rng.choice(np.flatnonzero(environment.action_masks())))
                else:
                    action, _ = model.predict(
                        observation, deterministic=True, action_masks=environment.action_masks()
                    )
                observation, _, terminated, truncated, info = environment.step(action)
            trajectory = hashlib.sha256(
                json.dumps(environment.journal, sort_keys=True).encode()
            ).hexdigest()
            results.append(info | {"truncated": truncated, "trajectory_digest": trajectory})
            if on_episode is not None:
                on_episode(report())
    finally:
        environment.max_steps = previous_limit
        if previous_timeout is not None:
            environment.game.timeout = previous_timeout
        environment.drain_episodes()
    return report()


def fingerprint(scope="first_combat"):
    from ai4sts2.game import ROOT, game_path

    hashes = {}
    for key, path in {
        "game": game_path() / "data_sts2_linuxbsd_x86_64/sts2.dll",
        "mod": ROOT / "artifacts/dist/ai4sts2/ai4sts2.dll",
        "dependencies": ROOT / "uv.lock",
    }.items():
        with path.open("rb") as stream:
            hashes[key] = hashlib.file_digest(stream, "sha256").hexdigest()
    source_hash = hashlib.sha256()
    for path in sorted((ROOT / "training/ai4sts2").glob("*.py")):
        source_hash.update(path.name.encode())
        source_hash.update(path.read_bytes())
    hashes["trainer"] = source_hash.hexdigest()
    return hashes | {"schema": SCHEMA, "scope": scope, "ascension": 10}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", dir=path.parent, prefix=path.name, delete=False
    ) as stream:
        temporary = stream.name
        try:
            stream.write(json.dumps(value, indent=2) + "\n")
            stream.close()
            Path(temporary).replace(path)
        finally:
            Path(temporary).unlink(missing_ok=True)
