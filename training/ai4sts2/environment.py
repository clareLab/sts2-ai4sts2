import hashlib
import json
import math
import time

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from ai4sts2.execution import REFERENCE
from ai4sts2.game import OfficialGame

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


def encode(state):
    actions = state["actions"]
    if len(actions) > MAX_ACTIONS:
        raise ValueError(f"Action capacity exceeded: {len(actions)} > {MAX_ACTIONS}")
    if not actions and not state["terminated"]:
        raise ValueError("Non-terminal state has no legal action.")
    observation = public_fields(state["observation"])
    for pile in ("draw", "discard", "exhaust", "deck"):
        if pile in observation:
            observation[pile] = sorted(
                observation[pile], key=lambda card: json.dumps(card, sort_keys=True)
            )
    matrix = np.zeros((MAX_ACTIONS, ACTION_FEATURES), dtype=np.float32)
    for index, action in enumerate(actions):
        matrix[index] = features(public_fields(action), ACTION_FEATURES)
    return {"state": features(observation, STATE_FEATURES), "actions": matrix}


def seed_string(split, seed):
    if split not in {"train", "validation", "test", "calibration"}:
        raise ValueError("Unknown seed split.")
    return hashlib.sha256(f"ai4sts2:{split}:{seed}".encode()).hexdigest()[:16].upper()


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
    ):
        super().__init__()
        if scope not in {"run", "first_combat"}:
            raise ValueError("Unknown episode scope.")
        self.game = (
            worker_factory(executable, execution=execution)
            if worker_factory is OfficialGame
            else worker_factory(executable)
        )
        self.scope = scope
        self.encoding_seconds = 0.0
        self.rng = np.random.default_rng(seed)
        self.max_steps = max_steps
        self.state = None
        self.steps = 0
        self.character = None
        self.action_space = spaces.Discrete(MAX_ACTIONS)
        self.observation_space = spaces.Dict(
            {
                "state": spaces.Box(-1, 1, (STATE_FEATURES,), dtype=np.float32),
                "actions": spaces.Box(-1, 1, (MAX_ACTIONS, ACTION_FEATURES), dtype=np.float32),
            }
        )

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.character = options.get("character") or CHARACTERS[int(self.rng.integers(5))]
        if self.character not in CHARACTERS:
            raise ValueError("Unknown character.")
        split = options.get("split", "train")
        episode_seed = int(self.rng.integers(2**63)) if seed is None else seed
        self.state = self.game.request(
            "reset",
            {
                "character": self.character,
                "seed": seed_string(split, episode_seed),
                "scope": self.scope,
            },
        )
        self.steps = 0
        return self.encode(), {"scope": self.scope, "character": self.character}

    def encode(self):
        started = time.perf_counter()
        observation = encode(self.state)
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
            or not 0 <= action < len(self.state["actions"])
        ):
            raise ValueError("Illegal action.")
        self.state = self.game.request(
            "step", {"revision": self.state["revision"], "action": action}
        )
        self.steps += 1
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
        }
        return self.encode(), reward, terminated, truncated, info

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


def evaluate(model, environment, seed=0, split="validation", max_steps=256):
    results = []
    previous_limit = environment.max_steps
    environment.max_steps = max_steps
    try:
        for index, character in enumerate(CHARACTERS):
            observation, _ = environment.reset(
                seed=seed + index, options={"character": character, "split": split}
            )
            terminated = truncated = False
            while not (terminated or truncated):
                if model is None:
                    action = probe_action(environment.state["actions"])
                else:
                    action, _ = model.predict(
                        observation, deterministic=True, action_masks=environment.action_masks()
                    )
                observation, _, terminated, truncated, info = environment.step(action)
            results.append(info | {"truncated": truncated})
    finally:
        if model is not None:
            model._last_obs = None
        environment.max_steps = previous_limit
    return {
        "scope": environment.scope,
        "certifying": False,
        "episodes": results,
        "win_rate": sum(r["victory"] for r in results) / len(results),
        "truncated_episodes": sum(r["truncated"] for r in results),
    }


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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)
