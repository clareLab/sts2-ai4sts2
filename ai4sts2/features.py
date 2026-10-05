import json
import tempfile
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

CATEGORICAL = {
    "character",
    "room",
    "decision",
    "boss",
    "second_boss",
    "act_model",
    "event",
    "zone",
    "model",
    "type",
    "rarity",
    "target",
    "enchantment",
    "affliction",
    "status",
    "usage",
    "side",
    "intent",
    "intent2",
    "item",
    "kind",
    "key",
}
REFERENCES = {
    "power": {"owner": "creature"},
    "option": {
        "card": "card",
        "creature": "creature",
        "relic": "relic",
        "potion": "potion",
        "node": "node",
        "cell": "cell",
    },
}
OFFERS = ("card", "relic", "potion")
ORDER = ("global", "card", "relic", "potion", "creature", "power", "orb", "node", "cell", "option")
KEYWORD_BITS = 16


class Vocabulary:
    def __init__(self, path=None):
        self.path = Path(path) if path else None
        self.words = ["<none>"]
        if self.path and self.path.exists():
            self.words = json.loads(self.path.read_text())
        self.ids = {word: index for index, word in enumerate(self.words)}
        self.dirty = False

    def __len__(self):
        return len(self.words)

    def id(self, word):
        index = self.ids.get(word)
        if index is None:
            index = self.ids[word] = len(self.words)
            self.words.append(word)
            self.dirty = True
        return index

    def save(self):
        if not self.path or not self.dirty:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.NamedTemporaryFile("w", dir=self.path.parent, delete=False) as stream:
            json.dump(self.words, stream)
        Path(stream.name).replace(self.path)
        self.dirty = False


class Translator:
    def __init__(self, vocabulary):
        self.vocabulary = vocabulary
        self.table = np.zeros(1, dtype=np.int64)

    def sync(self, words):
        if len(words) > len(self.table):
            fresh = [self.vocabulary.id(word) for word in words[len(self.table) :]]
            self.table = np.concatenate([self.table, np.asarray(fresh, dtype=np.int64)])

    def __call__(self, values):
        return self.table[values.astype(np.int64)]


@dataclass
class Layout:
    categorical: dict
    numeric: dict
    references: dict
    variables: list

    @classmethod
    def from_schema(cls, schema):
        categorical, numeric, references = {}, {}, {}
        variables = []
        for table in ORDER:
            columns = schema[table]
            refs = REFERENCES.get(table, {})
            references[table] = [(columns.index(c), target) for c, target in refs.items()]
            categorical[table] = [i for i, c in enumerate(columns) if c in CATEGORICAL]
            skip = {i for i, _ in references[table]} | set(categorical[table])
            if table == "card":
                variables = [
                    (
                        columns.index(f"var{i}"),
                        columns.index(f"var{i}_base"),
                        columns.index(f"var{i}_preview"),
                    )
                    for i in range(sum(c.startswith("var") and c[3:].isdigit() for c in columns))
                ]
                skip |= {i for triple in variables for i in triple}
                skip.add(columns.index("keywords"))
            if table in OFFERS:
                skip.add(columns.index("offer"))
            numeric[table] = [i for i in range(len(columns)) if i not in skip]
        return cls(categorical, numeric, references, variables)

    def width(self, table):
        extra = KEYWORD_BITS if table == "card" else 0
        return len(self.numeric[table]) + extra


def symlog(values):
    return np.sign(values) * np.log1p(np.abs(values))


class Batch:
    def __init__(self, observations, translators, layout, schema, combat_map=False):
        self.size = len(observations)
        tokens = {table: [] for table in ORDER}
        counts = np.zeros((self.size, len(ORDER)), dtype=np.int64)
        for b, (observation, translate) in enumerate(zip(observations, translators, strict=True)):
            in_combat = observation["global"][0, schema["global"].index("in_combat")] > 0
            for t, table in enumerate(ORDER):
                values = observation[table]
                if table in {"node"} and in_combat and not combat_map:
                    values = values[:0]
                counts[b, t] = len(values)
                tokens[table].append((b, values, translate))
        self.lengths = counts.sum(axis=1)
        offsets = np.zeros_like(counts)
        offsets[:, 1:] = np.cumsum(counts, axis=1)[:, :-1]
        self.width = int(self.lengths.max())
        self.tables = {}
        for t, table in enumerate(ORDER):
            entries = tokens[table]
            rows = [values for _, values, _ in entries]
            stacked = (
                np.concatenate(rows) if rows else np.zeros((0, len(schema[table])), np.float32)
            )
            batch_index = np.concatenate([np.full(len(v), b) for b, v, _ in entries])
            local = np.concatenate([np.arange(len(v)) for _, v, _ in entries])
            position = offsets[batch_index, t] + local
            categorical = np.zeros((len(stacked), len(layout.categorical[table])), dtype=np.int64)
            start = 0
            for _b, values, translate in entries:
                if len(values):
                    categorical[start : start + len(values)] = translate(
                        values[:, layout.categorical[table]]
                    )
                start += len(values)
            numeric = symlog(stacked[:, layout.numeric[table]]).astype(np.float32)
            item = {
                "categorical": torch.from_numpy(categorical),
                "numeric": torch.from_numpy(numeric),
                "position": torch.from_numpy(batch_index * self.width + position),
            }
            if table == "card":
                keywords = stacked[:, schema["card"].index("keywords")].astype(np.int64)
                bits = (keywords[:, None] >> np.arange(KEYWORD_BITS)) & 1
                item["numeric"] = torch.from_numpy(
                    np.concatenate([numeric, bits.astype(np.float32)], axis=1)
                )
                start = 0
                var_ids = np.zeros((len(stacked), len(layout.variables)), dtype=np.int64)
                for _b, rows_, translate in entries:
                    if len(rows_):
                        var_ids[start : start + len(rows_)] = translate(
                            rows_[:, [v[0] for v in layout.variables]]
                        )
                    start += len(rows_)
                values = np.stack([stacked[:, [v[1], v[2]]] for v in layout.variables], axis=1)
                item["variables"] = torch.from_numpy(var_ids)
                item["variable_values"] = torch.from_numpy(symlog(values).astype(np.float32))
            for column, target in layout.references[table]:
                target_index = ORDER.index(target)
                refs = stacked[:, column].astype(np.int64)
                valid = (refs >= 0) & (refs < counts[batch_index, target_index])
                flat = (
                    batch_index * self.width
                    + offsets[batch_index, target_index]
                    + np.where(valid, refs, 0)
                )
                item.setdefault("references", []).append(
                    (torch.from_numpy(np.where(valid, flat, -1)), target)
                )
            if table in OFFERS:
                offers = stacked[:, schema[table].index("offer")].astype(np.int64)
                option_index = ORDER.index("option")
                valid = (offers >= 0) & (offers < counts[batch_index, option_index])
                flat = (
                    batch_index * self.width
                    + offsets[batch_index, option_index]
                    + np.where(valid, offers, 0)
                )
                item["offers"] = torch.from_numpy(np.where(valid, flat, -1))
            self.tables[table] = item
        option = ORDER.index("option")
        self.options = counts[:, option]
        self.option_start = offsets[:, option]
        self.global_position = torch.from_numpy(np.arange(self.size) * self.width + offsets[:, 0])
        mask = np.arange(self.width)[None, :] < self.lengths[:, None]
        self.mask = torch.from_numpy(mask)
        width = int(self.options.max())
        self.option_width = width
        index = self.option_start[:, None] + np.arange(width)[None, :]
        valid = np.arange(width)[None, :] < self.options[:, None]
        self.option_positions = torch.from_numpy(
            np.arange(self.size)[:, None] * self.width + np.where(valid, index, 0)
        )
        self.option_mask = torch.from_numpy(valid)
