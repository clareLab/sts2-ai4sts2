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


POOLED = ("deck", "draw", "discard", "exhaust")
TOKENS = ORDER[:2] + ("pile",) + ORDER[2:]


def length(observation, schema):
    in_combat = observation["global"][0, schema["global"].index("in_combat")] > 0
    return sum(
        len(v) for k, v in observation.items() if k != "edge" and not (in_combat and k == "node")
    )


class Batch:
    def __init__(self, observations, translators, layout, schema, vocabulary, combat_map=False):
        self.size = len(observations)
        card_columns = schema["card"]
        zone_column = card_columns.index("zone")
        offer_column = card_columns.index("offer")
        option_card = schema["option"].index("card")
        pooled_ids = np.array([vocabulary.id(f"zone.{zone}") for zone in POOLED])
        in_combat = schema["global"].index("in_combat")
        entries = []
        counts = np.zeros((self.size, len(TOKENS)), dtype=np.int64)
        for b, (observation, translate) in enumerate(zip(observations, translators, strict=True)):
            entry = {"translate": translate}
            for table in ORDER:
                values = observation[table]
                if table == "node" and observation["global"][0, in_combat] > 0 and not combat_map:
                    values = values[:0]
                entry[table] = values
            cards = entry["card"]
            zones = translate(cards[:, zone_column]) if len(cards) else np.zeros(0, np.int64)
            keep = ~np.isin(zones, pooled_ids) | (cards[:, offer_column] >= 0)
            referenced = entry["option"][:, option_card].astype(np.int64)
            keep[referenced[(referenced >= 0) & (referenced < len(cards))]] = True
            individual = int(keep.sum())
            slots = np.empty(len(cards), dtype=np.int64)
            slots[keep] = np.arange(individual)
            weights = np.ones(len(cards), dtype=np.float32)
            piles = []
            for zone in pooled_ids:
                members = ~keep & (zones == zone)
                if members.any():
                    slots[members] = individual + len(piles)
                    weights[members] = 1.0 / members.sum()
                    piles.append((int(zone), int(members.sum())))
            entry["slots"], entry["weights"], entry["piles"] = slots, weights, piles
            for t, table in enumerate(TOKENS):
                if table == "card":
                    counts[b, t] = individual
                elif table == "pile":
                    counts[b, t] = len(piles)
                else:
                    counts[b, t] = len(entry[table])
            entries.append(entry)
        self.lengths = counts.sum(axis=1)
        offsets = np.zeros_like(counts)
        offsets[:, 1:] = np.cumsum(counts, axis=1)[:, :-1]
        self.width = int(self.lengths.max())
        base = np.arange(self.size) * self.width
        starts = base[:, None] + offsets
        self.tables = {}
        positions, row_starts, stacks = {}, {}, {}
        for table in ORDER:
            t = TOKENS.index(table)
            arrays = [entry[table] for entry in entries]
            sizes = np.array([len(v) for v in arrays], dtype=np.int64)
            row_starts[table] = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
            stacked = np.concatenate(arrays)
            stacks[table] = stacked
            batch_index = np.repeat(np.arange(self.size), sizes)
            if table == "card":
                slots = np.concatenate([entry["slots"] for entry in entries])
            else:
                slots = np.arange(len(stacked)) - row_starts[table][batch_index]
            positions[table] = starts[batch_index, t] + slots
            categorical = np.concatenate(
                [
                    entry["translate"](entry[table][:, layout.categorical[table]])
                    for entry in entries
                ]
            )
            numeric = symlog(stacked[:, layout.numeric[table]]).astype(np.float32)
            item = {
                "categorical": torch.from_numpy(categorical.astype(np.int64)),
                "numeric": torch.from_numpy(numeric),
                "position": torch.from_numpy(positions[table].astype(np.int64)),
            }
            if table == "card":
                keywords = stacked[:, card_columns.index("keywords")].astype(np.int64)
                bits = ((keywords[:, None] >> np.arange(KEYWORD_BITS)) & 1).astype(np.float32)
                item["numeric"] = torch.from_numpy(np.concatenate([numeric, bits], axis=1))
                columns = [v[0] for v in layout.variables]
                variables = np.concatenate(
                    [entry["translate"](entry["card"][:, columns]) for entry in entries]
                )
                values = np.stack([stacked[:, [v[1], v[2]]] for v in layout.variables], axis=1)
                item["variables"] = torch.from_numpy(variables.astype(np.int64))
                item["variable_values"] = torch.from_numpy(symlog(values).astype(np.float32))
                item["weights"] = torch.from_numpy(np.concatenate([e["weights"] for e in entries]))
            if table in OFFERS:
                offers = stacked[:, schema[table].index("offer")].astype(np.int64)
                option = TOKENS.index("option")
                valid = (offers >= 0) & (offers < counts[batch_index, option])
                resolved = np.where(valid, starts[batch_index, option] + offers, -1)
                item["offers"] = torch.from_numpy(resolved.astype(np.int64))
            item["batch"] = batch_index
            self.tables[table] = item
        for table in ORDER:
            item = self.tables[table]
            batch_index = item.pop("batch")
            for column, target in layout.references[table]:
                refs = stacks[table][:, column].astype(np.int64)
                sizes = np.diff(np.append(row_starts[target], len(stacks[target])))
                valid = (refs >= 0) & (refs < sizes[batch_index])
                resolved = np.full(len(refs), -1, dtype=np.int64)
                if valid.any():
                    index = row_starts[target][batch_index[valid]] + refs[valid]
                    resolved[valid] = positions[target][index]
                item.setdefault("references", []).append((torch.from_numpy(resolved), target))
        piles = [(b, zone, count) for b, e in enumerate(entries) for zone, count in e["piles"]]
        rank = np.zeros(len(piles), dtype=np.int64)
        seen = {}
        for i, (b, _, _) in enumerate(piles):
            rank[i] = seen.get(b, 0)
            seen[b] = rank[i] + 1
        pile_batch = np.array([b for b, _, _ in piles], dtype=np.int64)
        self.tables["pile"] = {
            "categorical": torch.tensor([[z] for _, z, _ in piles], dtype=torch.int64).reshape(
                -1, 1
            ),
            "numeric": torch.tensor(
                [[np.log1p(c)] for _, _, c in piles], dtype=torch.float32
            ).reshape(-1, 1),
            "position": torch.from_numpy(
                (starts[pile_batch, TOKENS.index("pile")] + rank).astype(np.int64)
                if piles
                else np.zeros(0, np.int64)
            ),
        }
        option = TOKENS.index("option")
        self.options = counts[:, option]
        self.global_position = torch.from_numpy(base + offsets[:, 0])
        self.mask = torch.from_numpy(np.arange(self.width)[None, :] < self.lengths[:, None])
        width = int(self.options.max())
        index = offsets[:, option][:, None] + np.arange(width)[None, :]
        valid = np.arange(width)[None, :] < self.options[:, None]
        self.option_positions = torch.from_numpy(base[:, None] + np.where(valid, index, 0))
        self.option_mask = torch.from_numpy(valid)
