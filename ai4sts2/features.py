import hashlib
from dataclasses import dataclass

import numpy as np
import torch

CAPACITY = 1 << 15
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
    "tip0",
    "tip1",
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
TOKENS = ORDER[:2] + ("pile",) + ORDER[2:]
PILES = ("draw", "discard", "exhaust")
KEYWORD_BITS = 16


def hashes(word):
    if word is None:
        return 0, 0
    digest = hashlib.blake2b(word.encode(), digest_size=8).digest()
    first, second = int.from_bytes(digest[:4], "little"), int.from_bytes(digest[4:], "little")
    return 1 + first % (CAPACITY - 1), 1 + second % (CAPACITY - 1)


@dataclass
class Layout:
    hashed: dict
    categorical: dict
    numeric: dict
    references: dict
    variables: list
    columns: dict
    piles: np.ndarray
    deck: int

    @classmethod
    def from_schema(cls, schema):
        hashed, categorical, numeric, references = {}, {}, {}, {}
        variables = []
        for table, columns in schema.items():
            width = len(columns)
            references[table] = [
                (columns.index(c), target) for c, target in REFERENCES.get(table, {}).items()
            ]
            plain = [i for i, c in enumerate(columns) if c in CATEGORICAL]
            slots = sum(c.startswith("var") and c[3:].isdigit() for c in columns)
            names = [columns.index(f"var{i}") for i in range(slots)]
            hashed[table] = plain + names
            second = {column: width + k for k, column in enumerate(hashed[table])}
            categorical[table] = plain + [second[c] for c in plain]
            skip = set(hashed[table]) | {i for i, _ in references[table]}
            if table == "card":
                variables = [(name, second[name], name + 1, name + 2) for name in names]
                skip |= {i for v in variables for i in v[2:]} | {columns.index("keywords")}
            if table in OFFERS:
                skip.add(columns.index("offer"))
            numeric[table] = [i for i in range(width) if i not in skip]
        index = {table: {c: i for i, c in enumerate(columns)} for table, columns in schema.items()}
        piles = np.array([hashes(f"zone.{zone}")[0] for zone in PILES])
        deck = hashes("zone.deck")[0]
        return cls(hashed, categorical, numeric, references, variables, index, piles, deck)

    def width(self, table):
        if table == "pile":
            return 1
        return len(self.numeric[table]) + (KEYWORD_BITS + 1 if table == "card" else 0)


class Translator:
    def __init__(self):
        self.table = np.zeros((1, 2), dtype=np.float32)

    def sync(self, words):
        if len(words) > len(self.table):
            fresh = np.array([hashes(word) for word in words[len(self.table) :]], np.float32)
            self.table = np.concatenate([self.table, fresh.reshape(-1, 2)])

    def canonical(self, observation, layout):
        result = {}
        for table, values in observation.items():
            columns = layout.hashed[table]
            if columns:
                ids = self.table[values[:, columns].astype(np.int64)]
                values = np.concatenate([values, ids[:, :, 1]], axis=1)
                values[:, columns] = ids[:, :, 0]
            result[table] = values
        return result


def symlog(values):
    return np.sign(values) * np.log1p(np.abs(values))


def length(observation):
    return sum(len(v) for k, v in observation.items() if k in ORDER)


class Batch:
    FIELDS = ("global_position", "mask", "option_positions", "option_mask", "receivers", "senders")

    def __init__(self, observations, layout):
        self.size = len(observations)
        card = layout.columns["card"]
        zone, offer = card["zone"], card["offer"]
        option_card = layout.columns["option"]["card"]
        entries = []
        counts = np.zeros((self.size, len(TOKENS)), dtype=np.int64)
        for b, observation in enumerate(observations):
            entry = dict(observation)
            cards = entry["card"]
            zones = cards[:, zone]
            keep = cards[:, offer] >= 0
            referenced = entry["option"][:, option_card].astype(np.int64)
            keep[referenced[(referenced >= 0) & (referenced < len(cards))]] = True
            pooled = ~keep & np.isin(zones, layout.piles)
            grouped = ~keep & (zones == layout.deck)
            slots = np.zeros(len(cards), dtype=np.int64)
            weights = np.ones(len(cards), dtype=np.float32)
            multiplicity = np.zeros(len(cards), dtype=np.float32)
            individual = np.flatnonzero(~pooled & ~grouped)
            slots[individual] = np.arange(len(individual))
            used = len(individual)
            if grouped.any():
                members = np.flatnonzero(grouped)
                _, inverse, sizes = np.unique(
                    cards[members], axis=0, return_inverse=True, return_counts=True
                )
                slots[members] = used + inverse
                weights[members] = 1.0 / sizes[inverse]
                multiplicity[members] = np.log(sizes[inverse])
                used += len(sizes)
            piles = []
            for pile in layout.piles:
                members = pooled & (zones == pile)
                if members.any():
                    slots[members] = len(piles)
                    weights[members] = 1.0 / members.sum()
                    piles.append((int(pile), int(members.sum())))
            entry["slots"], entry["weights"], entry["piles"] = slots, weights, piles
            entry["pooled"], entry["multiplicity"] = pooled, multiplicity
            for t, table in enumerate(TOKENS):
                if table == "card":
                    counts[b, t] = used
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
        positions, row_starts, stacks, batches = {}, {}, {}, {}
        receivers, senders = [], []
        option = TOKENS.index("option")
        for table in ORDER + ("edge",):
            arrays = [entry[table] for entry in entries]
            sizes = np.array([len(v) for v in arrays], dtype=np.int64)
            row_starts[table] = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
            stacks[table] = stacked = np.concatenate(arrays)
            batches[table] = batch_index = np.repeat(np.arange(self.size), sizes)
            if table == "edge":
                continue
            t = TOKENS.index(table)
            if table == "card":
                slots = np.concatenate([entry["slots"] for entry in entries])
                pooled = np.concatenate([entry["pooled"] for entry in entries])
                positions[table] = np.where(
                    pooled,
                    starts[batch_index, TOKENS.index("pile")] + slots,
                    starts[batch_index, t] + slots,
                )
            else:
                slots = np.arange(len(stacked)) - row_starts[table][batch_index]
                positions[table] = starts[batch_index, t] + slots
            numeric = symlog(stacked[:, layout.numeric[table]]).astype(np.float32)
            item = {
                "categorical": torch.from_numpy(
                    stacked[:, layout.categorical[table]].astype(np.int64)
                ),
                "numeric": torch.from_numpy(numeric),
                "position": torch.from_numpy(positions[table].astype(np.int64)),
            }
            if table == "card":
                keywords = stacked[:, card["keywords"]].astype(np.int64)
                bits = ((keywords[:, None] >> np.arange(KEYWORD_BITS)) & 1).astype(np.float32)
                multiplicity = np.concatenate([e["multiplicity"] for e in entries])[:, None]
                item["numeric"] = torch.from_numpy(
                    np.concatenate([numeric, bits, multiplicity], axis=1)
                )
                names = stacked[:, [i for v in layout.variables for i in v[:2]]]
                values = np.stack([stacked[:, [v[2], v[3]]] for v in layout.variables], axis=1)
                item["variables"] = torch.from_numpy(
                    names.astype(np.int64).reshape(len(stacked), len(layout.variables), 2)
                )
                item["variable_values"] = torch.from_numpy(symlog(values).astype(np.float32))
                item["weights"] = torch.from_numpy(np.concatenate([e["weights"] for e in entries]))
            if table in OFFERS:
                offers = stacked[:, layout.columns[table]["offer"]].astype(np.int64)
                valid = (offers >= 0) & (offers < counts[batch_index, option])
                receivers.append(starts[batch_index[valid], option] + offers[valid])
                senders.append(positions[table][valid])
            self.tables[table] = item
        for table in ORDER:
            batch_index = batches[table]
            for column, target in layout.references[table]:
                refs = stacks[table][:, column].astype(np.int64)
                available = np.diff(np.append(row_starts[target], len(stacks[target])))
                valid = (refs >= 0) & (refs < available[batch_index])
                rows = row_starts[target][batch_index[valid]] + refs[valid]
                receivers.append(positions[table][valid])
                senders.append(positions[target][rows])
        edges = stacks["edge"].astype(np.int64)
        if len(edges):
            first = row_starts["node"][batches["edge"]]
            parent, child = layout.columns["edge"]["parent"], layout.columns["edge"]["child"]
            receivers.append(positions["node"][first + edges[:, parent]])
            senders.append(positions["node"][first + edges[:, child]])
        self.receivers = torch.from_numpy(np.concatenate(receivers).astype(np.int64))
        self.senders = torch.from_numpy(np.concatenate(senders).astype(np.int64))
        piles = [(b, pile, count) for b, e in enumerate(entries) for pile, count in e["piles"]]
        self.tables["pile"] = {
            "categorical": torch.tensor([[p] for _, p, _ in piles], dtype=torch.int64).reshape(
                -1, 1
            ),
            "numeric": torch.tensor(
                [[np.log1p(c)] for _, _, c in piles], dtype=torch.float32
            ).reshape(-1, 1),
            "position": torch.from_numpy(
                np.array(
                    [starts[b, TOKENS.index("pile")] + r for b, r in self._ranks(piles)],
                    dtype=np.int64,
                )
            ),
        }
        self.options = counts[:, option]
        self.global_position = torch.from_numpy(base + offsets[:, 0])
        self.mask = torch.from_numpy(np.arange(self.width)[None, :] < self.lengths[:, None])
        width = int(self.options.max())
        index = offsets[:, option][:, None] + np.arange(width)[None, :]
        valid = np.arange(width)[None, :] < self.options[:, None]
        self.option_positions = torch.from_numpy(base[:, None] + np.where(valid, index, 0))
        self.option_mask = torch.from_numpy(valid)

    @staticmethod
    def _ranks(piles):
        seen = {}
        for b, _, _ in piles:
            yield b, seen.get(b, 0)
            seen[b] = seen.get(b, 0) + 1

    def to(self, device):
        for item in self.tables.values():
            for key, value in item.items():
                item[key] = value.to(device, non_blocking=True)
        for key in self.FIELDS:
            setattr(self, key, getattr(self, key).to(device, non_blocking=True))
        return self
