import torch
from torch import nn

from ai4sts2.features import ORDER


class Net(nn.Module):
    def __init__(self, layout, capacity=16384, width=128, depth=3, heads=4):
        super().__init__()
        self.width = width
        self.embed = nn.Embedding(capacity, width, padding_idx=0)
        self.kind = nn.Embedding(len(ORDER), width)
        self.numeric = nn.ModuleDict(
            {table: nn.Linear(layout.width(table), width) for table in ORDER if layout.width(table)}
        )
        self.variable = nn.Linear(2, width)
        layer = nn.TransformerEncoderLayer(
            width, heads, 2 * width, dropout=0.0, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.policy = nn.Linear(width, 1)
        self.value = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, 1))

    def forward(self, batch):
        flat = torch.zeros(batch.size * batch.width, self.width)
        for index, table in enumerate(ORDER):
            item = batch.tables[table]
            count = len(item["position"])
            if not count:
                continue
            tokens = self.kind.weight[index].expand(count, -1) + self.embed(
                item["categorical"]
            ).sum(1)
            if table in self.numeric:
                tokens = tokens + self.numeric[table](item["numeric"])
            if table == "card":
                gates = torch.tanh(self.variable(item["variable_values"]))
                tokens = tokens + (self.embed(item["variables"]) * gates).sum(1)
            flat = flat.index_copy(0, item["position"], tokens)
        related = torch.zeros_like(flat)
        for table in ORDER:
            item = batch.tables[table]
            for references, _target in item.get("references", []):
                valid = references >= 0
                if valid.any():
                    related = related.index_add(0, item["position"][valid], flat[references[valid]])
            offers = item.get("offers")
            if offers is not None:
                valid = offers >= 0
                if valid.any():
                    related = related.index_add(0, offers[valid], flat[item["position"][valid]])
        hidden = (flat + related).view(batch.size, batch.width, self.width)
        hidden = self.norm(self.encoder(hidden, src_key_padding_mask=~batch.mask))
        hidden = hidden.reshape(-1, self.width)
        logits = self.policy(hidden[batch.option_positions]).squeeze(-1)
        logits = logits.masked_fill(~batch.option_mask, float("-inf"))
        value = self.value(hidden[batch.global_position]).squeeze(-1)
        return logits, value
