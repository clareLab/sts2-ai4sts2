import torch
from torch import nn

from ai4sts2.features import CAPACITY, TOKENS


class Net(nn.Module):
    def __init__(self, layout, bins, width=128, depth=3, heads=4):
        super().__init__()
        self.width = width
        self.embed = nn.Embedding(CAPACITY, width, padding_idx=0)
        self.kind = nn.Embedding(len(TOKENS), width)
        widths = {table: layout.width(table) for table in TOKENS}
        self.numeric = nn.ModuleDict(
            {table: nn.Linear(size, width) for table, size in widths.items() if size}
        )
        self.variable = nn.Linear(2, width)
        layer = nn.TransformerEncoderLayer(
            width, heads, 2 * width, dropout=0.0, batch_first=True, norm_first=True
        )
        self.encoder = nn.TransformerEncoder(layer, depth, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(width)
        self.policy = nn.Linear(width, 1)
        self.value = nn.Sequential(nn.Linear(width, width), nn.GELU(), nn.Linear(width, bins))

    def forward(self, batch):
        flat = self.embed.weight.new_zeros(batch.size * batch.width, self.width)
        for index, table in enumerate(TOKENS):
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
                tokens = tokens + (self.embed(item["variables"]).sum(2) * gates).sum(1)
                tokens = tokens * item["weights"].unsqueeze(-1)
            flat = flat.index_add(0, item["position"], tokens)
        flat = flat.index_add(0, batch.receivers, flat[batch.senders])
        hidden = flat.view(batch.size, batch.width, self.width)
        hidden = self.norm(self.encoder(hidden, src_key_padding_mask=~batch.mask))
        hidden = hidden.reshape(-1, self.width)
        logits = self.policy(hidden[batch.option_positions]).squeeze(-1)
        logits = logits.masked_fill(~batch.option_mask, float("-inf"))
        value = self.value(hidden[batch.global_position])
        return logits, value
