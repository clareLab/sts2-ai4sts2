import torch
import torch.nn.functional as F
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
        hidden = self.norm(self.encode(flat, batch))
        logits = self.policy(hidden[batch.option_positions]).squeeze(-1)
        logits = logits.masked_fill(~batch.option_mask, float("-inf"))
        value = self.value(hidden[batch.global_position])
        return logits, value

    def encode(self, flat, batch):
        x = flat[batch.tokens]
        allowed = batch.mask[:, None, None, :]
        for layer in self.encoder.layers:
            attention = layer.self_attn
            projected = F.linear(layer.norm1(x), attention.in_proj_weight, attention.in_proj_bias)
            padded = projected.new_zeros(len(flat), 3 * self.width).index_copy(
                0, batch.tokens, projected
            )
            q, k, v = padded.view(batch.size, batch.width, 3, attention.num_heads, -1).permute(
                2, 0, 3, 1, 4
            )
            mixed = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed).transpose(1, 2)
            x = x + attention.out_proj(mixed.reshape(len(flat), self.width)[batch.tokens])
            x = x + layer.linear2(F.relu(layer.linear1(layer.norm2(x))))
        return flat.new_zeros(flat.shape).index_copy(0, batch.tokens, x)
