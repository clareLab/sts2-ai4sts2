import hashlib
import math
from functools import lru_cache

import numpy as np
import torch
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

STATE_NODES = 4096
ACTION_NODES = 256
NODE_SIZE = 17
MAX_DEPTH = 16


@lru_cache(maxsize=16384)
def identity(value, size):
    return tuple(hashlib.blake2b(value.encode(), digest_size=size).digest())


def number(value):
    if not math.isfinite(value):
        raise ValueError("Non-finite observation.")
    return math.copysign(math.log1p(abs(value)), value) / 5


def tree(value, capacity):
    output = np.zeros((capacity, NODE_SIZE), dtype=np.float32)
    count = 0

    def visit(item, parent, depth, key, position):
        nonlocal count
        if count >= capacity or depth > MAX_DEPTH:
            raise ValueError("Structured observation capacity exceeded.")
        index = count
        count += 1
        row = output[index]
        row[0:2] = parent, depth
        row[3:7] = identity(key, 4)
        row[16] = number(position)
        if item is None:
            row[2] = 1
        elif isinstance(item, dict):
            row[2] = 2
            for child_key in sorted(item):
                visit(item[child_key], index, depth + 1, child_key, 0)
        elif isinstance(item, list):
            row[2] = 3
            for child_index, child in enumerate(item):
                visit(child, index, depth + 1, "", child_index)
        elif isinstance(item, bool):
            row[2] = 4
            row[15] = float(item)
        elif isinstance(item, (int, float)):
            row[2] = 5
            row[15] = number(item)
        elif isinstance(item, str):
            row[2] = 6
            row[7:15] = identity(item, 8)
        else:
            raise TypeError(type(item))

    visit(value, 0, 0, "", 0)
    return output


class TreeNetwork(nn.Module):
    def __init__(self, width=32):
        super().__init__()
        self.width = width
        self.bytes = nn.Embedding(256, 4)
        self.types = nn.Embedding(7, 4)
        self.node = nn.Sequential(nn.Linear(55 + width, width), nn.Tanh())

    def forward(self, trees):
        batch, capacity, _ = trees.shape
        flat = trees.reshape(-1, NODE_SIZE)
        indices = torch.nonzero(flat[:, 2], as_tuple=True)[0]
        output = trees.new_zeros((batch, self.width))
        if indices.numel() == 0:
            return output
        nodes = flat[indices]
        groups = indices // capacity
        parents = torch.searchsorted(indices, groups * capacity + nodes[:, 0].long())
        depths = nodes[:, 1].long()
        embedded = torch.cat(
            (
                self.bytes(nodes[:, 3:15].long()).flatten(1),
                self.types(nodes[:, 2].long()),
                nodes[:, 15:17],
            ),
            dim=1,
        )
        counts = torch.bincount(parents[depths > 0], minlength=len(nodes)).unsqueeze(1)
        children = trees.new_zeros((len(nodes), self.width))
        for depth in range(int(depths.max()), -1, -1):
            selected = torch.nonzero(depths == depth, as_tuple=True)[0]
            features = torch.cat(
                (
                    embedded[selected],
                    children[selected] / counts[selected].clamp_min(1),
                    counts[selected].to(trees.dtype).log1p() / 5,
                ),
                dim=1,
            )
            encoded = self.node(features)
            if depth:
                children = children.index_add(0, parents[selected], encoded)
            else:
                output = output.index_copy(0, groups[selected], encoded)
        return output


class TreeFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        self.state_size = 512
        self.action_count = observation_space["actions"].shape[0]
        self.action_size = 64
        super().__init__(observation_space, self.state_size + self.action_count * self.action_size)
        self.encoder = TreeNetwork()
        self.state_projection = nn.Sequential(nn.Linear(32, self.state_size), nn.Tanh())
        self.action_projection = nn.Sequential(nn.Linear(32, self.action_size), nn.Tanh())

    def forward(self, observations):
        state = self.state_projection(self.encoder(observations["state"]))
        actions = observations["actions"]
        batch, count, capacity, size = actions.shape
        encoded = self.encoder(actions.reshape(batch * count, capacity, size))
        encoded = self.action_projection(encoded).reshape(batch, count, self.action_size)
        present = actions[:, :, 0, 2].ne(0).unsqueeze(-1)
        return torch.cat((state, (encoded * present).flatten(1)), dim=1)
