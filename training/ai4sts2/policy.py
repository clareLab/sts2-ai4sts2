import math

import torch
from sb3_contrib.common.maskable.distributions import MaskableCategoricalDistribution
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn


class ActionFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        super().__init__(
            observation_space,
            sum(math.prod(space.shape) for space in observation_space.spaces.values()),
        )

    def forward(self, observations):
        return torch.cat((observations["state"], observations["actions"].flatten(1)), dim=1)


class ActionNetwork(nn.Module):
    def __init__(self, observation_space, width):
        super().__init__()
        self.state_size = observation_space["state"].shape[0]
        self.action_count, self.action_size = observation_space["actions"].shape
        self.latent_dim_pi = self.latent_dim_vf = width
        self.state_encoder = nn.Sequential(nn.Linear(self.state_size, width), nn.Tanh())
        self.action_encoder = nn.Sequential(nn.Linear(self.action_size, width), nn.Tanh())
        self.actor = nn.Sequential(nn.Linear(width * 3 + 1, width), nn.Tanh())
        self.critic = nn.Sequential(nn.Linear(width * 2 + 1, width), nn.Tanh())

    def encode(self, features):
        state = self.state_encoder(features[:, : self.state_size])
        actions = features[:, self.state_size :].reshape(-1, self.action_count, self.action_size)
        present = actions.ne(0).any(dim=-1, keepdim=True)
        encoded = self.action_encoder(actions)
        count = present.sum(dim=1)
        pooled = (encoded * present).sum(dim=1) / count.clamp_min(1)
        context = torch.cat((state, pooled, count / self.action_count), dim=1)
        return encoded, context

    def actor_features(self, encoded, context):
        context = context.unsqueeze(1).expand(-1, self.action_count, -1)
        return self.actor(torch.cat((encoded, context), dim=-1))

    def forward(self, features):
        encoded, context = self.encode(features)
        return self.actor_features(encoded, context), self.critic(context)

    def forward_actor(self, features):
        return self.actor_features(*self.encode(features))

    def forward_critic(self, features):
        return self.critic(self.encode(features)[1])


class SharedCategorical(MaskableCategoricalDistribution):
    def proba_distribution_net(self, latent_dim):
        return nn.Sequential(nn.Linear(latent_dim, 1), nn.Flatten(start_dim=1))


class SharedActionPolicy(MaskableMultiInputActorCriticPolicy):
    def __init__(self, *args, width=64, **kwargs):
        if not isinstance(width, int) or width < 1:
            raise ValueError("Use a positive network width.")
        self.width = width
        kwargs["features_extractor_class"] = ActionFeatures
        super().__init__(*args, **kwargs)

    def _build_mlp_extractor(self):
        self.mlp_extractor = ActionNetwork(self.observation_space, self.width).to(self.device)
        self.action_dist = SharedCategorical(self.action_space.n)

    def _get_constructor_parameters(self):
        return super()._get_constructor_parameters() | {"width": self.width}
