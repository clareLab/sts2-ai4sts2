import math

import numpy as np
import torch
from sb3_contrib.common.maskable.distributions import MaskableCategoricalDistribution
from sb3_contrib.common.maskable.policies import MaskableMultiInputActorCriticPolicy
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor
from torch import nn

from ai4sts2.encoding import TreeFeatures


def evaluation_action(model, observation, mask, rng, deterministic=True):
    if model is None:
        return int(rng.choice(np.flatnonzero(mask)))
    if deterministic:
        action, _ = model.predict(observation, deterministic=True, action_masks=mask)
        return int(action)
    model.policy.set_training_mode(False)
    with torch.no_grad():
        tensors, _ = model.policy.obs_to_tensor(observation)
        distribution = model.policy.get_distribution(tensors, action_masks=mask)
        probabilities = distribution.distribution.probs[0].cpu().numpy().astype(np.float64)
    if (
        probabilities.shape != mask.shape
        or not np.isfinite(probabilities).all()
        or (probabilities < 0).any()
        or (probabilities[~mask] != 0).any()
        or probabilities.sum() <= 0
    ):
        raise ValueError("Invalid masked policy probabilities.")
    return int(rng.choice(len(probabilities), p=probabilities / probabilities.sum()))


class ActionFeatures(BaseFeaturesExtractor):
    def __init__(self, observation_space):
        self.state_size = observation_space["state"].shape[0]
        self.action_count, self.action_size = observation_space["actions"].shape
        super().__init__(
            observation_space,
            sum(math.prod(space.shape) for space in observation_space.spaces.values()),
        )

    def forward(self, observations):
        return torch.cat((observations["state"], observations["actions"].flatten(1)), dim=1)


class ActionNetwork(nn.Module):
    def __init__(self, extractor, width):
        super().__init__()
        self.state_size = extractor.state_size
        self.action_count, self.action_size = extractor.action_count, extractor.action_size
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
    def __init__(self, *args, width=64, encoding="hash", temperature=1.0, **kwargs):
        if not isinstance(width, int) or width < 1:
            raise ValueError("Use a positive network width.")
        self.width = width
        if encoding not in {"hash", "tree"}:
            raise ValueError("Unknown observation encoding.")
        self.encoding = encoding
        self.temperature = temperature
        kwargs["features_extractor_class"] = TreeFeatures if encoding == "tree" else ActionFeatures
        super().__init__(*args, **kwargs)

    @property
    def temperature(self):
        return self._temperature

    @temperature.setter
    def temperature(self, value):
        if not math.isfinite(value) or value <= 0:
            raise ValueError("Use a finite positive sampling temperature.")
        self._temperature = float(value)

    def _build_mlp_extractor(self):
        self.mlp_extractor = ActionNetwork(self.features_extractor, self.width).to(self.device)
        self.action_dist = SharedCategorical(self.action_space.n)

    def _get_action_dist_from_latent(self, latent_pi):
        return self.action_dist.proba_distribution(
            action_logits=self.action_net(latent_pi) / self.temperature
        )

    def _get_constructor_parameters(self):
        return super()._get_constructor_parameters() | {
            "width": self.width,
            "encoding": self.encoding,
            "temperature": self.temperature,
        }
