from typing import Callable, Iterable, Sequence, Tuple

import jax.numpy as jnp
from chex import Array
from distrax._src.distributions.distribution import Distribution
from distrax._src.distributions.log_stddev_normal import LogStddevNormal
from flax import linen as nn
from flax import struct

from rbig_jax.transforms.base import Bijector, BijectorChain


@struct.dataclass
class ConditionalGaussianizationFlow(BijectorChain):
    bijectors: Iterable[Bijector]
    base_dist: Distribution = struct.field(pytree_node=False)
    encoder: Callable

    def score_samples(self, inputs, outputs):

        # forward propagation
        z, log_det = self.forward_and_log_det(inputs)

        # encode params
        y_dist = self.encoder.forward(outputs)

        # calculate latent probability
        latent_prob = y_dist.log_prob(z)
        # calculate log probability
        log_prob = latent_prob.sum(axis=1) + log_det.sum(axis=1)

        return log_prob

    def score(self, inputs, outputs):
        return -jnp.mean(self.score_samples(inputs, outputs))

    def sample(self, outputs: Array, seed: int, n_samples: int):
        # encode params
        y_dist = self.encoder.forward(outputs)

        # generate Gaussian samples
        X_g_samples = y_dist.sample(seed=seed, sample_shape=n_samples)
        # # inverse transformation
        return self.inverse(X_g_samples)


@struct.dataclass
class ConditionalModel:
    params: dict
    model: Callable = struct.field(pytree_node=False)

    def forward(self, inputs) -> Tuple[Array, Array]:
        # forward pass for params
        outputs = self.model.apply(self.params, inputs)
        # split params
        split = outputs.shape[1] // 2

        # compute means and log stds
        means = outputs[..., :split]
        log_stds = outputs[..., split:]
        dist = LogStddevNormal(loc=means, log_scale=log_stds)

        return dist


class ExplicitMLP(nn.Module):
    features: Sequence[int]

    def setup(self):
        # we automatically know what to do with lists, dicts of submodules
        self.layers = [nn.Dense(feat) for feat in self.features]
        # for single submodules, we would just write:
        # self.layer1 = nn.Dense(self, feat1)

    def __call__(self, inputs):
        x = inputs
        for i, lyr in enumerate(self.layers):
            x = lyr(x)
            if i != len(self.layers) - 1:
                x = nn.relu(x)
        return x


class DualHeadProbNet(nn.Module):
    core_features: Sequence[int]
    head_features: Sequence[int]

    def setup(self):
        self.core_net = ExplicitMLP(features=self.core_features)
        self.loc_net = ExplicitMLP(features=self.head_features)
        self.std_net = ExplicitMLP(features=self.head_features)

    def __call__(self, inputs):

        x = self.core_net(inputs)

        loc_preds = self.loc_net(x)
        stds_preds = self.std_net(x)

        return jnp.concatenate([loc_preds, stds_preds], axis=1)
