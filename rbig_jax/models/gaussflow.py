import jax.numpy as jnp
from typing import Iterable
from chex import Array, dataclass
from rbig_jax.transforms.base import BijectorChain, Bijector
from distrax._src.distributions.distribution import Distribution


@dataclass
class GaussianizationFlow(BijectorChain):
    bijectors: Iterable[Bijector]
    base_dist: Distribution

    def score_samples(self, inputs):

        # forward propagation
        z, log_det = self.forward_and_log_det(inputs)

        # calculate latent probability
        latent_prob = self.base_dist.log_prob(z)

        # calculate log probability
        log_prob = latent_prob.sum(axis=1) + log_det.sum(axis=1)

        return log_prob

    def score(self, inputs):
        return -jnp.mean(self.score_samples(inputs))

    def sample(self, seed: int, n_samples: int):
        # generate Gaussian samples
        X_g_samples = self.base_dist.sample(seed=seed, sample_shape=n_samples)
        # # inverse transformation
        return self.inverse(X_g_samples)


