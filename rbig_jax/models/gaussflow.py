import objax
from objax import Module
from typing import List
import jax


class GaussianizationFlow(Module):
    def __init__(
        self, n_features: int, bijections: List[Module], base_dist: None, generator=None
    ):
        # create Gaussinization block
        self.n_features = n_features
        self.bijections = bijections
        if base_dist is None:
            base_dist = jax.scipy.stats.norm

        self.base_dist = base_dist

        if generator is None:
            generator = objax.random.Generator(123)
        self.generator = generator

    def __call__(self, X):
        return self.bijections(X)

    def transform(self, X):
        return self.bijections.transform(X)

    def inverse_transform(self, X):
        return self.bijections.inverse(X)

    def log_det_jacobian(self, X):
        _, logabsdet = self.bijections(X)
        return logabsdet

    def score_samples(self, X):
        z, logabsdet = self.bijections(X)

        log_prob = self.base_dist.logpdf(z)

        return log_prob.sum(axis=1) + logabsdet

    def score(self, X):
        return -self.score_samples(X).mean()

    def sample(self, n_samples: int):

        X_gauss = objax.random.normal(
            (n_samples, self.n_features), generator=self.generator
        )
        return self.inverse_transform(X_gauss)
