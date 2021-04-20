from typing import List

import jax
import objax
from objax import Module

from rbig_jax.transforms.base import CompositeTransform


class GaussianizationFlow:
    def __init__(
        self, bijections: List, base_dist: None,
    ):
        # create init f bijections
        self.init_func = CompositeTransform(bijections)

        # create base distribution
        if base_dist is None:
            base_dist = jax.scipy.stats.norm

        self.base_dist = base_dist

    def init_params(self, rng, n_features):

        params, forward_transform, inverse_transform = self.init_func(rng, n_features)
        self.n_features = n_features
        self.forward_chain = forward_transform
        self.inverse_chain = inverse_transform

        return params

    def __call__(self, params, X):

        return self.forward_chain(params, X)

    def transform(self, params, X):
        X, _ = self.forward_chain(params, X)
        return X

    def inverse_transform(self, params, X):
        X, _ = self.inverse_chain(params, X)
        return X

    def log_det_jacobian(self, params, X):
        _, logabsdet = self.forward_chain(params, X)
        return logabsdet

    def score_samples(self, params, X):
        z, logabsdet = self.forward_chain(params, X)

        log_prob = self.base_dist.logpdf(z)

        return log_prob.sum(axis=1) + logabsdet

    def score(self, params, X):
        return -self.score_samples(params, X).mean()

    def sample(self, rng, params, n_samples: int):

        X_gauss = jax.random.normal(rng, shape=(n_samples, self.n_features))
        return self.inverse_transform(params, X_gauss)
