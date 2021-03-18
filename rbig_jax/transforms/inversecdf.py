from typing import Callable, Optional, Tuple

import jax
import jax.numpy as np
from jax.random import PRNGKey
from chex import Array, dataclass


def InverseGaussCDF(eps: float = 1e-5) -> Callable:
    def init_func(rng: PRNGKey, n_features: int, **kwargs):
        def forward_func(
            params: Optional[dataclass], inputs: Array, **kwargs
        ) -> Tuple[Array, Array]:
            inputs = np.clip(inputs, eps, 1 - eps)

            outputs = invgausscdf_forward_transform(inputs)

            logabsdet = -jax.scipy.stats.norm.logpdf(outputs)

            return outputs, logabsdet.sum(axis=1)

        def inverse_func(
            params: Optional[dataclass], inputs: Array, **kwargs
        ) -> Tuple[Array, Array]:

            outputs = invgausscdf_inverse_transform(inputs)

            logabsdet = -jax.scipy.stats.norm.logpdf(outputs)

            return outputs, logabsdet.sum(axis=1)

        return (), forward_func, inverse_func

    return init_func


def InitInverseGaussCDF(eps: float = 1e-5,) -> Tuple:

    # TODO a bin initialization function

    def init_fun(inputs):

        # clip inputs within boundary
        inputs = np.clip(inputs, eps, 1 - eps)

        # forward transformation
        outputs = invgausscdf_forward_transform(inputs)

        return outputs, ()

    def forward_transform(params, inputs):

        # clip inputs within [0,1] boundary
        inputs = np.clip(inputs, eps, 1 - eps)

        return invgausscdf_forward_transform(inputs)

    def gradient_transform(params, inputs):

        # forward transformation
        outputs = forward_transform(params, inputs)

        # calculate gradient
        logabsdet = -jax.scipy.stats.norm.logpdf(outputs)

        return outputs, logabsdet

    def inverse_transform(params, inputs):
        return invgausscdf_inverse_transform(inputs)

    return init_fun, forward_transform, gradient_transform, inverse_transform


def invgausscdf_forward_transform(X):

    return jax.scipy.stats.norm.ppf(X)


def invgausscdf_inverse_transform(X):
    return jax.scipy.stats.norm.cdf(X)

