from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from chex import Array, dataclass
from jax.random import PRNGKey
from distrax._src.bijectors.bijector import Bijector as distaxBijector
from rbig_jax.transforms.base import InitFunctionsPlus, InitLayersFunctions


class InverseGaussCDF(distaxBijector):
    def __init__(self, eps: float = 1e-5):
        self.eps = eps

    def forward_and_log_det(self, inputs: Array) -> Tuple[Array, Array]:

        # clip inputs within boundary
        inputs = jnp.clip(inputs, self.eps, 1 - self.eps)

        # forward transformation
        outputs = jax.scipy.stats.norm.ppf(inputs)

        # gradient transformation
        # print(outputs.min(), outputs.max())
        # logabsdet = -jax.scipy.stats.norm.logpdf(outputs)
        logabsdet = -_stable_log_pdf(outputs)

        return outputs, logabsdet

    def inverse_and_log_det(self, inputs: Array) -> Tuple[Array, Array]:

        outputs = jax.scipy.stats.norm.cdf(inputs)

        # gradient transformation
        logabsdet = -jax.scipy.stats.norm.logpdf(outputs)

        return outputs, logabsdet


import math

_half_log2pi = 0.5 * math.log(2 * math.pi)
# _half_log2pi = jnp.log(jnp.sqrt(2 * jnp.pi))


def _stable_log_pdf(inputs):

    log_unnormalized = -0.5 * jnp.square(inputs)

    log_normalization = _half_log2pi
    return log_unnormalized - log_normalization


def InitInverseGaussCDF(eps: float = 1e-5, jitted=False):

    # initialize bijector
    bijector = InverseGaussCDF(eps=eps)

    if jitted:
        f = jax.jit(bijector.forward)
    else:
        f = bijector.forward

    def init_bijector(inputs, **kwargs):

        return InverseGaussCDF(eps=eps)

    def bijector_and_transform(inputs, **kwargs):
        outputs = f(inputs)
        return outputs, InverseGaussCDF(eps=eps)

    def transform(inputs, **kwargs):
        outputs = f(inputs)
        return outputs, ()

    def params(inputs, **kwargs):
        return ()

    def params_and_transform(inputs, **kwargs):
        outputs = f(inputs)
        return outputs, ()

    return InitLayersFunctions(
        bijector=init_bijector,
        bijector_and_transform=bijector_and_transform,
        transform=transform,
        params=params,
        params_and_transform=params_and_transform,
    )


# def InverseGaussCDF(eps: float = 1e-5) -> Callable:
#     def init_func(rng: PRNGKey, n_features: int, **kwargs):
#         def forward_func(
#             params: Optional[dataclass], inputs: Array, **kwargs
#         ) -> Tuple[Array, Array]:
#             inputs = jnp.clip(inputs, eps, 1 - eps)

#             outputs = invgausscdf_forward_transform(inputs)

#             logabsdet = -jax.scipy.stats.norm.logpdf(outputs)

#             return outputs, logabsdet.sum(axis=1)

#         def inverse_func(
#             params: Optional[dataclass], inputs: Array, **kwargs
#         ) -> Tuple[Array, Array]:

#             outputs = invgausscdf_inverse_transform(inputs)

#             logabsdet = -jax.scipy.stats.norm.logpdf(outputs)

#             return outputs, logabsdet.sum(axis=1)

#         return (), forward_func, inverse_func

#     return init_func


# def InitInverseGaussCDF(eps: float = 1e-5,) -> Tuple:

#     # TODO a bin initialization function

#     def init_fun(inputs):

#         # clip inputs within boundary
#         inputs = jnp.clip(inputs, eps, 1 - eps)

#         # forward transformation
#         outputs = invgausscdf_forward_transform(inputs)

#         return outputs, ()

#     def forward_transform(params, inputs):

#         # clip inputs within [0,1] boundary
#         inputs = jnp.clip(inputs, eps, 1 - eps)

#         return invgausscdf_forward_transform(inputs)

#     def gradient_transform(params, inputs):

#         # forward transformation
#         outputs = forward_transform(params, inputs)

#         # calculate gradient
#         logabsdet = -jax.scipy.stats.norm.logpdf(outputs)

#         return outputs, logabsdet

#     def inverse_transform(params, inputs):
#         return invgausscdf_inverse_transform(inputs)

#     return init_fun, forward_transform, gradient_transform, inverse_transform


def invgausscdf_forward_transform(X):

    return jax.scipy.stats.norm.ppf(X)


def invgausscdf_inverse_transform(X):
    return jax.scipy.stats.norm.cdf(X)
