import math
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from chex import Array, dataclass
from distrax._src.bijectors.bijector import Bijector as distaxBijector
from jax.random import PRNGKey

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


class GaussCDF(distaxBijector):
    def __init__(self, eps: float = 1e-5):
        self.eps = eps

    def forward_and_log_det(self, inputs: Array) -> Tuple[Array, Array]:

        outputs = jax.scipy.stats.norm.cdf(inputs)

        # gradient transformation
        logabsdet = -jax.scipy.stats.norm.logpdf(outputs)

        return outputs, logabsdet

    def inverse_and_log_det(self, inputs: Array) -> Tuple[Array, Array]:

        # clip inputs within boundary
        inputs = jnp.clip(inputs, self.eps, 1 - self.eps)

        # forward transformation
        outputs = jax.scipy.stats.norm.ppf(inputs)

        # gradient transformation
        # print(outputs.min(), outputs.max())
        # logabsdet = -jax.scipy.stats.norm.logpdf(outputs)
        logabsdet = -_stable_log_pdf(outputs)

        return outputs, logabsdet



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
        return outputs

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


def InitGaussCDF(eps: float = 1e-5, jitted=False):

    # initialize bijector
    bijector = GaussCDF(eps=eps)

    if jitted:
        f = jax.jit(bijector.forward)
    else:
        f = bijector.forward

    def init_bijector(inputs, **kwargs):

        return GaussCDF(eps=eps)

    def bijector_and_transform(inputs, **kwargs):
        outputs = f(inputs)
        return outputs, GaussCDF(eps=eps)

    def transform(inputs, **kwargs):
        outputs = f(inputs)
        return outputs

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


def invgausscdf_forward_transform(X):

    return jax.scipy.stats.norm.ppf(X)


def invgausscdf_inverse_transform(X):
    return jax.scipy.stats.norm.cdf(X)
