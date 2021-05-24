from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from chex import Array, dataclass
from distrax._src.bijectors.inverse import Inverse
from distrax._src.bijectors.sigmoid import (Sigmoid, _more_stable_sigmoid,
                                            _more_stable_softplus)
from flax import struct
from jax.nn import sigmoid, softplus
from jax.random import PRNGKey

from rbig_jax.transforms.base import Bijector, InitLayersFunctions

EPS = 1e-5
TEMPERATURE = 1.0


def safe_log(x):
    x = jnp.clip(x, a_min=1e-22)
    return jnp.log(x)


@struct.dataclass
class LogitTemperature(Bijector):
    temperature: Array

    def forward(self, inputs: Array, **kwargs) -> Array:

        inputs = jnp.clip(inputs, EPS, 1 - EPS)

        outputs = (1.0 / self.temperature) * (safe_log(inputs) - jnp.log1p(-inputs))

        return outputs

    def forward_and_log_det(self, inputs: Array, **kwargs) -> Tuple[Array, Array]:

        inputs = jnp.clip(inputs, EPS, 1 - EPS)

        outputs = (1.0 / self.temperature) * (safe_log(inputs) - jnp.log1p(-inputs))

        logabsdet = -self.inverse_log_det_jacobian(outputs, **kwargs)

        return outputs, logabsdet

    def inverse(self, inputs: Array, **kwargs) -> Array:

        inputs = self.temperature * inputs

        # forward transformation
        outputs = _more_stable_sigmoid(inputs)

        return outputs

    def inverse_and_log_det(self, inputs: Array, **kwargs) -> Tuple[Array, Array]:

        inputs = self.temperature * inputs

        # forward transformation
        outputs = _more_stable_sigmoid(inputs)

        # abs log determinant jacobian
        logabsdet = self.inverse_log_det_jacobian(inputs, **kwargs)

        return outputs, logabsdet

    def inverse_log_det_jacobian(self, inputs: Array, **kwargs) -> Array:

        # abs log determinant jacobian
        logabsdet = (
            safe_log(self.temperature)
            - _more_stable_softplus(-inputs)
            - _more_stable_softplus(inputs)
        )

        return logabsdet


def InitSigmoidTransform(eps: float = 1e-5, jitted: bool = False):

    if jitted:
        f = jax.jit(Sigmoid().forward)
    else:
        f = Sigmoid().forward

    def transform(inputs, **kwargs):
        inputs = jnp.clip(inputs, eps, 1 - eps)

        outputs = f(inputs)

        return outputs

    def bijector(inputs=None, **kwargs):

        return Sigmoid()

    def transform_and_bijector(inputs, **kwargs):
        inputs = jnp.clip(inputs, eps, 1 - eps)
        outputs = f(inputs)
        return outputs, Sigmoid()

    def transform_gradient_bijector(inputs, **kwargs):
        inputs = jnp.clip(inputs, eps, 1 - eps)
        bijector = Sigmoid()

        outputs, logabsdet = bijector.forward_and_log_det(inputs)

        return outputs, logabsdet, bijector

    return InitLayersFunctions(
        transform=transform,
        bijector=bijector,
        transform_and_bijector=transform_and_bijector,
        transform_gradient_bijector=transform_gradient_bijector,
    )


def InitLogitTransform(jitted: bool = False):

    if jitted:
        f = jax.jit(Inverse(Sigmoid()).forward)
    else:
        f = Inverse(Sigmoid()).forward

    def transform(inputs, **kwargs):

        outputs = f(inputs)

        return outputs

    def bijector(inputs=None, **kwargs):

        return Inverse(Sigmoid())

    def transform_and_bijector(inputs, **kwargs):
        outputs = f(inputs)
        return outputs, Inverse(Sigmoid())

    def transform_gradient_bijector(inputs, **kwargs):
        bijector = Inverse(Sigmoid())

        outputs, logabsdet = bijector.forward_and_log_det(inputs)

        return outputs, logabsdet, bijector

    return InitLayersFunctions(
        transform=transform,
        bijector=bijector,
        transform_and_bijector=transform_and_bijector,
        transform_gradient_bijector=transform_gradient_bijector,
    )


def InitLogitTempTransform(temperature: float = 1.0, jitted: bool = False):

    if jitted:
        f = jax.jit(LogitTemperature(temperature=temperature).forward)
    else:
        f = LogitTemperature(temperature=temperature).forward

    def transform(inputs, **kwargs):

        outputs = f(inputs)

        return outputs

    def bijector(inputs=None, **kwargs):

        return LogitTemperature(temperature=temperature)

    def transform_and_bijector(inputs, **kwargs):
        outputs = f(inputs)
        return outputs, LogitTemperature(temperature=temperature)

    def transform_gradient_bijector(inputs, **kwargs):
        bijector = LogitTemperature(temperature=temperature)

        outputs, logabsdet = bijector.forward_and_log_det(inputs)

        return outputs, logabsdet, bijector

    return InitLayersFunctions(
        transform=transform,
        bijector=bijector,
        transform_and_bijector=transform_and_bijector,
        transform_gradient_bijector=transform_gradient_bijector,
    )
