from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from chex import Array, dataclass
from distrax._src.bijectors.inverse import Inverse
from distrax._src.bijectors.sigmoid import (
    Sigmoid,
    _more_stable_sigmoid,
    _more_stable_softplus,
)
from jax.nn import sigmoid, softplus
from jax.random import PRNGKey

from rbig_jax.transforms.base import Bijector, InitLayersFunctions

EPS = 1e-5
TEMPERATURE = 1.0


def safe_log(x):
    x = jnp.clip(x, a_min=1e-22)
    return jnp.log(x)


@dataclass
class Logit(Bijector):
    def forward_and_log_det(self, inputs: Array, **kwargs) -> Tuple[Array, Array]:

        # inputs = jnp.clip(inputs, EPS, 1 - EPS)
        outputs = -safe_log(jnp.reciprocal(inputs) - 1.0)

        # outputs = jnp.log(inputs) - jnp.log1p(-inputs)

        logabsdet = -safe_log(inputs) - safe_log(1.0 - inputs)

        # # abs log determinant jacobian
        # logabsdet = -self.inverse_log_det_jacobian(outputs)

        return outputs, logabsdet

    def inverse_log_det_jacobian(self, inputs: Array, **kwargs) -> Array:

        # authors implementation
        logabsdet = _more_stable_softplus(inputs) + _more_stable_softplus(-inputs)

        # # distrax implementation
        # logabsdet = -_more_stable_softplus(-inputs) - _more_stable_softplus(inputs)
        return logabsdet

    def inverse_and_log_det(self, inputs: Array, **kwargs) -> Tuple[Array, Array]:

        # forward transformation
        outputs = _more_stable_sigmoid(inputs)

        # abs log determinant jacobian
        logabsdet = self.inverse_log_det_jacobian(inputs)

        return outputs, logabsdet


def InitSigmoidTransform(jitted: bool = False):

    if jitted:
        f = jax.jit(Sigmoid().forward)
    else:
        f = Sigmoid().forward

    def transform(inputs, **kwargs):

        outputs = f(inputs)

        return outputs

    def bijector(inputs=None, **kwargs):

        return Sigmoid()

    def transform_and_bijector(inputs, **kwargs):
        outputs = f(inputs)
        return outputs, Sigmoid()

    def transform_gradient_bijector(inputs, **kwargs):
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
