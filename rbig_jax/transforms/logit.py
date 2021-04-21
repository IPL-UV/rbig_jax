from typing import Callable, Optional, Tuple

import jax
import jax.numpy as np
from chex import Array, dataclass
from distrax._src.bijectors.sigmoid import _more_stable_sigmoid, _more_stable_softplus
from jax.nn import sigmoid, softplus
from jax.random import PRNGKey

from rbig_jax.transforms.base import Bijector, InitFunctionsPlus
from distrax._src.bijectors.sigmoid import Sigmoid
from distrax._src.bijectors.inverse import Inverse

EPS = 1e-5
TEMPERATURE = 1.0


@dataclass
class Logit(Bijector):
    def forward_and_log_det(self, inputs: Array, **kwargs) -> Tuple[Array, Array]:

        # inputs = np.clip(inputs, EPS, 1 - EPS)

        outputs = np.log(inputs) - np.log1p(-inputs)

        # abs log determinant jacobian
        logabsdet = -self.inverse_log_det_jacobian(outputs)

        return outputs, logabsdet

    def inverse_log_det_jacobian(self, inputs: Array, **kwargs) -> Array:

        # abs log determinant jacobian
        logabsdet = -_more_stable_softplus(-inputs) - _more_stable_softplus(inputs)
        return logabsdet

    def inverse_and_log_det(self, inputs: Array, **kwargs) -> Tuple[Array, Array]:

        # forward transformation
        outputs = _more_stable_sigmoid(inputs)

        # abs log determinant jacobian
        logabsdet = self.inverse_log_det_jacobian(inputs)

        return outputs, logabsdet


def InitSigmoidTransform(jitted: bool = False):

    # initialize bijector
    bijector = Sigmoid()

    if jitted:
        f = jax.jit(bijector.forward)
    else:
        f = bijector.forward

    def init_layer(rng: PRNGKey, n_features: int, **kwargs):

        return bijector

    def init_params(inputs):
        outputs = f(inputs)
        return outputs, ()

    def init_transform(inputs):

        outputs = f(inputs)
        return outputs

    def init_bijector(inputs):
        outputs = f(inputs)
        return outputs, bijector

    return InitFunctionsPlus(
        init_params=init_params,
        init_bijector=init_bijector,
        init_transform=init_transform,
        init_layer=init_layer,
    )


def InitLogitTransform(jitted: bool = False):

    # initialize bijector
    bijector = Inverse(Sigmoid())

    if jitted:
        f = jax.jit(bijector.forward)
    else:
        f = bijector.forward

    def init_layer(rng: PRNGKey, n_features: int, **kwargs):

        return bijector

    def init_params(inputs):
        outputs = f(inputs)
        return outputs, ()

    def init_transform(inputs):

        outputs = f(inputs)
        return outputs

    def init_bijector(inputs):
        outputs = f(inputs)
        return outputs, bijector

    return InitFunctionsPlus(
        init_params=init_params,
        init_bijector=init_bijector,
        init_transform=init_transform,
        init_layer=init_layer,
    )
