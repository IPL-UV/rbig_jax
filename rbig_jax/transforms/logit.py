from typing import Callable, Optional, Tuple

import jax.numpy as np
from chex import Array, dataclass
from distrax._src.bijectors.sigmoid import _more_stable_sigmoid, _more_stable_softplus
from jax.nn import sigmoid, softplus
from jax.random import PRNGKey

from rbig_jax.transforms.base import Bijector
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


def InitLogit(eps: float = 1e-5, temperature: float = 1.0):
    def init_func(rng: PRNGKey, n_features: int, **kwargs):

        return Inverse(Sigmoid())

    return init_func


def InitLogitTransform():
    def init_func(X: Array, rng: PRNGKey, n_features: int, **kwargs):
        bijector = Inverse(Sigmoid())

        outputs = bijector.forward(X)
        return outputs, bijector

    return init_func
