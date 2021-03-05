from typing import Tuple

import jax
import jax.numpy as np
import objax
from objax import StateVar
from objax.typing import JaxArray

from rbig_jax.transforms.base import Transform


class InverseGaussCDF(Transform):
    def __init__(
        self, eps=1e-6,
    ):
        super().__init__()
        self.eps = StateVar(np.array(eps))

    def __call__(self, inputs: JaxArray) -> Tuple[JaxArray, JaxArray]:

        inputs = np.clip(inputs, self.eps.value, 1 - self.eps.value)

        outputs = jax.scipy.stats.norm.ppf(inputs)

        logabsdet = -jax.scipy.stats.norm.logpdf(outputs)

        return outputs, logabsdet.sum(axis=1)

    def transform(self, inputs: JaxArray) -> JaxArray:

        inputs = np.clip(inputs, self.eps.value, 1 - self.eps.value)

        outputs = jax.scipy.stats.norm.ppf(inputs)
        return outputs

    def inverse(self, inputs: JaxArray) -> JaxArray:
        return jax.scipy.stats.norm.cdf(inputs)


def InitInverseGaussCDF(eps: float = 1e-5,):

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

