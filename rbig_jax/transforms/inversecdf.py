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


def invgausscdf_forward_transform(X):

    return jax.scipy.stats.norm.ppf(X)


def invgausscdf_inverse_transform(X):
    return jax.scipy.stats.norm.cdf(X)


def get_params(X):

    # forward transformation
    X = jax.scipy.stats.norm.ppf(X)

    # Jacobian
    log_det_jacobian = jax.scipy.stats.norm.logpdf(X)

    def forward_function(X):
        # get the
        return jax.scipy.stats.norm.ppf(X)

    def inverse_function(X):
        return jax.scipy.stats.norm.cdf(X)

    return X, log_det_jacobian, forward_function, inverse_function
