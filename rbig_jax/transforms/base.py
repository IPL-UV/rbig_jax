from typing import Callable, List, Sequence, Tuple

import jax
import jax.numpy as np
import objax
from objax.module import Module
from chex import Array, dataclass
from jax.random import PRNGKey
import abc


@dataclass
class TransformInfo:
    name: str
    input_shape: Tuple
    output_shape: Tuple


@dataclass
class Bijector:
    def forward(self, x: Array) -> Array:
        """Computes y = f(x)."""
        y, _ = self.forward_and_log_det(x)
        return y

    def inverse(self, y: Array) -> Array:
        """Computes x = f^{-1}(y)."""
        x, _ = self.inverse_and_log_det(y)
        return x

    def forward_log_det_jacobian(self, x: Array) -> Array:
        """Computes log|det J(f)(x)|."""
        _, logdet = self.forward_and_log_det(x)
        return logdet

    def inverse_log_det_jacobian(self, y: Array) -> Array:
        """Computes log|det J(f^{-1})(y)|."""
        _, logdet = self.inverse_and_log_det(y)
        return logdet

    @abc.abstractmethod
    def forward_and_log_det(self, x: Array) -> Tuple[Array, Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""

    def inverse_and_log_det(self, y: Array) -> Tuple[Array, Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        raise NotImplementedError(
            f"Bijector {self.name} does not implement `inverse_and_log_det`."
        )


class Transform(Module):
    """Base class for all transformation"""

    def forward(self, inputs: Array) -> Tuple[Array, Array]:
        raise NotImplementedError()

    def inverse(self, inputs: Array) -> Array:
        raise NotImplementedError


def CompositeTransform(bijectors: Sequence[Callable]):
    def init_fun(rng: PRNGKey, n_features: int, **kwargs):

        # initialize params stores
        all_params, forward_funs, inverse_funs = [], [], []
        # create keys
        rng, *layer_rngs = jax.random.split(rng, num=len(bijectors) + 1)
        for i_rng, init_f in zip(layer_rngs, bijectors):

            param, forward_f, inverse_f = init_f(rng=i_rng, n_features=n_features)

            all_params.append(param)
            forward_funs.append(forward_f)
            inverse_funs.append(inverse_f)

        def bijector_chain(params, bijectors, inputs, **kwargs):
            logabsdet = np.zeros(inputs.shape[0])
            for bijector, param in zip(bijectors, params):
                inputs, ilogabsdet = bijector(param, inputs, **kwargs)
                logabsdet += ilogabsdet
            return inputs, logabsdet

        def forward_func(params, inputs, **kwargs):
            return bijector_chain(params, forward_funs, inputs, **kwargs)

        def inverse_func(params, inputs, **kwargs):
            return bijector_chain(params[::-1], inverse_funs[::-1], inputs, **kwargs)

        return all_params, forward_func, inverse_func

    return init_fun
