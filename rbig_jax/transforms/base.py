from typing import Callable, List, Sequence, Tuple

import jax
import jax.numpy as np
import objax
from objax.module import Module
from objax.typing import JaxArray
from jax.random import PRNGKey
from distrax._src.utils import jittable
import abc
from chex import Array, dataclass


class Bijector(jittable.Jittable, metaclass=abc.ABCMeta):
    def __init__(self):
        pass

    def init_params(self, rng: PRNGKey, n_features: int) -> dataclass:
        """Computes y = f(x) and log|det J(f)(x)|."""

    def forward(self, params: dataclass, x: Array) -> Array:
        """Computes y = f(x)."""
        y, _ = self.forward_and_log_det(params, x)
        return y

    def inverse(self, params: dataclass, y: Array) -> Array:
        """Computes x = f^{-1}(y)."""
        x, _ = self.inverse_and_log_det(params, y)
        return x

    def forward_log_det_jacobian(self, params: dataclass, x: Array) -> Array:
        """Computes log|det J(f)(x)|."""
        _, logdet = self.forward_and_log_det(params, x)
        return logdet

    def inverse_log_det_jacobian(self, params: dataclass, y: Array) -> Array:
        """Computes log|det J(f^{-1})(y)|."""
        _, logdet = self.inverse_and_log_det(params, y)
        return logdet

    @abc.abstractmethod
    def forward_and_log_det(self, params: dataclass, x: Array) -> Tuple[Array, Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""

    def inverse_and_log_det(self, params: dataclass, y: Array) -> Tuple[Array, Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        raise NotImplementedError(
            f"Bijector {self.name} does not implement `inverse_and_log_det`."
        )

    @property
    def name(self) -> str:
        """Name of the bijector."""
        return self.__class__.__name__


class Transform(Module):
    """Base class for all transformation"""

    def forward(self, inputs: JaxArray) -> Tuple[JaxArray, JaxArray]:
        raise NotImplementedError()

    def inverse(self, inputs: JaxArray) -> JaxArray:
        raise NotImplementedError


# class Chain(Bijector):
#     def __init__(self, bijectors: Sequence[Bijector]):
#         self._bijectors = bijectors
#         super().__init__()

#     @property
#     def bijectors(self):
#         return self._bijectors

#     def forward(self, params: Sequence[dataclass], x: Array) -> Array:
#         for iparams, bijector in zip(params, self._bijectors):
#             x = bijector.forward(iparams, x)

#         return x

#     def inverse(self, params: Sequence[dataclass], x: Array) -> Array:
#         for iparams, bijector in zip(reversed(params), reversed(self._bijectors)):
#             x = bijector.inverse(iparams, x)

#         return x

#     def forward_and_log_det(
#         self, params: Sequence[dataclass], x: Array
#     ) -> Tuple[Array, Array]:
#         """Computes y = f(x) and log|det J(f)(x)|."""
#         x, log_det = self._bijectors[-1].forward_and_log_det(params[-1], x)
#         for iparams, bijector in zip(params[1:], self._bijectors[1:]):
#             x, ld = bijector.forward_and_log_det(iparams, x)
#             log_det += ld
#         return x, log_det

#     def inverse_and_log_det(
#         self, params: Sequence[dataclass], y: Array
#     ) -> Tuple[Array, Array]:
#         """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
#         y, log_det = self._bijectors[-1].inverse_and_log_det(params[-1], y)
#         for iparams, bijector in zip(
#             reversed(params[1:]), reversed(self._bijectors[1:])
#         ):
#             y, ld = bijector.inverse_and_log_det(iparams, y)
#             log_det += ld
#         return y, log_det

def Chain

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
