import abc
from typing import Callable, Iterable, List, Optional, Sequence, Tuple, NamedTuple

import jax
import jax.numpy as jnp
from chex import Array, dataclass
from jax.random import PRNGKey
from distrax._src.utils.math import sum_last
from distrax._src.bijectors.bijector import Bijector as FixedBijector
from distrax._src.utils import jittable
import abc


@dataclass
class TransformInfo:
    name: str
    input_shape: Tuple
    output_shape: Tuple


class InitFunctions(NamedTuple):
    init_params: Callable
    init_bijector: Callable
    init_transform: Callable


@dataclass
class Bijector:
    def forward(self, inputs: Array) -> Array:
        """Computes y = f(x)."""
        outputs, _ = self.forward_and_log_det(inputs)
        return outputs

    def inverse(self, inputs: Array) -> Array:
        """Computes x = f^{-1}(y)."""
        outputs, _ = self.inverse_and_log_det(inputs)
        return outputs

    def forward_log_det_jacobian(self, inputs: Array) -> Array:
        """Computes log|det J(f)(x)|."""
        _, logdet = self.forward_and_log_det(inputs)
        return logdet

    def inverse_log_det_jacobian(self, inputs: Array) -> Array:
        """Computes log|det J(f^{-1})(y)|."""
        _, logdet = self.inverse_and_log_det(inputs)
        return logdet

    @abc.abstractmethod
    def forward_and_log_det(self, inputs: Array) -> Tuple[Array, Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""

    def inverse_and_log_det(self, inputs: Array) -> Tuple[Array, Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        raise NotImplementedError(
            f"Bijector {self.name} does not implement `inverse_and_log_det`."
        )


class NonTrainableBijector(jittable.Jittable, metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def forward(self, inputs: Array) -> Array:
        return NotImplementedError()

    @abc.abstractmethod
    def forward_and_log_det(self, inputs: Array) -> Tuple[Array, Array]:
        return NotImplementedError()

    @abc.abstractmethod
    def forward_and_bijector(self, inputs: Array) -> Tuple[Array, Bijector]:
        return NotImplementedError()

    @abc.abstractmethod
    def forward_log_det_bijector(self, inputs: Array) -> Tuple[Array, Array, Bijector]:
        return NotImplementedError()

    @abc.abstractmethod
    def bijector(self, inputs: Array) -> Bijector:
        return NotImplementedError()


class InitLayersFunctions(NamedTuple):
    bijector: Optional[Callable]
    bijector_and_transform: Optional[Callable]
    transform: Optional[Callable]
    params: Optional[Callable]
    params_and_transform: Optional[Callable]


class InitFunctionsPlus(NamedTuple):
    init_params: Callable
    init_bijector: Callable
    init_transform: Callable
    init_layer: Callable


@dataclass(frozen=True)
class HyperParams:
    params: dataclass


@dataclass
class InverseBijector:
    bijector: dataclass

    def forward(self, x: Array) -> Array:
        """Computes y = f(x)."""
        y, _ = self.bijector.inverse_and_log_det(x)
        return y

    def inverse(self, y: Array) -> Array:
        """Computes x = f^{-1}(y)."""
        x, _ = self.bijector.forward_and_log_det(y)
        return x

    def forward_log_det_jacobian(self, x: Array) -> Array:
        """Computes log|det J(f)(x)|."""
        _, logdet = self.bijector.inverse_and_log_det(x)
        return logdet

    def inverse_log_det_jacobian(self, y: Array) -> Array:
        """Computes log|det J(f^{-1})(y)|."""
        _, logdet = self.bijector.forward_and_log_det(y)
        return logdet

    def forward_and_log_det(self, x: Array) -> Tuple[Array, Array]:
        """Computes y = f(x) and log|det J(f)(x)|."""
        outputs, logdet = self.bijector.inverse_and_log_det(x)
        return outputs, logdet

    def inverse_and_log_det(self, y: Array) -> Tuple[Array, Array]:
        """Computes x = f^{-1}(y) and log|det J(f^{-1})(y)|."""
        outputs, logdet = self.bijector.forward_and_log_det(y)
        return outputs, logdet


@dataclass
class BijectorChain:
    bijectors: Iterable[Bijector]

    def forward_and_log_det(self, inputs: Array) -> Tuple[Array, Array]:

        outputs = inputs
        total_logabsdet = jnp.zeros_like(outputs)
        # total_logabsdet = jnp.zeros((outputs.shape[0],))
        # total_logabsdet = jnp.expand_dims(total_logabsdet, axis=1)
        for ibijector in self.bijectors:
            outputs, logabsdet = ibijector.forward_and_log_det(outputs)
            total_logabsdet += logabsdet  # sum_last(logabsdet, ndims=logabsdet.ndim)

        return outputs, total_logabsdet

    def inverse_and_log_det(self, inputs: Array) -> Tuple[Array, Array]:

        outputs = inputs
        total_logabsdet = jnp.zeros_like(outputs)
        # total_logabsdet = jnp.expand_dims(total_logabsdet, axis=1)
        for ibijector in reversed(self.bijectors):
            outputs, logabsdet = ibijector.inverse_and_log_det(outputs)
            total_logabsdet += logabsdet  # sum_last(logabsdet, ndims=logabsdet.ndim)

        return outputs, total_logabsdet

    def forward(self, inputs: Array) -> Array:

        outputs = inputs
        for ibijector in self.bijectors:
            outputs = ibijector.forward(outputs)

        return outputs

    def inverse(self, inputs: Array) -> Array:

        outputs = inputs
        for ibijector in reversed(self.bijectors):
            outputs = ibijector.inverse(outputs)

        return outputs

    def forward_log_det_jacobian(self, inputs: Array) -> Array:

        outputs = inputs
        total_logabsdet = jnp.zeros_like(outputs)
        for ibijector in self.bijectors:
            outputs, logabsdet = ibijector.forward_and_log_det(outputs)
            total_logabsdet += logabsdet

        return total_logabsdet

    def inverse_log_det_jacobian(self, inputs: Array) -> Tuple[Array, Array]:

        outputs = inputs
        total_logabsdet = jnp.zeros_like(outputs)
        # total_logabsdet = jnp.expand_dims(total_logabsdet, axis=1)
        for ibijector in reversed(self.bijectors):
            outputs, logabsdet = ibijector.inverse_and_log_det(outputs)
            total_logabsdet += logabsdet  # sum_last(logabsdet, ndims=logabsdet.ndim)

        return total_logabsdet


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
            logabsdet = jnp.zeros(inputs.shape[0])
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


def cascade_forward_and_log_det(
    bijectors: dataclass, inputs: Array
) -> Tuple[Array, Array]:
    outputs = inputs
    total_logabsdet = jnp.zeros_like(outputs)
    for ibijector in bijectors:
        outputs, logabsdet = ibijector.forward_and_log_det(outputs)
        total_logabsdet += logabsdet
    return outputs, total_logabsdet


def cascade_inverse_and_log_det(
    bijectors: dataclass, inputs: Array
) -> Tuple[Array, Array]:
    outputs = inputs
    total_logabsdet = jnp.zeros_like(outputs)
    for ibijector in bijectors[::-1]:
        outputs, logabsdet = ibijector.inverse_and_log_det(outputs)
        total_logabsdet += logabsdet
    return outputs, total_logabsdet
