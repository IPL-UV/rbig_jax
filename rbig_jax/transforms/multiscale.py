from rbig_jax.transforms.reshape import init_scale_function
from typing import Iterable, Tuple, Optional, NamedTuple, Callable, List
from chex import dataclass, Array
from rbig_jax.transforms.base import Bijector
import jax.numpy as jnp


class RescaleParams(NamedTuple):
    fh: int
    fw: int
    Hn: int
    Wn: int
    W: int
    H: int
    C: int


class RescaleFunctions(NamedTuple):
    forward: Callable
    inverse: Callable
    params: RescaleParams


@dataclass
class MultiScaleBijectorChain:
    bijectors: Iterable[Bijector]
    filter_shape: Tuple[int, int]
    image_shape: Tuple

    def __post_init__(self):
        self.ms_reshape = init_scale_function(
            self.filter_shape, self.image_shape, batch=False
        )

    def forward_and_log_det(self, inputs: Array) -> Tuple[Array, Array]:

        # rescale data
        inputs = self.ms_reshape.forward(inputs)

        outputs = inputs
        total_logabsdet = jnp.zeros_like(outputs)
        # total_logabsdet = jnp.zeros((outputs.shape[0],))
        # total_logabsdet = jnp.expand_dims(total_logabsdet, axis=1)
        for ibijector in self.bijectors:
            outputs, logabsdet = ibijector.forward_and_log_det(outputs)
            total_logabsdet += logabsdet  # sum_last(logabsdet, ndims=logabsdet.ndim)

        # unrescale data
        outputs = self.ms_reshape.inverse(outputs)
        total_logabsdet = self.ms_reshape.inverse(total_logabsdet)
        return outputs, total_logabsdet

    def inverse_and_log_det(self, inputs: Array) -> Tuple[Array, Array]:

        # rescale data
        inputs = self.ms_reshape.forward(inputs)

        outputs = inputs
        total_logabsdet = jnp.zeros_like(outputs)
        # total_logabsdet = jnp.expand_dims(total_logabsdet, axis=1)
        for ibijector in reversed(self.bijectors):
            outputs, logabsdet = ibijector.inverse_and_log_det(outputs)
            total_logabsdet += logabsdet  # sum_last(logabsdet, ndims=logabsdet.ndim)

        # unrescale data
        outputs = self.ms_reshape.inverse(outputs)
        total_logabsdet = self.ms_reshape.inverse(total_logabsdet)

        return outputs, total_logabsdet

    def forward(self, inputs: Array) -> Array:

        # rescale data
        inputs = self.ms_reshape.forward(inputs)

        outputs = inputs
        for ibijector in self.bijectors:
            outputs = ibijector.forward(outputs)
        # unrescale data
        outputs = self.ms_reshape.inverse(outputs)

        return outputs

    def inverse(self, inputs: Array) -> Array:
        # rescale data
        inputs = self.ms_reshape.forward(inputs)

        outputs = inputs
        for ibijector in reversed(self.bijectors):
            outputs = ibijector.inverse(outputs)

        # unrescale data
        outputs = self.ms_reshape.inverse(outputs)
        return outputs

    def forward_log_det_jacobian(self, inputs: Array) -> Array:
        # rescale data
        inputs = self.ms_reshape.forward(inputs)
        outputs = inputs
        total_logabsdet = jnp.zeros_like(outputs)
        for ibijector in self.bijectors:
            outputs, logabsdet = ibijector.forward_and_log_det(outputs)
            total_logabsdet += logabsdet

        # unrescale data
        total_logabsdet = self.ms_reshape.inverse(total_logabsdet)
        return total_logabsdet

    def inverse_log_det_jacobian(self, inputs: Array) -> Tuple[Array, Array]:

        # rescale data
        inputs = self.ms_reshape.forward(inputs)
        outputs = inputs
        total_logabsdet = jnp.zeros_like(outputs)
        # total_logabsdet = jnp.expand_dims(total_logabsdet, axis=1)
        for ibijector in reversed(self.bijectors):
            outputs, logabsdet = ibijector.inverse_and_log_det(outputs)
            total_logabsdet += logabsdet  # sum_last(logabsdet, ndims=logabsdet.ndim)

        # unrescale data
        total_logabsdet = self.ms_reshape.inverse(total_logabsdet)

        return total_logabsdet


@dataclass
class MultiScaleRBIGBlockInit:
    init_functions: List[dataclass]
    filter_shape: Tuple[int, int]
    image_shape: Tuple

    def __post_init__(self):
        self.ms_reshape = init_scale_function(self.filter_shape, self.image_shape)

    def forward_and_params(self, inputs: Array) -> Tuple[Array, Array]:

        # rescale data
        inputs = self.ms_reshape.forward(inputs)

        outputs = inputs
        bijectors = []

        # loop through bijectors
        for ibijector in self.init_functions:

            # transform and params
            outputs, ibjector = ibijector.bijector_and_transform(outputs)

            # accumulate params
            bijectors.append(ibjector)

        # unrescale data
        outputs = self.ms_reshape.inverse(outputs)

        # create bijector chain
        bijectors = MultiScaleBijectorChain(
            bijectors=bijectors,
            filter_shape=self.filter_shape,
            image_shape=self.image_shape,
        )

        return outputs, bijectors

    def forward(self, inputs: Array) -> Array:

        # rescale data
        inputs = self.ms_reshape.forward(inputs)

        outputs = inputs
        for ibijector in self.init_functions:
            outputs = ibijector.transform(outputs)

        # unrescale data
        outputs = self.ms_reshape.inverse(outputs)
        return outputs
