from dataclasses import field
from typing import Callable, Iterable, List, NamedTuple, Optional, Tuple

import jax.numpy as jnp
from chex import Array, dataclass

from rbig_jax.transforms.base import Bijector, BijectorChain
from rbig_jax.transforms.reshape import init_scale_function
from einops import rearrange


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


from distrax._src.utils import jittable
import abc

from flax import struct


@struct.dataclass
class MultiScaleBijector:
    bijectors: List[dataclass]
    squeeze: Callable = struct.field(pytree_node=False)
    unsqueeze: Callable = struct.field(pytree_node=False)

    def forward_and_log_det(self, inputs: Array) -> Tuple[Array, Array]:

        # rescale data
        inputs = self.squeeze(inputs)

        # bijector chain transform
        outputs, logabsdet = self.bijectors.forward_and_log_det(inputs)

        # unrescale data
        outputs = self.unsqueeze(outputs)
        logabsdet = self.unsqueeze(logabsdet)

        return outputs, logabsdet

    def inverse_and_log_det(self, inputs: Array) -> Tuple[Array, Array]:

        # rescale data
        inputs = self.squeeze(inputs)

        # bijector chain transform
        outputs, logabsdet = self.bijectors.inverse_and_log_det(inputs)

        # unrescale data
        outputs = self.unsqueeze(outputs)
        logabsdet = self.unsqueeze(logabsdet)

        return outputs, logabsdet

    def forward(self, inputs: Array) -> Array:

        # rescale data
        inputs = self.squeeze(inputs)
        # bijector chain transform
        outputs = self.bijectors.forward(inputs)

        # unrescale data
        outputs = self.unsqueeze(outputs)

        return outputs

    def inverse(self, inputs: Array) -> Array:
        # rescale data
        inputs = self.squeeze(inputs)

        # bijector chain transform
        outputs = self.bijectors.inverse(inputs)

        # unrescale data
        outputs = self.unsqueeze(outputs)
        return outputs

    def forward_log_det_jacobian(self, inputs: Array) -> Array:
        # rescale data
        inputs = self.squeeze(inputs)

        # bijector chain transform
        logabsdet = self.bijectors.forward_log_det_jacobian(inputs)

        # unrescale data
        logabsdet = self.unsqueeze(logabsdet)
        return logabsdet

    def inverse_log_det_jacobian(self, inputs: Array) -> Tuple[Array, Array]:

        # rescale data
        inputs = self.squeeze(inputs)

        # bijector chain transform
        logabsdet = self.bijectors.inverse_log_det_jacobian(inputs)

        # unrescale data
        logabsdet = self.unsqueeze(logabsdet)

        return logabsdet


@struct.dataclass
class MultiScaleRBIGBlockInit:
    init_functions: List[dataclass]
    filter_shape: Tuple[int, int] = struct.field(pytree_node=False)
    image_shape: Tuple = struct.field(pytree_node=False)

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
        bijectors = MultiScaleBijector(
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


def init_rescale_params(filter_shape, image_shape):

    fh, fw = filter_shape
    H = image_shape.H
    W = image_shape.W
    C = image_shape.C
    # # do some checks!
    # assert H / fh !% 0
    # assert W / fw !% 0

    Hn = H // fh
    Wn = W // fw

    rescale_params = RescaleParams(fh=fh, fw=fw, H=H, W=W, C=C, Hn=Hn, Wn=Wn,)

    return rescale_params

