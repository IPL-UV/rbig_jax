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


# @dataclass
# class MultiScaleRBIGBlockInit:
#     init_functions: List[dataclass]
#     filter_shape: Tuple[int, int]
#     image_shape: Tuple

#     def __post_init__(self, init_functions: List[dataclass], filter_shape: Tuple[int,int], image_shape: Tuple):
#         self.ms_reshape = init_scale_function(self.filter_shape, self.image_shape)
#         self.ms_reshape = init_scale_function(
#             self.filter_shape, self.image_shape, batch=False
#         )


# @dataclass
# class MultiScaleBijector:
#     bijectors: List[dataclass]
#     filter_shape: Tuple[int, int]
#     image_shape: Tuple
#     rescale_params: NamedTuple = field(default=None)

from distrax._src.utils import jittable
import abc

from flax import struct


@struct.dataclass
class MultiScaleBijector:
    bijectors: List[dataclass]
    squeeze: Callable = struct.field(pytree_node=False)
    unsqueeze: Callable = struct.field(pytree_node=False)
    # # class MultiScaleBijector(jittable.Jittable, metaclass=abc.ABCMeta):
    # def __init__(self, bijectors: List[dataclass], filter_shape, image_shape, batch):
    #     self.bijectors = bijectors
    #     squeeze_params = init_scale_function(
    #         filter=filter_shape, image_shape=image_shape, batch=batch
    #     )
    #     self.squeeze = squeeze_params.forward
    #     self.unsqueeze = squeeze_params.inverse

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

    # # @staticmethod
    # def squeeze(
    #     self, inputs: Array,
    # ):
    #     return rearrange(
    #         inputs,
    #         "B (Hn fh Wn fw C) -> (B Hn Wn) (fh fw C)",
    #         fh=self.rescale_params.fh,
    #         fw=self.rescale_params.fw,
    #         C=self.rescale_params.C,
    #         Wn=self.rescale_params.Wn,
    #         Hn=self.rescale_params.Hn,
    #     )

    # # @staticmethod
    # def unsqueeze(
    #     self, inputs: Array,
    # ):

    #     temp = rearrange(
    #         inputs,
    #         "(B Hn Wn) (fh fw C) -> B Hn Wn fh fw C",
    #         #             B=inputs.shape[0],
    #         C=self.rescale_params.C,
    #         Hn=self.rescale_params.Hn,
    #         Wn=self.rescale_params.Wn,
    #         fh=self.rescale_params.fh,
    #         fw=self.rescale_params.fw,
    #     )
    #     return rearrange(
    #         temp,
    #         "B Hn Wn fh fw C -> B (Hn fh Wn fw C)",
    #         #             B=inputs.shape[0],
    #         C=self.rescale_params.C,
    #         Hn=self.rescale_params.Hn,
    #         Wn=self.rescale_params.Wn,
    #         fh=self.rescale_params.fh,
    #         fw=self.rescale_params.fw,
    #     )


# @dataclass
# class MultiScaleRBIGBlockInit:
#     init_functions: List[dataclass]
#     filter_shape: Tuple[int, int]
#     image_shape: Tuple

#     def __post_init__(self):
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
