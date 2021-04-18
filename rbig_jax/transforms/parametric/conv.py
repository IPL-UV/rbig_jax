from typing import Callable, Tuple

import jax
import numpy as np
from chex import Array, dataclass
from jax.lax import conv_general_dilated
from jax.random import PRNGKey
from rbig_jax.transforms.base import Bijector

from rbig_jax.transforms.parametric.householder import (
    householder_inverse_transform,
    householder_transform,
)


@dataclass
class Conv1x1HouseHolder(Bijector):
    weight: Array

    def forward_and_log_det(self, inputs: Array, **kwargs) -> Tuple[Array, Array]:
        """Forward transformation with the logabsdet.
        This does the forward transformation and returns the transformed
        variable as well as the log absolute determinant. This is useful
        in a larger normalizing flow and for calculating the density.

        Parameters
        ----------
        inputs : Array
            input array of size (n_samples, height, width, n_channels)
        Returns
        -------
        outputs: Array
            output array of size (n_samples, n_channels, height, width)
        logabsdet : Array
            the log absolute determinant of size (n_samples,)
        """
        *_, C = inputs.shape
        weight_init = np.eye(C)

        # weight matrix, householder product
        kernel = householder_transform(weight_init, self.weight)

        # forward transformation
        outputs = convolutions_1x1(x=inputs, kernel=kernel)

        # initialize logabsdet with batch dimension
        log_abs_det = np.zeros_like(inputs)

        return outputs, log_abs_det

    def inverse_and_log_det(self, inputs: Array, **kwargs) -> Tuple[Array, Array]:
        """Inverse transformation
        Parameters
        ----------
        inputs : Array
            input array of size (n_samples, height, width, n_channels)
        Returns
        -------
        outputs: Array
            output array of size (n_samples, height, width, n_channels)
        logabsdet : Array
            the log absolute determinant of size (n_samples,)
        """
        *_, C = inputs.shape
        weight_init = np.eye(C)
        kernel = householder_inverse_transform(weight_init, self.weight)

        outputs = convolutions_1x1(x=inputs, kernel=kernel)

        # initialize logabsdet with batch dimension
        log_abs_det = np.zeros_like(inputs)

        return outputs, log_abs_det


def InitConv1x1Householder(n_reflections: int):
    """1x1 Convolution w/ Orthogonal Constraint
    This class will perform a 1x1 convolutions given an input
    array and a kernel. The output will be the same size as the input
    array. This will do the __call__ transformation (with the logabsdet)
    as well as the forward transformation (just the input) and the
    inverse transformation (just the input).

    Parameters
    ----------
    n_reflections : int
        the number of householder reflections to use

    """

    def init_func(rng: PRNGKey, shape: Tuple, **kwargs) -> Conv1x1HouseHolder:

        # extract shape
        *_, C = shape

        # initialize the householder rotation matrix
        V = jax.nn.initializers.orthogonal()(key=rng, shape=(n_reflections, C))

        # create bijector
        return Conv1x1HouseHolder(weight=V)

    return init_func


def convolutions_1x1(x: Array, kernel: Array) -> Array:
    """1x1 Convolution function
    This function will perform a 1x1 convolutions given an input
    array and a kernel. The output will be the same size as the input
    array.

    Parameters
    ----------
    x: Array
        input array for convolutions of shape
        (n_samples, height, width, n_channels)
    kernel: Array
        input kernel of shape (n_channels, n_channels)
    Returns
    -------
    output: Array
        the output array after the convolutions of shape
        (n_samples, height, width, n_channels)

    References
    ----------
    * https://hackerstreak.com/1x1-convolution/
    * https://iamaaditya.github.io/2016/03/one-by-one-convolution/
    * https://sebastianraschka.com/faq/docs/fc-to-conv.html
    """
    return conv_general_dilated(
        lhs=x,
        rhs=kernel[..., None, None],
        window_strides=(1, 1),
        padding="SAME",
        lhs_dilation=(1, 1),
        rhs_dilation=(1, 1),
        dimension_numbers=("NHWC", "IOHW", "NHWC"),
    )


# @dataclass
# class Conv1x1Params:
#     weight: Array


# @dataclass
# class Conv1x1(Bijector):
#     weight: Array

#     def forward_and_log_det(self, inputs: Array, **kwargs) -> Tuple[Array, Array]:
#         """Forward transformation with the logabsdet.
#         This does the forward transformation and returns the transformed
#         variable as well as the log absolute determinant. This is useful
#         in a larger normalizing flow and for calculating the density.

#         Parameters
#         ----------
#         inputs : Array
#             input array of size (n_samples, height, width, n_channels)
#         Returns
#         -------
#         outputs: Array
#             output array of size (n_samples, n_channels, height, width)
#         logabsdet : Array
#             the log absolute determinant of size (n_samples,)
#         """
#         *_, C = inputs.shape
#         weight_init = np.eye(C)

#         # weight matrix, householder product
#         kernel = householder_transform(weight_init, self.weight)

#         # forward transformation
#         outputs = convolutions_1x1(x=inputs, kernel=kernel)

#         # initialize logabsdet with batch dimension
#         log_abs_det = np.zeros_like(inputs)

#         return outputs, log_abs_det

#     def inverse_and_log_det(self, inputs: Array, **kwargs) -> Tuple[Array, Array]:
#         """Inverse transformation
#         Parameters
#         ----------
#         inputs : Array
#             input array of size (n_samples, height, width, n_channels)
#         Returns
#         -------
#         outputs: Array
#             output array of size (n_samples, height, width, n_channels)
#         logabsdet : Array
#             the log absolute determinant of size (n_samples,)
#         """
#         *_, C = inputs.shape
#         weight_init = np.eye(C)
#         kernel = householder_inverse_transform(weight_init, self.weight)

#         outputs = convolutions_1x1(x=inputs, kernel=kernel)

#         # initialize logabsdet with batch dimension
#         log_abs_det = np.zeros_like(inputs)

#         return outputs, log_abs_det

# def Conv1x1(n_channels: int) -> None:
#     """1x1 Convolution
#     This function will perform a 1x1 convolutions given an input
#     array and a kernel. The output will be the same size as the input
#     array. This will do the __call__ transformation (with the logabsdet)
#     as well as the forward transformation (just the input) and the
#     inverse transformation (just the input).

#     Parameters
#     ----------
#     n_channels : int
#         the input channels for the image to be used

#     """

#     def init_func(rng: PRNGKey, shape: int, **kwargs) -> Conv1x1:
#         *_, C = shape
#         # initialize the householder rotation matrix
#         V = jax.nn.initializers.orthogonal()(key=rng, shape=(C, C))

#         return Conv1x1(weight=V)

#         def forward_func(
#             params: dataclass, inputs: Array, **kwargs
#         ) -> Tuple[Array, Array]:
#             """Forward transformation with the logabsdet.
#             This does the forward transformation and returns the transformed
#             variable as well as the log absolute determinant. This is useful
#             in a larger normalizing flow and for calculating the density.

#             Parameters
#             ----------
#             inputs : Array
#                 input array of size (n_samples, n_channels, height, width)
#             Returns
#             -------
#             outputs: Array
#                 output array of size (n_samples, n_channels, height, width)
#             logabsdet : Array
#                 the log absolute determinant of size (n_samples,)
#             """
#             n_samples, height, width, _ = inputs.shape

#             # forward transformation
#             outputs = convolutions_1x1(x=inputs, kernel=params.weight)

#             # calculate log determinant jacobian
#             log_abs_det = np.ones(n_samples)
#             log_abs_det = (
#                 log_abs_det * height * width * np.linalg.slogdet(params.weight)[1]
#             )

#             return outputs, log_abs_det

#         def inverse_func(
#             params: dataclass, inputs: Array, **kwargs
#         ) -> Tuple[Array, Array]:
#             """Inverse transformation
#             Parameters
#             ----------
#             inputs : Array
#                 input array of size (n_samples, n_channels, height, width)
#             Returns
#             -------
#             outputs: Array
#                 output array of size (n_samples, n_channels, height, width)
#             logabsdet : Array
#                 the log absolute determinant of size (n_samples,)
#             """
#             n_samples, height, width, _ = inputs.shape

#             outputs = convolutions_1x1(x=inputs, kernel=params.weight.T)

#             # calculate log determinant jacobian
#             log_abs_det = np.ones(n_samples)
#             log_abs_det = (
#                 log_abs_det * height * width * np.linalg.slogdet(params.weight)[1]
#             )

#             return outputs, log_abs_det

#         return init_params, forward_func, inverse_func

#     return init_func

