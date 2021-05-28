from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from chex import Array, dataclass
from flax import struct
from jax.lax import conv_general_dilated
from jax.random import PRNGKey
from rbig_jax.transforms.base import Bijector, InitLayersFunctions
from rbig_jax.transforms.parametric.householder import (
    householder_inverse_transform,
    householder_transform,
    init_householder_weights,
)


@struct.dataclass
class Conv1x1Householder(Bijector):
    V: Array

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
        weight_init = jnp.eye(C)

        # weight matrix, householder product
        kernel = householder_transform(weight_init, self.V)

        # forward transformation
        outputs = convolutions_1x1(x=inputs, kernel=kernel)

        # initialize logabsdet with batch dimension
        log_abs_det = jnp.zeros_like(inputs)

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
        weight_init = jnp.eye(C)
        kernel = householder_inverse_transform(weight_init, self.V)

        outputs = convolutions_1x1(x=inputs, kernel=kernel)

        # initialize logabsdet with batch dimension
        log_abs_det = jnp.zeros_like(inputs)

        return outputs, log_abs_det


def InitConv1x1Householder(n_reflections: int, method: str = "random") -> Callable:
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

    def bijector(
        inputs: Array, n_channels: int, rng: PRNGKey = None, **kwargs
    ) -> Conv1x1Householder:

        # initialize weight matrix
        V = init_householder_weights(
            rng=rng,
            n_features=n_channels,
            n_reflections=n_reflections,
            method=method,
            X=inputs,
        )

        # initialize bijector
        bijector = Conv1x1Householder(V=V)

        return bijector

    def transform_and_bijector(
        inputs: Array, n_channels: int, rng: PRNGKey = None, **kwargs
    ) -> Tuple[Array, Conv1x1Householder]:

        # initialize weight matrix
        V = init_householder_weights(
            rng=rng,
            n_features=n_channels,
            n_reflections=n_reflections,
            method=method,
            X=inputs,
        )

        # initialize bijector
        bijector = Conv1x1Householder(V=V)

        # forward transform
        outputs = bijector.forward(inputs=inputs)

        return outputs, bijector

    def transform(
        inputs: Array, n_channels: int, rng: PRNGKey = None, **kwargs
    ) -> Array:

        # initialize weight matrix
        V = init_householder_weights(
            rng=rng,
            n_features=n_channels,
            n_reflections=n_reflections,
            method=method,
            X=inputs,
        )

        # initialize bijector
        bijector = Conv1x1Householder(V=V)

        # forward transform
        outputs = bijector.forward(inputs=inputs)

        return outputs

    def transform_gradient_bijector(
        inputs: Array, n_channels: int, rng: PRNGKey = None, **kwargs
    ) -> Tuple[Array, Conv1x1Householder]:

        # initialize weight matrix
        V = init_householder_weights(
            rng=rng,
            n_features=n_channels,
            n_reflections=n_reflections,
            method=method,
            X=inputs,
        )

        # initialize bijector
        bijector = Conv1x1Householder(V=V)

        # forward transform
        outputs, logabsdet = bijector.forward_and_log_det(inputs=inputs)

        return outputs, logabsdet, bijector

    return InitLayersFunctions(
        transform=transform,
        bijector=bijector,
        transform_and_bijector=transform_and_bijector,
        transform_gradient_bijector=transform_gradient_bijector,
    )


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
