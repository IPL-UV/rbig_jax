from typing import Callable, Tuple
import jax
from jax.lax import conv_general_dilated
import numpy as np
from rbig_jax.transforms.parametric.householder import (
    householder_transform,
    householder_inverse_transform,
)
from chex import Array, dataclass
from jax.random import PRNGKey


@dataclass
class Conv1x1Params:
    weight: Array


def Conv1x1(n_channels: int) -> None:
    """1x1 Convolution
    This function will perform a 1x1 convolutions given an input
    array and a kernel. The output will be the same size as the input
    array. This will do the __call__ transformation (with the logabsdet)
    as well as the forward transformation (just the input) and the
    inverse transformation (just the input).

    Parameters
    ----------
    n_channels : int
        the input channels for the image to be used
    
    """

    def init_func(
        rng: PRNGKey, n_features: int, **kwargs
    ) -> Tuple[Conv1x1Params, Callable, Callable]:

        # initialize the householder rotation matrix
        V = jax.nn.initializers.orthogonal()(key=rng, shape=(n_channels, n_channels))

        init_params = Conv1x1Params(weight=V)

        def forward_func(
            params: dataclass, inputs: Array, **kwargs
        ) -> Tuple[Array, Array]:
            """Forward transformation with the logabsdet.
            This does the forward transformation and returns the transformed
            variable as well as the log absolute determinant. This is useful
            in a larger normalizing flow and for calculating the density.

            Parameters
            ----------
            inputs : Array
                input array of size (n_samples, n_channels, height, width)
            Returns
            -------
            outputs: Array
                output array of size (n_samples, n_channels, height, width)
            logabsdet : Array
                the log absolute determinant of size (n_samples,)
            """
            n_samples, height, width, _ = inputs.shape

            # forward transformation
            outputs = convolutions_1x1(x=inputs, kernel=params.weight)

            # calculate log determinant jacobian
            log_abs_det = np.ones(n_samples)
            log_abs_det = (
                log_abs_det * height * width * np.linalg.slogdet(params.weight)[1]
            )

            return outputs, log_abs_det

        def inverse_func(
            params: dataclass, inputs: Array, **kwargs
        ) -> Tuple[Array, Array]:
            """Inverse transformation
            Parameters
            ----------
            inputs : Array
                input array of size (n_samples, n_channels, height, width)
            Returns
            -------
            outputs: Array
                output array of size (n_samples, n_channels, height, width)
            logabsdet : Array
                the log absolute determinant of size (n_samples,)
            """
            n_samples, height, width, _ = inputs.shape

            outputs = convolutions_1x1(x=inputs, kernel=params.weight.T)

            # calculate log determinant jacobian
            log_abs_det = np.ones(n_samples)
            log_abs_det = (
                log_abs_det * height * width * np.linalg.slogdet(params.weight)[1]
            )

            return outputs, log_abs_det

        return init_params, forward_func, inverse_func

    return init_func


def Conv1x1Householder(n_reflections: int, n_channels: int):
    """1x1 Convolution w/ Orthogonal Constraint
    This class will perform a 1x1 convolutions given an input
    array and a kernel. The output will be the same size as the input
    array. This will do the __call__ transformation (with the logabsdet)
    as well as the forward transformation (just the input) and the
    inverse transformation (just the input).

    Parameters
    ----------
    n_channels : int
        the input channels for the image to be used

    """

    def init_func(
        rng: PRNGKey, n_features: int, **kwargs
    ) -> Tuple[Conv1x1Params, Callable, Callable]:

        # initialize the householder rotation matrix
        V = jax.nn.initializers.orthogonal()(key=rng, shape=(n_reflections, n_channels))

        init_params = Conv1x1Params(weight=V)

        weight_init = np.eye(n_channels)

        def forward_func(
            params: dataclass, inputs: Array, **kwargs
        ) -> Tuple[Array, Array]:
            """Forward transformation with the logabsdet.
            This does the forward transformation and returns the transformed
            variable as well as the log absolute determinant. This is useful
            in a larger normalizing flow and for calculating the density.

            Parameters
            ----------
            inputs : Array
                input array of size (n_samples, n_channels, height, width)
            Returns
            -------
            outputs: Array
                output array of size (n_samples, n_channels, height, width)
            logabsdet : Array
                the log absolute determinant of size (n_samples,)
            """
            # initialize logabsdet with batch dimension
            log_abs_det = np.zeros(inputs.shape[0])

            # weight matrix, householder product
            kernel = householder_transform(weight_init, params.weight)

            # forward transformation
            outputs = convolutions_1x1(x=inputs, kernel=kernel)

            return outputs, log_abs_det

        def inverse_func(
            params: dataclass, inputs: Array, **kwargs
        ) -> Tuple[Array, Array]:
            """Inverse transformation
            Parameters
            ----------
            inputs : Array
                input array of size (n_samples, n_channels, height, width)
            Returns
            -------
            outputs: Array
                output array of size (n_samples, n_channels, height, width)
            logabsdet : Array
                the log absolute determinant of size (n_samples,)
            """
            kernel = householder_inverse_transform(weight_init, params.weight)

            outputs = convolutions_1x1(x=inputs, kernel=kernel)

            log_abs_det = np.zeros(inputs.shape[0])

            return outputs, log_abs_det

        return init_params, forward_func, inverse_func

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
