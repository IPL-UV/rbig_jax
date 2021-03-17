from typing import Tuple
import jax
from jax.lax import conv_general_dilated
import numpy as np
from objax.typing import JaxArray
import objax
from rbig_jax.transforms.parametric.householder import (
    householder_transform,
    householder_inverse_transform,
)


class Conv1x1(objax.Module):
    """1x1 Convolution
    This class will perform a 1x1 convolutions given an input
    array and a kernel. The output will be the same size as the input
    array. This will do the __call__ transformation (with the logabsdet)
    as well as the forward transformation (just the input) and the
    inverse transformation (just the input).

    Parameters
    ----------
    n_channels : int
        the input channels for the image to be used
    
    Attributes
    ----------
    weight : objax.TrainVar
        the kernel weight matrix of size (n_channels, n_channels)
    """

    def __init__(self, n_channels: int):
        self.weight = objax.TrainVar(objax.nn.init.orthogonal((n_channels, n_channels)))

    def __call__(self, inputs: JaxArray) -> Tuple[JaxArray, JaxArray]:
        """Forward transformation with the logabsdet.
        This does the forward transformation and returns the transformed
        variable as well as the log absolute determinant. This is useful
        in a larger normalizing flow and for calculating the density.

        Parameters
        ----------
        inputs : JaxArray
            input array of size (n_samples, n_channels, height, width)
        Returns
        -------
        outputs: JaxArray
            output array of size (n_samples, n_channels, height, width)
        logabsdet : Jaxarray
            the log absolute determinant of size (n_samples,)
        """
        n_samples, height, width, _ = inputs.shape

        # forward transformation
        outputs = convolutions_1x1(x=inputs, kernel=self.weight.value)

        # calculate log determinant jacobian
        log_abs_det = np.ones(n_samples)
        log_abs_det = (
            log_abs_det * height * width * np.linalg.slogdet(self.weight.value)[1]
        )

        return outputs, log_abs_det

    def transform(self, inputs: JaxArray) -> JaxArray:
        """Forward transformation w/o
        Parameters
        ----------
        inputs : JaxArray
            input array of size (n_samples, n_channels, height, width)
        Returns
        -------
        outputs: JaxArray
            output array of size (n_samples, n_channels, height, width)
        """
        outputs = convolutions_1x1(x=inputs, kernel=self.weight.value)
        return outputs

    def inverse(self, inputs: JaxArray) -> JaxArray:
        """Inverse transformation
        Parameters
        ----------
        inputs : JaxArray
            input array of size (n_samples, n_channels, height, width)
        Returns
        -------
        outputs: JaxArray
            output array of size (n_samples, n_channels, height, width)
        """
        outputs = convolutions_1x1(x=inputs, kernel=self.weight.value.T)
        return outputs


class Conv1x1Householder(objax.Module):
    """1x1 Convolution
    This class will perform a 1x1 convolutions given an input
    array and a kernel. The output will be the same size as the input
    array. This will do the __call__ transformation (with the logabsdet)
    as well as the forward transformation (just the input) and the
    inverse transformation (just the input).

    Parameters
    ----------
    n_channels : int
        the input channels for the image to be used
    
    Attributes
    ----------
    weight : objax.TrainVar
        the kernel weight matrix of size (n_channels, n_channels)
    """

    def __init__(self, n_channels: int, n_reflections: int = 10):

        self.weight = objax.TrainVar(
            objax.nn.init.orthogonal((n_reflections, n_channels))
        )
        self.weight_init = np.eye(n_channels)

    def __call__(self, inputs: JaxArray) -> Tuple[JaxArray, JaxArray]:
        """Forward transformation with the logabsdet.
        This does the forward transformation and returns the transformed
        variable as well as the log absolute determinant. This is useful
        in a larger normalizing flow and for calculating the density.

        Parameters
        ----------
        inputs : JaxArray
            input array of size (n_samples, n_channels, height, width)
        Returns
        -------
        outputs: JaxArray
            output array of size (n_samples, n_channels, height, width)
        logabsdet : Jaxarray
            the log absolute determinant of size (n_samples,)
        """
        # initialize logabsdet with batch dimension
        log_abs_det = np.zeros(inputs.shape[0])

        # weight matrix, householder product
        kernel = householder_transform(self.weight_init, self.weight.value)

        # forward transformation
        outputs = convolutions_1x1(x=inputs, kernel=kernel)

        return outputs, log_abs_det

    def transform(self, inputs: JaxArray) -> JaxArray:
        """Forward transformation w/o
        Parameters
        ----------
        inputs : JaxArray
            input array of size (n_samples, n_channels, height, width)
        Returns
        -------
        outputs: JaxArray
            output array of size (n_samples, n_channels, height, width)
        """
        # weight matrix, householder product
        kernel = householder_transform(self.weight_init, self.weight.value)

        outputs = convolutions_1x1(x=inputs, kernel=kernel)
        return outputs

    def inverse(self, inputs: JaxArray) -> JaxArray:
        """Inverse transformation
        Parameters
        ----------
        inputs : JaxArray
            input array of size (n_samples, n_channels, height, width)
        Returns
        -------
        outputs: JaxArray
            output array of size (n_samples, n_channels, height, width)
        """
        # weight matrix, householder inverse product
        kernel = householder_inverse_transform(self.weight_init, self.weight.value)

        outputs = convolutions_1x1(x=inputs, kernel=kernel)

        return outputs


def convolutions_1x1(x: JaxArray, kernel: JaxArray) -> JaxArray:
    """1x1 Convolution function
    This function will perform a 1x1 convolutions given an input
    array and a kernel. The output will be the same size as the input
    array.

    Parameters
    ----------
    x: JaxArray
        input array for convolutions of shape
        (n_samples, height, width, n_channels)
    kernel: JaxArray
        input kernel of shape (n_channels, n_channels)
    Returns
    -------
    output: JaxArray
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
