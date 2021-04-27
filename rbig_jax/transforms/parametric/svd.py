from typing import Callable, Tuple

import jax
import jax.numpy as jnp
import numpy as np
from chex import Array, dataclass
from jax.lax import conv_general_dilated
from jax.random import PRNGKey

from rbig_jax.transforms.parametric.householder import (
    householder_inverse_transform, householder_transform)


@dataclass
class SVDParams:
    V: Array
    U: Array
    S: Array


def SVD(n_reflections: int, eps: float = 1e-3, identity_init: bool = True):
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
    ) -> Tuple[SVDParams, Callable, Callable]:

        assert n_reflections % 2 == 0

        u_rng, s_rng, v_rng = jax.random.split(rng, 3)

        # initialize the householder rotation matrix, U
        U = jax.nn.initializers.orthogonal()(
            key=u_rng, shape=(n_reflections, n_features)
        )
        # initialize the householder rotation matrix, V^T
        V = jax.nn.initializers.orthogonal()(
            key=v_rng, shape=(n_reflections, n_features)
        )
        # initialize the diagonal matrix, S
        if identity_init:
            constant = jnp.log(jnp.exp(1 - eps) - 1)
            S = constant * jnp.ones(shape=(n_features,))
        else:
            stdv = 1.0 / jnp.sqrt(n_features)
            S = jax.random.uniform(
                key=s_rng, shape=(n_features,), minval=-stdv, maxval=stdv
            )

        init_params = SVDParams(V=V, U=U, S=S)

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
            # calculate U matrix
            outputs = jax.vmap(householder_transform, in_axes=(0, None))(
                inputs, params.V
            )

            # multiply by diagonal, S
            diagonal = _transform_diagonal(params.S, eps)
            outputs *= diagonal

            # multiply by LHS, V
            outputs = jax.vmap(householder_transform, in_axes=(0, None))(
                outputs, params.U
            )

            # initialize logabsdet with batch dimension
            log_abs_det = diagonal.sum() * np.ones(inputs.shape[0])

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
            # multiply by U
            outputs = jax.vmap(householder_inverse_transform, in_axes=(0, None))(
                inputs, params.U
            )

            # divide by diagonal, S
            diagonal = _transform_diagonal(params.S, eps)
            outputs /= diagonal

            # multiply by U
            outputs = jax.vmap(householder_inverse_transform, in_axes=(0, None))(
                outputs, params.V
            )

            # initialize logabsdet with batch dimension
            log_abs_det = -diagonal.sum() * np.ones(inputs.shape[0])

            return outputs, log_abs_det

        return init_params, forward_func, inverse_func

    return init_func


def _transform_diagonal(S: Array, eps: float) -> Array:

    return eps + jax.nn.softplus(S)
