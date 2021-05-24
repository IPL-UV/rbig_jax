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
    householder_inverse_transform, householder_transform)


@struct.dataclass
class SVDTransform(Bijector):
    V: Array
    U: Array
    S: Array
    eps: Array = struct.field(pytree_node=False)

    def forward_and_log_det(self, inputs: Array) -> Tuple[Array, Array]:
        # calculate U matrix
        outputs = jax.vmap(householder_transform, in_axes=(0, None))(inputs, self.V)

        # multiply by diagonal, S
        diagonal = _transform_diagonal(self.S, self.eps)
        outputs *= diagonal

        # multiply by LHS, V
        outputs = jax.vmap(householder_transform, in_axes=(0, None))(outputs, self.U)

        # initialize logabsdet with batch dimension
        log_abs_det = diagonal.sum() * np.ones(inputs.shape[0])

        return outputs, log_abs_det

    def inverse_and_log_det(self, inputs: Array) -> Tuple[Array, Array]:
        # multiply by U
        outputs = jax.vmap(householder_inverse_transform, in_axes=(0, None))(
            inputs, self.U
        )

        # divide by diagonal, S
        diagonal = _transform_diagonal(self.S, self.eps)
        outputs /= diagonal

        # multiply by U
        outputs = jax.vmap(householder_inverse_transform, in_axes=(0, None))(
            outputs, self.V
        )

        # initialize logabsdet with batch dimension
        log_abs_det = -diagonal.sum() * np.ones(inputs.shape[0])

        return outputs, log_abs_det

    def forward(self, inputs: Array) -> Tuple[Array, Array]:
        # calculate U matrix
        outputs = jax.vmap(householder_transform, in_axes=(0, None))(inputs, self.V)

        # multiply by diagonal, S
        diagonal = _transform_diagonal(self.S, self.eps)
        outputs *= diagonal

        # multiply by LHS, V
        outputs = jax.vmap(householder_transform, in_axes=(0, None))(outputs, self.U)

        return outputs

    def inverse(self, inputs: Array) -> Tuple[Array, Array]:
        outputs = jax.vmap(householder_inverse_transform, in_axes=(0, None))(
            inputs, self.V
        )
        return outputs

    def forward_log_det_jacobian(self, inputs: Array) -> Tuple[Array, Array]:
        # divide by diagonal, S
        diagonal = _transform_diagonal(self.S, self.eps)

        # initialize logabsdet with batch dimension
        log_abs_det = diagonal.sum() * np.ones(inputs.shape[0])

        return log_abs_det

    def inverse_log_det_jacobian(self, inputs: Array) -> Tuple[Array, Array]:
        # divide by diagonal, S
        diagonal = _transform_diagonal(self.S, self.eps)
        # initialize logabsdet with batch dimension
        log_abs_det = -diagonal.sum() * np.ones(inputs.shape[0])

        return log_abs_det


def InitSVDTransform(
    n_reflections: int, method: str = "random", eps: float = 1e-5
) -> Callable:
    """Performs the householder transformation.

    This is a useful method to parameterize an orthogonal matrix.
    
    Parameters
    ----------
    n_features : int
        the number of features of the data
    n_reflections: int
        the number of householder reflections
    """

    def bijector(
        inputs: Array, n_features: int, rng: PRNGKey = None, **kwargs
    ) -> SVDTransform:

        # initialize weight matrix
        U, S, V = init_svd_weights(
            rng=rng,
            n_features=n_features,
            n_reflections=n_reflections,
            method=method,
            X=inputs,
        )

        # initialize bijector
        bijector = SVDTransform(U=U, S=S, V=V, eps=eps)

        return bijector

    def transform_and_bijector(
        inputs: Array, n_features: int, rng: PRNGKey = None, **kwargs
    ) -> Tuple[Array, SVDTransform]:

        # initialize weight matrix
        U, S, V = init_svd_weights(
            rng=rng,
            n_features=n_features,
            n_reflections=n_reflections,
            method=method,
            X=inputs,
        )

        # initialize bijector
        bijector = SVDTransform(U=U, S=S, V=V, eps=eps)

        # forward transform
        outputs = bijector.forward(inputs=inputs)

        return outputs, bijector

    def transform(
        inputs: Array, n_features: int, rng: PRNGKey = None, **kwargs
    ) -> Array:

        # initialize weight matrix
        U, S, V = init_svd_weights(
            rng=rng,
            n_features=n_features,
            n_reflections=n_reflections,
            method=method,
            X=inputs,
        )

        # initialize bijector
        bijector = SVDTransform(U=U, S=S, V=V, eps=eps)

        # forward transform
        outputs = bijector.forward(inputs=inputs)

        return outputs

    def transform_gradient_bijector(
        inputs: Array, n_features: int, rng: PRNGKey = None, **kwargs
    ) -> Tuple[Array, SVDTransform]:

        # initialize weight matrix
        U, S, V = init_svd_weights(
            rng=rng,
            n_features=n_features,
            n_reflections=n_reflections,
            method=method,
            X=inputs,
        )

        # initialize bijector
        bijector = SVDTransform(U=U, S=S, V=V, eps=eps)

        # forward transform
        outputs, logabsdet = bijector.forward_and_log_det(inputs=inputs)

        return outputs, logabsdet, bijector

    return InitLayersFunctions(
        transform=transform,
        bijector=bijector,
        transform_and_bijector=transform_and_bijector,
        transform_gradient_bijector=transform_gradient_bijector,
    )


def init_svd_weights(
    rng,
    n_features,
    n_reflections,
    method: str = "random",
    X=None,
    identity_init: bool = True,
) -> jnp.ndarray:

    if method == "random":

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
            constant = _transform_diagonal(1.0, 1e-5)
            S = constant * jnp.ones(shape=(n_features,))
        else:
            stdv = 1.0 / jnp.sqrt(n_features)
            S = jax.random.uniform(
                key=s_rng, shape=(n_features,), minval=-stdv, maxval=stdv
            )

    elif method == "pca":

        # center the data
        X = X - jnp.mean(X, axis=0)

        U, S, V = jnp.linalg.svd(X, full_matrices=True, compute_uv=True)

    else:
        raise ValueError(f"Unrecognized init method: {method}")
    return U, S, V


def _transform_diagonal(S: Array, eps: float) -> Array:

    return eps + jax.nn.softplus(S)
