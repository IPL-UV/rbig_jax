import collections
from rbig_jax.transforms.rotation import compute_projection
from typing import Callable, Optional, Tuple

import jax
import jax.numpy as jnp
from chex import Array, dataclass
from jax.random import PRNGKey

from rbig_jax.transforms.base import Bijector, InitLayersFunctions

# RotParams = collections.namedtuple("Params", ["projection"])


@dataclass
class HouseHolder(Bijector):
    V: Array

    def forward_and_log_det(self, inputs: Array) -> Tuple[Array, Array]:
        # forward transformation with batch dimension
        outputs = jax.vmap(householder_transform, in_axes=(0, None))(inputs, self.V)

        # log abs det, all zeros
        logabsdet = jnp.zeros_like(inputs)

        return outputs, logabsdet

    def inverse_and_log_det(self, inputs: Array) -> Tuple[Array, Array]:
        outputs = jax.vmap(householder_inverse_transform, in_axes=(0, None))(
            inputs, self.V
        )

        # log abs det, all zeros
        logabsdet = jnp.zeros_like(inputs)

        return outputs, logabsdet


def InitHouseHolder(n_reflections: int, method: str = "random") -> Callable:
    """Performs the householder transformation.

    This is a useful method to parameterize an orthogonal matrix.
    
    Parameters
    ----------
    n_features : int
        the number of features of the data
    n_reflections: int
        the number of householder reflections
    """

    def init_bijector(
        inputs: Array, rng: PRNGKey = None, n_features: int = None, **kwargs
    ) -> HouseHolder:

        V = init_householder_weights(
            rng=rng,
            n_features=n_features,
            n_reflections=n_reflections,
            method=method,
            X=inputs,
        )
        bijector = HouseHolder(V=V)

        return bijector

    def bijector_and_transform(
        inputs: Array, rng: PRNGKey = None, n_features: int = None, **kwargs
    ) -> Tuple[Array, HouseHolder]:

        # init bijector
        bijector = init_bijector(
            rng=rng, n_features=n_features, inputs=inputs, **kwargs,
        )

        # forward transform
        outputs = bijector.forward(inputs=inputs)

        return outputs, bijector

    def transform(
        inputs: Array, rng: PRNGKey = None, n_features: int = None, **kwargs
    ) -> Array:

        # init bijector
        outputs = init_bijector(
            rng=rng, n_features=n_features, inputs=inputs, **kwargs,
        ).forward(inputs=inputs)

        return outputs

    return InitLayersFunctions(
        bijector=init_bijector,
        bijector_and_transform=bijector_and_transform,
        transform=transform,
        params=None,
        params_and_transform=None,
    )


def init_householder_weights(
    rng, n_reflections, n_features, method: str = "random", X=None
):
    if method == "random":
        # initialize mixture
        V = jax.nn.initializers.orthogonal()(key=rng, shape=(n_reflections, n_features))

    elif method == "pca":

        V = compute_projection(X)
    else:
        raise ValueError(f"Unrecognized init method: {method}")
    return V


def householder_product(inputs: Array, q_vector: Array) -> Array:
    """
    Args:
        inputs (Array) : inputs for the householder product
        (D,)
        q_vector (Array): vector to be multiplied
        (D,)
    
    Returns:
        outputs (Array) : outputs after the householder product
    """
    # norm for q_vector
    squared_norm = jnp.sum(q_vector ** 2)
    # inner product
    temp = jnp.dot(inputs, q_vector)
    # outer product
    temp = jnp.outer(temp, (2.0 / squared_norm) * q_vector).squeeze()
    # update
    output = inputs - temp
    return output


def _householder_product_body(carry: Array, inputs: Array) -> Tuple[Array, int]:
    """Helper function for the scan product"""
    return householder_product(carry, inputs), 0


def householder_transform(inputs: Array, vectors: Array) -> Array:
    """
    Args:
        inputs (Array) : inputs for the householder product
            (D,)
        q_vector (Array): vectors to be multiplied in the 
            (D,K)
    
    Returns:
        outputs (Array) : outputs after the householder product
            (D,)
    """
    return jax.lax.scan(_householder_product_body, inputs, vectors)[0]


def householder_inverse_transform(inputs: Array, vectors: Array) -> Array:
    """
    Args:
        inputs (Array) : inputs for the householder product
            (D,)
        q_vector (Array): vectors to be multiplied in the reverse order
            (D,K)
    
    Returns:
        outputs (Array) : outputs after the householder product
            (D,)
    """
    return jax.lax.scan(_householder_product_body, inputs, vectors[::-1])[0]
