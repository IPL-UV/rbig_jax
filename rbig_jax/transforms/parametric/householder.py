import collections
from functools import partial
from typing import Tuple, Callable

import jax
import jax.numpy as np
from jax.random import PRNGKey
from chex import Array, dataclass

RotParams = collections.namedtuple("Params", ["projection"])


@dataclass
class HouseHolderParams:
    V: Array


def HouseHolder(n_reflections: int) -> Callable:
    """Performs the householder transformation.

    This is a useful method to parameterize an orthogonal matrix.
    
    Parameters
    ----------
    n_features : int
        the number of features of the data
    n_reflections: int
        the number of householder reflections
    """

    def init_func(
        rng: PRNGKey, n_features: int, **kwargs
    ) -> Tuple[HouseHolderParams, Callable, Callable]:

        # initialize the householder rotation matrix
        V = jax.nn.initializers.orthogonal()(key=rng, shape=(n_reflections, n_features))

        init_params = HouseHolderParams(V=V)

        def forward_func(params, inputs: Array, **kwargs) -> Tuple[Array, Array]:

            # forward transformation with batch dimension
            outputs = jax.vmap(householder_transform, in_axes=(0, None))(
                inputs, params.V
            )

            # log abs det, all zeros
            logabsdet = np.zeros(inputs.shape[0])

            return outputs, logabsdet

        def inverse_func(params, inputs: Array, **kwargs) -> Tuple[Array, Array]:

            outputs = jax.vmap(householder_inverse_transform, in_axes=(0, None))(
                inputs, params.V
            )

            # log abs det, all zeros
            logabsdet = np.zeros(inputs.shape[0])

            return outputs, logabsdet

        return init_params, forward_func, inverse_func

    return init_func


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
    squared_norm = np.sum(q_vector ** 2)
    # inner product
    temp = np.dot(inputs, q_vector)
    # outer product
    temp = np.outer(temp, (2.0 / squared_norm) * q_vector).squeeze()
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
