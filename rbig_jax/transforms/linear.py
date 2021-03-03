import collections
from functools import partial
from typing import Tuple

import jax
import jax.numpy as np
import objax
from objax.typing import JaxArray
from rbig_jax.transforms.base import Transform

RotParams = collections.namedtuple("Params", ["projection"])


class HouseHolder(Transform):
    def __init__(self, n_features: int, n_reflections: int, generator=None) -> None:
        self.n_features = n_features
        self.n_reflections = n_reflections

        if generator is None:
            generator = objax.random.Generator(123)

        self.V = objax.TrainVar(
            objax.random.normal((n_reflections, n_features), generator=generator)
        )

    def __call__(self, inputs: JaxArray) -> Tuple[JaxArray, JaxArray]:

        # forward transformation with batch dimension
        outputs = jax.vmap(householder_transform, in_axes=(0, None))(
            inputs, self.V.value
        )

        # log abs det, all zeros
        logabsdet = np.zeros(inputs.shape[0])

        return outputs, logabsdet

    def transform(self, inputs: JaxArray) -> JaxArray:

        outputs = jax.vmap(householder_transform, in_axes=(0, None))(
            inputs, self.V.value
        )

        return outputs

    def inverse(self, inputs: JaxArray) -> JaxArray:

        outputs = jax.vmap(householder_inverse_transform, in_axes=(0, None))(
            inputs, self.V.value
        )
        return outputs


def householder_product(inputs: JaxArray, q_vector: JaxArray) -> JaxArray:
    """
    Args:
        inputs (JaxArray) : inputs for the householder product
        (D,)
        q_vector (JaxArray): vector to be multiplied
        (D,)
    
    Returns:
        outputs (JaxArray) : outputs after the householder product
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


def _householder_product_body(carry, inputs):
    """Helper function for the scan product"""
    return householder_product(carry, inputs), 0


def householder_transform(inputs: JaxArray, vectors: JaxArray) -> JaxArray:
    """
    Args:
        inputs (JaxArray) : inputs for the householder product
            (D,)
        q_vector (JaxArray): vectors to be multiplied in the 
            (D,K)
    
    Returns:
        outputs (JaxArray) : outputs after the householder product
            (D,)
    """
    return jax.lax.scan(_householder_product_body, inputs, vectors)[0]


def householder_inverse_transform(inputs: JaxArray, vectors: JaxArray) -> JaxArray:
    """
    Args:
        inputs (JaxArray) : inputs for the householder product
            (D,)
        q_vector (JaxArray): vectors to be multiplied in the reverse order
            (D,K)
    
    Returns:
        outputs (JaxArray) : outputs after the householder product
            (D,)
    """
    return jax.lax.scan(_householder_product_body, inputs, vectors[::-1])[0]


def get_pca_params(X):

    R = compute_projection(X)

    return np.dot(X, X), RotParams(projection=R)


def svd_transform(X: np.ndarray) -> np.ndarray:
    """Compute PCA projection matrix
    Using SVD, this computes the PCA components for
    a dataset X and computes the projection matrix
    needed to do the PCA decomposition.

    Parameters
    ----------
    X : np.ndarray, (n_samples, n_features)
        the data to calculate to PCA projection matrix
    
    Returns
    -------
    VT : np.ndarray, (n_features, n_features)
        the projection matrix (V.T) for the PCA decomposition

    Notes
    -----
    Can find the original implementation here:
    https://bit.ly/2EBDV9o
    """

    # center the data
    X = X - np.mean(X, axis=0)

    # Compute SVD
    _, _, VT = np.linalg.svd(X, full_matrices=False, compute_uv=True)

    return X @ VT.T


def svd_tranform_gradient(X: np.ndarray) -> np.ndarray:
    """Log Determinant Jacobian of a linear transform"""
    return np.zeros_like(X)


def compute_projection(X: np.ndarray) -> np.ndarray:
    """Compute PCA projection matrix
    Using SVD, this computes the PCA components for
    a dataset X and computes the projection matrix
    needed to do the PCA decomposition.

    Parameters
    ----------
    X : np.ndarray, (n_samples, n_features)
        the data to calculate to PCA projection matrix
    
    Returns
    -------
    VT : np.ndarray, (n_features, n_features)
        the projection matrix (V.T) for the PCA decomposition

    Notes
    -----
    Can find the original implementation here:
    https://bit.ly/2EBDV9o
    """

    # center the data
    X = X - np.mean(X, axis=0)

    # Compute SVD
    _, _, VT = np.linalg.svd(X, full_matrices=False, compute_uv=True)

    return X @ VT.T


def compute_projection_v1(X: np.ndarray) -> np.ndarray:
    """Compute PCA projection matrix
    Using SVD, this computes the PCA components for
    a dataset X and computes the projection matrix
    needed to do the PCA decomposition.

    Parameters
    ----------
    X : np.ndarray, (n_samples, n_features)
        the data to calculate to PCA projection matrix
    
    Returns
    -------
    VT : np.ndarray, (n_features, n_features)
        the projection matrix (V.T) for the PCA decomposition

    Notes
    -----
    Can find the original implementation here:
    https://bit.ly/2EBDV9o
    """

    # center the data
    X = X - np.mean(X, axis=0)

    # Compute SVD
    _, _, VT = np.linalg.svd(X, full_matrices=False, compute_uv=True)

    return VT.T


def init_pca_params(X):

    # compute projection matrix
    R = compute_projection(X)

    return (
        np.dot(X, R),
        RotParams(R),
        partial(forward_transform, R=R),
        partial(inverse_transform, R=R),
    )


def forward_transform(X, R):
    return np.dot(X, R), np.zeros(X.shape)


def inverse_transform(X, R):
    return np.dot(X, R.T)


def rot_forward_transform(X, params):
    return np.dot(X, params.R)


def rot_inverse_transform(X, params):
    return np.dot(X, params.R.T)


def rot_gradient_transform(X, params):
    return np.ones_like(X)
