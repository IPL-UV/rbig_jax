import collections
from functools import partial

import jax
import jax.numpy as np

RotParams = collections.namedtuple("Params", ["projection"])


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

