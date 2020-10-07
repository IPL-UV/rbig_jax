import collections

import jax.numpy as np

RotParams = collections.namedtuple("Params", ["rotation"])


def get_pca_params(X):

    R = compute_projection(X)

    return np.dot(X, R), RotParams(rotation=R)


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

    return VT.T


def rot_forward_transform(X, params):
    return np.dot(X, params.rotation)


def rot_inverse_transform(X, params):
    return np.dot(X, params.rotation.T)


def rot_gradient_transform(X, params):
    return np.ones_like(X)
