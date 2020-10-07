import jax.numpy as np
import numpy as onp
import chex
from rbig_jax.transforms.rotation import (
    get_pca_params,
    rot_forward_transform,
    rot_inverse_transform,
    rot_gradient_transform,
)

rng = onp.random.RandomState(123)
X = rng.randn(100, 3)


def test_pca_forward_shape():

    X_r, _ = get_pca_params(X)

    chex.assert_equal_shape([X_r, X])


def test_pca_forward():

    X_r, params = get_pca_params(X)

    X_r_trans = rot_forward_transform(X, params)

    chex.assert_tree_all_close(X_r, X_r_trans)


def test_pca_inverse_shape():

    _, params = get_pca_params(X)

    X_approx = rot_inverse_transform(X, params)

    chex.assert_equal_shape([X, X_approx])


def test_pca_inverse():

    X_r, params = get_pca_params(X)

    X_approx = rot_inverse_transform(X_r, params)

    chex.assert_tree_all_close(X, X_approx, atol=1e-5)


def test_pca_gradient_shape():

    X_r, params = get_pca_params(X)

    X_grad = rot_gradient_transform(X_r, params)

    chex.assert_equal_shape([X, X_grad])


def test_pca_gradient():

    X_r, params = get_pca_params(X)

    X_grad = rot_gradient_transform(X_r, params)

    # calculate determinant manually
    real_grad = np.linalg.det(params.rotation)
    real_grad = real_grad * np.ones_like(X_grad, dtype=np.float64)

    chex.assert_tree_all_close(real_grad, X_grad, atol=1e-5)
