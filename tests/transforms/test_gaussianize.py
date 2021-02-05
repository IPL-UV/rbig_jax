import chex
import jax.numpy as np
import numpy as onp
import pytest

from rbig_jax.transforms.gaussianize import (forward_gaussianize_transform,
                                             get_gauss_params_hist,
                                             get_gauss_params_kde,
                                             inverse_gaussianize_transform)
from rbig_jax.transforms.histogram import get_hist_params
from rbig_jax.transforms.kde import get_kde_params

# TODO: Think about a test for the uniformization gradient

rng = onp.random.RandomState(123)
X = rng.randn(1_000)


def test_hist_gaussianize_shape():

    X_g, params = get_gauss_params_hist(
        X, support_extension=10, precision=1000, alpha=1e-5
    )

    X_g_trans = forward_gaussianize_transform(X, params)

    chex.assert_equal_shape([X, X_g, X_g_trans])


def test_hist_gaussianize_params():

    X_g, params = get_gauss_params_hist(
        X, support_extension=10, precision=1000, alpha=1e-5
    )

    _, params_hist = get_hist_params(
        X, support_extension=10, precision=1000, alpha=1e-5
    )

    chex.assert_tree_all_close(params, params_hist)


def test_hist_gaussianize_bounds():

    X_g, params = get_gauss_params_hist(
        X, support_extension=10, precision=1000, alpha=1e-5
    )

    assert X_g.min() > -np.inf
    assert X_g.max() < np.inf

    X_g_trans = forward_gaussianize_transform(X, params)

    assert X_g_trans.min() > -np.inf
    assert X_g_trans.max() < np.inf


def test_hist_gaussianize():

    X_g, params = get_gauss_params_hist(
        X, support_extension=10, precision=1000, alpha=1e-5
    )

    X_g_trans = forward_gaussianize_transform(X, params)

    chex.assert_tree_all_close(X_g, X_g_trans)


def test_hist_gaussianize_inverse_shape():

    X_g, params = get_gauss_params_hist(
        X, support_extension=10, precision=1000, alpha=1e-5
    )

    X_approx = inverse_gaussianize_transform(X_g, params)

    chex.assert_equal_shape([X, X_g, X_approx])


@pytest.mark.parametrize(
    "X,atol",
    [(rng.randn(100), 1e-6), (rng.randn(1_000), 1e-5), (rng.randn(10_000), 1e-4)],
)
def test_hist_gaussianize_inverse(X, atol):

    X_g, params = get_gauss_params_hist(
        X, support_extension=10, precision=1000, alpha=1e-5
    )

    X_approx = inverse_gaussianize_transform(X_g, params)

    chex.assert_tree_all_close(X, X_approx, atol=atol)


def test_kde_gaussianize_shape():

    X_g, params = get_gauss_params_kde(X, support_extension=10, precision=1000)

    X_g_trans = forward_gaussianize_transform(X, params)

    chex.assert_equal_shape([X, X_g, X_g_trans])


def test_kde_gaussianize_params():

    X_g, params = get_gauss_params_kde(X, support_extension=10, precision=1000)

    _, params_hist = get_kde_params(X, support_extension=10, precision=1000,)

    chex.assert_tree_all_close(params, params_hist)


def test_kde_gaussianize_bounds():

    X_g, params = get_gauss_params_kde(X, support_extension=10, precision=1000)

    assert X_g.min() > -np.inf
    assert X_g.max() < np.inf

    X_g_trans = forward_gaussianize_transform(X, params)

    assert X_g_trans.min() > -np.inf
    assert X_g_trans.max() < np.inf


def test_kde_gaussianize():

    X_g, params = get_gauss_params_kde(X, support_extension=10, precision=1000)

    X_g_trans = forward_gaussianize_transform(X, params)

    chex.assert_tree_all_close(X_g, X_g_trans)


def test_kde_gaussianize_inverse_shape():

    X_g, params = get_gauss_params_kde(X, support_extension=10, precision=1000)

    X_approx = inverse_gaussianize_transform(X_g, params)

    chex.assert_equal_shape([X, X_g, X_approx])


@pytest.mark.parametrize(
    "X,atol",
    [(rng.randn(100), 1e-6), (rng.randn(1_000), 1e-5), (rng.randn(10_000), 1e-4)],
)
def test_kde_gaussianize_inverse(X, atol):

    X_g, params = get_gauss_params_kde(X, support_extension=20, precision=1000)

    X_approx = inverse_gaussianize_transform(X_g, params)

    chex.assert_tree_all_close(X, X_approx, atol=atol)
