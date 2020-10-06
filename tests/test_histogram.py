import jax.numpy as np
import numpy as onp
import chex
from rbig_jax.transforms.histogram import get_hist_params

rng = onp.random.RandomState(123)


def test_hist_params_transform():

    X = rng.randn(100)

    X_u, _, params = get_hist_params(X, support_extension=10, precision=50, alpha=1e-5)

    X_u_trans = np.interp(X, params.support, params.quantiles)

    chex.assert_tree_all_close(X_u, X_u_trans)


def test_hist_params_inv_transform():

    X = rng.randn(100)

    X_u, _, params = get_hist_params(X, support_extension=10, precision=50, alpha=1e-5)

    X_approx = np.interp(X_u, params.quantiles, params.support)

    onp.testing.assert_array_almost_equal(X, onp.array(X_approx))
