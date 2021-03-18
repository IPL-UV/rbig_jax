import chex
import jax
import jax.numpy as np
import numpy as onp
import objax
import pytest
from jax import random

from rbig_jax.transforms.logit import Logit

seed = 123
rng = onp.random.RandomState(123)

KEY = jax.random.PRNGKey(seed)


# def test_hist_params_transform():

#     X_u = rng.uniform(100)

#     model = Logit()

#     X_g = model(X_u)

#     X_approx = model.inverse(X_g)

#     chex.assert_tree_all_close(X_u, X_approx)


@pytest.mark.parametrize("n_features", [1, 3, 10])
@pytest.mark.parametrize("n_samples", [1, 3, 10])
def test_logit_shape(n_samples, n_features):

    params_rng, data_rng = jax.random.split(KEY, 2)

    x = jax.random.uniform(data_rng, shape=(n_samples, n_features))

    # create layer
    init_func = Logit(eps=1e-5, temperature=1)

    # create layer
    params, forward_f, inverse_f = init_func(rng=params_rng, n_features=n_features,)

    # forward transformation
    z, log_abs_det = forward_f(params, x)

    # checks
    chex.assert_equal_shape([z, x])
    chex.assert_shape(log_abs_det, (n_samples,))

    # inverse transformation
    x_approx, log_abs_det = inverse_f(params, z)

    # checks
    chex.assert_equal_shape([x_approx, x])


@pytest.mark.parametrize("n_features", [1, 3, 10])
@pytest.mark.parametrize("n_samples", [1, 3, 10])
def test_logit_approx(n_samples, n_features):

    params_rng, data_rng = jax.random.split(KEY, 2)

    x = jax.random.uniform(data_rng, shape=(n_samples, n_features))

    # create layer
    init_func = Logit(eps=1e-5, temperature=1)

    # create layer
    params, forward_f, inverse_f = init_func(rng=params_rng, n_features=n_features,)

    # forward transformation
    z, _ = forward_f(params, x)

    # inverse transformation
    x_approx, _ = inverse_f(params, z)

    # checks
    chex.assert_tree_all_close(x, x_approx, atol=1e-3)
