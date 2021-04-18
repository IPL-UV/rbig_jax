import chex
import jax
import jax.numpy as np
import numpy as onp
import pytest
from jax import random

from rbig_jax.transforms.reshape import Squeeze, _get_new_shapes

seed = 123
rng = onp.random.RandomState(123)

KEY = jax.random.PRNGKey(seed)


@pytest.mark.parametrize("filters", [(2, 2), (4, 4), (7, 7), (14, 14), (28, 28)])
def test_reshape_shape(filters):

    n_dims = (1, 28, 28, 1)
    new_shape = _get_new_shapes(28, 28, 1, filters)
    params_rng, data_rng = jax.random.split(KEY, 2)

    x = jax.random.uniform(data_rng, shape=n_dims)

    # create layer
    init_func = Squeeze(filter_shape=filters, collapse=None)

    # create layer
    params, forward_f, inverse_f = init_func(rng=params_rng, shape=n_dims,)

    # forward transformation
    z, log_abs_det = forward_f(params, x)

    # checks
    chex.assert_equal_shape([z, log_abs_det])
    chex.assert_rank(z, 4)
    chex.assert_equal(z.shape[1:], new_shape)

    # inverse transformation
    x_approx, log_abs_det = inverse_f(params, z)

    # checks
    chex.assert_equal_shape([x_approx, x])
    chex.assert_tree_all_close(x_approx, x)


@pytest.mark.parametrize("filters", [(2, 2), (4, 4), (7, 7), (14, 14), (28, 28)])
def test_reshape_shape_forward(filters):

    n_dims = (1, 28, 28, 1)
    new_shape = _get_new_shapes(28, 28, 1, filters)
    params_rng, data_rng = jax.random.split(KEY, 2)

    x = jax.random.uniform(data_rng, shape=n_dims)

    # create layer
    init_func = Squeeze(filter_shape=filters, collapse=None, return_outputs=True)

    # create layer
    z_, params, forward_f, inverse_f = init_func(rng=params_rng, shape=n_dims, inputs=x)

    # forward transformation
    z, log_abs_det = forward_f(params, x)

    # checks
    chex.assert_tree_all_close(z, z_)
    chex.assert_equal_shape([z, log_abs_det, z_])
    chex.assert_rank(z, 4)
    chex.assert_equal(z.shape[1:], new_shape)

    # inverse transformation
    x_approx, log_abs_det = inverse_f(params, z)

    # checks
    chex.assert_equal_shape([x_approx, x])
    chex.assert_tree_all_close(x_approx, x)


@pytest.mark.parametrize("filters", [(2, 2), (4, 4), (7, 7), (14, 14), (28, 28)])
@pytest.mark.parametrize(
    "collapse", ["spatial", "channels", "features", "width", "height"]
)
def test_reshape_shape_collapse(filters, collapse):

    n_dims = (1, 28, 28, 1)
    params_rng, data_rng = jax.random.split(KEY, 2)

    x = jax.random.uniform(data_rng, shape=n_dims)

    # create layer
    init_func = Squeeze(filter_shape=filters, collapse=collapse)

    # create layer
    params, forward_f, inverse_f = init_func(rng=params_rng, shape=n_dims,)

    # forward transformation
    z, log_abs_det = forward_f(params, x)

    # checks
    chex.assert_equal_shape([z, log_abs_det])
    chex.assert_rank(z, 2)

    # inverse transformation
    x_approx, log_abs_det = inverse_f(params, z)

    # checks
    chex.assert_equal_shape([x_approx, x])
    chex.assert_tree_all_close(x_approx, x)
