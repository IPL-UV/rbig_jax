import chex
import jax
import jax.numpy as np
import numpy as onp
import pytest

from rbig_jax.transforms.parametric.conv import Conv1x1

seed = 123
rng = onp.random.RandomState(123)
KEY = jax.random.PRNGKey(seed)


@pytest.mark.parametrize("n_channels", [1, 3, 12])
@pytest.mark.parametrize("hw", [(1, 1), (5, 5), (12, 12)])
@pytest.mark.parametrize("n_samples", [1, 5, 10])
def test_conv1x1_shape(n_channels, hw, n_samples):

    params_rng, data_rng = jax.random.split(KEY, 2)

    x = jax.random.normal(data_rng, shape=(n_samples, hw[0], hw[1], n_channels))

    # create layer
    init_func = Conv1x1(n_channels=n_channels)

    # create layer
    params, forward_f, inverse_f = init_func(rng=params_rng, n_features=n_channels)

    # forward transformation
    z, log_abs_det = forward_f(params, x)

    # print(z.shape, log_abs_det.shape)

    # checks
    chex.assert_equal_shape([z, x])
    chex.assert_shape(np.atleast_1d(log_abs_det), (n_samples,))

    # inverse transformation
    x_approx, log_abs_det = inverse_f(params, z)

    # checks
    chex.assert_equal_shape([x_approx, x])
    chex.assert_shape(np.atleast_1d(log_abs_det), (n_samples,))


@pytest.mark.parametrize("n_channels", [1, 3, 12])
@pytest.mark.parametrize("hw", [(1, 1), (5, 5), (12, 12)])
@pytest.mark.parametrize("n_samples", [1, 5, 10])
def test_conv1x1_approx(n_channels, hw, n_samples):

    params_rng, data_rng = jax.random.split(KEY, 2)

    x = jax.random.normal(data_rng, shape=(n_samples, hw[0], hw[1], n_channels))

    # create layer
    init_func = Conv1x1(n_channels=n_channels)

    # create layer
    params, forward_f, inverse_f = init_func(rng=params_rng, n_features=n_channels)

    # forward transformation
    z, log_abs_det = forward_f(params, x)

    # inverse transformation
    x_approx, log_abs_det = inverse_f(params, z)

    # checks
    chex.assert_tree_all_close(x, x_approx, atol=1e-5)
