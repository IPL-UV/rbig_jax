import chex
import jax
import jax.numpy as np
import numpy as onp
import objax
import pytest

from rbig_jax.transforms.parametric.svd import SVD

seed = 123
rng = onp.random.RandomState(123)
generator = objax.random.Generator(123)

KEY = jax.random.PRNGKey(seed)


# @pytest.mark.parametrize("n_samples", [1, 10])
# @pytest.mark.parametrize("n_features", [1, 10, 100])
# @pytest.mark.parametrize("n_reflections", [2, 4, 10])
# def test_householder_bijector_shape(n_samples, n_features, n_reflections):

#     params_rng, data_rng = jax.random.split(KEY, 2)

#     x = jax.random.normal(data_rng, shape=(n_samples, n_features))

#     init_func = SVD(n_reflections=n_reflections)

#     # create layer
#     params, forward_f, inverse_f = init_func(rng=params_rng, n_features=n_features)

#     # forward transformation
#     z, log_abs_det = forward_f(params, x)

#     # checks
#     chex.assert_equal_shape([z, x])
#     chex.assert_shape(log_abs_det, (n_samples,))

#     # inverse transformation
#     x_approx, log_abs_det = inverse_f(params, z)

#     # checks
#     chex.assert_equal_shape([x_approx, x])


@pytest.mark.parametrize("n_samples", [1, 10])
@pytest.mark.parametrize("n_features", [1, 10, 100])
@pytest.mark.parametrize("n_reflections", [2, 4, 10])
def test_householder_bijector_approx(n_samples, n_features, n_reflections):

    params_rng, data_rng = jax.random.split(KEY, 2)

    x = jax.random.normal(data_rng, shape=(n_samples, n_features))

    init_func = SVD(n_reflections=n_reflections)

    # create layer
    params, forward_f, inverse_f = init_func(rng=params_rng, n_features=n_features)

    # forward transformation
    z, _ = forward_f(params, x)

    # inverse transformation
    x_approx, _ = inverse_f(params, z)

    # checks
    chex.assert_tree_all_close(x, x_approx, atol=1e-5)
