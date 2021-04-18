import chex
import jax
import jax.numpy as np
import numpy as onp
import pytest

from rbig_jax.transforms.base import CompositeTransform
from rbig_jax.transforms.logit import Logit
from rbig_jax.transforms.parametric.householder import HouseHolder
from rbig_jax.transforms.parametric.mixture import MixtureLogisticCDF

seed = 123
rng = onp.random.RandomState(123)

KEY = jax.random.PRNGKey(seed)


@pytest.mark.parametrize("n_features", [1, 3,])
@pytest.mark.parametrize("n_samples", [1, 5, 10])
def test_composite_shape(n_samples, n_features):

    params_rng, data_rng = jax.random.split(KEY, 2)

    x = jax.random.normal(data_rng, shape=(n_samples, n_features))
    # create layer
    init_func = CompositeTransform(
        [MixtureLogisticCDF(n_components=5), Logit(), HouseHolder(n_reflections=2),]
    )

    # create layer
    params, forward_f, inverse_f = init_func(rng=params_rng, n_features=n_features)

    # forward transformation
    z, log_abs_det = forward_f(params, x)

    # checks
    chex.assert_equal_shape([z, x])
    chex.assert_shape(log_abs_det, (n_samples,))

    # forward transformation
    x_approx, log_abs_det = inverse_f(params, z)

    # checks
    chex.assert_equal_shape([x_approx, x])


@pytest.mark.parametrize("n_features", [1, 3,])
@pytest.mark.parametrize("n_components", [1, 5,])
@pytest.mark.parametrize("n_samples", [1, 5, 10])
def test_composite_shape_jitted(n_samples, n_features, n_components):

    params_rng, data_rng = jax.random.split(KEY, 2)

    x = jax.random.normal(data_rng, shape=(n_samples, n_features))
    # create layer
    init_func = CompositeTransform(
        [MixtureLogisticCDF(n_components=5), Logit(), HouseHolder(n_reflections=2),]
    )

    # create layer
    params, forward_f, inverse_f = init_func(rng=params_rng, n_features=n_features)

    # forward transformation
    z, log_abs_det = forward_f(params, x)
    transform_f_jitted = jax.jit(forward_f)
    z_, log_abs_det_ = transform_f_jitted(params, x)

    # checks
    chex.assert_tree_all_close(z_, z, rtol=1e-4)
    chex.assert_tree_all_close(log_abs_det_, log_abs_det, rtol=1e-4)

    # forward transformation
    x_approx, ilog_abs_det = inverse_f(params, z)
    inverse_f_jitted = jax.jit(inverse_f)
    x_approx_, ilog_abs_det_ = inverse_f_jitted(params, z_)

    # checks
    chex.assert_tree_all_close(x_approx, x_approx_, rtol=1e-4)
    chex.assert_tree_all_close(ilog_abs_det, ilog_abs_det_, rtol=1e-4)
