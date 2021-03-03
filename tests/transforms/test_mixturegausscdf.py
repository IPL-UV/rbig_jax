import chex
import jax
import jax.numpy as np
import numpy as onp
import objax
import pytest
from jax import random

from rbig_jax.transforms.mixture import MixtureGaussianCDF, MixtureLogisticCDF

seed = 123
rng = onp.random.RandomState(123)
generator = objax.random.Generator(123)

# TODO: test naive mixture gaussian shape
# TODO: test naive mixture gaussian approximation
# TODO: test naive mixture gaussian vmap shape
# TODO: test naive mixture gaussian vmap approximation


@pytest.mark.parametrize("n_samples", [1, 5, 10])
@pytest.mark.parametrize("n_features", [1, 5, 10])
@pytest.mark.parametrize("n_components", [1, 10])
def test_mixturegausscdf_bijector_shape(n_samples, n_features, n_components):

    x = objax.random.normal((n_samples, n_features,), generator=generator)

    # create layer
    model = MixtureGaussianCDF(n_features, n_components)

    # forward transformation
    z, log_abs_det = model(x)

    # checks
    chex.assert_equal_shape([z, x])
    chex.assert_shape(log_abs_det, (n_samples,))

    # forward transformation
    x_approx = model.inverse(z)

    # checks
    chex.assert_equal_shape([x, x_approx])


@pytest.mark.parametrize("n_samples", [1, 5, 10])
@pytest.mark.parametrize("n_features", [1, 5, 10])
@pytest.mark.parametrize("n_components", [1, 10])
def test_mixturegausscdf_bijector_approx(n_samples, n_features, n_components):

    x = objax.random.normal((n_samples, n_features,), generator=generator)

    # create layer
    model = MixtureGaussianCDF(n_features, n_components)

    # forward transformation
    z, _ = model(x)

    # forward transformation
    x_approx = model.inverse(z)

    # checks
    chex.assert_tree_all_close(x, x_approx, rtol=1e-3)
