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


@pytest.mark.parametrize("n_features", [1, 3, 10])
@pytest.mark.parametrize("n_components", [1, 3, 10, 100])
def test_mixturegausscdf_shape(n_features, n_components):

    x = objax.random.normal((n_features,), generator=generator)

    # create layer
    model = MixtureGaussianCDF(n_features, n_components)

    # forward transformation
    z, log_abs_det = model(x)

    # checks
    chex.assert_equal_shape([z, x])
    chex.assert_equal_shape([log_abs_det, x])


@pytest.mark.parametrize("n_features", [1, 5, 10])
@pytest.mark.parametrize("n_components", [1, 10])
def test_mixturegausscdf_vmap_shape(n_features, n_components):

    n_samples = 10
    x = objax.random.normal((n_samples, n_features,), generator=generator)

    # create layer
    model = MixtureGaussianCDF(n_features, n_components)

    # forward transformation
    z, log_abs_det = objax.Vectorize(model, batch_axis=(0,))(x)

    # checks
    chex.assert_equal_shape([z, x])
    chex.assert_equal_shape([log_abs_det, x])


@pytest.mark.parametrize("n_features", [1, 3, 10])
@pytest.mark.parametrize("n_components", [1, 3, 10, 100])
def test_mixturegausscdf_shape_inverse(n_features, n_components):

    x = objax.random.normal((n_features,), generator=generator)

    # create layer
    model = MixtureGaussianCDF(n_features, n_components)

    # forward transformation
    z, _ = model(x)

    # inverse transformation
    x_approx = model.inverse(z)

    # checks
    chex.assert_equal_shape([x_approx, x])


@pytest.mark.parametrize("n_features", [1, 5, 10])
@pytest.mark.parametrize("n_components", [1, 10])
def test_mixturegausscdf_vmap_shape_inverse(n_features, n_components):

    n_samples = 10
    x = objax.random.normal((n_samples, n_features,), generator=generator)

    # create layer
    model = MixtureGaussianCDF(n_features, n_components)

    # forward transformation
    z, log_abs_det = jax.vmap(model)(x)

    # inverse transformation
    x_approx = jax.vmap(model.inverse)(z)

    # checks
    chex.assert_equal_shape([x_approx, x])


@pytest.mark.parametrize("n_features", [1, 3, 10])
@pytest.mark.parametrize("n_components", [1, 3, 10, 100])
def test_mixturegausscdf_approx(n_features, n_components):

    x = objax.random.normal((n_features,), generator=generator)

    # create layer
    model = MixtureGaussianCDF(n_features, n_components)

    # forward transformation
    z, _ = model(x)

    # inverse transformation
    x_approx = model.inverse(z)

    # checks
    chex.assert_tree_all_close(x, x_approx, atol=1e-3)


@pytest.mark.parametrize("n_features", [1, 5, 10])
@pytest.mark.parametrize("n_components", [1, 5])
def test_mixturegausscdf_vmap_approx(n_features, n_components):

    n_samples = 10
    x = objax.random.normal((n_samples, n_features,), generator=generator)

    # create layer
    model = MixtureGaussianCDF(n_features, n_components)

    # forward transformation
    z, _ = jax.vmap(model)(x)

    # inverse transformation
    x_approx = jax.vmap(model.inverse)(z)

    # checks
    chex.assert_tree_all_close(x, x_approx, rtol=1e-3)


@pytest.mark.parametrize("n_features", [1, 3, 10])
@pytest.mark.parametrize("n_components", [1, 3, 10, 100])
def test_mixturelogisticcdf_shape(n_features, n_components):

    x = objax.random.normal((n_features,), generator=generator)

    # create layer
    model = MixtureLogisticCDF(n_features, n_components)

    # forward transformation
    z, log_abs_det = model(x)

    # checks
    chex.assert_equal_shape([z, x])
    chex.assert_equal_shape([log_abs_det, x])


@pytest.mark.parametrize("n_features", [1, 5, 10])
@pytest.mark.parametrize("n_components", [1, 10])
def test_mixturelogisticcdf_vmap_shape(n_features, n_components):

    n_samples = 10
    x = objax.random.normal((n_samples, n_features,), generator=generator)

    # create layer
    model = MixtureLogisticCDF(n_features, n_components)

    # forward transformation
    z, log_abs_det = jax.vmap(model)(x)

    # checks
    chex.assert_equal_shape([z, x])
    chex.assert_equal_shape([log_abs_det, x])


@pytest.mark.parametrize("n_features", [1, 3, 10])
@pytest.mark.parametrize("n_components", [1, 3, 10, 100])
def test_mixturelogisticcdf_shape_inverse(n_features, n_components):

    x = objax.random.normal((n_features,), generator=generator)

    # create layer
    model = MixtureLogisticCDF(n_features, n_components)

    # forward transformation
    z, _ = model(x)

    # inverse transformation
    x_approx = model.inverse(z)

    # checks
    chex.assert_equal_shape([x_approx, x])


@pytest.mark.parametrize("n_features", [1, 5, 10])
@pytest.mark.parametrize("n_components", [1, 10])
def test_mixturelogisticcdf_vmap_shape_inverse(n_features, n_components):

    n_samples = 10
    x = objax.random.normal((n_samples, n_features,), generator=generator)

    # create layer
    model = MixtureLogisticCDF(n_features, n_components)

    # forward transformation
    z, log_abs_det = jax.vmap(model)(x)

    # inverse transformation
    x_approx = jax.vmap(model.inverse)(z)

    # checks
    chex.assert_equal_shape([x_approx, x])


@pytest.mark.parametrize("n_features", [1, 3, 10])
@pytest.mark.parametrize("n_components", [1, 3, 10, 100])
def test_mixturelogisticcdf_approx(n_features, n_components):

    x = objax.random.normal((n_features,), generator=generator)

    # create layer
    model = MixtureLogisticCDF(n_features, n_components)

    # forward transformation
    z, _ = model(x)

    # inverse transformation
    x_approx = model.inverse(z)

    # checks
    chex.assert_tree_all_close(x, x_approx, atol=1e-3)


@pytest.mark.parametrize("n_features", [1, 5, 10])
@pytest.mark.parametrize("n_components", [1, 5])
def test_mixturelogisticcdf_vmap_approx(n_features, n_components):

    n_samples = 10
    x = objax.random.normal((n_samples, n_features,), generator=generator)

    # create layer
    model = MixtureLogisticCDF(n_features, n_components)

    # forward transformation
    z, _ = jax.vmap(model)(x)

    # inverse transformation
    x_approx = jax.vmap(model.inverse)(z)

    # checks
    chex.assert_tree_all_close(x, x_approx, rtol=1e-3)
