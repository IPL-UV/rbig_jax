import chex
import jax
import jax.numpy as np
import numpy as onp
import objax
import pytest
from jax import random
from jax.interpreters.batching import batch

from rbig_jax.transforms.base import CompositeTransform
from rbig_jax.transforms.logit import Logit
from rbig_jax.transforms.mixture import MixtureGaussianCDF

seed = 123
rng = onp.random.RandomState(123)
generator = objax.random.Generator(123)


@pytest.mark.parametrize("n_features", [1, 3,])
@pytest.mark.parametrize("n_components", [1, 5,])
@pytest.mark.parametrize("n_samples", [1, 5, 10])
def test_composite_shape(n_samples, n_features, n_components):

    x = objax.random.normal((n_samples, n_features,), generator=generator)

    # create layer
    transform = CompositeTransform(
        [MixtureGaussianCDF(n_features, n_components), Logit()]
    )

    # forward transformation
    z, log_abs_det = transform(x)

    # checks
    chex.assert_equal_shape([z, x])
    chex.assert_shape(log_abs_det, (n_samples,))

    # forward transformation
    z = transform.transform(x)

    # checks
    chex.assert_equal_shape([z, x])

    # inverse transformation
    x_approx = transform.inverse(z)

    # checks
    chex.assert_equal_shape([x_approx, x])


@pytest.mark.parametrize("n_features", [1, 3,])
@pytest.mark.parametrize("n_components", [1, 5,])
@pytest.mark.parametrize("n_samples", [1, 5, 10])
def test_composite_shape_jitted(n_samples, n_features, n_components):

    x = objax.random.normal((n_samples, n_features,), generator=generator)

    # create layer
    transform = CompositeTransform(
        [MixtureGaussianCDF(n_features, n_components), Logit()]
    )

    # forward transformation
    jit_net = objax.Jit(transform, transform.vars())
    z, log_abs_det = jit_net(x)

    # checks
    chex.assert_equal_shape([z, x])
    chex.assert_shape(log_abs_det, (n_samples,))

    # forward transformation
    z = transform.transform(x)

    # checks
    chex.assert_equal_shape([z, x])

    # inverse transformation
    x_approx = transform.inverse(z)

    # checks
    chex.assert_equal_shape([x_approx, x])
