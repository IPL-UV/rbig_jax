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
from rbig_jax.transforms.mixture import MixtureGaussianCDF, MixtureLogisticCDF

seed = 123
rng = onp.random.RandomState(123)
generator = objax.random.Generator(123)


@pytest.mark.parametrize("n_features", [1, 3,])
@pytest.mark.parametrize("n_components", [1, 1,])
def test_composite_shape(n_features, n_components):

    x = objax.random.normal((n_features,), generator=generator)

    # create layer
    transform = CompositeTransform(
        [MixtureGaussianCDF(n_features, n_components), Logit()]
    )

    # forward transformation
    z, log_abs_det = transform(x)

    # checks
    chex.assert_equal_shape([z, x])
    chex.assert_equal_shape([log_abs_det, x])

    # forward transformation
    z = transform.transform(x)

    # checks
    chex.assert_equal_shape([z, x])

    # inverse transformation
    x_approx = transform.inverse(z)

    # checks
    chex.assert_equal_shape([x_approx, x])


@pytest.mark.parametrize("n_features", [1, 3,])
@pytest.mark.parametrize("n_components", [1, 1,])
def test_composite_vmap_shape(n_features, n_components):

    n_samples = 10
    x = objax.random.normal((n_samples, n_features,), generator=generator)

    # create layer
    transform = CompositeTransform(
        [MixtureGaussianCDF(n_features, n_components), Logit()]
    )

    # forward transformation
    v_net = objax.Vectorize(transform, batch_axis=(0,))
    jit_net = objax.Jit(v_net)
    z, log_abs_det = jit_net(x)

    # checks
    chex.assert_equal_shape([z, x])
    chex.assert_equal_shape([log_abs_det, x])

    # forward transformation
    z = jax.vmap(transform.transform)(x)

    # checks
    chex.assert_equal_shape([z, x])

    # inverse transformation
    x_approx = jax.vmap(transform.inverse)(z)

    # checks
    chex.assert_equal_shape([x_approx, x])
