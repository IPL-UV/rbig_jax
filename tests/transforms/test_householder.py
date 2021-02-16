import chex
import jax
import jax.numpy as np
import numpy as onp
import objax
import pytest

from rbig_jax.transforms.linear import HouseHolder

seed = 123
rng = onp.random.RandomState(123)
generator = objax.random.Generator(123)


@pytest.mark.parametrize("n_features", [1, 10, 100])
@pytest.mark.parametrize("n_reflections", [1, 3, 10])
def test_householder_shape(n_features, n_reflections):

    x = objax.random.normal((n_features,), generator=generator)

    # create layer
    model = HouseHolder(
        n_features=n_features, n_reflections=n_reflections, generator=generator
    )

    # forward transformation
    z, log_abs_det = model(x)

    # checks
    chex.assert_equal_shape([z, x])
    chex.assert_equal_shape([log_abs_det, x])

    # forward transformation
    z = model.transform(x)

    # checks
    chex.assert_equal_shape([z, x])

    # inverse transformation
    x_approx = model.inverse(z)

    # checks
    chex.assert_equal_shape([x_approx, x])


@pytest.mark.parametrize("n_features", [1, 10, 100])
@pytest.mark.parametrize("n_reflections", [1, 3, 10])
def test_householder_vmap_shape(n_features, n_reflections):

    n_samples = 10
    x = objax.random.normal((n_samples, n_features,), generator=generator)

    # create layer
    model = HouseHolder(
        n_features=n_features, n_reflections=n_reflections, generator=generator
    )

    # forward transformation
    z, log_abs_det = objax.Vectorize(model, vc=model.vars())(x)

    # checks
    chex.assert_equal_shape([z, x])
    chex.assert_equal_shape([log_abs_det, x])

    # forward transformation
    z = objax.Vectorize(model.transform, vc=model.vars())(x)

    # checks
    chex.assert_equal_shape([z, x])

    # inverse transformation
    x_approx = objax.Vectorize(model.inverse, vc=model.vars())(z)

    # checks
    chex.assert_equal_shape([x_approx, x])


@pytest.mark.parametrize("n_features", [1, 10, 100])
@pytest.mark.parametrize("n_reflections", [1, 3, 10])
def test_householder_approx(n_features, n_reflections):

    n_samples = 50
    x = objax.random.normal((n_samples, n_features,), generator=generator)

    # create layer
    model = HouseHolder(
        n_features=n_features, n_reflections=n_reflections, generator=generator
    )
    # forward transformation
    z, log_abs_det = objax.Vectorize(model, vc=model.vars())(x)

    # checks
    chex.assert_equal_shape([z, x])
    chex.assert_equal_shape([log_abs_det, x])

    # inverse transformation
    x_approx = objax.Vectorize(model.inverse, vc=model.vars())(z)

    # checks
    chex.assert_tree_all_close(x, x_approx, atol=1e-5)
