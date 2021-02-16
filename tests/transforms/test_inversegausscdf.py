import chex
import jax
import jax.numpy as np
import numpy as onp
import objax
import pytest
from jax import random

from rbig_jax.transforms.inversecdf import InverseGaussCDF

seed = 123
rng = onp.random.RandomState(123)
generator = objax.random.Generator(123)


# def test_hist_params_transform():

#     X_u = rng.uniform(100)

#     model = Logit()

#     X_g = model(X_u)

#     X_approx = model.inverse(X_g)

#     chex.assert_tree_all_close(X_u, X_approx)


@pytest.mark.parametrize("n_features", [1, 3, 10])
def test_logit_shape(n_features):

    x = objax.random.normal((n_features,), generator=generator)

    # create layer
    model = InverseGaussCDF()

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


@pytest.mark.parametrize("n_features", [1, 3, 10])
def test_logit_approx(n_features):

    x = objax.random.uniform((n_features,), generator=generator)

    # clip elements
    eps = 1e-6
    x = np.clip(x, a_min=eps, a_max=1 - eps)

    # create layer
    model = InverseGaussCDF(eps=eps)

    # forward transformation
    z, _ = model(x)

    # inverse transformation
    x_approx = model.inverse(z)

    # checks
    chex.assert_tree_all_close(x, x_approx, atol=1e-3)
